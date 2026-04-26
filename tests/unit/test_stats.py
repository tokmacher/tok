"""Tests for tok.stats — savings ledger load/save/merge."""

import pytest

from tok.stats import SavingsTracker


@pytest.fixture
def tracker(tmp_path):
    """Create a tracker with temp files."""
    return SavingsTracker(
        savings_file=str(tmp_path / "tok_savings.tok"),
        ledger_path=tmp_path / "global_savings.tok",
    )


class TestSavingsTracker:
    def test_load_empty_stats(self, tracker) -> None:
        stats = tracker.load_stats()
        assert "session_start" in stats
        assert stats["models"] == {}

    def test_save_and_load_roundtrip(self, tracker) -> None:
        stats = {
            "session_start": "2024-01-01T00:00:00Z",
            "models": {
                "claude-sonnet-4": {
                    "calls": 5,
                    "actual_input_tokens": 1000,
                    "actual_output_tokens": 500,
                    "cache_read_tokens": 100,
                    "cache_write_tokens": 50,
                    "input_saved_tokens": 200,
                    "output_saved_tokens": 100,
                    "actual_cost_usd": 0.01,
                    "baseline_cost_usd": 0.02,
                    "behavior_signals": {"repeat_file_read": 2},
                }
            },
        }
        tracker.save_stats(stats)
        loaded = tracker.load_stats()
        assert loaded["models"]["claude-sonnet-4"]["calls"] == 5
        assert loaded["models"]["claude-sonnet-4"]["actual_input_tokens"] == 1000
        assert loaded["models"]["claude-sonnet-4"]["behavior_signals"]["repeat_file_read"] == 2

    def test_baseline_cost_excludes_caching_discount(self, tracker) -> None:
        # Cache tokens must be charged at their actual rates in both actual and baseline
        # costs. Previously the baseline charged cache_read/write at inp_rate, inflating
        # cost savings by attributing the API's caching discount to Tok.
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=200,
            cache_read=5000,
            cache_write=500,
            input_saved=300,
            output_saved=0,
        )
        # inp=3.00, out=15.00, cr=0.30, cw=3.75 per MTok for claude-sonnet-4
        M = 1_000_000
        expected_actual = (1000 * 3.00 + 200 * 15.00 + 5000 * 0.30 + 500 * 3.75) / M
        expected_baseline = (1300 * 3.00 + 200 * 15.00 + 5000 * 0.30 + 500 * 3.75) / M
        stats = tracker.load_stats()
        m = stats["models"]["claude-sonnet-4"]
        assert abs(m["actual_cost_usd"] - expected_actual) < 1e-9
        assert abs(m["baseline_cost_usd"] - expected_baseline) < 1e-9
        # Token savings pct and cost savings pct should be close (within 5pp)
        summary = tracker.session_summary()
        assert summary is not None
        assert abs(summary["savings_pct"] - summary["cost_savings_pct"]) < 5.0

    def test_record_call(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=500,
            cache_read=0,
            cache_write=0,
            input_saved=200,
            output_saved=100,
            behavior_signals={"python_c_workaround": 1},
        )
        stats = tracker.load_stats()
        assert stats["models"]["claude-sonnet-4"]["calls"] == 1
        assert stats["models"]["claude-sonnet-4"]["behavior_signals"]["python_c_workaround"] == 1

    def test_session_summary_includes_quality_and_reason(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=20,
            output_saved=10,
            behavior_signals={
                "tok_fallback_activated": 1,
                "baseline_only_session": 1,
            },
        )

        summary = tracker.session_summary()

        assert summary is not None
        assert summary["session_quality"] == "degraded"
        assert summary["last_degradation_reason"] == "baseline fallback"
        assert summary["fail_open_count"] == 0
        assert summary["reacquisition_count"] == 0

    def test_session_summary_includes_recovery_and_repair_counters(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=20,
            output_saved=10,
            behavior_signals={
                "stream_recovery_started": 1,
                "stream_recovery_success_text": 1,
                "stream_recovery_success_tool_use": 1,
                "stream_recovery_fallback": 1,
                "stream_recovery_empty_success": 1,
                "stream_recovery_read_error": 1,
                "tok_bridge_tool_history_repaired": 1,
                "tok_bridge_tool_history_pairing_repaired": 1,
                "tok_bridge_invalid_tool_history_quarantined": 1,
                "tok_bridge_invalid_tool_history_blocked": 1,
                "tok_bridge_invalid_tool_history_session_reset": 1,
            },
        )

        summary = tracker.session_summary()

        assert summary is not None
        assert summary["stream_recovery_attempt_count"] == 1
        assert summary["stream_recovery_success_text_count"] == 1
        assert summary["stream_recovery_success_tool_use_count"] == 1
        assert summary["stream_recovery_fallback_count"] == 1
        assert summary["stream_recovery_empty_success_count"] == 1
        assert summary["stream_recovery_read_error_count"] == 1
        assert summary["tool_history_repaired_count"] == 1
        assert summary["tool_history_pairing_repaired_count"] == 1
        assert summary["tool_history_quarantined_count"] == 1
        assert summary["tool_history_blocked_count"] == 1
        assert summary["invalid_tool_history_session_reset_count"] == 1

    def test_session_summary_surfaces_provider_pairing_disagreement(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=20,
            output_saved=10,
            behavior_signals={
                "fail_open_retry_upstream_pairing_disagreement": 1,
                "tok_bridge_provider_pairing_risk_detected": 1,
            },
        )

        summary = tracker.session_summary()

        assert summary is not None
        assert summary["provider_pairing_disagreement_count"] == 2
        assert summary["session_quality"] == "watch"
        assert summary["last_degradation_reason"] == "request-shape incompatibility"

    def test_session_summary_surfaces_request_policy_counters(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=20,
            output_saved=10,
            behavior_signals={
                "request_policy_natural_first": 2,
                "request_policy_tool_compatible": 3,
                "request_policy_escalations": 1,
                "request_policy_deescalations": 1,
            },
        )

        summary = tracker.session_summary()

        assert summary is not None
        assert summary["request_policy_natural_first_count"] == 2
        assert summary["request_policy_tool_compatible_count"] == 3
        assert summary["request_policy_escalations_count"] == 1
        assert summary["request_policy_deescalations_count"] == 1

    def test_session_summary_surfaces_interleaving_downgrade_counters(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=20,
            output_saved=10,
            behavior_signals={
                "tok_bridge_assistant_tool_use_text_interleaving_blocked": 2,
                "preflight_block_original_payload": 1,
                "preflight_block_rewritten_payload": 1,
                "request_policy_interleaving_downgrades": 2,
            },
        )

        summary = tracker.session_summary()

        assert summary is not None
        assert summary["assistant_tool_use_text_interleaving_blocked_count"] == 2
        assert summary["preflight_block_original_payload_count"] == 1
        assert summary["preflight_block_rewritten_payload_count"] == 1
        assert summary["request_policy_interleaving_downgrades_count"] == 2
        assert summary["session_quality"] == "watch"
        assert summary["last_degradation_reason"] == "request-shape incompatibility"

    def test_session_summary_surfaces_request_policy_recovery_attribution(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=20,
            output_saved=10,
            behavior_signals={
                "request_policy_reason_stream_recovery": 2,
                "request_policy_reason_tool_recovery": 1,
                "request_policy_reason_structured_tool_loop": 1,
                "request_policy_held_by_recovery": 3,
            },
        )

        summary = tracker.session_summary()

        assert summary is not None
        assert summary["request_policy_reason_stream_recovery_count"] == 2
        assert summary["request_policy_reason_tool_recovery_count"] == 1
        assert summary["request_policy_reason_structured_tool_loop_count"] == 1
        assert summary["request_policy_held_by_recovery_count"] == 3
        assert summary["last_degradation_reason"] == "recovery holdover"

    def test_last_session_summary_defaults_missing_degradation_fields(self, tracker) -> None:
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 1\n\n@per_session_log\n  # old format\n  2026-03-18T10:00:00Z;bbb22222;5;1000;0.010000;0.025000;0.015000;700;1;0"
            "\n"
        )

        summary = tracker.last_session_summary()

        assert summary is not None
        assert summary["session_quality"] == "clean"
        assert summary["last_degradation_reason"] == ""

    def test_merge_session_to_ledger(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=500,
            cache_read=0,
            cache_write=0,
            input_saved=200,
            output_saved=100,
        )
        tracker.merge_session_to_ledger()
        assert tracker.ledger_path.exists()
        content = tracker.ledger_path.read_text()
        assert "sessions: 1" in content

    def test_merge_session_to_ledger_persists_malformed_tok_subtypes(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=5,
            behavior_signals={
                "malformed_tok_response": 1,
                "malformed_tok_non_inverted_msg": 1,
                "malformed_tok_bad_header": 1,
            },
        )

        tracker.merge_session_to_ledger()

        content = tracker.ledger_path.read_text()
        assert "malformed_tok_response: 1" in content
        assert "malformed_tok_non_inverted_msg: 1" in content
        assert "malformed_tok_bad_header: 1" in content

    def test_format_session_empty(self, tracker) -> None:
        assert tracker.format_session() is None

    def test_format_session_with_data(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=500,
            cache_read=0,
            cache_write=0,
            input_saved=200,
            output_saved=100,
        )
        result = tracker.format_session()
        assert result is not None
        assert "Current session:" in result
        assert "- calls: 1" in result
        assert "baseline tokens:" in result

    def test_format_compact_session_summary_with_data(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=500,
            cache_read=0,
            cache_write=0,
            input_saved=200,
            output_saved=100,
            behavior_signals={"tok_fallback_activated": 1},
        )
        result = tracker.format_compact_session_summary()
        assert result is not None
        assert "Saved $" in result
        assert "status=active and helping" in result
        assert "fallbacks=1" in result

    def test_format_ledger_empty(self, tracker) -> None:
        assert tracker.format_ledger() is None

    def test_recent_summary_aggregates_latest_sessions(self, tracker) -> None:
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 3\n  total_turns: 12\n  total_tokens: 2400\n  total_cost_usd: 0.024000\n  estimated_baseline_cost_usd: 0.042000\n  tokens_saved: 1200\n  cost_saved_usd: 0.018000\n  savings_pct: 42.9\n\n@per_session_log\n  2026-03-17T10:00:00Z;aaa11111;4;800;0.008000;0.014000;0.006000;400;1;0\n  2026-03-18T10:00:00Z;bbb22222;4;800;0.008000;0.016000;0.008000;500;1;0\n  2026-03-19T10:00:00Z;ccc33333;4;800;0.008000;0.012000;0.004000;300;1;0"
            "\n"
        )

        summary = tracker.recent_summary(2)

        assert summary is not None
        assert summary["sessions"] == 2
        assert summary["date_start"] == "2026-03-18T10:00:00Z"
        assert summary["date_end"] == "2026-03-19T10:00:00Z"
        assert summary["actual_tokens"] == 1600
        assert summary["baseline_tokens"] == 2400
        assert summary["tokens_saved"] == 800

    def test_since_summary_filters_by_iso_date(self, tracker) -> None:
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 3\n  total_turns: 12\n  total_tokens: 2400\n  total_cost_usd: 0.024000\n  estimated_baseline_cost_usd: 0.042000\n  tokens_saved: 1200\n  cost_saved_usd: 0.018000\n  savings_pct: 42.9\n\n@per_session_log\n  2026-03-17T10:00:00Z;aaa11111;4;800;0.008000;0.014000;0.006000;400;1;0\n  2026-03-18T10:00:00Z;bbb22222;4;800;0.008000;0.016000;0.008000;500;1;0\n  2026-03-19T10:00:00Z;ccc33333;4;800;0.008000;0.012000;0.004000;300;1;0"
            "\n"
        )

        summary = tracker.since_summary("2026-03-18")

        assert summary is not None
        assert summary["label"] == "Since 2026-03-18"
        assert summary["sessions"] == 2
        assert summary["date_start"] == "2026-03-18T10:00:00Z"

    def test_behavior_signals_aggregate(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=5,
            behavior_signals={"repeat_file_read": 2},
        )
        tracker.record_call(
            model="claude-haiku-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=5,
            behavior_signals={"repeat_file_read": 1, "repeat_search": 3},
        )
        signals = tracker.behavior_signals()
        assert signals["repeat_file_read"] == 3
        assert signals["repeat_search"] == 3

    def test_behavior_signals_include_memory_metrics(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=5,
            behavior_signals={
                "durable_promotions": 2,
                "hot_entries": 5,
                "cold_start_structured_memory": 1,
            },
        )
        signals = tracker.behavior_signals()
        assert signals["durable_promotions"] == 2
        assert signals["hot_entries"] == 5
        assert signals["cold_start_structured_memory"] == 1

    def test_behavior_summary_clean(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=5,
            behavior_signals={
                "cold_start_structured_memory": 1,
                "durable_promotions": 2,
            },
        )
        summary = tracker.behavior_summary()
        assert summary["status"] == "clean"
        assert summary["memory_lift"] == 3

    def test_behavior_summary_noisy(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=5,
            behavior_signals={
                "repeat_file_read": 2,
                "repeat_search": 1,
                "python_c_workaround": 1,
                "non_tok_response": 1,
            },
        )
        summary = tracker.behavior_summary()
        assert summary["status"] == "noisy"
        assert summary["invisible_pressure"] == 5

    def test_behavior_summary_counts_contract_drift_as_pressure(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=5,
            behavior_signals={
                "fail_open_compat_response": 1,
                "malformed_tok_response": 1,
            },
        )
        summary = tracker.behavior_summary()
        assert summary["status"] == "watch"
        assert summary["invisible_pressure"] == 2

    def test_tool_compatible_prose_does_not_trigger_drift_degradation(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=5,
            behavior_signals={
                "tool_compatible_response": 1,
            },
        )
        summary = tracker.behavior_summary()
        assert summary["status"] == "clean"
        assert summary["degradation_reason"] == ""

    def test_merge_session_to_ledger_does_not_clear_session_file(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=500,
            cache_read=0,
            cache_write=0,
            input_saved=200,
            output_saved=100,
        )
        # Verify session has data before merge
        session_before = tracker.format_session()
        assert session_before is not None
        assert "- calls: 1" in session_before

        tracker.merge_session_to_ledger()

        # Session file should still contain data after merge
        session_after = tracker.format_session()
        assert session_after is not None
        assert "- calls: 1" in session_after

        # But reset_session_stats should clear it
        tracker.reset_session_stats()
        session_after_reset = tracker.format_session()
        assert session_after_reset is None

    def test_trend_summary_reads_recent_session_log(self, tracker) -> None:
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 2\n\n@per_session_log\n  # format: date;session_id;turns;tokens;cost_usd;baseline_cost_usd;saved_usd;tokens_saved;invisible_pressure;non_tok_response\n  2026-03-17T10:00:00Z;aaa11111;5;1000;0.010000;0.020000;0.010000;500;3;1\n  2026-03-18T10:00:00Z;bbb22222;5;1000;0.010000;0.025000;0.015000;700;1;0"
            "\n"
        )

        trend = tracker.trend_summary()

        assert trend["sessions_considered"] == 2
        assert trend["direction"] == "improving"
        assert trend["avg_invisible_pressure"] == 2.0

    def test_trend_summary_memory_lift_velocity(self, tracker) -> None:
        # Sessions with increasing memory lift should yield positive velocity
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 3\n\n@per_session_log\n  # format: date;session_id;turns;tokens;cost_usd;baseline_cost_usd;saved_usd;tokens_saved;invisible_pressure;non_tok_response;memory_lift\n  2026-03-17T10:00:00Z;aaa11111;5;1000;0.010000;0.020000;0.010000;500;1;0;1\n  2026-03-18T10:00:00Z;bbb22222;5;1000;0.010000;0.020000;0.010000;500;1;0;3\n  2026-03-19T10:00:00Z;ccc33333;5;1000;0.010000;0.020000;0.010000;500;1;0;5"
            "\n"
        )

        trend = tracker.trend_summary()

        assert trend["sessions_considered"] == 3
        assert trend["avg_memory_lift"] == 3.0
        assert trend["memory_lift_velocity"] > 0  # rising trend

    def test_trend_summary_memory_lift_backward_compat(self, tracker) -> None:
        # Old log format without memory_lift field should default to 0
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 2\n\n@per_session_log\n  # old format without memory_lift\n  2026-03-17T10:00:00Z;aaa11111;5;1000;0.010000;0.020000;0.010000;500;2;0\n  2026-03-18T10:00:00Z;bbb22222;5;1000;0.010000;0.025000;0.015000;700;1;0"
            "\n"
        )

        trend = tracker.trend_summary()

        assert trend["sessions_considered"] == 2
        assert trend["avg_memory_lift"] == 0.0
        assert trend["memory_lift_velocity"] == 0.0

    def test_format_last_session_reads_latest_completed_session(self, tracker) -> None:
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 2\n  total_turns: 10\n  total_tokens: 2000\n  estimated_baseline_cost_usd: 0.045000\n  total_cost_usd: 0.020000\n  tokens_saved: 1200\n  cost_saved_usd: 0.025000\n  savings_pct: 55.0\n\n@per_session_log\n  # format: date;session_id;turns;tokens;cost_usd;baseline_cost_usd;saved_usd;tokens_saved;invisible_pressure;non_tok_response\n  2026-03-17T10:00:00Z;aaa11111;5;1000;0.010000;0.020000;0.010000;500;3;1\n  2026-03-18T10:00:00Z;bbb22222;5;1000;0.010000;0.025000;0.015000;700;1;0"
            "\n"
        )

        result = tracker.format_last_session()

        assert result is not None
        assert "Last completed session:" in result
        assert "2026-03-18T10:00:00Z" in result
        assert "saved: 700 (41.2%)" in result
