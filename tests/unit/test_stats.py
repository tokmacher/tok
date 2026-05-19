"""Tests for tok.stats — savings ledger load/save/merge."""

import os

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

    def test_order_only_tool_result_repair_does_not_degrade_session_quality(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=20,
            output_saved=10,
            behavior_signals={
                "tok_bridge_tool_result_order_repaired": 1,
                "tok_bridge_tool_result_pairing_repaired": 1,
                "tool_result_order_repair_non_degrading": 1,
            },
        )

        summary = tracker.session_summary()

        assert summary is not None
        assert summary["session_quality"] == "clean"
        assert summary["last_degradation_reason"] == ""
        assert summary["tool_history_pairing_repaired_count"] == 0

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

    def test_merge_session_to_ledger_replaces_same_session_snapshot(self, tracker) -> None:
        stats = tracker.load_stats()
        stats["session_start"] = "2026-04-28T12:00:00Z"
        tracker.save_stats(stats)
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=100,
            cache_read=0,
            cache_write=0,
            input_saved=200,
            output_saved=0,
        )
        tracker.merge_session_to_ledger()

        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=500,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=100,
            output_saved=0,
        )
        tracker.merge_session_to_ledger()

        content = tracker.ledger_path.read_text()
        assert "sessions: 1" in content
        assert "total_turns: 2" in content
        assert "total_prompt_tokens: 1500" in content
        assert "total_completion_tokens: 150" in content
        assert "total_tokens: 1650" in content
        assert "tokens_saved: 300" in content
        assert content.count("2026-04-28T12:00:00Z;") == 1

    def test_merge_session_to_ledger_repairs_inflated_header_totals(self, tracker) -> None:
        tracker.ledger_path.write_text(
            "@lifetime_savings\n"
            "  sessions: 99\n"
            "  total_turns: 999\n"
            "  total_prompt_tokens: 999000\n"
            "  total_completion_tokens: 999000\n"
            "  total_tokens: 999000\n"
            "  total_cost_usd: 999.000000\n"
            "  estimated_baseline_cost_usd: 999.000000\n"
            "  tokens_saved: 999000\n"
            "  net_tokens_saved: 999000\n"
            "  cost_saved_usd: 999.000000\n"
            "  savings_pct: 99.0\n\n"
            "@per_session_log\n"
            "  2026-04-28T10:00:00Z;old11111;2;1000;0.010000;0.020000;0.010000;500;0;0;0;0;0;0;;900;100;0;0;800;500;300;0;0;0\n"
        )
        stats = tracker.load_stats()
        stats["session_start"] = "2026-04-28T12:00:00Z"
        tracker.save_stats(stats)
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=100,
            cache_read=0,
            cache_write=0,
            input_saved=400,
            output_saved=0,
        )

        tracker.merge_session_to_ledger()

        content = tracker.ledger_path.read_text()
        assert "sessions: 2" in content
        assert "total_turns: 3" in content
        assert "total_prompt_tokens: 1900" in content
        assert "total_completion_tokens: 200" in content
        assert "total_tokens: 2100" in content
        assert "tokens_saved: 900" in content
        assert "cost_saved_usd: 0.011200" in content

    def test_lifetime_summary_dedupes_session_log_rows(self, tracker) -> None:
        tracker.ledger_path.write_text(
            "@lifetime_savings\n"
            "  sessions: 3\n"
            "  total_turns: 7\n"
            "  total_tokens: 7000\n"
            "  total_cost_usd: 0.070000\n"
            "  estimated_baseline_cost_usd: 0.140000\n"
            "  tokens_saved: 7000\n"
            "  cost_saved_usd: 0.070000\n"
            "\n"
            "@per_session_log\n"
            "  2026-04-28T12:00:00Z;same1111;1;1000;0.010000;0.020000;0.010000;500;0;0\n"
            "  2026-04-28T12:00:00Z;same1111;2;2500;0.025000;0.050000;0.025000;1500;0;0\n"
            "  2026-04-28T13:00:00Z;other222;1;1000;0.010000;0.030000;0.020000;1000;0;0\n"
        )

        summary = tracker.lifetime_summary()

        assert summary is not None
        assert summary["sessions"] == 2
        assert summary["total_turns"] == 3
        assert summary["actual_tokens"] == 3500
        assert summary["baseline_tokens"] == 6000
        assert summary["tokens_saved"] == 2500
        assert summary["cost_saved_usd"] == pytest.approx(0.045)

    def test_lifetime_summary_does_not_overlay_stale_flushed_session_file(self, tmp_path) -> None:
        ledger_path = tmp_path / "global_savings.tok"
        savings_file = tmp_path / "session_stats" / "live.tok"
        savings_file.parent.mkdir()

        tracker = SavingsTracker(savings_file=str(savings_file), ledger_path=ledger_path)
        stats = tracker.load_stats()
        stats["session_start"] = "2026-05-14T10:00:00Z"
        tracker.save_stats(stats)
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=100,
            cache_read=0,
            cache_write=0,
            input_saved=400,
            output_saved=0,
        )
        tracker.merge_session_to_ledger()

        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=5000,
            actual_output=500,
            cache_read=0,
            cache_write=0,
            input_saved=2000,
            output_saved=0,
        )
        newer = savings_file.stat().st_mtime + 10
        os.utime(ledger_path, (newer, newer))

        summary = tracker.lifetime_summary()

        assert summary is not None
        assert summary["sessions"] == 1
        assert summary["total_turns"] == 1
        assert summary["actual_tokens"] == 1100
        assert summary["tokens_saved"] == 400

    def test_tracker_flushes_lifetime_ledger_periodically(self, tmp_path, monkeypatch) -> None:
        from tok.stats import SavingsTracker

        monkeypatch.setenv("TOK_LIFETIME_FLUSH_EVERY_TURNS", "1")
        ledger_path = tmp_path / "global_savings.tok"
        savings_file = str(tmp_path / "session_stats.tok")

        tracker = SavingsTracker(savings_file=savings_file, ledger_path=ledger_path)
        tracker.record_call(
            model="claude-3-5-sonnet-latest",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=25,
            output_saved=10,
        )

        text = ledger_path.read_text()
        assert "@per_session_log" in text

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

    def test_session_summary_aggregates_evidence_safety_signals(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=5,
            behavior_signals={
                "evidence_exact_observed": 2,
                "evidence_first_exact_observed": 1,
                "evidence_non_exact_reference_emitted": 1,
                "evidence_non_exact_summary_emitted": 1,
                "evidence_non_exact_skeleton_emitted": 1,
                "evidence_exact_reacquisition_required": 2,
                "evidence_exact_reacquisition_satisfied": 1,
                "evidence_compression_blocked_for_safety": 1,
                "evidence_tool_result_compression_skipped": 1,
                "evidence_history_compression_skipped": 1,
            },
        )

        summary = tracker.session_summary()

        assert summary is not None
        assert summary["evidence_exact_observed_count"] == 2
        assert summary["evidence_first_exact_observed_count"] == 1
        assert summary["evidence_non_exact_reference_count"] == 1
        assert summary["evidence_non_exact_summary_count"] == 1
        assert summary["evidence_non_exact_skeleton_count"] == 1
        assert summary["evidence_exact_reacquisition_required_count"] == 2
        assert summary["evidence_exact_reacquisition_satisfied_count"] == 1
        assert summary["evidence_compression_blocked_for_safety_count"] == 1
        assert summary["evidence_tool_result_compression_skipped_count"] == 1
        assert summary["evidence_history_compression_skipped_count"] == 1
        assert summary["evidence_safety_event_count"] == 12

    def test_merge_session_to_ledger_persists_selected_evidence_safety_counters(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=5,
            behavior_signals={
                "evidence_exact_observed": 2,
                "evidence_first_exact_observed": 1,
                "evidence_non_exact_reference_emitted": 1,
                "evidence_non_exact_summary_emitted": 5,
                "evidence_exact_reacquisition_required": 2,
                "evidence_exact_reacquisition_satisfied": 1,
                "evidence_compression_blocked_for_safety": 1,
            },
        )

        tracker.merge_session_to_ledger()

        content = tracker.ledger_path.read_text()
        assert "evidence_exact_observed: 2" in content
        assert "evidence_first_exact_observed: 1" in content
        assert "evidence_non_exact_reference_emitted: 1" in content
        assert "evidence_exact_reacquisition_required: 2" in content
        assert "evidence_exact_reacquisition_satisfied: 1" in content
        assert "evidence_compression_blocked_for_safety: 1" in content
        assert "evidence_non_exact_summary_emitted:" not in content

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

    # ------------------------------------------------------------------
    # lifetime_summary inflight overlay tests
    # ------------------------------------------------------------------

    def test_lifetime_summary_overlays_inflight_session_on_completed_log(self, tracker) -> None:
        """In-flight session data must be added to lifetime totals even before shutdown."""
        # Simulate one previously completed session in the ledger
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 1\n\n@per_session_log\n"
            "  2026-04-01T10:00:00Z;prev1111;3;3000;0.030000;0.060000;0.030000;1500;0;0\n"
        )
        # Record a new in-flight call (not yet merged to ledger)
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=200,
            cache_read=0,
            cache_write=0,
            input_saved=400,
            output_saved=0,
        )

        summary = tracker.lifetime_summary()

        assert summary is not None
        # Two sessions: one completed + one in-flight
        assert summary["sessions"] == 2
        # actual_tokens: 3000 (completed) + 1200 (inflight: 1000 input + 200 output)
        assert summary["actual_tokens"] == 4200
        # tokens_saved: 1500 (completed) + 400 (inflight input_saved)
        assert summary["tokens_saved"] == 1900
        assert summary["baseline_tokens"] == summary["actual_tokens"] + summary["tokens_saved"]

    def test_lifetime_summary_no_double_count_when_session_already_flushed(self, tracker) -> None:
        """If the current session was already merged (e.g. via periodic flush), it must not be counted twice."""
        import hashlib

        date_str = "2026-04-02T09:00:00Z"
        sess_id = hashlib.md5(f"{date_str}:{tracker.savings_file}".encode(), usedforsecurity=False).hexdigest()[:8]

        # The session is already in the per_session_log with its real sess_id
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 1\n\n@per_session_log\n"
            f"  {date_str};{sess_id};2;2000;0.020000;0.040000;0.020000;1000;0;0\n"
        )
        # Set session_start to the same date so the sess_id matches
        stats = tracker.load_stats()
        stats["session_start"] = date_str
        tracker.save_stats(stats)
        # Record another call (same session, already flushed once)
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=500,
            actual_output=100,
            cache_read=0,
            cache_write=0,
            input_saved=200,
            output_saved=0,
        )

        summary = tracker.lifetime_summary()

        assert summary is not None
        # Must still show only 1 session — the flushed entry already covers it
        assert summary["sessions"] == 1

    def test_lifetime_summary_replaces_flushed_row_with_live_snapshot(self, tracker) -> None:
        """A live session file is a newer snapshot of the same session, not a duplicate or stale row."""
        import hashlib

        date_str = "2026-04-02T09:00:00Z"
        sess_id = hashlib.md5(f"{date_str}:{tracker.savings_file}".encode(), usedforsecurity=False).hexdigest()[:8]
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 1\n\n@per_session_log\n"
            f"  {date_str};{sess_id};2;2000;0.020000;0.040000;0.020000;1000;0;0\n"
        )
        stats = tracker.load_stats()
        stats["session_start"] = date_str
        tracker.save_stats(stats)
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=2000,
            actual_output=200,
            cache_read=0,
            cache_write=0,
            input_saved=1500,
            output_saved=0,
        )

        summary = tracker.lifetime_summary()

        assert summary is not None
        assert summary["sessions"] == 1
        assert summary["tokens_saved"] == 1500
        assert summary["actual_tokens"] == 2200

    def test_lifetime_summary_includes_live_bridge_session_stats_files(self, tracker, tmp_path) -> None:
        """tok stats must include keyed bridge sessions under the ledger's session_stats directory."""
        stats_dir = tracker.ledger_path.parent / "session_stats"
        stats_dir.mkdir()
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 1\n\n@per_session_log\n"
            "  2026-04-01T10:00:00Z;prev1111;3;3000;0.030000;0.060000;0.030000;1500;0;0\n"
        )
        live = type(tracker)(
            savings_file=str(stats_dir / "live-session.tok"),
            ledger_path=tracker.ledger_path,
        )
        live.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=200,
            cache_read=0,
            cache_write=0,
            input_saved=400,
            output_saved=50,
        )

        summary = tracker.lifetime_summary()

        assert summary is not None
        assert summary["sessions"] == 2
        assert summary["tokens_saved"] == 1950
        assert summary["actual_tokens"] == 4200

    def test_lifetime_summary_ignores_empty_session_log_rows(self, tracker) -> None:
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 2\n\n@per_session_log\n"
            "  2026-04-01T10:00:00Z;empty000;0;0;0.000000;0.000000;0.000000;0;0;0\n"
            "  2026-04-01T10:01:00Z;real1111;3;3000;0.030000;0.060000;0.030000;1500;0;0\n"
        )

        summary = tracker.lifetime_summary(include_inflight=False)

        assert summary is not None
        assert summary["sessions"] == 1
        assert summary["total_turns"] == 3
        assert summary["actual_tokens"] == 3000

    def test_lifetime_summary_shows_inflight_when_no_ledger_exists(self, tracker) -> None:
        """If the ledger file does not exist yet, return the in-flight session as the sole entry."""
        assert not tracker.ledger_path.exists()
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=800,
            actual_output=200,
            cache_read=0,
            cache_write=0,
            input_saved=300,
            output_saved=0,
        )

        summary = tracker.lifetime_summary()

        assert summary is not None
        assert summary["sessions"] == 1
        assert summary["tokens_saved"] == 300
        # actual_tokens = 800 input + 200 output
        assert summary["actual_tokens"] == 1000

    def test_lifetime_summary_returns_none_when_no_ledger_and_no_inflight(self, tracker) -> None:
        """Both ledger and in-flight session absent → None."""
        assert not tracker.ledger_path.exists()
        summary = tracker.lifetime_summary()
        assert summary is None

    def test_inflight_session_agg_returns_none_when_no_calls(self, tracker) -> None:
        """_inflight_session_agg must return None before any calls are recorded."""
        assert tracker._inflight_session_agg() is None

    def test_inflight_session_agg_reflects_recorded_calls(self, tracker) -> None:
        """_inflight_session_agg must reflect the live session without touching the ledger."""
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=500,
            actual_output=100,
            cache_read=0,
            cache_write=0,
            input_saved=200,
            output_saved=50,
        )

        agg = tracker._inflight_session_agg()

        assert agg is not None
        assert agg["calls"] == 1
        assert agg["saved_tokens"] == 250  # input_saved + output_saved
        assert agg["sess_id"] != ""
        assert not tracker.ledger_path.exists()  # must not trigger a write


class TestPeakSavingsPct:
    def test_peak_set_on_first_call(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4-6",
            actual_input=1000,
            actual_output=100,
            cache_read=0,
            cache_write=0,
            input_saved=500,
            output_saved=0,
        )
        summary = tracker.session_summary()
        assert summary is not None
        assert "peak_savings_pct" in summary
        assert float(summary["peak_savings_pct"]) > 0

    def test_peak_does_not_decline(self, tracker) -> None:
        # First call: high savings (500 saved / 1500 baseline ≈ 33%)
        tracker.record_call(
            model="claude-sonnet-4-6",
            actual_input=1000,
            actual_output=0,
            cache_read=0,
            cache_write=0,
            input_saved=500,
            output_saved=0,
        )
        summary_after_1 = tracker.session_summary()
        peak_after_1 = float(summary_after_1["peak_savings_pct"])

        # Subsequent calls: nothing saved → cumulative % drops
        for _ in range(10):
            tracker.record_call(
                model="claude-sonnet-4-6",
                actual_input=5000,
                actual_output=0,
                cache_read=0,
                cache_write=0,
                input_saved=0,
                output_saved=0,
            )

        summary_after_many = tracker.session_summary()
        current_pct = float(summary_after_many["savings_pct"])
        peak_after_many = float(summary_after_many["peak_savings_pct"])

        assert current_pct < peak_after_1, "savings_pct should have declined"
        assert peak_after_many == peak_after_1, "peak_savings_pct must not decline"

    def test_peak_persists_across_load(self, tracker) -> None:
        tracker.record_call(
            model="claude-sonnet-4-6",
            actual_input=1000,
            actual_output=0,
            cache_read=0,
            cache_write=0,
            input_saved=400,
            output_saved=0,
        )
        peak = float(tracker.session_summary()["peak_savings_pct"])

        # Reload from disk
        reloaded = type(tracker)(
            savings_file=tracker.savings_file,
            ledger_path=tracker.ledger_path,
        )
        summary = reloaded.session_summary()
        assert summary is not None
        assert float(summary["peak_savings_pct"]) == peak


class TestNetTokensSavedSubtraction:
    """Regression: _subtract_session_from_ledger must subtract net, not gross."""

    def test_net_tokens_saved_subtracts_net_not_gross(self, tracker) -> None:
        from tok.utils.savings_tracker import SavingsTracker as _ST

        ledger = {
            "sessions": 2,
            "total_turns": 10,
            "total_tokens": 5000,
            "total_prompt_tokens": 4000,
            "total_completion_tokens": 1000,
            "tokens_saved": 2000,
            "net_tokens_saved": 1800,
            "total_cost_usd": 0.1,
            "estimated_baseline_cost_usd": 0.2,
            "cost_saved_usd": 0.1,
            "baseline_prompt_tokens": 4000,
            "prepared_prompt_tokens": 3000,
            "saved_prompt_tokens": 1000,
            "hot_hint_tokens_added": 0,
            "reacquisition_tokens_avoided_estimate": 0,
            "reacquisition_cost_tokens": 0,
        }
        entry = {
            "turns": 5,
            "tokens": 2500,
            "prompt_tokens": 2000,
            "completion_tokens": 500,
            "tokens_saved": 1000,
            "actual_cost_usd": 0.05,
            "baseline_cost_usd": 0.1,
            "saved_usd": 0.05,
            "reacquisition_cost_tokens": 200,
        }
        _ST._subtract_session_from_ledger(ledger, entry)

        assert ledger["net_tokens_saved"] == 1800 - (1000 - 200)

    def test_net_tokens_saved_clamps_to_zero(self, tracker) -> None:
        from tok.utils.savings_tracker import SavingsTracker as _ST

        ledger = {
            "sessions": 1,
            "total_turns": 5,
            "total_tokens": 2500,
            "total_prompt_tokens": 2000,
            "total_completion_tokens": 500,
            "tokens_saved": 1000,
            "net_tokens_saved": 100,
            "total_cost_usd": 0.05,
            "estimated_baseline_cost_usd": 0.1,
            "cost_saved_usd": 0.05,
            "baseline_prompt_tokens": 2000,
            "prepared_prompt_tokens": 1500,
            "saved_prompt_tokens": 500,
            "hot_hint_tokens_added": 0,
            "reacquisition_tokens_avoided_estimate": 0,
            "reacquisition_cost_tokens": 0,
        }
        entry = {
            "turns": 5,
            "tokens": 2500,
            "prompt_tokens": 2000,
            "completion_tokens": 500,
            "tokens_saved": 1000,
            "actual_cost_usd": 0.05,
            "baseline_cost_usd": 0.1,
            "saved_usd": 0.05,
            "reacquisition_cost_tokens": 0,
        }
        _ST._subtract_session_from_ledger(ledger, entry)

        assert ledger["net_tokens_saved"] == 0

    def test_net_tokens_saved_uses_reacquisition_cost(self, tracker) -> None:
        from tok.utils.savings_tracker import SavingsTracker as _ST

        ledger = {
            "sessions": 1,
            "total_turns": 5,
            "total_tokens": 2500,
            "total_prompt_tokens": 2000,
            "total_completion_tokens": 500,
            "tokens_saved": 1000,
            "net_tokens_saved": 800,
            "total_cost_usd": 0.05,
            "estimated_baseline_cost_usd": 0.1,
            "cost_saved_usd": 0.05,
            "baseline_prompt_tokens": 2000,
            "prepared_prompt_tokens": 1500,
            "saved_prompt_tokens": 500,
            "hot_hint_tokens_added": 0,
            "reacquisition_tokens_avoided_estimate": 0,
            "reacquisition_cost_tokens": 0,
        }
        entry = {
            "turns": 5,
            "tokens": 2500,
            "prompt_tokens": 2000,
            "completion_tokens": 500,
            "tokens_saved": 1000,
            "actual_cost_usd": 0.05,
            "baseline_cost_usd": 0.1,
            "saved_usd": 0.05,
            "reacquisition_cost_tokens": 500,
        }
        _ST._subtract_session_from_ledger(ledger, entry)

        assert ledger["net_tokens_saved"] == 800 - (1000 - 500)


class TestLifetimeSummaryNoDoubleCountHealthOverlay:
    """Regression: lifetime_summary already includes inflight data; stats_command must not add it again."""

    def test_lifetime_summary_includes_inflight_without_health_overlay(self, tracker) -> None:
        tracker.ledger_path.write_text(
            "@lifetime_savings\n  sessions: 1\n\n@per_session_log\n"
            "  2026-04-01T10:00:00Z;prev1111;3;3000;0.030000;0.060000;0.030000;1500;0;0\n"
        )
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=200,
            cache_read=0,
            cache_write=0,
            input_saved=400,
            output_saved=0,
        )

        summary = tracker.lifetime_summary()

        assert summary is not None
        assert summary["sessions"] == 2
        assert summary["actual_tokens"] == 4200
        assert summary["tokens_saved"] == 1900


def test_lifetime_summary_always_includes_net_tokens_saved(tmp_path) -> None:
    """net_tokens_saved must be present in all non-None lifetime_summary return paths."""
    from tok.utils.savings_tracker import SavingsTracker

    tracker = SavingsTracker(
        savings_file=str(tmp_path / "stats.tok"),
        ledger_path=tmp_path / "lifetime.tok",
    )
    tracker.record_call(
        model="claude-sonnet-4",
        actual_input=800,
        actual_output=200,
        cache_read=0,
        cache_write=0,
        input_saved=300,
        output_saved=0,
    )
    summary = tracker.lifetime_summary()
    assert summary is not None
    assert "net_tokens_saved" in summary
    assert summary["net_tokens_saved"] >= 0
