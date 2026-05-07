from __future__ import annotations

from tok.runtime._diagnostics import DiagnosticsSnapshot


def test_diagnostics_snapshot_from_session_basic_fields() -> None:
    snap = DiagnosticsSnapshot.from_session(
        port=9090,
        api_base="https://api.example.com",
        request_policy_default="natural_first",
        mode_label="tool-compatible",
        baseline_only=False,
        persistence_failures=2,
        session_summary={
            "tokens_saved": 1000,
            "savings_pct": 12.5,
            "actual_tokens": 8000,
            "baseline_tokens": 9000,
            "calls": 5,
            "session_quality": "clean",
            "last_degradation_reason": "",
        },
        signals={},
    )
    assert snap.port == 9090
    assert snap.api_base == "https://api.example.com"
    assert snap.request_policy == "natural_first"
    assert snap.mode == "tool-compatible"
    assert snap.baseline_only is False
    assert snap.persistence_failures == 2
    assert snap.session_tokens_saved == 1000
    assert snap.session_savings_pct == 12.5
    assert snap.actual_tokens == 8000
    assert snap.baseline_tokens == 9000
    assert snap.calls == 5
    assert snap.session_quality == "clean"


def test_diagnostics_snapshot_from_session_fallback_count_takes_max() -> None:
    # fallback_count should be max(session_summary value, tok_fallback_activated signal)
    snap_summary_wins = DiagnosticsSnapshot.from_session(
        port=0,
        api_base="",
        request_policy_default="",
        mode_label="",
        baseline_only=False,
        persistence_failures=0,
        session_summary={"fallback_count": 5},
        signals={"tok_fallback_activated": 3},
    )
    assert snap_summary_wins.fallback_count == 5

    snap_signal_wins = DiagnosticsSnapshot.from_session(
        port=0,
        api_base="",
        request_policy_default="",
        mode_label="",
        baseline_only=False,
        persistence_failures=0,
        session_summary={"fallback_count": 1},
        signals={"tok_fallback_activated": 7},
    )
    assert snap_signal_wins.fallback_count == 7


def test_diagnostics_snapshot_from_session_cost_savings_pct_fallback() -> None:
    # cost_savings_pct falls back to savings_pct when absent
    snap = DiagnosticsSnapshot.from_session(
        port=0,
        api_base="",
        request_policy_default="",
        mode_label="",
        baseline_only=False,
        persistence_failures=0,
        session_summary={"savings_pct": 20.0},
        signals={},
    )
    assert snap.session_cost_savings_pct == 20.0

    snap2 = DiagnosticsSnapshot.from_session(
        port=0,
        api_base="",
        request_policy_default="",
        mode_label="",
        baseline_only=False,
        persistence_failures=0,
        session_summary={"savings_pct": 20.0, "cost_savings_pct": 15.0},
        signals={},
    )
    assert snap2.session_cost_savings_pct == 15.0


def test_diagnostics_snapshot_from_session_signal_fields() -> None:
    # Signal keys must match what the gateway actually emits.
    snap = DiagnosticsSnapshot.from_session(
        port=0,
        api_base="",
        request_policy_default="",
        mode_label="",
        baseline_only=False,
        persistence_failures=0,
        session_summary={
            # tool_history counts come from session_summary (not signals)
            "tool_history_repaired_count": 2,
        },
        signals={
            "repeat_search": 3,
            "repeat_file_read": 4,
            "repeat_target_hot": 1,
            # state_resend uses the _turn suffix
            "state_resend_full_turn": 5,
            # hot_recent_hint uses the _injected suffix
            "hot_recent_hint_injected": 7,
        },
    )
    assert snap.repeat_search_count == 3
    assert snap.repeat_file_read_count == 4
    assert snap.repeat_target_hot_count == 1
    assert snap.tool_history_repaired_count == 2
    assert snap.state_resend_full_count == 5
    assert snap.hot_recent_hint_count == 7
    # repeated_active_file_reads maps to the repeat_file_read signal
    assert snap.repeated_active_file_reads == 4


def test_diagnostics_snapshot_from_session_output_is_roundtrippable() -> None:
    snap = DiagnosticsSnapshot.from_session(
        port=8080,
        api_base="https://x.invalid",
        request_policy_default="forced_baseline",
        mode_label="baseline",
        baseline_only=True,
        persistence_failures=1,
        session_summary={"tokens_saved": 500, "calls": 3, "session_quality": "degraded"},
        signals={"repeat_search": 2},
    )
    payload = snap.to_health_response()
    roundtripped = DiagnosticsSnapshot.from_health_response(payload).to_health_response()
    assert roundtripped == payload


def test_from_session_output_matches_expected_health_fields() -> None:
    """C1 regression: from_session() produces the same field set as the health endpoint."""
    session_summary = {
        "tokens_saved": 1200,
        "savings_pct": 15.0,
        "cost_savings_pct": 14.0,
        "actual_tokens": 8000,
        "baseline_tokens": 9200,
        "calls": 7,
        "session_quality": "clean",
        "last_degradation_reason": "",
        "fallback_count": 2,
        "tool_history_repaired_count": 3,
        "stream_recovery_success_text_count": 1,
        "request_policy_escalations_count": 2,
    }
    signals = {
        "tok_fallback_activated": 1,
        "repeat_search": 5,
        "repeat_file_read": 6,
        "hot_recent_hint_injected": 4,
        "state_resend_full_turn": 2,
        "state_resend_delta_turn": 1,
        "stream_recovery_started": 3,
        "tok_bridge_invalid_tool_history_blocked": 1,
        "preflight_block_original_payload": 2,
    }
    snap = DiagnosticsSnapshot.from_session(
        port=9090,
        api_base="https://api.anthropic.com",
        request_policy_default="natural_first",
        mode_label="tool-compatible",
        baseline_only=False,
        persistence_failures=0,
        session_summary=session_summary,
        signals=signals,
    )
    payload = snap.to_health_response()

    # Verify all DiagnosticsSnapshot fields appear in the response
    assert set(payload.keys()) == set(DiagnosticsSnapshot.__dataclass_fields__.keys())

    # Verify key field values are correctly derived
    assert payload["fallback_count"] == 2  # max(session_summary=2, signal=1)
    assert payload["session_tokens_saved"] == 1200
    assert payload["session_cost_savings_pct"] == 14.0
    assert payload["repeat_search_count"] == 5
    assert payload["repeat_file_read_count"] == 6
    assert payload["repeated_active_file_reads"] == 6  # same signal as repeat_file_read
    assert payload["hot_recent_hint_count"] == 4
    assert payload["state_resend_full_count"] == 2
    assert payload["state_resend_delta_count"] == 1
    assert payload["stream_recovery_attempt_count"] == 3  # max(0, signal=3)
    assert payload["tool_history_repaired_count"] == 3  # from session_summary
    assert payload["tool_history_blocked_count"] == 1  # max(0, signal=1)
    assert payload["preflight_block_original_payload_count"] == 2
    assert payload["request_policy_escalations_count"] == 2  # from session_summary


def test_diagnostics_snapshot_to_health_response_has_all_fields() -> None:
    snap = DiagnosticsSnapshot(
        port=9090,
        api_base="https://example.invalid",
        mode="tool-compatible",
        request_policy="natural_first",
        baseline_only=False,
        persistence_failures=0,
        fallback_count=1,
        calls=2,
        session_quality="clean",
    )
    payload = snap.to_health_response()
    assert set(payload.keys()) == set(DiagnosticsSnapshot.__dataclass_fields__.keys())
    assert payload["port"] == 9090
    assert payload["api_base"] == "https://example.invalid"


def test_diagnostics_snapshot_roundtrip_from_health_response() -> None:
    original = DiagnosticsSnapshot(
        port=1,
        api_base="x",
        mode="m",
        request_policy="p",
        baseline_only=True,
        fallback_count=3,
        actual_tokens=4,
        session_savings_pct=5.0,
        session_quality="degraded",
        calls=6,
        current_mode="FULL_TOK",
    )
    payload = original.to_health_response()
    snap = DiagnosticsSnapshot.from_health_response(payload)
    assert snap.to_health_response() == payload
