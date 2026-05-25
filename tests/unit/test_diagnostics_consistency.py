"""Section 5.1.2 RED tests: Diagnostics Unification.

Verifies that DiagnosticsSnapshot is the single source of truth for all three
diagnostic commands (tok bridge status, tok doctor, tok stats). All shared field
names must be consistent; savings_source must be present.
"""

from __future__ import annotations

import dataclasses

from tok.runtime._diagnostics import DiagnosticsSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(**overrides: object) -> DiagnosticsSnapshot:
    base = {
        "status": "ok",
        "bridge": "tok",
        "port": 9090,
        "api_base": "http://localhost:9090",
        "mode": "compressed",
        "request_policy": "natural_first",
        "baseline_only": False,
        "persistence_failures": 0,
        "fallback_count": 2,
        "actual_tokens": 1000,
        "baseline_tokens": 2000,
        "session_tokens_saved": 900,
        "session_net_tokens_saved": 800,
        "reacquisition_cost_tokens": 100,
        "baseline_prompt_tokens": 1800,
        "prepared_prompt_tokens": 900,
        "saved_prompt_tokens": 900,
        "session_savings_pct": 45.0,
        "session_cost_savings_pct": 42.0,
        "actual_cost_usd": 0.01,
        "baseline_cost_usd": 0.02,
        "cost_saved_usd": 0.01,
        "session_quality": "degraded",
        "last_degradation_reason": "fallback_activated",
        "calls": 5,
        "smoothness_score": 80,
        "labour_index": 10,
        "current_mode": "compressed",
        "fail_open_count": 1,
    }
    base.update(overrides)
    return DiagnosticsSnapshot(**base)


# ---------------------------------------------------------------------------
# RED test 1: savings_source field must exist on DiagnosticsSnapshot
# ---------------------------------------------------------------------------


def test_diagnostics_snapshot_has_savings_source_field() -> None:
    """DiagnosticsSnapshot must carry a savings_source field for ledger attribution."""
    fields = {f.name for f in dataclasses.fields(DiagnosticsSnapshot)}
    assert "savings_source" in fields, (
        "DiagnosticsSnapshot.savings_source is missing. "
        "Add savings_source: str = 'session_tracker' to distinguish live bridge "
        "readings from ledger-reconstructed summaries."
    )


# ---------------------------------------------------------------------------
# RED test 2: to_health_response must include all fields consumed by bridge_status
# ---------------------------------------------------------------------------

_BRIDGE_STATUS_REQUIRED_KEYS = {
    # Core session economics
    "session_tokens_saved",
    "session_net_tokens_saved",
    "actual_tokens",
    "baseline_tokens",
    "reacquisition_cost_tokens",
    "session_savings_pct",
    "session_cost_savings_pct",
    "actual_cost_usd",
    "baseline_cost_usd",
    "cost_saved_usd",
    # Quality / policy
    "session_quality",
    "last_degradation_reason",
    "request_policy",
    "fallback_count",
    "baseline_only",
    # Evidence safety
    "evidence_exact_observed_count",
    "evidence_non_exact_reference_count",
    "evidence_non_exact_summary_count",
    "evidence_non_exact_skeleton_count",
    "evidence_exact_reacquisition_required_count",
    "evidence_exact_reacquisition_satisfied_count",
    "evidence_compression_blocked_for_safety_count",
    # Preflight / recovery
    "preflight_block_original_payload_count",
    "preflight_block_rewritten_payload_count",
    "stream_recovery_empty_success_count",
    "stream_recovery_read_error_count",
    "request_policy_held_by_recovery_count",
    # Meta
    "calls",
    "goal",
}


def test_to_health_response_covers_bridge_status_keys() -> None:
    """to_health_response() must include every key consumed by bridge_status."""
    snap = _make_snapshot()
    health = snap.to_health_response()
    missing = _BRIDGE_STATUS_REQUIRED_KEYS - set(health)
    assert not missing, f"to_health_response() is missing keys needed by bridge_status: {sorted(missing)}"


# ---------------------------------------------------------------------------
# RED test 3: field-name consistency between bridge_status and stats session dict
# ---------------------------------------------------------------------------

_SHARED_SESSION_FIELDS: dict[str, str] = {
    # key-in-bridge-status -> must equal same key in stats
    "session_quality": "session_quality",
    "last_degradation_reason": "last_degradation_reason",
    "fallback_count": "fallback_count",
    "baseline_only": "baseline_only",
    "actual_tokens": "actual_tokens",
    "baseline_tokens": "baseline_tokens",
    "reacquisition_cost_tokens": "reacquisition_cost_tokens",
    "actual_cost_usd": "actual_cost_usd",
    "baseline_cost_usd": "baseline_cost_usd",
    "cost_saved_usd": "cost_saved_usd",
}


def test_bridge_status_and_stats_use_consistent_field_names() -> None:
    """bridge_status and stats must use the same key names for shared session fields.

    This test ensures the canonical field names are defined on DiagnosticsSnapshot
    and that both CLI paths read from it rather than building their own dicts.
    """
    snap = _make_snapshot()
    health = snap.to_health_response()

    # All shared field names must be present in to_health_response so that
    # both bridge_status and stats can derive their outputs from the same object.
    missing = set(_SHARED_SESSION_FIELDS) - set(health)
    assert not missing, f"Shared session fields absent from to_health_response(): {sorted(missing)}"

    # Verify that stats's 'degradation_reason' inconsistency is resolved:
    # the canonical name must be 'last_degradation_reason' (matching bridge_status).
    assert "last_degradation_reason" in health, (
        "to_health_response() must use 'last_degradation_reason' (not 'degradation_reason') "
        "to stay consistent with bridge_status output"
    )


# ---------------------------------------------------------------------------
# RED test 4: doctor ok=False on degraded session
# ---------------------------------------------------------------------------


def test_diagnostics_snapshot_exposes_is_degraded_for_doctor() -> None:
    """DiagnosticsSnapshot must expose is_degraded so doctor can set ok=False."""
    snap = _make_snapshot(session_quality="degraded", fail_open_count=1)
    assert hasattr(snap, "is_degraded"), (
        "DiagnosticsSnapshot.is_degraded property missing. "
        "Doctor command needs it to set ok=False without re-implementing the logic."
    )
    assert snap.is_degraded is True


def test_diagnostics_snapshot_clean_session_is_not_degraded() -> None:
    """Clean sessions must not be flagged as degraded."""
    snap = _make_snapshot(session_quality="clean", fail_open_count=0, fallback_count=0)
    assert hasattr(snap, "is_degraded"), "DiagnosticsSnapshot.is_degraded property missing"
    assert snap.is_degraded is False


# ---------------------------------------------------------------------------
# RED test 5: zero-turn session produces safe defaults
# ---------------------------------------------------------------------------


def test_zero_turn_snapshot_is_safe() -> None:
    """A fresh zero-turn DiagnosticsSnapshot must not raise and must not be degraded."""
    snap = DiagnosticsSnapshot()
    health = snap.to_health_response()
    assert isinstance(health, dict)
    assert health.get("status") == "ok"
    # Zero-turn: savings_source must default to a non-empty string
    if hasattr(snap, "savings_source"):
        assert snap.savings_source, "savings_source must not be empty on a default snapshot"


# ---------------------------------------------------------------------------
# RED test 6: savings + bad baseline edge case
# ---------------------------------------------------------------------------


def test_snapshot_with_savings_but_zero_baseline_is_safe() -> None:
    """Snapshot with tokens_saved > 0 but baseline_tokens == 0 must not crash."""
    snap = _make_snapshot(session_tokens_saved=500, baseline_tokens=0)
    health = snap.to_health_response()
    assert isinstance(health, dict)
    pct = health.get("session_savings_pct", None)
    assert pct is not None


# ---------------------------------------------------------------------------
# RED test 7: savings_source field has correct default
# ---------------------------------------------------------------------------


def test_savings_source_default_value() -> None:
    """savings_source must default to 'session_tracker' on a fresh snapshot."""
    snap = DiagnosticsSnapshot()
    assert hasattr(snap, "savings_source"), "DiagnosticsSnapshot.savings_source missing"
    assert snap.savings_source == "session_tracker", (
        f"Expected savings_source='session_tracker', got {snap.savings_source!r}"
    )
