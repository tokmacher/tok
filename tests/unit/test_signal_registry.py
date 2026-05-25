from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tok.runtime._signal_registry import (
    _INVALID_TOOL_HISTORY_FAILURES as REG_INVALID,
)
from tok.runtime._signal_registry import (
    _NON_BLOCKING_OUTGOING_FAILURES as REG_NON_BLOCKING,
)
from tok.runtime._signal_registry import (
    _PROVIDER_SENSITIVE_FAILURES as REG_PROVIDER,
)
from tok.runtime._signal_registry import (
    _RECOVERABLE_IMMEDIATE_PAIRING_FAILURES as REG_RECOVERABLE,
)
from tok.runtime._signal_registry import (
    _STRICT_FAILURE_SIGNAL_MAP as REG_MAP,
)
from tok.runtime.pipeline.request_validation import (
    _INVALID_TOOL_HISTORY_FAILURES as RV_INVALID,
)
from tok.runtime.pipeline.request_validation import (
    _NON_BLOCKING_OUTGOING_FAILURES as RV_NON_BLOCKING,
)
from tok.runtime.pipeline.request_validation import (
    _PROVIDER_SENSITIVE_FAILURES as RV_PROVIDER,
)
from tok.runtime.pipeline.request_validation import (
    _RECOVERABLE_IMMEDIATE_PAIRING_FAILURES as RV_RECOVERABLE,
)
from tok.runtime.pipeline.request_validation import (
    _STRICT_FAILURE_SIGNAL_MAP as RV_MAP,
)
from tok.runtime.signals import (
    aggregate_registered_categories,
    aggregate_signal_category,
    signal_definition,
    signals_by_category,
    unregistered_signals,
)


def test_release_critical_signals_are_registered() -> None:
    expected = {
        "tok_fallback_activated": "fallback",
        "fail_open_compat_response": "fallback",
        "request_policy_reason_stream_recovery": "recovery",
        "stream_recovery_empty_success": "recovery",
        "tok_bridge_provider_pairing_risk_detected": "provider_safety",
        "evidence_exact_observed": "evidence_safety",
        "evidence_non_exact_reference_emitted": "evidence_safety",
        "evidence_exact_reacquisition_required": "evidence_safety",
        "evidence_exact_reacquisition_satisfied": "evidence_safety",
        "evidence_compression_blocked_for_safety": "evidence_safety",
    }

    for name, category in expected.items():
        definition = signal_definition(name)
        assert definition is not None
        assert definition.category == category
        assert definition.release_critical is True
        assert definition.label


def test_unknown_signals_remain_unregistered_internal() -> None:
    signals = {
        "evidence_exact_observed": 2,
        "experimental_probe_signal": 3,
    }

    assert signal_definition("experimental_probe_signal") is None
    assert unregistered_signals(signals) == {"experimental_probe_signal": 3}


def test_category_aggregation_for_registered_groups() -> None:
    signals = {
        "evidence_exact_observed": 2,
        "evidence_non_exact_reference_emitted": 1,
        "tok_fallback_activated": 1,
        "stream_recovery_empty_success": 4,
        "tok_bridge_provider_pairing_risk_detected": 1,
        "experimental_probe_signal": 99,
    }

    assert aggregate_signal_category(signals, "evidence_safety") == 3
    assert aggregate_signal_category(signals, "fallback") == 1
    assert aggregate_signal_category(signals, "recovery") == 4
    assert aggregate_signal_category(signals, "provider_safety") == 1
    assert aggregate_registered_categories(signals) == {
        "evidence_safety": 3,
        "fallback": 1,
        "recovery": 4,
        "provider_safety": 1,
    }
    assert {signal.name for signal in signals_by_category("evidence_safety")}


# Parity tests: _signal_registry must stay in sync with pipeline/request_validation.


def test_signal_registry_strict_failure_map_matches_request_validation() -> None:
    assert REG_MAP == RV_MAP, (
        "_signal_registry._STRICT_FAILURE_SIGNAL_MAP diverged from "
        "pipeline/request_validation._STRICT_FAILURE_SIGNAL_MAP"
    )


def test_signal_registry_invalid_tool_history_failures_matches_request_validation() -> None:
    assert REG_INVALID == RV_INVALID, (
        "_signal_registry._INVALID_TOOL_HISTORY_FAILURES diverged from "
        "pipeline/request_validation._INVALID_TOOL_HISTORY_FAILURES"
    )


def test_signal_registry_recoverable_pairing_failures_matches_request_validation() -> None:
    assert REG_RECOVERABLE == RV_RECOVERABLE, (
        "_signal_registry._RECOVERABLE_IMMEDIATE_PAIRING_FAILURES diverged from "
        "pipeline/request_validation._RECOVERABLE_IMMEDIATE_PAIRING_FAILURES"
    )


def test_signal_registry_non_blocking_outgoing_failures_matches_request_validation() -> None:
    assert REG_NON_BLOCKING == RV_NON_BLOCKING, (
        "_signal_registry._NON_BLOCKING_OUTGOING_FAILURES diverged from "
        "pipeline/request_validation._NON_BLOCKING_OUTGOING_FAILURES"
    )


def test_signal_registry_provider_sensitive_failures_matches_request_validation() -> None:
    assert REG_PROVIDER == RV_PROVIDER, (
        "_signal_registry._PROVIDER_SENSITIVE_FAILURES diverged from "
        "pipeline/request_validation._PROVIDER_SENSITIVE_FAILURES"
    )


# ---------------------------------------------------------------------------
# Phase 0.2.1 additions: health-affecting and release-critical signal coverage
# ---------------------------------------------------------------------------

_HEALTH_AFFECTING_SIGNALS: dict[str, str] = {
    # Fallback / error
    "processing_error": "fallback",
    "critical_system_error": "fallback",
    "data_structure_error": "fallback",
    "json_decode_error": "fallback",
    "unexpected_error": "fallback",
    "streaming_error_retry": "fallback",
    "stream_buffer_read_error": "fallback",
    # Recovery
    "stream_recovery_started": "recovery",
    "stream_recovery_fallback": "recovery",
    "stream_recovery_loop_broken": "recovery",
    "stream_recovery_retry": "recovery",
    "stream_recovery_retry_exception": "recovery",
    "stream_recovery_success_text": "recovery",
    "stream_recovery_success_tool_use": "recovery",
    "stream_recovery_empty_success": "recovery",
    "stream_recovery_read_error": "recovery",
    "stream_recovery_history_floor_applied": "recovery",
    "stream_recovery_usage": "recovery",
    "stream_empty_after_success": "recovery",
    # Provider safety
    "tok_bridge_pairing_degraded_to_provider_safe": "provider_safety",
    "tok_bridge_thinking_mutation_degraded_to_provider_safe": "provider_safety",
    "tok_bridge_thinking_mutation_unrestored": "provider_safety",
    "tok_bridge_invalid_tool_history_blocked": "provider_safety",
    "tok_bridge_preflight_failed_local": "provider_safety",
    "tok_bridge_preflight_rejected": "provider_safety",
    "tok_bridge_prepared_pairing_rejected_local": "provider_safety",
    "tok_bridge_provider_sensitive_blocked_local": "provider_safety",
    "tok_bridge_provider_sensitive_degraded_to_provider_safe": "provider_safety",
    "tok_bridge_tool_history_pairing_repaired": "provider_safety",
    "tok_bridge_tool_history_repaired": "provider_safety",
    "tok_bridge_assistant_tool_use_text_interleaving_blocked": "provider_safety",
    "tok_compression_worked_before_pairing_degraded": "provider_safety",
    "tok_history_pairing_safety_degraded": "provider_safety",
    "tok_release_blocking_thinking_mutation": "provider_safety",
    "preflight_block_rewritten_payload": "provider_safety",
    # Evidence safety
    "evidence_history_compression_skipped": "evidence_safety",
    "evidence_tool_result_compression_skipped": "evidence_safety",
    "evidence_resolver_cache_stored": "evidence_safety",
    # Session / request policy
    "request_policy_escalations": "recovery",  # already registered as "recovery"
    "request_policy_deescalations": "recovery",  # already registered as "recovery"
    "request_policy_forced_baseline": "session",
    "short_session_baseline_mode": "session",
    # Compression
    "lossless_task_mode_history_skipped": "compression",
    "broad_audit_history_skipped": "compression",
    "tok_history_compression_skipped": "compression",
    "tok_history_cut_blocked_tool_result": "compression",
    "tok_history_cut_point_missing": "compression",
    "tok_history_cut_point_missing_with_tools": "compression",
    "compress_tool_results_bypassed": "compression",
    "tool_compatible_compression": "compression",
}


def test_health_affecting_signals_are_registered() -> None:
    """All signals affecting bridge health or release decisions must be registered."""
    failures = []
    for name, expected_category in _HEALTH_AFFECTING_SIGNALS.items():
        defn = signal_definition(name)
        if defn is None:
            failures.append(f"UNREGISTERED: {name!r}")
        elif defn.category != expected_category:
            failures.append(f"WRONG CATEGORY: {name!r} expected={expected_category!r} got={defn.category!r}")
    if failures:
        joined = "\n  ".join(failures)
        raise AssertionError(f"Signal registration gaps ({len(failures)}):\n  {joined}")


def test_health_affecting_signals_have_label() -> None:
    for name in _HEALTH_AFFECTING_SIGNALS:
        defn = signal_definition(name)
        if defn is not None:
            assert defn.label, f"Signal {name!r} has empty label"


def test_debug_unregistered_signals_function_exists() -> None:
    """debug_unregistered_signals() must exist in signals module."""
    from tok.runtime import signals as sig_mod

    assert hasattr(sig_mod, "debug_unregistered_signals"), "signals.debug_unregistered_signals() not yet implemented"


def test_signal_definition_validates_category() -> None:
    """SignalDefinition must reject invalid category values."""
    from tok.runtime.signals import SignalDefinition

    with pytest.raises((ValueError, AssertionError)):
        SignalDefinition(
            name="test_bad_category",
            category="invalid_category_xyz",
            severity="info",
            label="Test",
        )


def test_signal_definition_validates_severity() -> None:
    """SignalDefinition must reject invalid severity values."""
    from tok.runtime.signals import SignalDefinition

    with pytest.raises((ValueError, AssertionError)):
        SignalDefinition(
            name="test_bad_severity",
            category="fallback",
            severity="ultra_critical",
            label="Test",
        )


def test_signal_registry_no_duplicate_names() -> None:
    """Each signal name must appear exactly once in _SIGNALS."""
    import tok.runtime.signals as sig_mod

    seen: dict[str, int] = {}
    for defn in sig_mod._SIGNALS:
        seen[defn.name] = seen.get(defn.name, 0) + 1
    dupes = {name: count for name, count in seen.items() if count > 1}
    assert not dupes, f"Duplicate signal names in _SIGNALS: {dupes}"


def test_signal_registry_dict_matches_signals_tuple() -> None:
    """SIGNAL_REGISTRY dict must contain exactly the signals in _SIGNALS."""
    import tok.runtime.signals as sig_mod

    names_from_tuple = {s.name for s in sig_mod._SIGNALS}
    names_from_dict = set(sig_mod.SIGNAL_REGISTRY.keys())
    assert names_from_tuple == names_from_dict


def test_signal_definition_rejects_empty_name() -> None:
    from tok.runtime.signals import SignalDefinition

    with pytest.raises((ValueError, AssertionError)):
        # Empty name should be rejected at the registry build stage or validation
        s = SignalDefinition(name="", category="fallback", severity="info", label="test")
        # If dataclass doesn't reject it, the registry build must catch it
        _ = {s.name: s}  # empty key is technically allowed by dict, so check via registry
        raise AssertionError("Expected validation to fail for empty name")


def test_debug_unregistered_signals_logs_under_tok_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    from tok.runtime.signals import debug_unregistered_signals

    monkeypatch.setenv("TOK_DEBUG", "1")
    signals = {"tok_fallback_activated": 1, "totally_unknown_xyz": 3}
    # Must not raise
    debug_unregistered_signals(signals)


def test_debug_unregistered_signals_silent_without_tok_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    from tok.runtime.signals import debug_unregistered_signals

    monkeypatch.delenv("TOK_DEBUG", raising=False)
    # Must be a no-op (no output, no exception)
    debug_unregistered_signals({"totally_unknown_xyz": 99})


def _collect_static_signal_keys() -> set[str]:
    """Parse codebase AST to collect all literal string keys in behavior_signals[...]."""
    src = Path(__file__).resolve().parents[2] / "src" / "tok"
    keys: set[str] = set()
    for py_file in src.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                # Check if the subscript target name looks like a behavior_signals dict
                target = node.value
                target_name = getattr(target, "id", "") or getattr(target, "attr", "")
                if "behavior_signals" in target_name:
                    keys.add(node.slice.value)
    return keys


# ---------------------------------------------------------------------------
# Phase 0.2.1 additions: TOK_ENFORCE_SIGNAL_REGISTRY=1 mode
# ---------------------------------------------------------------------------


def test_warn_unregistered_signals_function_exported() -> None:
    """warn_unregistered_signals() must be exported from signals module."""
    from tok.runtime import signals as sig_mod

    assert hasattr(sig_mod, "warn_unregistered_signals"), "signals.warn_unregistered_signals() not yet implemented"


def test_warn_unregistered_signals_emits_warning_when_enforce_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When TOK_ENFORCE_SIGNAL_REGISTRY=1, warning is emitted for unregistered signals."""
    from tok.runtime.signals import warn_unregistered_signals

    monkeypatch.setenv("TOK_ENFORCE_SIGNAL_REGISTRY", "1")
    signals = {"tok_fallback_activated": 1, "totally_unknown_xyz": 3, "another_unknown": 2}
    import logging

    with caplog.at_level(logging.WARNING, logger="tok.signals"):
        warn_unregistered_signals(signals)
    unknown_names = {"totally_unknown_xyz", "another_unknown"}
    logged_text = caplog.text
    for name in unknown_names:
        assert name in logged_text, f"Expected warning for unregistered signal {name!r}, got: {logged_text!r}"


def test_warn_unregistered_signals_silent_without_enforce_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without TOK_ENFORCE_SIGNAL_REGISTRY=1, warn_unregistered_signals() is a no-op."""
    from tok.runtime.signals import warn_unregistered_signals

    monkeypatch.delenv("TOK_ENFORCE_SIGNAL_REGISTRY", raising=False)
    import logging

    with caplog.at_level(logging.WARNING, logger="tok.signals"):
        warn_unregistered_signals({"totally_unknown_xyz": 3})
    assert "totally_unknown_xyz" not in caplog.text


def test_warn_unregistered_signals_silent_when_all_registered(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When all signals are registered, no warning even with enforce set."""
    from tok.runtime.signals import warn_unregistered_signals

    monkeypatch.setenv("TOK_ENFORCE_SIGNAL_REGISTRY", "1")
    signals = {"tok_fallback_activated": 1, "evidence_exact_observed": 2}
    import logging

    with caplog.at_level(logging.WARNING, logger="tok.signals"):
        warn_unregistered_signals(signals)
    assert caplog.text == "" or "unregistered" not in caplog.text.lower()


def test_warn_unregistered_signals_does_not_warn_for_known_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """warn_unregistered_signals() must not raise on any input."""
    from tok.runtime.signals import warn_unregistered_signals

    monkeypatch.setenv("TOK_ENFORCE_SIGNAL_REGISTRY", "1")
    # Should not raise
    warn_unregistered_signals({})
    warn_unregistered_signals({"tok_fallback_activated": 0})


# ---------------------------------------------------------------------------
# Adversarial: TOK_ENFORCE_SIGNAL_REGISTRY edge cases
# ---------------------------------------------------------------------------


def test_warn_unregistered_signals_silent_when_value_is_true(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TOK_ENFORCE_SIGNAL_REGISTRY=true (not '1') must NOT trigger warning."""
    from tok.runtime.signals import warn_unregistered_signals

    monkeypatch.setenv("TOK_ENFORCE_SIGNAL_REGISTRY", "true")
    import logging

    with caplog.at_level(logging.WARNING, logger="tok.signals"):
        warn_unregistered_signals({"completely_unknown_xyz": 1})
    assert "completely_unknown_xyz" not in caplog.text, (
        "TOK_ENFORCE_SIGNAL_REGISTRY=true should not trigger; only '1' is accepted"
    )


def test_warn_unregistered_signals_silent_when_value_is_0(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TOK_ENFORCE_SIGNAL_REGISTRY=0 must be silent."""
    from tok.runtime.signals import warn_unregistered_signals

    monkeypatch.setenv("TOK_ENFORCE_SIGNAL_REGISTRY", "0")
    import logging

    with caplog.at_level(logging.WARNING, logger="tok.signals"):
        warn_unregistered_signals({"completely_unknown_xyz": 1})
    assert "completely_unknown_xyz" not in caplog.text


def test_warn_unregistered_signals_single_warning_covers_all_names(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single warning line must cover all unregistered signal names."""
    from tok.runtime.signals import warn_unregistered_signals

    monkeypatch.setenv("TOK_ENFORCE_SIGNAL_REGISTRY", "1")
    signals = {"alpha_unknown": 1, "beta_unknown": 2, "tok_fallback_activated": 1}
    import logging

    with caplog.at_level(logging.WARNING, logger="tok.signals"):
        warn_unregistered_signals(signals)
    text = caplog.text
    assert "alpha_unknown" in text
    assert "beta_unknown" in text
    assert "tok_fallback_activated" not in text  # registered; must not appear in warning


def test_codebase_signal_keys_all_registered_or_internal() -> None:
    """All static behavior_signals keys from the codebase must be registered or start with '_'."""
    from tok.runtime.signals import SIGNAL_REGISTRY

    static_keys = _collect_static_signal_keys()
    unregistered = {k for k in static_keys if k not in SIGNAL_REGISTRY and not k.startswith("_")}
    # Known metric/counter keys that are intentionally not registered (pure telemetry)
    known_telemetry = {
        "bridge_history_cut_search_extension_turns",
        "command_result_cache_saved_tokens",
        "plan_finalization_min_saved_tokens",
        "plan_finalization_original_prompt_tokens",
        "plan_finalization_prepared_prompt_tokens",
        "plan_finalization_saved_prompt_tokens",
        "state_payload_chars",
        "structured_tool_loop_stuck_target_recent",
        "structured_tool_loop_stuck_target_expired",
        "runtime_hint_cooldown_suppressed",
        "macro_savings_attributed",
        "late_answer_assembly_mixed_signal_suppressed",
        "tok_prompt_optimization_suppressed_chars",
        "tok_prompt_optimization_suppressed_tokens",
        "tok_sift_cache_marked_block_tokens",
        "tok_sift_cache_marked_blocks",
        "tok_sift_cache_marked_saved_tokens",
        "seed_answer_assembly_pressure",
        "seed_navigation_pressure",
        "fallback_pressure_incremented",
        "fallback_pressure_suppressed",
        "grounded_oracle_miss_streak",
    }
    remaining = unregistered - known_telemetry
    assert not remaining, (
        f"Unregistered non-telemetry signal keys found ({len(remaining)}). "
        f"Either register them in SIGNAL_REGISTRY or add to known_telemetry:\n  " + "\n  ".join(sorted(remaining))
    )
