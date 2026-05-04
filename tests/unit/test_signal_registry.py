from __future__ import annotations

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
