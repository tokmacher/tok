"""Internal registry for behavior signals that matter to diagnostics and release gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SignalDefinition:
    """Metadata for behavior signals with release or diagnostic meaning."""

    name: str
    category: str
    severity: str
    label: str
    affects_health: bool = False
    release_critical: bool = False


def _signal(
    name: str,
    *,
    category: str,
    severity: str,
    label: str,
    affects_health: bool = False,
    release_critical: bool = True,
) -> SignalDefinition:
    return SignalDefinition(
        name=name,
        category=category,
        severity=severity,
        label=label,
        affects_health=affects_health,
        release_critical=release_critical,
    )


_SIGNALS = (
    _signal(
        "tok_fallback_activated",
        category="fallback",
        severity="warning",
        label="Tok fallback activated",
        affects_health=True,
    ),
    _signal(
        "baseline_only_session",
        category="fallback",
        severity="warning",
        label="Baseline-only session",
        affects_health=True,
    ),
    _signal(
        "non_tok_response",
        category="fallback",
        severity="warning",
        label="Non-Tok response",
        affects_health=True,
    ),
    _signal(
        "fail_open_compat_response",
        category="fallback",
        severity="warning",
        label="Compatibility fallback",
        affects_health=True,
    ),
    _signal(
        "malformed_tok_response",
        category="fallback",
        severity="warning",
        label="Malformed Tok response",
        affects_health=True,
    ),
    _signal(
        "malformed_tok_hybrid_tool",
        category="fallback",
        severity="warning",
        label="Malformed hybrid tool response",
        affects_health=True,
    ),
    _signal(
        "malformed_tok_non_inverted_msg",
        category="fallback",
        severity="warning",
        label="Malformed non-inverted message",
        affects_health=True,
    ),
    _signal(
        "malformed_tok_markdown_fallback",
        category="fallback",
        severity="warning",
        label="Markdown fallback response",
        affects_health=True,
    ),
    _signal(
        "malformed_tok_bad_header",
        category="fallback",
        severity="warning",
        label="Malformed Tok header",
        affects_health=True,
    ),
    _signal(
        "request_policy_escalations",
        category="recovery",
        severity="info",
        label="Request policy escalations",
    ),
    _signal(
        "request_policy_deescalations",
        category="recovery",
        severity="info",
        label="Request policy deescalations",
    ),
    _signal(
        "request_policy_reason_stream_recovery",
        category="recovery",
        severity="info",
        label="Stream recovery policy hold",
    ),
    _signal(
        "request_policy_reason_tool_recovery",
        category="recovery",
        severity="info",
        label="Tool recovery policy hold",
    ),
    _signal(
        "request_policy_reason_structured_tool_loop",
        category="recovery",
        severity="info",
        label="Structured tool-loop policy hold",
    ),
    _signal(
        "request_policy_held_by_recovery",
        category="recovery",
        severity="info",
        label="Policy held by recovery",
    ),
    _signal(
        "request_policy_recovery_sticky_continuations",
        category="recovery",
        severity="info",
        label="Sticky recovery continuations",
    ),
    _signal(
        "stream_recovery_started",
        category="recovery",
        severity="info",
        label="Stream recovery started",
    ),
    _signal(
        "stream_recovery_retry",
        category="recovery",
        severity="info",
        label="Stream recovery retry",
    ),
    _signal(
        "stream_recovery_empty_success",
        category="recovery",
        severity="info",
        label="Empty stream recovered",
    ),
    _signal(
        "stream_recovery_read_error",
        category="recovery",
        severity="info",
        label="Stream read error recovered",
    ),
    _signal(
        "stream_recovery_success_text",
        category="recovery",
        severity="info",
        label="Stream recovered as text",
    ),
    _signal(
        "stream_recovery_success_tool_use",
        category="recovery",
        severity="info",
        label="Stream recovered as tool use",
    ),
    _signal(
        "stream_recovery_fallback",
        category="recovery",
        severity="warning",
        label="Stream recovery fallback",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_provider_sensitive_degraded_to_provider_safe",
        category="provider_safety",
        severity="warning",
        label="Provider-sensitive request degraded safely",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_provider_sensitive_blocked_local",
        category="provider_safety",
        severity="warning",
        label="Provider-sensitive request blocked locally",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_provider_pairing_risk_detected",
        category="provider_safety",
        severity="warning",
        label="Provider pairing risk detected",
        affects_health=True,
    ),
    _signal(
        "fail_open_retry_upstream_pairing_disagreement",
        category="provider_safety",
        severity="warning",
        label="Upstream pairing disagreement",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_assistant_tool_use_text_interleaving_blocked",
        category="provider_safety",
        severity="warning",
        label="Assistant tool/text interleaving blocked",
        affects_health=True,
    ),
    _signal(
        "tok_bridge_invalid_tool_history_blocked",
        category="provider_safety",
        severity="warning",
        label="Invalid tool history blocked",
        affects_health=True,
    ),
    _signal(
        "evidence_exact_observed",
        category="evidence_safety",
        severity="info",
        label="Exact evidence observed",
    ),
    _signal(
        "evidence_first_exact_observed",
        category="evidence_safety",
        severity="info",
        label="First exact evidence observed",
    ),
    _signal(
        "evidence_non_exact_reference_emitted",
        category="evidence_safety",
        severity="info",
        label="Non-exact evidence reference emitted",
    ),
    _signal(
        "evidence_non_exact_summary_emitted",
        category="evidence_safety",
        severity="info",
        label="Non-exact evidence summary emitted",
    ),
    _signal(
        "evidence_non_exact_skeleton_emitted",
        category="evidence_safety",
        severity="info",
        label="Non-exact evidence skeleton emitted",
    ),
    _signal(
        "evidence_exact_reacquisition_required",
        category="evidence_safety",
        severity="info",
        label="Exact evidence reacquisition required",
    ),
    _signal(
        "evidence_exact_reacquisition_satisfied",
        category="evidence_safety",
        severity="info",
        label="Exact evidence reacquisition satisfied",
    ),
    _signal(
        "evidence_compression_blocked_for_safety",
        category="evidence_safety",
        severity="info",
        label="Evidence compression blocked for safety",
    ),
    _signal(
        "evidence_tool_result_compression_skipped",
        category="evidence_safety",
        severity="info",
        label="Tool-result compression skipped for evidence safety",
    ),
    _signal(
        "evidence_history_compression_skipped",
        category="evidence_safety",
        severity="info",
        label="History compression skipped for evidence safety",
    ),
)

SIGNAL_REGISTRY: dict[str, SignalDefinition] = {signal.name: signal for signal in _SIGNALS}

EVIDENCE_SAFETY_SIGNAL_NAMES: tuple[str, ...] = tuple(
    signal.name for signal in _SIGNALS if signal.category == "evidence_safety"
)


def signal_definition(name: str) -> SignalDefinition | None:
    """Return registered metadata for a signal, or None for accepted internal signals."""

    return SIGNAL_REGISTRY.get(name)


def is_registered_signal(name: str) -> bool:
    return name in SIGNAL_REGISTRY


def signals_by_category(category: str) -> tuple[SignalDefinition, ...]:
    return tuple(signal for signal in _SIGNALS if signal.category == category)


def aggregate_signal_category(signals: Mapping[str, int], category: str) -> int:
    names = {signal.name for signal in signals_by_category(category)}
    return sum(int(value) for name, value in signals.items() if name in names)


def aggregate_registered_categories(signals: Mapping[str, int]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for name, value in signals.items():
        definition = signal_definition(name)
        if definition is None:
            continue
        totals[definition.category] = totals.get(definition.category, 0) + int(value)
    return totals


def unregistered_signals(signals: Mapping[str, int]) -> dict[str, int]:
    return {name: int(value) for name, value in signals.items() if name not in SIGNAL_REGISTRY}


__all__ = [
    "EVIDENCE_SAFETY_SIGNAL_NAMES",
    "SIGNAL_REGISTRY",
    "SignalDefinition",
    "aggregate_registered_categories",
    "aggregate_signal_category",
    "is_registered_signal",
    "signal_definition",
    "signals_by_category",
    "unregistered_signals",
]
