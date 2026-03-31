"""Semantic validation, protocol drift detection, and reasoning metrics."""

import re
from typing import Any

MEMORY_LIFT_SIGNALS = (
    "cold_start_structured_memory",
    "durable_promotions",
    "hot_promotions",
    "file_snapshot_recorded",
    "search_snapshot_recorded",
)

SEMANTIC_REGRESSION_SIGNALS = (
    "non_tok_response",
    "fail_open_compat_response",
    "malformed_tok_response",
    "tool_contract_failure",
    "blocker_rediscovery",
    "repeat_file_read",
    "repeat_search",
)


def calculate_invisible_pressure(signals: dict[str, int]) -> int:
    return sum(
        signals.get(name, 0)
        for name in (
            "repeat_file_read",
            "repeat_search",
            "python_c_workaround",
            "stderr_workaround",
            "non_tok_response",
            "fail_open_compat_response",
            "malformed_tok_response",
            "tool_contract_failure",
            "cold_start_wire_fallback",
            "semantic_drift_detected",
            "semantic_pressure_detected",
        )
    )


def calculate_memory_lift(signals: dict[str, int]) -> int:
    return sum(signals.get(name, 0) for name in MEMORY_LIFT_SIGNALS)


def calculate_semantic_regression_score(signals: dict[str, int]) -> int:
    return sum(signals.get(name, 0) for name in SEMANTIC_REGRESSION_SIGNALS)


def pressure_score(signals: dict[str, int]) -> int:
    """Compute an aggregate protocol-pressure score for optimization gates."""
    return calculate_invisible_pressure(signals)


class SemanticValidator:
    """Monitors and corrects protocol drift (the 'Reflex' layer)."""

    def __init__(self) -> None:
        self.drift_count = 0

    def validate_drift(
        self, text: str, behavior_signals: dict[str, Any]
    ) -> dict[str, Any]:
        """Detect and log semantic drift patterns."""
        drift_signals = {}

        if "|> SNAP" in text:
            drift_signals["tok_memory_snap_triggered"] = 1

        # 1. Detect redundant human prose (Leakage)
        # Case A: Multi-word conversational phrases that signal verbose filler
        if re.search(
            r"(?i)\b(I have|I'll have|I'll|successfully|requested|certainly|summarized|investigate)\b",
            text,
        ):
            if len(text.split()) > 10:
                drift_signals["semantic_drift_detected"] = 1

        # Case B: Long non-Tok responses (absence of protocol markers)
        # If the response is over 40 words and contains no >>> marker, it is a prose leak.
        # This triggers for "Victorian Poet" or overly creative / non-mechanical filler.
        if ">>>" not in text and len(text.split()) > 40:
            drift_signals["semantic_drift_detected"] = 1

        # Case C: Bullet-list prose without Tok markers — gradual drift indicator.
        # A response with multiple "- " bullet lines but no @msg or >>> is leaking
        # narrative structure into the protocol layer.
        bullet_lines = [
            ln for ln in text.splitlines() if ln.lstrip().startswith("- ")
        ]
        if len(bullet_lines) >= 2 and "@msg" not in text and ">>>" not in text:
            drift_signals["semantic_drift_detected"] = 1

        # Case D: @msg block with plain paragraph body instead of |> prefix.
        # Detects drift where the model starts using @msg but abandons the |> convention.
        # Fixed: must allow leading whitespace as Tok is often indented.
        if "@msg" in text and re.search(r"@msg[^\n]*\n\s*[^| ]", text):
            drift_signals["semantic_pressure_detected"] = 1

        # 2. Detect protocol friction (raw markdown headers)
        if re.search(r"^(#|##|###) ", text, re.MULTILINE):
            drift_signals["semantic_pressure_detected"] = 1

        # 4. Repeated tool patterns indicating 'Cognitive Tax'
        if (
            behavior_signals.get("repeat_file_read", 0) > 1
            or behavior_signals.get("repeat_search", 0) > 1
        ):
            drift_signals["semantic_pressure_detected"] = 1

        if drift_signals:
            self.drift_count += 1

        return drift_signals
