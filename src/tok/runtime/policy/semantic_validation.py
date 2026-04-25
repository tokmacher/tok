"""Semantic validation, protocol drift detection, and reasoning metrics."""

import re
from typing import Any

from tok.runtime.policy.translator import IS_TOK
from tok.utils.event_logging import log_drift_detected

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


def semantic_pressure_score(signals: dict[str, int]) -> int:
    """Compute an aggregate protocol-pressure score for optimization gates."""
    return calculate_invisible_pressure(signals)


class SemanticValidator:
    """Monitors and corrects protocol drift (the 'Reflex' layer)."""

    def __init__(self) -> None:
        pass

    def validate_drift(self, text: str, behavior_signals: dict[str, Any]) -> dict[str, Any]:
        """Detect and log semantic drift patterns."""
        drift_signals = {}

        if "|> SNAP" in text:
            drift_signals["tok_memory_snap_triggered"] = 1

        has_tok_markers = bool(IS_TOK.search(text))

        # 1. Detect redundant human prose (Leakage)
        # Case A: Unambiguous filler phrases with no Tok markers in the response.
        # Narrowed from the original broad list to avoid false positives on
        # legitimate natural-language content (e.g. "successfully", "investigate").
        if not has_tok_markers and re.search(
            r"(?i)\b(certainly|of course|sure thing|no problem|I'll be happy|absolutely)\b",
            text,
        ):
            if len(text.split()) > 10:
                drift_signals["semantic_drift_detected"] = 1
                log_drift_detected("prose_leakage", f"{len(text.split())} words")

        # Case B: Long response with zero Tok markers — genuine prose leak.
        # Gated on IS_TOK so that well-formed natural-language responses (health
        # checks, clarifications) don't inflate semantic_drift_count.
        if not has_tok_markers and len(text.split()) > 40:
            drift_signals["semantic_drift_detected"] = 1
            log_drift_detected("long_prose", f"{len(text.split())} words no markers")

        # Case C: Bullet-list prose — only a drift signal when no Tok markers present.
        bullet_lines = [ln for ln in text.splitlines() if ln.lstrip().startswith("- ")]
        if len(bullet_lines) >= 2 and not has_tok_markers:
            drift_signals["semantic_drift_detected"] = 1
            log_drift_detected("bullet_prose", f"{len(bullet_lines)} bullets")

        # Case D: @msg block with plain paragraph body instead of |> prefix.
        # Detects drift where the model starts using @msg but abandons the |> convention.
        # Fixed: must allow leading whitespace as Tok is often indented.
        if "@msg" in text and re.search(r"@msg[^\n]*\n\s*[^| ]", text):
            drift_signals["semantic_pressure_detected"] = 1

        # 2. Detect protocol friction (raw markdown headers)
        if re.search(r"^(#|##|###) ", text, re.MULTILINE):
            drift_signals["semantic_pressure_detected"] = 1

        # 4. Repeated tool patterns indicating 'Cognitive Tax'
        if behavior_signals.get("repeat_file_read", 0) > 1 or behavior_signals.get("repeat_search", 0) > 1:
            drift_signals["semantic_pressure_detected"] = 1

        return drift_signals
