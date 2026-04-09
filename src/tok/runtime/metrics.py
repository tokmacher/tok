"""Runtime telemetry, metrics, and cost calculation."""

import logging
from typing import TYPE_CHECKING, Any

from tok.utils.telemetry import emit_event_sync

if TYPE_CHECKING:
    from tok.runtime.core import RuntimeSession

logger = logging.getLogger("tok.runtime")


def calculate_usage_cost(prompt_tokens: int, completion_tokens: int, pricing: dict[str, float]) -> float:
    """Compute USD cost for a given token usage and pricing model."""
    return (prompt_tokens / 1_000_000) * pricing.get("prompt", 0) + (completion_tokens / 1_000_000) * pricing.get(
        "completion", 0
    )


def report_protocol_drift(
    model: str,
    merged_signals: dict[str, int],
    mode: str,
    session: "RuntimeSession",
    content_blocks: list[dict[str, Any]],
) -> None:
    """Orchestrate protocol drift telemetry events."""
    from .policy.semantic_validation import (
        calculate_invisible_pressure,
        calculate_memory_lift,
        calculate_semantic_regression_score,
    )

    signals_to_report = {
        k: v
        for k, v in merged_signals.items()
        if k
        in (
            "semantic_drift_detected",
            "semantic_pressure_detected",
            "non_tok_response",
            "malformed_tok_response",
            "tool_contract_failure",
            "answer_ready_turn",
            "answer_ready_tool_violation",
            "answer_ready_mixed_turn_violation",
            "answer_ready_failed_to_answer",
            "answer_ready_repair_requested",
            "answer_ready_repair_active",
            "answer_ready_repair_resolved",
            "answer_ready_repair_failed",
            "late_answer_assembly_repair_requested",
            "late_answer_assembly_repair_active",
            "late_answer_assembly_repair_tool_only",
            "late_answer_assembly_repair_answer_only",
            "late_answer_assembly_repair_answer_only_requested",
            "late_answer_assembly_repair_answer_only_resolved",
            "late_answer_assembly_repair_answer_only_failed",
            "late_answer_assembly_repair_resolved",
            "late_answer_assembly_repair_failed",
            "late_answer_followthrough_requested",
            "late_answer_followthrough_active",
            "late_answer_followthrough_resolved",
            "late_answer_followthrough_failed",
            "late_answer_followthrough_after_tool_only_repair",
            "mixed_answer_tool_event",
            "mixed_tool_visible_text",
            "answer_phase_tool_intent_quarantined",
            "answer_phase_non_labeled_fallback_applied",
            "answer_phase_fallback_failed_no_anchor",
            "tok_drift_healed",
        )
    }

    if not signals_to_report:
        return

    emit_event_sync(
        "protocol_drift",
        {
            "signals": signals_to_report,
            "mode": mode,
            "tool_density": getattr(session, "_current_tool_density", 0),
            "context_char_count": getattr(session, "_current_context_char_count", 0),
            "invisible_pressure": calculate_invisible_pressure(merged_signals),
            "semantic_regression": calculate_semantic_regression_score(merged_signals),
            "memory_lift": calculate_memory_lift(merged_signals),
            "reasoning_depth": session.reasoning_depth_per_token(),
            "active_tools": getattr(session, "_active_tools", []),
            "current_tools": [
                block["name"] for block in content_blocks if block.get("type") == "tool_use" and block.get("name")
            ],
        },
        model=model,
    )

    if merged_signals.get("tok_drift_healed"):
        emit_event_sync(
            "drift_healed",
            {"healed": True},
            model=model,
        )
