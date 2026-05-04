"""Logic for detecting and correcting late-session protocol drift."""


def _is_late_answer_assembly_context(signals: dict[str, int]) -> bool:
    """Check if the current signal state warrants a late answer repair attempt."""
    return any(
        signals.get(key)
        for key in (
            "payload_pressure_ready",
            "compaction_eligible_ready",
            "late_staged_retry_context",
            "late_retry_contract_stage_tool_only",
            "late_retry_contract_stage_answer_only",
        )
    )


def _late_answer_assembly_repair_mode(signals: dict[str, int]) -> str:
    """Determine the repair mode (tool_only vs answer_only) based on signals.

    Priority order (first match wins):
      1. toolless_fresh_answer_event  -> tool_only
      2. mixed_answer_tool_event / answer_ready_mixed_turn_violation -> answer_only
      3. late_retry_contract_stage_answer_only + answer_ready_failed_to_answer -> answer_only
      4. answer_ready_failed_to_answer -> tool_only

    When both toolless_fresh_answer_event and mixed_answer_tool_event are set,
    tool_only wins and mixed_answer_tool_event is suppressed — this is intentional
    but should be tracked via late_answer_assembly_mixed_signal_suppressed.
    """
    if signals.get("toolless_fresh_answer_event"):
        return "tool_only"
    if signals.get("mixed_answer_tool_event") or signals.get("answer_ready_mixed_turn_violation"):
        return "answer_only"
    if signals.get("late_retry_contract_stage_answer_only") and signals.get("answer_ready_failed_to_answer"):
        return "answer_only"
    if signals.get("answer_ready_failed_to_answer"):
        return "tool_only"
    return ""


def _late_answer_assembly_repair_satisfied(repair_mode: str, *, has_tool: bool, has_answer_text: bool) -> bool:
    """Return True if the response satisfies the requested late repair mode."""
    if repair_mode == "tool_only":
        return has_tool and not has_answer_text
    if repair_mode == "answer_only":
        return (not has_tool) and has_answer_text
    return False


def _mark_late_answer_assembly_mode_signal(behavior_signals: dict[str, int], repair_mode: str) -> None:
    """Set the specific repair-mode signal for telemetry."""
    if repair_mode == "tool_only":
        behavior_signals["late_answer_assembly_repair_tool_only"] = 1
    elif repair_mode == "answer_only":
        behavior_signals["late_answer_assembly_repair_answer_only"] = 1


def _mark_late_answer_assembly_mode_counters(
    behavior_signals: dict[str, int], *, repair_mode: str, outcome: str
) -> None:
    """Set outcome counters for late answer repair (mostly for answer_only trials)."""
    if repair_mode != "answer_only":
        return
    if outcome == "requested":
        behavior_signals["late_answer_assembly_repair_answer_only_requested"] = 1
    elif outcome == "resolved":
        behavior_signals["late_answer_assembly_repair_answer_only_resolved"] = 1
    elif outcome == "failed":
        behavior_signals["late_answer_assembly_repair_answer_only_failed"] = 1


def _mark_late_answer_assembly_suppressed_mixed_signal(
    behavior_signals: dict[str, int],
    signals: dict[str, int],
) -> None:
    """Track when mixed_answer_tool_event is suppressed by toolless_fresh_answer_event."""
    if signals.get("toolless_fresh_answer_event") and signals.get("mixed_answer_tool_event"):
        behavior_signals["late_answer_assembly_mixed_signal_suppressed"] = (
            behavior_signals.get("late_answer_assembly_mixed_signal_suppressed", 0) + 1
        )
