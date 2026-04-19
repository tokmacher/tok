from __future__ import annotations

from typing import Any


def fields_match_expected(expected_fields: dict[str, str], observed_fields: dict[str, str]) -> bool:
    if not expected_fields:
        return bool(observed_fields.get("file") or observed_fields.get("verification"))
    expected_file = expected_fields.get("file", "").lower()
    expected_verification = expected_fields.get("verification", "").lower()
    observed_file = observed_fields.get("file", "").lower()
    observed_verification = observed_fields.get("verification", "").lower()
    if expected_file and expected_file not in observed_file:
        return False
    return not (expected_verification and expected_verification not in observed_verification)


def payload_pressure_reached(
    total_evidence_chars: int, validated_anchor_count: int, min_payload_pressure_bytes: int
) -> bool:
    return validated_anchor_count >= 2 and total_evidence_chars >= min_payload_pressure_bytes


def task_answer_validated(
    *,
    task: Any,
    expected_fields: dict[str, str],
    observed_fields: dict[str, str],
    attempt_tool_count: int,
    attempt_tool_names: set[str],
    validated_reacquisition: bool,
) -> bool:
    if not fields_match_expected(expected_fields, observed_fields):
        return False
    if task.forbid_reacquisition and validated_reacquisition:
        return False
    if task.require_fresh_evidence and attempt_tool_count < max(task.require_tool_count, 1):
        return False
    if task.required_tool_names and not (set(task.required_tool_names) & set(attempt_tool_names)):
        return False
    return not (task.phase_name == "retention-probe" and attempt_tool_count > 0)


def turn_has_valid_supporting_tool_backing(turn: Any) -> bool:
    if not turn.tool_uses or turn.validated_target_exact_reacquired:
        return False
    from ..utils import _is_supported_read_only_tool_name

    if any(turn.output_behavior_signals.get(key) for key in ("unsupported_tool_event", "bad_tool_args_event")):
        return False
    return any(_is_supported_read_only_tool_name(block.get("name", "")) for block in turn.tool_uses)


def seed_tool_summary(
    *,
    task: Any,
    expected_fields: dict[str, str],
    tool_uses: list[dict[str, Any]],
    attempt_tool_count_before_turn: int,
    attempt_seed_evidence_sufficient_before_turn: bool,
) -> dict[str, int]:
    summary = {
        "seed_search_tools_used": 0,
        "seed_direct_read_tools_used": 0,
        "seed_evidence_sufficient": 0,
        "repeated_seed_search_without_read": 0,
        "repeated_seed_tool_after_evidence": 0,
    }
    if task.phase_name != "anchor-seed":
        return summary
    expected_file = expected_fields.get("file", "").strip().lower()
    saw_search = False
    saw_direct_read = False
    for block in tool_uses:
        name = str(block.get("name", "")).strip().lower()
        tool_input = block.get("input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}
        path_text = (
            str(
                tool_input.get("path")
                or tool_input.get("file_path")
                or tool_input.get("search_path")
                or tool_input.get("text")
                or ""
            )
            .strip()
            .lower()
        )
        if name in {"grep_search", "search", "grep", "rg"}:
            summary["seed_search_tools_used"] += 1
            saw_search = True
        if name in {"view_file", "read"}:
            summary["seed_direct_read_tools_used"] += 1
            saw_direct_read = True
            if expected_file and expected_file in path_text:
                summary["seed_evidence_sufficient"] = 1
    if saw_search and not saw_direct_read and attempt_tool_count_before_turn > 0:
        summary["repeated_seed_search_without_read"] = 1
    if attempt_seed_evidence_sufficient_before_turn and tool_uses:
        summary["repeated_seed_tool_after_evidence"] = 1
    return summary
