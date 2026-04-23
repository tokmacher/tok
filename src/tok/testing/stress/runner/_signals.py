from __future__ import annotations

from typing import Any


def tool_signature(block: dict[str, Any]) -> tuple[str, str]:
    name = str(block.get("name", "")).strip().lower()
    tool_input = block.get("input", {})
    if not isinstance(tool_input, dict):
        tool_input = {}
    path = str(tool_input.get("path") or tool_input.get("file_path") or tool_input.get("search_path") or "").strip()
    query = str(tool_input.get("query") or tool_input.get("pattern") or tool_input.get("search") or "").strip()
    return name, f"{path}|{query}"


def repetition_signals(
    conversation: list[dict[str, Any]],
    tool_uses: list[dict[str, Any]],
    *,
    target_already_validated: bool,
) -> dict[str, int]:
    if not target_already_validated:
        return {}
    prior_signatures: set[tuple[str, str]] = set()
    for message in conversation:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            prior_signatures.add(tool_signature(block))

    signals: dict[str, int] = {}
    for block in tool_uses:
        sig = tool_signature(block)
        if sig not in prior_signatures:
            continue
        name = sig[0]
        if name in {"grep_search", "search", "grep", "rg"}:
            signals["repeat_search"] = signals.get("repeat_search", 0) + 1
        elif name in {"view_file", "read"}:
            signals["repeat_file_read"] = signals.get("repeat_file_read", 0) + 1
        signals["reacquisition_cost_tokens"] = signals.get("reacquisition_cost_tokens", 0) + 50
    return signals


def resend_mode(behavior_signals: dict[str, int]) -> str:
    if behavior_signals.get("state_resend_full_turn"):
        return "full"
    if behavior_signals.get("state_resend_delta_turn"):
        return "delta"
    if behavior_signals.get("state_resend_suppressed_turn"):
        return "suppressed"
    return "none"


def resend_decision_reason(behavior_signals: dict[str, int]) -> str:
    explicit_reasons = (
        (
            "state_resend_reason_answer_anchor_present_kept_full",
            "answer_anchor_present_kept_full",
        ),
        ("state_resend_reason_delta_selected", "delta_resend_selected"),
        ("state_resend_reason_state_suppressed", "state_suppressed"),
        (
            "state_resend_reason_history_compression_skipped",
            "history_compression_skipped",
        ),
        (
            "state_resend_reason_tool_compatible_compression_without_resend_change",
            "tool_compatible_compression_without_resend_change",
        ),
        ("state_resend_reason_delta_not_smaller", "delta_not_smaller"),
        ("state_resend_reason_full_default", "full_resend_default"),
    )
    for key, label in explicit_reasons:
        if behavior_signals.get(key):
            return label
    if behavior_signals.get("state_resend_delta_turn"):
        return "delta_resend_selected"
    if behavior_signals.get("state_resend_suppressed_turn"):
        return "state_suppressed"
    if behavior_signals.get("state_resend_full_turn"):
        if behavior_signals.get("answer_anchor_present"):
            return "answer_anchor_present_kept_full"
        for key in sorted(behavior_signals):
            if key.startswith("tok_skip_") and behavior_signals.get(key):
                return key
        if behavior_signals.get("tool_compatible_compression"):
            return "tool_compatible_compression_without_resend_change"
        return "full_resend_default"
    return "no_resend_signal"


def update_failure_counter(
    consecutive_failures: int,
    *,
    protocol_failure: bool,
    tool_contract_failure: bool,
    suppress_failure_increment: bool = False,
    fallback_threshold: int = 3,
) -> tuple[int, bool, bool]:
    failure = bool(tool_contract_failure or protocol_failure)
    incremented = False
    if failure:
        if not suppress_failure_increment:
            consecutive_failures += 1
            incremented = True
    else:
        consecutive_failures = 0
    trigger_baseline = consecutive_failures >= fallback_threshold
    return consecutive_failures, incremented, trigger_baseline
