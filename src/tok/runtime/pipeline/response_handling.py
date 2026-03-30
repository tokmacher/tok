"""Helper functions for handling runtime responses."""

from __future__ import annotations
import copy
import logging
import re
from typing import TYPE_CHECKING, Any

from ..policy.semantic_validation import calculate_invisible_pressure
from ..types import ReplayGateResult

if TYPE_CHECKING:
    from ..core import RuntimeSession

logger = logging.getLogger("tok.runtime")


def has_forbidden_tok_hybrid_patterns(text: str) -> bool:
    """Return True if text contains raw JSON tool patterns that should be formatted via Tok."""
    lowered = text.lower()
    return any(
        pattern in lowered
        for pattern in (
            "@tool(json=",
            "@tool(json:",
            "@tool({",
            "@tool(",
            '"type": "tool_use"',
        )
    )


def has_non_inverted_assistant_message(text: str) -> bool:
    """Return True if an assistant message block is missing proper Tok inversion markers."""
    in_msg_assistant = False
    block_is_inverted = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">>>"):
            in_msg_assistant = False
            block_is_inverted = False
            continue
        if stripped.startswith("@"):
            in_msg_assistant = (
                stripped.startswith("@msg") and "role:assistant" in stripped
            )
            block_is_inverted = False
            continue

        if in_msg_assistant:
            if re.match(r"^\s+\|[#\d]?>", line):
                block_is_inverted = True
                continue
            if not block_is_inverted:
                return True
    return False


def has_markdown_fallback_after_tok_header(text: str) -> bool:
    """Return True if a Tok header is followed directly by markdown headers."""
    saw_header = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">>>"):
            saw_header = True
            continue
        if saw_header and re.match(r"^#{1,6}\s+", stripped):
            return True
    return False


def sort_cache_control_blocks(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort content blocks by cache_control TTL for Anthropic durability efficiency."""
    from ..config import TTL_SECONDS

    if not isinstance(messages, list):
        return messages
    sorted_messages = copy.deepcopy(messages)
    for msg in sorted_messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        msg["content"] = sorted(
            content,
            key=lambda block: (
                0
                if not isinstance(block, dict)
                else (
                    0
                    if "cache_control" not in block
                    else -TTL_SECONDS.get(
                        block.get("cache_control", {}).get("ttl", ""), 0
                    )
                )
            ),
        )
    return sorted_messages


def evaluate_replay_gate(
    replay_meta: dict[str, Any],
    *,
    savings_pct: float,
    behavior_signals: dict[str, int],
) -> ReplayGateResult:
    """Evaluate whether a replay should be granted based on performance gates."""
    invisible_pressure = calculate_invisible_pressure(behavior_signals)
    gate_checks = {
        "min_savings_pct": savings_pct
        >= float(replay_meta["min_savings_pct"]),
        "max_invisible_pressure": invisible_pressure
        <= int(replay_meta["max_invisible_pressure"]),
        "max_repeat_file_read": behavior_signals.get("repeat_file_read", 0)
        <= int(replay_meta["max_repeat_file_read"]),
        "max_repeat_search": behavior_signals.get("repeat_search", 0)
        <= int(replay_meta["max_repeat_search"]),
        "max_non_tok_response": behavior_signals.get("non_tok_response", 0)
        <= int(replay_meta["max_non_tok_response"]),
        "max_fail_open_compat_response": behavior_signals.get(
            "fail_open_compat_response", 0
        )
        <= int(replay_meta["max_fail_open_compat_response"]),
        "max_malformed_tok_response": behavior_signals.get(
            "malformed_tok_response", 0
        )
        <= int(replay_meta["max_malformed_tok_response"]),
        "max_blocker_rediscovery": behavior_signals.get(
            "blocker_rediscovery", 0
        )
        <= int(replay_meta["max_blocker_rediscovery"]),
    }
    failed = [name for name, ok in gate_checks.items() if not ok]
    return ReplayGateResult(
        passed=not failed,
        invisible_pressure=invisible_pressure,
        failed_checks=failed,
    )


def _handle_answer_ready_phase(
    session: RuntimeSession,
    merged_signals: dict[str, int],
    has_tool: bool,
    has_answer_text: bool,
) -> None:
    if merged_signals.get("answer_ready_turn"):
        if has_tool:
            merged_signals["answer_ready_tool_violation"] = 1
        if has_tool and has_answer_text:
            merged_signals["answer_ready_mixed_turn_violation"] = 1
            merged_signals["tool_contract_failure"] = 1
        elif not has_tool and not has_answer_text:
            merged_signals["answer_ready_failed_to_answer"] = 1

    answer_ready_failed = any(
        merged_signals.get(key)
        for key in (
            "answer_ready_tool_violation",
            "answer_ready_mixed_turn_violation",
            "answer_ready_failed_to_answer",
        )
    )

    if answer_ready_failed:
        merged_signals["answer_ready_repair_requested"] = 1
        if session._answer_ready_repair_active:
            merged_signals["answer_ready_repair_failed"] = 1
        session._answer_ready_repair_pending = True
    elif session._answer_ready_repair_active:
        merged_signals["answer_ready_repair_resolved"] = 1
        session._answer_ready_repair_pending = False
    else:
        session._answer_ready_repair_pending = False
    session._answer_ready_repair_active = False


def _handle_followthrough_phase(
    session: RuntimeSession,
    merged_signals: dict[str, int],
    has_tool: bool,
    has_answer_text: bool,
) -> bool:
    from ..policy.answer_repair import _late_answer_assembly_repair_satisfied

    followthrough_guard_blocked = False
    if session._late_answer_followthrough_active:
        if _late_answer_assembly_repair_satisfied(
            "answer_only",
            has_tool=has_tool,
            has_answer_text=has_answer_text,
        ):
            merged_signals["late_answer_followthrough_resolved"] = 1
        else:
            merged_signals["late_answer_followthrough_failed"] = 1
            followthrough_guard_blocked = True
        session._late_answer_followthrough_pending = False
    else:
        session._late_answer_followthrough_pending = False
    session._late_answer_followthrough_active = False

    return followthrough_guard_blocked


def _handle_late_assembly_phase(
    session: RuntimeSession,
    merged_signals: dict[str, int],
    has_tool: bool,
    has_answer_text: bool,
    tool_compatible: bool,
    followthrough_guard_blocked: bool,
) -> None:
    from ..policy.answer_repair import (
        _is_late_answer_assembly_context,
        _late_answer_assembly_repair_mode,
        _late_answer_assembly_repair_satisfied,
        _mark_late_answer_assembly_mode_counters,
    )

    late_answer_assembly_mode = ""
    if (
        tool_compatible
        and not session._baseline_only
        and _is_late_answer_assembly_context(merged_signals)
        and not followthrough_guard_blocked
        and not merged_signals.get("unsupported_tool_event")
        and not merged_signals.get("bad_tool_args_event")
        and not merged_signals.get("validated_target_exact_reacquired")
    ):
        late_answer_assembly_mode = _late_answer_assembly_repair_mode(
            merged_signals
        )

    if late_answer_assembly_mode:
        merged_signals["late_answer_assembly_repair_requested"] = 1
        _mark_late_answer_assembly_mode_counters(
            merged_signals,
            repair_mode=late_answer_assembly_mode,
            outcome="requested",
        )
        if merged_signals.get("late_freshness_signal_promoted"):
            merged_signals["late_freshness_signal_consumed_by_tok"] = 1
        if merged_signals.get("late_mixed_signal_promoted"):
            merged_signals["late_mixed_signal_consumed_by_tok"] = 1
        if session._late_answer_assembly_repair_active:
            merged_signals["late_answer_assembly_repair_failed"] = 1
            _mark_late_answer_assembly_mode_counters(
                merged_signals,
                repair_mode=session._late_answer_assembly_repair_mode_active,
                outcome="failed",
            )
        if session._late_answer_followthrough_pending:
            merged_signals["late_answer_followthrough_failed"] = 1
            session._late_answer_followthrough_pending = False
        session._late_answer_assembly_repair_pending = True
        session._late_answer_assembly_repair_mode_pending = (
            late_answer_assembly_mode
        )
    elif session._late_answer_assembly_repair_active:
        if _late_answer_assembly_repair_satisfied(
            session._late_answer_assembly_repair_mode_active,
            has_tool=has_tool,
            has_answer_text=has_answer_text,
        ):
            merged_signals["late_answer_assembly_repair_resolved"] = 1
            _mark_late_answer_assembly_mode_counters(
                merged_signals,
                repair_mode=session._late_answer_assembly_repair_mode_active,
                outcome="resolved",
            )
            if session._late_answer_assembly_repair_mode_active == "tool_only":
                merged_signals["late_answer_followthrough_requested"] = 1
                merged_signals[
                    "late_answer_followthrough_after_tool_only_repair"
                ] = 1
                session._late_answer_followthrough_pending = True
        session._late_answer_assembly_repair_pending = False
        session._late_answer_assembly_repair_mode_pending = ""
    else:
        session._late_answer_assembly_repair_pending = False
        session._late_answer_assembly_repair_mode_pending = ""
    session._late_answer_assembly_repair_active = False
    session._late_answer_assembly_repair_mode_active = ""


def handle_answer_repair(
    session: RuntimeSession,
    *,
    merged_signals: dict[str, int],
    has_tool: bool,
    has_answer_text: bool,
    tool_compatible: bool,
) -> None:
    """Update repair-pending flags and signals based on response content."""
    _handle_answer_ready_phase(
        session, merged_signals, has_tool, has_answer_text
    )
    followthrough_guard_blocked = _handle_followthrough_phase(
        session, merged_signals, has_tool, has_answer_text
    )
    _handle_late_assembly_phase(
        session,
        merged_signals,
        has_tool,
        has_answer_text,
        tool_compatible,
        followthrough_guard_blocked,
    )


def report_protocol_drift(
    *,
    model: str,
    merged_signals: dict[str, int],
    mode: str,
    session: RuntimeSession,
    content_blocks: list[dict[str, Any]],
) -> None:
    """Emit telemetry for protocol drift events."""
    # Move protocol drift telemetry logic here from process_response
    pass
