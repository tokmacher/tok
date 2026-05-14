from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tok.runtime.config import (
    TOK_REPEAT_COMMAND_SUPPRESSION_HINT,
    TOK_TOOL_REQUIRED_LATCH_THRESHOLD,
)
from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline.request_preparation import (
    _capture_repeat_target_snapshots,
    _has_unresolved_tool_required_conditions,
    _is_answer_ready_turn,
    _is_read_only_audit_turn,
)
from tok.runtime.types import RuntimeRequest

from ._prepare_init_context import _has_exact_search_evidence


@dataclass
class Step5Result:
    answer_ready: bool = False
    late_answer_followthrough_active: bool = False
    late_answer_assembly_repair_active: bool = False
    late_answer_assembly_repair_mode: str = ""
    answer_ready_repair_active: bool = False
    preserve_exact_search_evidence: bool = False
    has_answer_anchor: bool = False
    read_only_audit_turn: bool = False
    tool_required_latch_active: bool = False
    behavior_signals: dict[str, int] = field(default_factory=dict)
    runtime_hints: list[str] = field(default_factory=list)
    resend_signals: dict[str, int] = field(default_factory=dict)
    exact_search_evidence_keys_in_request: set[str] = field(default_factory=set)


def run_step_5(
    session: RuntimeSession,
    request: RuntimeRequest,
    translated_messages: list[dict[str, Any]],
    id_to_context: dict[str, dict[str, Any]],
    normalized_tool_events: list[Any],
    behavior_signals: dict[str, int],
    effective_tool_compatible: bool,
    initial_answer_facts_present: bool,
    initial_exact_search_evidence_present: bool,
    exact_search_evidence_keys_in_request: set[str],
    plan_finalization_turn: bool,
    initial_runtime_hints: list[str],
) -> Step5Result:
    runtime_hints = list(initial_runtime_hints)
    if session.consume_loop_detected():
        behavior_signals["loop_terminated"] = 1
    if behavior_signals.get("repeat_command_stable_no_change", 0) > 0:
        runtime_hints.append(TOK_REPEAT_COMMAND_SUPPRESSION_HINT)
        behavior_signals["repeat_command_suppression_hint_injected"] = 1
    resend_signals: dict[str, int] = {}
    has_answer_anchor = False
    preserve_exact_search_evidence = False
    read_only_audit_turn = effective_tool_compatible and _is_read_only_audit_turn(translated_messages)
    tool_required_unresolved = _has_unresolved_tool_required_conditions(translated_messages)
    if effective_tool_compatible and not session._baseline_only and tool_required_unresolved:
        session._tool_required_latch_streak += 1
        behavior_signals["tool_required_condition_unresolved"] = 1
    else:
        session._tool_required_latch_streak = 0
    tool_required_latch_active = (
        effective_tool_compatible
        and not session._baseline_only
        and session._tool_required_latch_streak >= TOK_TOOL_REQUIRED_LATCH_THRESHOLD
    )
    if tool_required_latch_active:
        behavior_signals["tool_required_latch_active"] = 1
    late_answer_followthrough_active = (
        effective_tool_compatible
        and session._late_answer_followthrough_pending
        and not session._baseline_only
        and not read_only_audit_turn
    )
    late_answer_assembly_repair_active = (
        effective_tool_compatible
        and session._late_answer_assembly_repair_pending
        and not session._baseline_only
        and not late_answer_followthrough_active
        and not read_only_audit_turn
    )
    late_answer_assembly_repair_mode = (
        session._late_answer_assembly_repair_mode_pending if late_answer_assembly_repair_active else ""
    )
    answer_ready_repair_active = (
        effective_tool_compatible
        and session._answer_ready_repair_pending
        and not session._baseline_only
        and not late_answer_followthrough_active
        and not late_answer_assembly_repair_active
        and not read_only_audit_turn
    )
    session._late_answer_followthrough_active = late_answer_followthrough_active
    session._answer_ready_repair_active = answer_ready_repair_active
    session._late_answer_assembly_repair_active = late_answer_assembly_repair_active
    session._late_answer_assembly_repair_mode_active = late_answer_assembly_repair_mode

    repeat_snapshot_signals = _capture_repeat_target_snapshots(translated_messages, id_to_context, session)
    if repeat_snapshot_signals:
        session._bump_signals(repeat_snapshot_signals)
        for key, value in repeat_snapshot_signals.items():
            behavior_signals[key] = behavior_signals.get(key, 0) + value

    answer_ready_turn = False
    if effective_tool_compatible:
        has_answer_facts = initial_answer_facts_present or any(
            entry.value.startswith("answer_")
            for bucket in (session.bridge_memory.hot, session.bridge_memory.durable)
            for entry in bucket.get("facts", [])
        )
        seen_exact_search_evidence = initial_exact_search_evidence_present or _has_exact_search_evidence(
            session._first_exact_evidence_seen | session._pending_exact_evidence_keys
        )
        has_answer_anchor = bool(has_answer_facts or seen_exact_search_evidence)
        answer_ready_turn = _is_answer_ready_turn(
            translated_messages,
            tool_compatible=effective_tool_compatible,
            has_answer_anchor=has_answer_anchor,
            baseline_only=session._baseline_only,
        )
        if tool_required_latch_active:
            answer_ready_turn = False
        preserve_exact_search_evidence = bool(answer_ready_turn and exact_search_evidence_keys_in_request)
    session._answer_phase_expected_this_turn = bool(answer_ready_turn)

    return Step5Result(
        answer_ready=answer_ready_turn,
        late_answer_followthrough_active=late_answer_followthrough_active,
        late_answer_assembly_repair_active=late_answer_assembly_repair_active,
        late_answer_assembly_repair_mode=late_answer_assembly_repair_mode,
        answer_ready_repair_active=answer_ready_repair_active,
        preserve_exact_search_evidence=preserve_exact_search_evidence,
        has_answer_anchor=has_answer_anchor,
        read_only_audit_turn=read_only_audit_turn,
        tool_required_latch_active=tool_required_latch_active,
        behavior_signals=behavior_signals,
        runtime_hints=runtime_hints,
        resend_signals=resend_signals,
        exact_search_evidence_keys_in_request=exact_search_evidence_keys_in_request,
    )
