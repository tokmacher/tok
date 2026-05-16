from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from tok.compression import EDIT_LIKE_TOOLS, text_of
from tok.macros.ir import Instruction
from tok.runtime.core import RuntimeSession
from tok.runtime.memory.session_state import extract_memory_items
from tok.runtime.pipeline.request_preparation import collect_transient_error_snippets
from tok.runtime.pipeline.response_processing import translate_request_results
from tok.runtime.pipeline.tool_processing import (
    build_tool_use_id_to_context,
    collect_behavior_signals,
    normalize_tool_events,
)
from tok.runtime.repeat_targets import evidence_identity_key, search_result_evidence_level
from tok.runtime.types import RuntimeRequest

_RECENT_COMMAND_WINDOW = 10
_DEFAULT_JIT_HIT_THRESHOLD = 3


def _env_int_or_default(name: str, default: int) -> int:
    from tok.utils.env_utils import env_int_or_default

    return env_int_or_default(name, default)


def _exact_search_evidence_keys_in_messages(
    messages: list[dict[str, Any]],
    tool_use_id_to_context: dict[str, dict[str, Any]],
) -> set[str]:
    evidence_keys: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_id = str(block.get("tool_use_id") or "")
            context = tool_use_id_to_context.get(tool_id)
            if not context:
                continue
            tool_name = str(context.get("name") or "").strip().lower()
            raw = text_of(cast("Any", block.get("content", ""))).strip()
            if not raw:
                continue
            if raw.startswith(">>>"):
                continue
            if search_result_evidence_level(raw) != "exact_content":
                continue
            args = context.get("args") if isinstance(context.get("args"), dict) else {}
            command = str(args.get("command") or args.get("cmd") or "").strip() or None
            key = evidence_identity_key(
                tool_name,
                path=str(context.get("path") or "").strip() or None,
                query=str(context.get("query") or "").strip() or None,
                command=command,
                args=args,
            )
            if key and key.startswith("search|"):
                evidence_keys.add(key)
    return evidence_keys


def _is_first_turn_broad_audit_batch(
    request: RuntimeRequest,
    session: RuntimeSession,
    normalized_tool_events: list[Any],
) -> bool:
    if not request.uses_first_turn_broad_audit_guard or session.bridge_memory.turn > 1:
        return False
    if session._stream_recovery_reacquisition_budget > 0 or session._stream_recovery_history_floor_budget > 0:
        return False
    file_read_count = sum(1 for event in normalized_tool_events if event.compressibility_class == "file_read")
    if file_read_count < 8:
        return False
    return True


def _edit_events_requiring_exact_reacquisition(
    session: RuntimeSession,
    normalized_tool_events: list[Any],
) -> dict[str, int]:
    signals: dict[str, int] = {}
    for event in normalized_tool_events:
        if not session._is_edit_tool_event(event):
            continue
        file_path = session._extract_file_path_from_event(event)
        if not file_path:
            continue
        exact_key = evidence_identity_key(
            "read_file",
            path=file_path,
            args=event.args if isinstance(event.args, dict) else None,
        )
        if not exact_key or not session.evidence_requires_reacquisition(exact_key):
            continue
        for key, value in session.require_exact_reacquisition(exact_key).items():
            signals[key] = signals.get(key, 0) + value
    return signals


@dataclass
class Step3Result:
    body: dict[str, Any] = field(default_factory=dict)
    plan_finalization_turn: bool = False
    behavior_signals: dict[str, int] = field(default_factory=dict)
    id_to_context: dict[str, dict[str, Any]] = field(default_factory=dict)
    normalized_tool_events: list[Any] = field(default_factory=list)
    broad_audit_batch: bool = False
    edit_reacquisition_signals: dict[str, int] = field(default_factory=dict)
    should_skip_history: bool = False
    skip_reason: str = ""
    history_skip_reason: str = ""
    injected_state_payload: str = ""
    exact_search_evidence_keys_in_request: set[str] = field(default_factory=set)
    suppress_reacquisition_once: bool = False
    stream_recovery_history_floor_active: bool = False
    runtime_hints: list[str] = field(default_factory=list)
    translated_messages: list[dict[str, Any]] = field(default_factory=list)


def run_step_3(
    request: RuntimeRequest,
    session: RuntimeSession,
    body: dict[str, Any],
    is_bridge_adapter: bool,
) -> Step3Result:
    from tok.runtime.pipeline.request_preparation import is_plan_or_answer_finalization_turn
    from tok.runtime.policy.macro_handling import _jit_context_matches

    translated_messages = translate_request_results(body.get("messages", []))
    body["messages"] = translated_messages
    plan_finalization_turn = request.uses_plan_finalization_guard and is_plan_or_answer_finalization_turn(
        translated_messages
    )

    rolling_cmds = session.bridge_memory.rolling_cmds
    jit_macro = None
    threshold = _env_int_or_default("TOK_JIT_HIT_THRESHOLD", _DEFAULT_JIT_HIT_THRESHOLD)
    if rolling_cmds:
        recent_instructions: list[Instruction] = []
        for entry in rolling_cmds[-_RECENT_COMMAND_WINDOW:]:
            parts = entry.value.strip().split()
            if not parts:
                continue
            recent_instructions.append(Instruction(op=parts[0], args=tuple(parts[1:])))

        jit_macro = session.bridge_memory.macro_registry.match_recent_sequence(recent_instructions)
        if jit_macro and jit_macro.hit_count >= threshold and _jit_context_matches(jit_macro, session):
            session.pending_behavior_signals["jit_offer_available"] = 1
            session.pending_behavior_signals[f"jit_offer_{jit_macro.name}"] = 1
            session._pending_macro_heal = jit_macro.name
            session._pending_macro_heal_turn = session.bridge_memory.turn
        elif jit_macro and not _jit_context_matches(jit_macro, session):
            session.pending_behavior_signals["jit_offer_context_filtered"] = 1

    _speculative_macro_hint: str | None = None
    if jit_macro and jit_macro.hit_count >= threshold and rolling_cmds and _jit_context_matches(jit_macro, session):
        from tok.runtime.policy.macro_handling import execute_macro_proactively

        _speculative_macro_hint = execute_macro_proactively(session, jit_macro, session.bridge_memory.rolling_cmds)

    id_to_context = build_tool_use_id_to_context(translated_messages, session)
    exact_search_evidence_keys_in_request = _exact_search_evidence_keys_in_messages(translated_messages, id_to_context)
    for ctx in id_to_context.values():
        path = ctx.get("path")
        if path:
            session._file_reads_by_turn[path] = session.bridge_memory.turn
    suppress_reacquisition_once = session._stream_recovery_reacquisition_budget > 0
    stream_recovery_history_floor_active = session._stream_recovery_history_floor_budget > 0
    if stream_recovery_history_floor_active:
        session._stream_recovery_history_floor_budget = max(0, session._stream_recovery_history_floor_budget - 1)
    behavior_signals = collect_behavior_signals(
        translated_messages,
        id_to_context,
        suppress_reacquisition_once=suppress_reacquisition_once,
    )
    if suppress_reacquisition_once:
        session._stream_recovery_reacquisition_budget = max(0, session._stream_recovery_reacquisition_budget - 1)
    behavior_signals["_project_markers_proxy"] = len(session._project_markers)
    for err_snippet in collect_transient_error_snippets(translated_messages):
        session.bridge_memory._upsert(session.bridge_memory.hot, "errs", err_snippet, score_delta=1)

    blockers, hypotheses = extract_memory_items(translated_messages)
    for blocker in blockers:
        session.bridge_memory._upsert(session.bridge_memory.hot, "blockers", blocker, score_delta=2)
    for hypothesis in hypotheses:
        session.bridge_memory._upsert(session.bridge_memory.hot, "questions", hypothesis, score_delta=2)

    normalized_tool_events = normalize_tool_events(translated_messages)
    broad_audit_batch = (
        not suppress_reacquisition_once
        and not stream_recovery_history_floor_active
        and _is_first_turn_broad_audit_batch(request, session, normalized_tool_events)
    )
    injected_state_payload = ""
    history_skip_reason = ""
    should_skip_history = False
    skip_reason = ""
    for event in normalized_tool_events:
        if event.name.lower() in EDIT_LIKE_TOOLS and event.path:
            session.bridge_memory.bump_file_heat(event.path, weight=2.0)
    edit_reacquisition_signals = _edit_events_requiring_exact_reacquisition(session, normalized_tool_events)
    if edit_reacquisition_signals:
        for key, value in edit_reacquisition_signals.items():
            behavior_signals[key] = behavior_signals.get(key, 0) + value
        should_skip_history = True
        skip_reason = "evidence_exact_reacquisition"
        history_skip_reason = skip_reason
    if broad_audit_batch:
        behavior_signals["broad_audit_tok_additions_suppressed"] = 1

    runtime_hints = [h for h in [_speculative_macro_hint] if h]

    return Step3Result(
        body=body,
        plan_finalization_turn=plan_finalization_turn,
        behavior_signals=behavior_signals,
        id_to_context=id_to_context,
        normalized_tool_events=normalized_tool_events,
        broad_audit_batch=broad_audit_batch,
        edit_reacquisition_signals=edit_reacquisition_signals,
        should_skip_history=should_skip_history,
        skip_reason=skip_reason,
        history_skip_reason=history_skip_reason,
        injected_state_payload=injected_state_payload,
        exact_search_evidence_keys_in_request=exact_search_evidence_keys_in_request,
        suppress_reacquisition_once=suppress_reacquisition_once,
        stream_recovery_history_floor_active=stream_recovery_history_floor_active,
        runtime_hints=runtime_hints,
        translated_messages=translated_messages,
    )
