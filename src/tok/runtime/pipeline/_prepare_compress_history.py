from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tok.runtime.core import RuntimeSession
from tok.runtime.types import RuntimeRequest

from ._prepare_bridge_cut_search import run_step_7a_bridge_cut_search
from ._prepare_translate_classify import _exact_search_evidence_keys_in_messages


@dataclass
class Step7Result:
    body: dict[str, Any] = field(default_factory=dict)
    recent: list[dict[str, Any]] = field(default_factory=list)
    tok_state: str = ""
    session_memory: str = ""
    compressed: bool = False
    behavior_signals: dict[str, int] = field(default_factory=dict)
    type_breakdown: dict[str, int] = field(default_factory=dict)
    should_skip_history: bool = False
    skip_reason: str = ""
    history_skip_reason: str = ""
    saved_tokens: int = 0
    injected_state_payload: str = ""
    keep_turns: int = 3
    bridge_keep_turns: int = 3


def run_step_7(
    *,
    session: RuntimeSession,
    request: RuntimeRequest,
    normalized_tool_events: list[Any],
    body: dict[str, Any],
    id_to_context: dict[str, dict[str, Any]],
    behavior_signals: dict[str, int],
    effective_tool_compatible: bool,
    mode: str,
    policy: Any,
    should_skip_history: bool,
    skip_reason: str,
    history_skip_reason: str,
    preserve_exact_search_evidence: bool,
    plan_finalization_turn: bool,
    broad_audit_batch: bool,
    edit_reacquisition_signals: dict[str, int],
    stream_recovery_history_floor_active: bool,
    session_memory: str,
    history_baseline_prompt_tokens: int,
    seen_mutation_pairs: set[tuple[str, str]] | None,
    saved_tokens: int,
    compressed: bool,
    current_pressure: float,
    request_policy: str,
    exact_search_evidence_keys_in_request: set[str],
    recent: list[dict[str, Any]],
    tok_state: str,
    type_breakdown: dict[str, int],
    keep_turns: int,
    bridge_keep_turns: int,
    bridge_profile: dict[str, Any],
    h_profile: dict[str, Any],
    _first_exact_evidence_seen_for_compression: frozenset[str],
) -> Step7Result:
    from tok.compression import compress_history, compress_recent_window
    from tok.runtime._history_slicing import _stream_recovery_winnowing_floor_messages
    from tok.runtime.config import _SHORT_SESSION_THRESHOLD
    from tok.runtime.core import logger
    from tok.runtime.pipeline.tool_processing import _should_skip_history_rewrite

    injected_state_payload = ""
    tok_state_out = tok_state
    recent_out = recent
    should_skip_history_out = should_skip_history
    skip_reason_out = skip_reason
    history_skip_reason_out = history_skip_reason
    compressed_out = compressed
    saved_tokens_out = saved_tokens
    session_memory_out = session_memory
    type_breakdown_out = dict(type_breakdown)
    behavior_signals_out = dict(behavior_signals)

    if not should_skip_history_out:
        if broad_audit_batch:
            should_skip_history_out = True
            skip_reason_out = "broad_audit"
            history_skip_reason_out = skip_reason_out
            behavior_signals_out["broad_audit_history_skipped"] = 1
        elif edit_reacquisition_signals:
            should_skip_history_out = True
            skip_reason_out = "evidence_exact_reacquisition"
            history_skip_reason_out = skip_reason_out
            behavior_signals_out["evidence_history_compression_skipped"] = 1
        elif plan_finalization_turn:
            should_skip_history_out = True
            skip_reason_out = "plan_finalization"
            history_skip_reason_out = skip_reason_out
            behavior_signals_out["plan_finalization_history_skipped"] = 1
        elif stream_recovery_history_floor_active:
            should_skip_history_out = True
            skip_reason_out = "stream_recovery_history_floor"
            history_skip_reason_out = skip_reason_out
            behavior_signals_out["stream_recovery_history_floor_applied"] = 1
        elif session.bridge_memory.turn < _SHORT_SESSION_THRESHOLD and not (
            request.supports_tool_pairs and _messages_contain_tool_material(body["messages"])
        ):
            should_skip_history_out = True
            skip_reason_out = "short_session"
            history_skip_reason_out = skip_reason_out
            behavior_signals_out["short_session_history_skipped"] = 1
        else:
            # Preserve the original skip-history heuristic based on tool activity.
            should_skip_history_out, skip_reason_out = _should_skip_history_rewrite(
                request.messages,
                normalized_tool_events,
                tool_compatible=effective_tool_compatible,
            )

    if should_skip_history_out:
        if stream_recovery_history_floor_active:
            floored_recent = _stream_recovery_winnowing_floor_messages(body["messages"])
            if floored_recent:
                if len(floored_recent) < len(body["messages"]):
                    compressed_out = True
                recent_out = floored_recent
                body["messages"] = recent_out
                behavior_signals_out["stream_recovery_history_floor_kept_context"] = 1
            else:
                recent_out = body["messages"]
                behavior_signals_out["stream_recovery_history_floor_noop"] = 1
        else:
            behavior_signals_out["tok_history_compression_skipped"] = (
                behavior_signals_out.get("tok_history_compression_skipped", 0) + 1
            )
            if skip_reason_out:
                behavior_signals_out[f"tok_skip_{skip_reason_out}"] = 1
                history_skip_reason_out = skip_reason_out
            recent_out = body["messages"]
    else:
        if skip_reason_out:
            behavior_signals_out[f"tok_soft_{skip_reason_out}"] = 1

        history_baseline_prompt_tokens_out = history_baseline_prompt_tokens
        h_profile_out = dict(h_profile)
        h_profile_out["_no_pointers"] = True
        recent_compressed, tok_state_out, suppressed_markers = compress_history(
            body["messages"],
            keep_turns=bridge_keep_turns,
            profile=bridge_profile if request.uses_cut_search else h_profile_out,
            prune_tool_results=True,
        )
        session._suppressed_failure_markers = frozenset(suppressed_markers)
        recent_compressed, recent_breakdown = compress_recent_window(
            recent_compressed,
            tool_use_id_to_context=id_to_context,
            tool_compatible=effective_tool_compatible,
            first_exact_evidence_seen=set(_first_exact_evidence_seen_for_compression),
            preserve_exact_search_evidence=preserve_exact_search_evidence,
            session_files_read=session._files_read_this_session,
            model_profile=session.effective_model_profile,
        )

        step7a_result = run_step_7a_bridge_cut_search(
            session=session,
            request=request,
            recent=recent_compressed,
            original_messages=body["messages"],
            system=body.get("system", ""),
            id_to_context=id_to_context,
            keep_turns=keep_turns,
            bridge_keep_turns=bridge_keep_turns,
            bridge_profile=bridge_profile,
            history_baseline_prompt_tokens=history_baseline_prompt_tokens_out,
            seen_mutation_pairs=seen_mutation_pairs,
            preserve_exact_search_evidence=preserve_exact_search_evidence,
            exact_search_evidence_keys_in_request=exact_search_evidence_keys_in_request,
            _first_exact_evidence_seen_for_compression=_first_exact_evidence_seen_for_compression,
            effective_tool_compatible=effective_tool_compatible,
        )

        recent_out = step7a_result.recent
        tok_state_out = step7a_result.tok_state
        recent_breakdown = step7a_result.recent_breakdown
        for key, value in step7a_result.behavior_signals.items():
            behavior_signals_out[key] = behavior_signals_out.get(key, 0) + value

        if preserve_exact_search_evidence:
            if not exact_search_evidence_keys_in_request.issubset(
                _exact_search_evidence_keys_in_messages(recent_out, id_to_context)
            ):
                recent_out = body["messages"]
                tok_state_out = ""
                recent_breakdown = {}
                should_skip_history_out = True
                skip_reason_out = "answer_ready_exact_search_evidence"
                history_skip_reason_out = skip_reason_out
                behavior_signals_out["answer_ready_exact_evidence_fallback_full_history"] = 1
                behavior_signals_out["answer_ready_exact_search_evidence_history_preserved"] = 1
                behavior_signals_out["tok_history_compression_skipped"] = (
                    behavior_signals_out.get("tok_history_compression_skipped", 0) + 1
                )
                behavior_signals_out[f"tok_skip_{skip_reason_out}"] = 1

        for k, v in recent_breakdown.items():
            type_breakdown_out[f"recent_{k}"] = type_breakdown_out.get(f"recent_{k}", 0) + v

    if not should_skip_history_out:
        if tok_state_out:
            _record_non_exact_history_evidence(session, tok_state_out)
            logger.info(f"HISTORY WINNOWING SUCCESS: msgs {len(body['messages'])} -> {len(recent_out)}")
            _in_active_tool_loop = any(
                behavior_signals_out.get(k, 0) > 0
                for k in (
                    "repeat_file_read",
                    "repeat_search",
                    "repeat_command",
                    "repeated_tool_call",
                )
            )
            from tok.runtime.smoothness.models import TokMode

            current_mode = session.current_tok_mode
            if _in_active_tool_loop and current_mode in (
                TokMode.GUARDED_TOK,
                TokMode.SMOOTH_MODE,
                TokMode.LOSSLESS_TASK_MODE,
            ):
                should_skip_history_out = True
                behavior_signals_out["smoothness_guarded_history_winnowing_skipped"] = 1
                logger.info(
                    "GUARDED_TOK: skipping history winnowing in active tool loop (mode=%s)",
                    session.current_tok_mode.value,
                )
            else:
                if _in_active_tool_loop:
                    behavior_signals_out["smoothness_history_winnowing_active_loop"] = 1
                body["messages"] = recent_out
                compressed_out = True
                if effective_tool_compatible:
                    behavior_signals_out["tool_compatible_compression"] = (
                        behavior_signals_out.get("tool_compatible_compression", 0) + 1
                    )
                session_memory_out = session.refresh_hot_memory(tok_state_out, model=request.model)
        else:
            behavior_signals_out["tok_history_cut_point_missing"] = 1
            tool_result_count = sum(
                1
                for m in body.get("messages", [])
                if m.get("role") == "user"
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                )
            )
            if tool_result_count > 0:
                behavior_signals_out["tok_history_cut_point_missing_with_tools"] = 1
                behavior_signals_out["tok_history_cut_blocked_tool_result"] = 1
            session_memory_out = session.refresh_hot_memory("", model=request.model)
    else:
        session_memory_out = session.refresh_hot_memory("", model=request.model)

    return Step7Result(
        body=body,
        recent=recent_out,
        tok_state=tok_state_out,
        session_memory=session_memory_out,
        compressed=compressed_out,
        behavior_signals=behavior_signals_out,
        type_breakdown=type_breakdown_out,
        should_skip_history=should_skip_history_out,
        skip_reason=skip_reason_out,
        history_skip_reason=history_skip_reason_out,
        saved_tokens=saved_tokens_out,
        injected_state_payload=injected_state_payload,
        keep_turns=keep_turns,
        bridge_keep_turns=bridge_keep_turns,
    )


def _messages_contain_tool_material(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") == "tool_result":
            return True
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"tool_use", "tool_result"}:
                return True
    return False


def _record_non_exact_history_evidence(session: RuntimeSession, tok_state: str) -> None:
    if not tok_state:
        return
    for match in _EVIDENCE_PATH_RE.finditer(tok_state):
        path = match.group(1)
        from tok.runtime.repeat_targets import evidence_identity_key

        exact_key = evidence_identity_key("read_file", path=path, args={"path": path})
        if exact_key:
            session.record_non_exact_evidence(exact_key, form="summary")


_EVIDENCE_PATH_RE = re.compile(
    r"(?<!\w)([\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|yml|sh|txt|css|html|sql|rs|go|rb))(?!\w)"
)
