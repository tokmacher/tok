from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tok.compression import inject_system_additions
from tok.runtime.core import RuntimeSession, UniversalTokRuntime
from tok.runtime.pipeline.request_preparation import _inject_system
from tok.runtime.types import RuntimeRequest


@dataclass
class Step8Result:
    body: dict[str, Any] = field(default_factory=dict)
    injected_state_payload: str = ""
    runtime_hints: list[str] = field(default_factory=list)
    behavior_signals: dict[str, int] = field(default_factory=dict)
    hot_hint_metrics: dict[str, int] = field(default_factory=dict)
    resend_signals: dict[str, int] = field(default_factory=dict)
    answer_ready: bool = False
    has_answer_anchor: bool = False
    session_memory: str = ""
    tok_state: str = ""


def run_step_8(
    runtime_self: UniversalTokRuntime,
    request: RuntimeRequest,
    session: RuntimeSession,
    body: dict[str, Any],
    session_memory: str,
    history_skip_reason: str | None,
    skip_reason: str,
    behavior_signals: dict[str, int],
    runtime_hints: list[str],
    effective_tool_compatible: bool,
    current_pressure: int,
    hot_hint_metrics: dict[str, int],
    translated_messages: list[dict[str, Any]],
    should_skip_history: bool,
    recent: list[dict[str, Any]],
    has_answer_anchor: bool,
) -> Step8Result:
    resend_signals: dict[str, int] = {}
    answer_ready = False
    if effective_tool_compatible:
        if skip_reason in {"short_session", "broad_audit"}:
            behavior_signals[f"{skip_reason}_system_additions_skipped"] = 1
            return Step8Result(
                body=body,
                behavior_signals=behavior_signals,
            )
        else:
            (
                injected_state_payload,
                runtime_hints,
                behavior_signals,
                hot_hint_metrics,
                _processed_body,
                resend_signals,
                answer_ready,
            ) = runtime_self._build_tool_compatible_resend(
                request,
                session,
                session_memory,
                history_skip_reason or skip_reason or None,
                behavior_signals,
                runtime_hints,
                current_pressure=current_pressure,
                hot_hint_metrics=hot_hint_metrics,
                translated_messages=translated_messages,
                should_skip_history=should_skip_history,
                _recent_messages=recent,
                has_answer_anchor_param=has_answer_anchor,
            )
            if session._answer_phase_expected_this_turn and runtime_hints:
                runtime_hints = []
            max_runtime_hints = 10
            if len(runtime_hints) > max_runtime_hints:
                runtime_hints = runtime_hints[:max_runtime_hints]
            body = _inject_system(
                body,
                injected_state_payload,
                runtime_hints,
                tool_compatible=True,
                grammar=bool(request.grammar),
                todo=request.todo or "",
                deltas=bool(request.deltas),
                pressure=current_pressure,
                behavior_signals=behavior_signals,
                current_turn=session.bridge_memory.turn,
                session=session,
            )
            has_answer_anchor = bool(behavior_signals.get("answer_anchor_present", 0))
            tok_state = injected_state_payload
    elif skip_reason in {"short_session", "broad_audit"}:
        behavior_signals[f"{skip_reason}_system_additions_skipped"] = 1
        return Step8Result(
            body=body,
            behavior_signals=behavior_signals,
            resend_signals=resend_signals,
            answer_ready=answer_ready,
            session_memory=session_memory,
        )
    else:
        max_runtime_hints = 10
        if len(runtime_hints) > max_runtime_hints:
            runtime_hints = runtime_hints[:max_runtime_hints]
        system_body = inject_system_additions(
            body,
            tok_state=session_memory,
            tool_compatible=False,
            pressure=current_pressure,
            runtime_hints=runtime_hints,
            behavior_signals=behavior_signals,
        )
        body["system"] = system_body.get("system", body.get("system", ""))
        tok_state = session_memory

    return Step8Result(
        body=body,
        injected_state_payload=tok_state,
        runtime_hints=runtime_hints,
        behavior_signals=behavior_signals,
        hot_hint_metrics=hot_hint_metrics,
        resend_signals=resend_signals,
        answer_ready=answer_ready,
        has_answer_anchor=has_answer_anchor,
        session_memory=session_memory,
        tok_state=tok_state,
    )
