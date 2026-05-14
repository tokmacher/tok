from __future__ import annotations

from typing import Any, Literal, cast

from tok.runtime.core import RuntimeSession, UniversalTokRuntime
from tok.runtime.types import PreparedRuntimeRequest, RuntimeRequest

from ._prepare_context import PreparePipelineContext


def _prepare_request_impl(
    runtime_self: UniversalTokRuntime,
    request: RuntimeRequest,
    session: RuntimeSession,
    *,
    result_cache: dict[str, Any] | None = None,
) -> PreparedRuntimeRequest:
    from ._prepare_init_context import run_step_1

    s1 = run_step_1(request, session)
    body = s1.body
    original_body = s1.original_body
    _thinking_snapshot = s1.thinking_snapshot
    _pre_existing_session_signals = s1.pre_existing_session_signals
    ctx = PreparePipelineContext(
        body=body,
        original_body=original_body,
        compressed=s1.compressed,
        pre_existing_session_signals=_pre_existing_session_signals,
    )
    seen_mutation_pairs = s1.seen_mutation_pairs
    last_user_msg = s1.last_user_msg
    is_bridge_adapter = s1.is_bridge_adapter
    initial_answer_facts_present = s1.initial_answer_facts_present
    initial_exact_search_evidence_present = s1.initial_exact_search_evidence_present

    from ._prepare_optimize_prompt import run_step_2

    s2 = run_step_2(request, session, ctx.body, last_user_msg, is_bridge_adapter, ctx.compressed)
    ctx.body = s2.body
    ctx.compressed = s2.compressed

    from ._prepare_translate_classify import run_step_3

    s3 = run_step_3(request, session, ctx.body, is_bridge_adapter)
    ctx.body = s3.body
    translated_messages = s3.translated_messages
    plan_finalization_turn = s3.plan_finalization_turn
    id_to_context = s3.id_to_context
    exact_search_evidence_keys_in_request = s3.exact_search_evidence_keys_in_request
    stream_recovery_history_floor_active = s3.stream_recovery_history_floor_active
    ctx.behavior_signals = s3.behavior_signals
    normalized_tool_events = s3.normalized_tool_events
    broad_audit_batch = s3.broad_audit_batch
    history_skip_reason = s3.history_skip_reason
    should_skip_history = s3.should_skip_history
    skip_reason = s3.skip_reason
    edit_reacquisition_signals = s3.edit_reacquisition_signals

    from ._prepare_resolve_policy import run_step_4

    s4 = run_step_4(
        request,
        session,
        translated_messages,
        normalized_tool_events,
        ctx.behavior_signals,
        should_skip_history,
        skip_reason,
        history_skip_reason,
        plan_finalization_turn,
    )
    mode = s4.mode
    policy = s4.policy
    ctx.saved_tokens = s4.saved_tokens
    ctx.type_breakdown = s4.type_breakdown
    ctx.hot_hint_metrics = s4.hot_hint_metrics
    should_skip_history = s4.should_skip_history
    skip_reason = s4.skip_reason
    history_skip_reason = s4.history_skip_reason
    current_pressure = s4.current_pressure
    request_policy = s4.request_policy
    effective_tool_compatible = s4.effective_tool_compatible
    request_policy_escalated = s4.request_policy_escalated

    from ._prepare_detect_answer_phase import run_step_5

    s5 = run_step_5(
        session,
        request,
        translated_messages,
        id_to_context,
        normalized_tool_events,
        ctx.behavior_signals,
        effective_tool_compatible,
        initial_answer_facts_present,
        initial_exact_search_evidence_present,
        exact_search_evidence_keys_in_request,
        plan_finalization_turn,
        list(s3.runtime_hints),
    )
    answer_ready = s5.answer_ready
    resend_signals = s5.resend_signals
    has_answer_anchor = s5.has_answer_anchor
    preserve_exact_search_evidence = s5.preserve_exact_search_evidence
    read_only_audit_turn = s5.read_only_audit_turn
    runtime_hints = s5.runtime_hints

    from ._prepare_compress_tool_results import run_step_6

    s6 = run_step_6(
        session,
        request,
        ctx.body,
        translated_messages,
        id_to_context,
        ctx.behavior_signals,
        effective_tool_compatible,
        preserve_exact_search_evidence,
        broad_audit_batch,
        edit_reacquisition_signals,
        stream_recovery_history_floor_active,
        plan_finalization_turn,
        mode,
        policy,
        exact_search_evidence_keys_in_request,
        current_pressure,
        ctx.saved_tokens,
        ctx.compressed,
        result_cache,
    )
    ctx.body = s6.body
    ctx.type_breakdown = s6.type_breakdown
    ctx.saved_tokens = s6.saved_tokens
    ctx.compressed = s6.compressed

    recent = ctx.body["messages"]
    tok_state = ""
    session_memory = ""
    keep_turns = session.adaptive_keep_turns()
    if session._tok_memory_snap_triggered:
        keep_turns = 0
        session._tok_memory_snap_triggered = 0

    history_baseline_prompt_tokens = session.prepared_prompt_tokens(ctx.body)
    h_profile: dict[str, Any] = dict(policy.history_profiles[mode])
    bridge_keep_turns = max(keep_turns, 4) if request.adapter_kind == "claude-bridge" else keep_turns
    bridge_profile = dict(h_profile)
    if request.adapter_kind == "claude-bridge":
        bridge_profile["_bridge_cut_search"] = 1

    if preserve_exact_search_evidence:
        first_exact_evidence_seen_for_compression = set(session._first_exact_evidence_seen)
        first_exact_evidence_seen_for_compression.difference_update(exact_search_evidence_keys_in_request)
    else:
        first_exact_evidence_seen_for_compression = set(session._first_exact_evidence_seen)

    from ._prepare_compress_history import run_step_7

    s7 = run_step_7(
        session=session,
        request=request,
        normalized_tool_events=normalized_tool_events,
        body=ctx.body,
        id_to_context=id_to_context,
        behavior_signals=ctx.behavior_signals,
        effective_tool_compatible=effective_tool_compatible,
        mode=mode,
        policy=policy,
        should_skip_history=should_skip_history,
        skip_reason=skip_reason,
        history_skip_reason=history_skip_reason,
        preserve_exact_search_evidence=preserve_exact_search_evidence,
        plan_finalization_turn=plan_finalization_turn,
        broad_audit_batch=broad_audit_batch,
        edit_reacquisition_signals=edit_reacquisition_signals,
        stream_recovery_history_floor_active=stream_recovery_history_floor_active,
        session_memory=session_memory,
        history_baseline_prompt_tokens=history_baseline_prompt_tokens,
        seen_mutation_pairs=seen_mutation_pairs,
        saved_tokens=ctx.saved_tokens,
        compressed=ctx.compressed,
        current_pressure=current_pressure,
        request_policy=request_policy,
        exact_search_evidence_keys_in_request=exact_search_evidence_keys_in_request,
        recent=recent,
        tok_state=tok_state,
        type_breakdown=ctx.type_breakdown,
        keep_turns=keep_turns,
        bridge_keep_turns=bridge_keep_turns,
        bridge_profile=bridge_profile,
        h_profile=h_profile,
        _first_exact_evidence_seen_for_compression=frozenset(first_exact_evidence_seen_for_compression),
    )
    ctx.body = s7.body
    recent = s7.recent
    tok_state = s7.tok_state
    session_memory = s7.session_memory
    ctx.compressed = s7.compressed
    ctx.behavior_signals = s7.behavior_signals
    ctx.type_breakdown = s7.type_breakdown
    should_skip_history = s7.should_skip_history
    skip_reason = s7.skip_reason
    history_skip_reason = s7.history_skip_reason
    ctx.saved_tokens = s7.saved_tokens

    from ._prepare_inject_system import run_step_8

    s8 = run_step_8(
        runtime_self,
        request,
        session,
        ctx.body,
        session_memory,
        history_skip_reason or skip_reason or None,
        skip_reason,
        ctx.behavior_signals,
        runtime_hints,
        effective_tool_compatible,
        current_pressure,
        ctx.hot_hint_metrics,
        translated_messages,
        should_skip_history,
        recent,
        has_answer_anchor,
    )
    ctx.body = s8.body
    ctx.behavior_signals = s8.behavior_signals
    ctx.hot_hint_metrics = s8.hot_hint_metrics
    resend_signals = s8.resend_signals
    answer_ready = s8.answer_ready
    has_answer_anchor = s8.has_answer_anchor
    session_memory = s8.session_memory

    for key, value in resend_signals.items():
        if value:
            session._bump_signals({key: value})

    if has_answer_anchor and answer_ready:
        session._answer_ready_repair_pending = False
        session._late_answer_followthrough_pending = False
        session._late_answer_assembly_repair_pending = False
    elif effective_tool_compatible and not session._baseline_only and not read_only_audit_turn:
        if has_answer_anchor and not answer_ready:
            session._answer_ready_repair_pending = True
        elif answer_ready and not has_answer_anchor:
            session._late_answer_followthrough_pending = True
    session._save_bridge_memory()

    from ._prepare_finalize import run_step_9

    s9 = run_step_9(
        runtime_self=runtime_self,
        request=request,
        session=session,
        body=ctx.body,
        original_body=ctx.original_body,
        thinking_snapshot=_thinking_snapshot,
        compressed=ctx.compressed,
        saved_tokens=ctx.saved_tokens,
        type_breakdown=ctx.type_breakdown,
        behavior_signals=ctx.behavior_signals,
        mode=mode,
        request_policy=cast(Literal["legacy_tool_compatible", "natural_first", "forced_baseline"], request_policy),
        effective_tool_compatible=effective_tool_compatible,
        request_policy_escalated=request_policy_escalated,
        normalized_tool_events=normalized_tool_events,
        baseline_prompt_tokens=0,
        prepared_prompt_tokens=0,
        hot_hint_metrics=ctx.hot_hint_metrics,
        seen_mutation_pairs=seen_mutation_pairs,
        _pre_existing_session_signals=ctx.pre_existing_session_signals,
    )
    return s9.prepared_request
