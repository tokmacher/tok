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
    from ._prepare_init_context import prepare_init_context

    ctx_init = prepare_init_context(request, session)
    body = ctx_init.body
    original_body = ctx_init.original_body
    _thinking_snapshot = ctx_init.thinking_snapshot
    _pre_existing_session_signals = ctx_init.pre_existing_session_signals
    ctx = PreparePipelineContext(
        body=body,
        original_body=original_body,
        compressed=ctx_init.compressed,
        pre_existing_session_signals=_pre_existing_session_signals,
    )
    seen_mutation_pairs = ctx_init.seen_mutation_pairs
    last_user_msg = ctx_init.last_user_msg
    is_bridge_adapter = ctx_init.is_bridge_adapter
    initial_answer_facts_present = ctx_init.initial_answer_facts_present
    initial_exact_search_evidence_present = ctx_init.initial_exact_search_evidence_present

    from ._prepare_optimize_prompt import prepare_optimize_prompt

    ctx_opt = prepare_optimize_prompt(request, session, ctx.body, last_user_msg, is_bridge_adapter, ctx.compressed)
    ctx.body = ctx_opt.body
    ctx.compressed = ctx_opt.compressed

    from ._prepare_translate_classify import prepare_translate_classify

    ctx_tc = prepare_translate_classify(request, session, ctx.body, is_bridge_adapter)
    ctx.body = ctx_tc.body
    translated_messages = ctx_tc.translated_messages
    plan_finalization_turn = ctx_tc.plan_finalization_turn
    context_dependency = ctx_tc.context_dependency
    id_to_context = ctx_tc.id_to_context
    exact_search_evidence_keys_in_request = ctx_tc.exact_search_evidence_keys_in_request
    stream_recovery_history_floor_active = ctx_tc.stream_recovery_history_floor_active
    ctx.behavior_signals = ctx_tc.behavior_signals
    normalized_tool_events = ctx_tc.normalized_tool_events
    broad_audit_batch = ctx_tc.broad_audit_batch
    history_skip_reason = ctx_tc.history_skip_reason
    should_skip_history = ctx_tc.should_skip_history
    skip_reason = ctx_tc.skip_reason
    edit_reacquisition_signals = ctx_tc.edit_reacquisition_signals

    from ._prepare_resolve_policy import prepare_resolve_policy

    ctx_policy = prepare_resolve_policy(
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
    mode = ctx_policy.mode
    policy = ctx_policy.policy
    ctx.saved_tokens = ctx_policy.saved_tokens
    ctx.type_breakdown = ctx_policy.type_breakdown
    ctx.hot_hint_metrics = ctx_policy.hot_hint_metrics
    should_skip_history = ctx_policy.should_skip_history
    skip_reason = ctx_policy.skip_reason
    history_skip_reason = ctx_policy.history_skip_reason
    current_pressure = ctx_policy.current_pressure
    request_policy = ctx_policy.request_policy
    effective_tool_compatible = ctx_policy.effective_tool_compatible
    request_policy_escalated = ctx_policy.request_policy_escalated

    from ._prepare_detect_answer_phase import prepare_detect_answer_phase

    ctx_answer = prepare_detect_answer_phase(
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
        list(ctx_tc.runtime_hints),
    )
    answer_ready = ctx_answer.answer_ready
    resend_signals = ctx_answer.resend_signals
    has_answer_anchor = ctx_answer.has_answer_anchor
    preserve_exact_search_evidence = ctx_answer.preserve_exact_search_evidence
    read_only_audit_turn = ctx_answer.read_only_audit_turn
    runtime_hints = ctx_answer.runtime_hints

    from ._prepare_compress_tool_results import prepare_compress_tool_results

    ctx_tool = prepare_compress_tool_results(
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
        context_dependency,
        mode,
        policy,
        exact_search_evidence_keys_in_request,
        current_pressure,
        ctx.saved_tokens,
        ctx.compressed,
        result_cache,
    )
    ctx.body = ctx_tool.body
    ctx.type_breakdown = ctx_tool.type_breakdown
    ctx.saved_tokens = ctx_tool.saved_tokens
    ctx.compressed = ctx_tool.compressed

    recent = ctx.body["messages"]
    tok_state = ""
    session_memory = ""
    keep_turns = session.adaptive_keep_turns()
    if session._tok_memory_snap_triggered:
        keep_turns = 0
        session._tok_memory_snap_triggered = 0

    history_baseline_prompt_tokens = session.prepared_prompt_tokens(ctx.body)
    h_profile: dict[str, Any] = dict(policy.history_profiles[mode])
    bridge_keep_turns = max(keep_turns, 4) if request.uses_cut_search else keep_turns
    bridge_profile = dict(h_profile)
    if request.uses_cut_search:
        bridge_profile["_bridge_cut_search"] = 1

    if preserve_exact_search_evidence:
        first_exact_evidence_seen_for_compression = set(session._first_exact_evidence_seen)
        first_exact_evidence_seen_for_compression.difference_update(exact_search_evidence_keys_in_request)
    else:
        first_exact_evidence_seen_for_compression = set(session._first_exact_evidence_seen)

    from ._prepare_compress_history import prepare_compress_history

    ctx_hist = prepare_compress_history(
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
        context_dependency=context_dependency,
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
    ctx.body = ctx_hist.body
    recent = ctx_hist.recent
    tok_state = ctx_hist.tok_state
    session_memory = ctx_hist.session_memory
    ctx.compressed = ctx_hist.compressed
    ctx.behavior_signals = ctx_hist.behavior_signals
    ctx.type_breakdown = ctx_hist.type_breakdown
    should_skip_history = ctx_hist.should_skip_history
    skip_reason = ctx_hist.skip_reason
    history_skip_reason = ctx_hist.history_skip_reason
    ctx.saved_tokens = ctx_hist.saved_tokens

    from ._prepare_inject_system import prepare_inject_system

    ctx_inject = prepare_inject_system(
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
    ctx.body = ctx_inject.body
    ctx.behavior_signals = ctx_inject.behavior_signals
    ctx.hot_hint_metrics = ctx_inject.hot_hint_metrics
    resend_signals = ctx_inject.resend_signals
    answer_ready = ctx_inject.answer_ready
    has_answer_anchor = ctx_inject.has_answer_anchor
    session_memory = ctx_inject.session_memory

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

    from ._prepare_finalize import prepare_finalize

    ctx_final = prepare_finalize(
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
    return ctx_final.prepared_request
