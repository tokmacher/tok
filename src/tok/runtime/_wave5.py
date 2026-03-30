from __future__ import annotations

"""Wave 5 helper implementations for runtime-core methods."""

import tok.runtime.core as _core

globals().update(vars(_core))


def observe_repeat_target_result_impl(
    session_self,
    *,
    tool_id: str,
    tool_name: str,
    path: str | None,
    query: str | None,
    command: str | None,
    raw_content: str,
    blocker_rediscovery: bool = False,
) -> dict[str, int]:
    """Record a new result-bearing logical target event for repeat-target control."""
    if tool_id and tool_id in session_self._observed_tool_result_ids:
        return {}
    (
        family,
        logical_target,
        display_target,
    ) = logical_target_key_from_context(
        tool_name,
        path=path,
        query=query,
        command=command,
    )
    if family not in {"file_read", "search", "command"}:
        return {}
    text = str(raw_content or "").strip()
    if not text:
        return {}
    if tool_id:
        session_self._observed_tool_result_ids.add(tool_id)

    evidence_intent = resolve_evidence_intent(
        tool_name, path=path, query=query, command=command
    )
    evidence_anchor = evidence_intent.anchor if evidence_intent else ""

    current_turn = max(1, session_self.bridge_memory.turn)
    token_cost = max(0, count_tokens(text))
    digest = stable_digest(text)
    summary = build_summary_for_family(
        family,
        text,
        file_max_chars=TOK_HOT_FILE_MAX_CHARS,
        file_max_lines=TOK_HOT_FILE_MAX_LINES,
        search_max_chars=TOK_HOT_SEARCH_MAX_CHARS,
        search_max_lines=TOK_HOT_SEARCH_MAX_LINES,
        command_max_chars=TOK_HOT_COMMAND_MAX_CHARS,
        command_max_lines=TOK_HOT_COMMAND_MAX_LINES,
    )
    if not summary:
        return {}

    key = f"{family}|{logical_target}"
    record = session_self._hot_summary_records.get(key)
    unchanged_result = bool(record and record.result_digest == digest)
    session_self._recent_repeat_target_events.append(
        RepeatTargetEvent(
            turn_index=current_turn,
            tool_family=family,
            logical_target=logical_target,
            display_target=display_target,
            token_cost=token_cost,
            result_digest=digest,
            unchanged_result=unchanged_result,
            evidence_anchor=evidence_anchor,
        )
    )
    recent_events = [
        event
        for event in session_self._recent_repeat_target_events
        if event.tool_family == family
        and event.logical_target == logical_target
        and current_turn - event.turn_index < TOK_REACQUIRE_WINDOW_TURNS
    ]
    stuck_events = [
        event
        for event in session_self._recent_repeat_target_events
        if event.tool_family == family
        and event.logical_target == logical_target
        and current_turn - event.turn_index
        < TOK_REACQUIRE_STUCK_WINDOW_TURNS
    ]
    repeat_count = len(recent_events)
    stuck_count = len(stuck_events)
    updated = HotSummaryRecord(
        tool_family=family,
        logical_target=logical_target,
        display_target=display_target or logical_target,
        summary=summary,
        token_cost=token_cost,
        result_digest=digest,
        last_seen_turn=current_turn,
        hot_promotion_turn=record.hot_promotion_turn if record else 0,
        stuck_promotion_turn=record.stuck_promotion_turn if record else 0,
        last_injected_turn=record.last_injected_turn if record else 0,
        repeat_count=repeat_count,
        recent_window_count=repeat_count,
        stuck_window_count=stuck_count,
        unchanged_result_count=(
            (record.unchanged_result_count + 1)
            if unchanged_result and record
            else (1 if unchanged_result else 0)
        ),
        evidence_intent=(
            record.evidence_intent if record else evidence_intent
        ),
    )
    signals: dict[str, int] = {}
    if session_self.is_predictive_cache_hit(family, logical_target):
        signals["predictive_cache_hits"] = 1
    hot_now = repeat_count >= TOK_REACQUIRE_TRIGGER_COUNT
    stuck_now = stuck_count >= TOK_REACQUIRE_STUCK_COUNT or (
        hot_now and blocker_rediscovery
    )
    if hot_now and not updated.hot_promotion_turn:
        updated.hot_promotion_turn = current_turn
        signals["repeat_target_hot"] = 1
    elif (
        hot_now
        and record
        and record.hot_promotion_turn < current_turn
        and repeat_count > record.repeat_count
    ):
        updated.hot_promotion_turn = current_turn
        signals["repeat_target_hot"] = 1
    if stuck_now and not updated.stuck_promotion_turn:
        updated.stuck_promotion_turn = current_turn
        signals["repeat_target_stuck"] = 1
    elif (
        stuck_now
        and record
        and record.stuck_promotion_turn < current_turn
        and stuck_count > record.stuck_window_count
    ):
        updated.stuck_promotion_turn = current_turn
        signals["repeat_target_stuck"] = 1

    session_self._hot_summary_records[key] = updated
    session_self._trim_repeat_target_state()

    if evidence_intent and evidence_anchor:
        novelty_keys = session_self._evidence_anchor_novelty_keys.setdefault(
            evidence_anchor, set()
        )
        if evidence_intent.novelty_key:
            if evidence_intent.novelty_key in novelty_keys:
                signals["evidence_novelty_missing"] = 1
            else:
                novelty_keys.add(evidence_intent.novelty_key)
        else:
            if novelty_keys:
                signals["evidence_novelty_missing"] = 1

        if signals.get("repeat_target_hot"):
            signals["evidence_anchor_hot"] = 1
        if signals.get("repeat_target_stuck"):
            signals["evidence_anchor_stuck"] = 1

        if evidence_intent.domain == "file_current" and evidence_anchor:
            parent_dir = str(Path(evidence_anchor).parent)
            if parent_dir and parent_dir != ".":
                neighborhood = session_self._evidence_neighborhoods.setdefault(
                    parent_dir, set()
                )
                neighborhood.add(evidence_anchor)
                recent_neighborhood_events = [
                    e
                    for e in session_self._recent_repeat_target_events
                    if e.evidence_anchor in neighborhood
                    and current_turn - e.turn_index
                    < TOK_NEIGHBORHOOD_WINDOW_TURNS
                ]
                if (
                    len(neighborhood) >= TOK_NEIGHBORHOOD_TRIGGER_ANCHORS
                    and len(recent_neighborhood_events)
                    >= TOK_NEIGHBORHOOD_TRIGGER_ANCHORS
                ):
                    signals["evidence_neighborhood_hot"] = 1

    if family == "file_read" and (
        signals.get("repeat_target_hot")
        or signals.get("repeat_target_stuck")
    ):
        warm_metrics = session_self.apply_predictive_cache_warming(
            logical_target
        )
        for metric_key, metric_value in warm_metrics.items():
            signals[metric_key] = signals.get(metric_key, 0) + metric_value
    return signals


def prepare_request_impl(
    runtime_self,
    request: RuntimeRequest,
    session: RuntimeSession,
    *,
    result_cache: dict[str, Any] | None = None,
) -> PreparedRuntimeRequest:
    body: dict[str, Any] = {
        "model": request.model,
        "messages": copy.deepcopy(request.messages),
    }
    if request.system is not None:
        body["system"] = copy.deepcopy(request.system)
    original_body = copy.deepcopy(body)
    compressed = False

    last_user_msg = ""
    if request.messages:
        for m in reversed(request.messages):
            if m.get("role") == "user":
                last_user_msg = text_of(cast(Any, m.get("content", "")))
                break

    if detect_prompt_bloat(body.get("system"), last_user_msg):
        session.pending_behavior_signals["tok_prompt_bloat_detected"] = 1
        current_sys = cast(Any, body.get("system", ""))
        cleaned_sys = clean_system_context(session.bridge_memory, current_sys)
        if cleaned_sys and cleaned_sys != current_sys:
            body["system"] = cleaned_sys
            session.pending_behavior_signals["tok_prompt_optimized"] = 1
            compressed = True
            logger.warning(
                "tok_prompt_optimized: system prompt reduced from %d to %d chars",
                len(
                    text_of(current_sys)
                    if isinstance(current_sys, list)
                    else str(current_sys)
                ),
                len(
                    text_of(cleaned_sys)
                    if isinstance(cleaned_sys, list)
                    else str(cleaned_sys)
                ),
            )

    translated_messages = translate_request_results(body.get("messages", []))
    body["messages"] = translated_messages

    rolling_cmds = session.bridge_memory.rolling_cmds
    if rolling_cmds:
        recent_instructions: list[Instruction] = []
        for entry in rolling_cmds[-10:]:
            parts = entry.value.strip().split()
            if not parts:
                continue
            recent_instructions.append(
                Instruction(op=parts[0], args=tuple(parts[1:]))
            )

        jit_macro = session.bridge_memory.macro_registry.match_recent_sequence(
            recent_instructions
        )
        threshold = int(os.getenv("TOK_JIT_HIT_THRESHOLD", "3"))
        if (
            jit_macro
            and jit_macro.hit_count >= threshold
            and _jit_context_matches(jit_macro, session)
        ):
            session.pending_behavior_signals["jit_offer_available"] = 1
            session.pending_behavior_signals[f"jit_offer_{jit_macro.name}"] = 1
            session._pending_macro_heal = jit_macro.name
            session._pending_macro_heal_turn = session.bridge_memory.turn
        elif jit_macro and not _jit_context_matches(jit_macro, session):
            session.pending_behavior_signals[
                "jit_offer_context_filtered"
            ] = 1

    _speculative_macro_hint: str | None = None
    if session.bridge_memory.load_global_macros:
        _spec_threshold = int(os.getenv("TOK_SPECULATIVE_HIT_THRESHOLD", "2"))
        _spec_names = [
            f"@{m.name}"
            for m in session.bridge_memory.macro_registry.macros.values()
            if m.hit_count >= _spec_threshold and _jit_context_matches(m, session)
        ]
        if _spec_names:
            _speculative_macro_hint = (
                "Available macros for current context: "
                + ", ".join(sorted(_spec_names))
                + ". Use @name to invoke."
            )
            session.pending_behavior_signals[
                "speculative_macros_injected"
            ] = len(_spec_names)

    id_to_context = build_tool_use_id_to_context(translated_messages)
    behavior_signals = collect_behavior_signals(translated_messages, id_to_context)
    behavior_signals["_project_markers_proxy"] = len(session._project_markers)
    for err_snippet in collect_transient_error_snippets(translated_messages):
        session.bridge_memory._upsert(
            session.bridge_memory.hot, "errs", err_snippet, score_delta=1
        )

    blockers, hypotheses = extract_memory_items(translated_messages)
    for blocker in blockers:
        session.bridge_memory._upsert(
            session.bridge_memory.hot, "blockers", blocker, score_delta=2
        )
    for hypothesis in hypotheses:
        session.bridge_memory._upsert(
            session.bridge_memory.hot, "questions", hypothesis, score_delta=2
        )

    normalized_tool_events = normalize_tool_events(translated_messages)
    runtime_hints: list[str] = []
    resend_signals: dict[str, int] = {}
    injected_state_payload = ""
    history_skip_reason = ""
    should_skip_history = False
    skip_reason = ""
    resend_reason: str | None = None
    previous_comparable: dict[str, list[str]] = {}

    for event in normalized_tool_events:
        if event.name.lower() in EDIT_LIKE_TOOLS and event.path:
            session.bridge_memory.bump_file_heat(event.path, weight=2.0)

    mode, policy = session.policy_snapshot(request.model)
    saved_tokens = 0
    type_breakdown: dict[str, int] = {}
    hot_hint_metrics: dict[str, int] = {
        "hot_recent_hint_injected": 0,
        "hot_hint_tokens_added": 0,
        "reacquisition_tokens_avoided_estimate": 0,
        "repeat_tool_collapse_applied": 0,
    }
    current_pressure = calculate_invisible_pressure(behavior_signals)

    if translated_messages:
        session.bridge_memory.turn += 1
        runtime_hints = (
            [_speculative_macro_hint] if _speculative_macro_hint else []
        )
        answer_ready = False
        resend_signals = {}
        resend_reason = None
        has_answer_anchor = False
        late_answer_followthrough_active = (
            request.tool_compatible
            and session._late_answer_followthrough_pending
            and not session._baseline_only
        )
        late_answer_assembly_repair_active = (
            request.tool_compatible
            and session._late_answer_assembly_repair_pending
            and not session._baseline_only
            and not late_answer_followthrough_active
        )
        late_answer_assembly_repair_mode = (
            session._late_answer_assembly_repair_mode_pending
            if late_answer_assembly_repair_active
            else ""
        )
        answer_ready_repair_active = (
            request.tool_compatible
            and session._answer_ready_repair_pending
            and not session._baseline_only
            and not late_answer_followthrough_active
            and not late_answer_assembly_repair_active
        )
        session._late_answer_followthrough_active = (
            late_answer_followthrough_active
        )
        session._answer_ready_repair_active = answer_ready_repair_active
        session._late_answer_assembly_repair_active = (
            late_answer_assembly_repair_active
        )
        session._late_answer_assembly_repair_mode_active = (
            late_answer_assembly_repair_mode
        )

        repeat_snapshot_signals = _capture_repeat_target_snapshots(
            translated_messages, id_to_context, session
        )
        if repeat_snapshot_signals:
            session._bump_signals(repeat_snapshot_signals)

        session._save_bridge_memory()
        body["messages"], type_breakdown = compress_tool_results(
            translated_messages,
            result_cache=(
                result_cache if result_cache is not None else session.result_cache
            ),
            tool_use_id_to_context=id_to_context,
            compression_level=policy.tool_levels[mode],
            semantic_hash_cache=session.semantic_hash_cache,
        )
        tool_saved = sum(type_breakdown.values()) // 4
        if tool_saved > 0:
            saved_tokens += tool_saved
            compressed = True
        file_cache_hits = sum(
            v for k, v in type_breakdown.items() if k.endswith("_cached")
        )
        if file_cache_hits > 0:
            behavior_signals["tool_result_cache_hit"] = (
                behavior_signals.get("tool_result_cache_hit", 0) + 1
            )
        semantic_dedup_hits = type_breakdown.get("semantic_dedup", 0)
        if semantic_dedup_hits > 0:
            behavior_signals["semantic_dedup_hit"] = (
                behavior_signals.get("semantic_dedup_hit", 0) + 1
            )
            from ..compression import _STABLE_RESULT_EXPLANATION

            runtime_hints.append(_STABLE_RESULT_EXPLANATION)

        recent: list[dict[str, Any]] = body["messages"]
        tok_state = ""
        keep_turns = session.adaptive_keep_turns()
        if session._tok_memory_snap_triggered:
            logger.info("Memory snap triggered: forcing keep_turns=0")
            keep_turns = 0
            session._tok_memory_snap_triggered = 0

        should_skip_history, skip_reason = _should_skip_history_rewrite(
            request.messages,
            normalized_tool_events,
            tool_compatible=request.tool_compatible,
        )

        if should_skip_history:
            behavior_signals["tok_history_compression_skipped"] = (
                behavior_signals.get("tok_history_compression_skipped", 0) + 1
            )
            if skip_reason:
                behavior_signals[f"tok_skip_{skip_reason}"] = 1
                history_skip_reason = skip_reason
            recent = body["messages"]
        else:
            if skip_reason:
                behavior_signals[f"tok_soft_{skip_reason}"] = 1

            h_profile: dict[str, Any] = dict(policy.history_profiles[mode])
            h_profile["_no_pointers"] = True
            recent, tok_state = compress_history(
                body["messages"],
                keep_turns=keep_turns,
                profile=h_profile,
                prune_tool_results=True,
            )
            recent, recent_breakdown = compress_recent_window(
                recent,
                tool_use_id_to_context=id_to_context,
                tool_compatible=request.tool_compatible,
            )
            if (
                request.adapter_kind == "claude-bridge"
                and not recent
                and body["messages"]
            ):
                recent, tok_state = compress_history(
                    body["messages"],
                    keep_turns=1,
                    profile=h_profile,
                    prune_tool_results=True,
                )
                recent, recent_breakdown = compress_recent_window(
                    recent,
                    tool_use_id_to_context=id_to_context,
                    tool_compatible=request.tool_compatible,
                )
                behavior_signals["bridge_minimum_tail_preserved"] = 1
            for k, v in recent_breakdown.items():
                type_breakdown[f"recent_{k}"] = (
                    type_breakdown.get(f"recent_{k}", 0) + v
                )

        if not should_skip_history:
            if tok_state:
                logger.error(
                    f"HISTORY WINNOWING SUCCESS: msgs {len(body['messages'])} -> {len(recent)}"
                )
                body["messages"] = recent
                compressed = True
                if request.tool_compatible:
                    behavior_signals["tool_compatible_compression"] = (
                        behavior_signals.get("tool_compatible_compression", 0) + 1
                    )
                session_memory = session.refresh_hot_memory(
                    tok_state, model=request.model
                )
            else:
                behavior_signals["tok_history_cut_point_missing"] = 1
                tool_result_count = sum(
                    1
                    for m in body.get("messages", [])
                    if m.get("role") == "user"
                    and any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in (
                            m.get("content")
                            if isinstance(m.get("content"), list)
                            else []
                        )
                    )
                )
                if tool_result_count > 0:
                    behavior_signals["tok_history_cut_point_missing_with_tools"] = (
                        1
                    )
                session_memory = session.refresh_hot_memory(
                    "", model=request.model
                )
        else:
            session_memory = session.refresh_hot_memory("", model=request.model)

        if request.tool_compatible:
            (
                injected_state_payload,
                runtime_hints,
                behavior_signals,
                hot_hint_metrics,
                processed_body,
                resend_signals,
                answer_ready,
            ) = runtime_self._build_tool_compatible_resend(
                request,
                session,
                session_memory,
                runtime_hints,
                behavior_signals,
                hot_hint_metrics,
                current_pressure=current_pressure,
                type_breakdown=type_breakdown,
            )
            if processed_body:
                body["system"] = processed_body.get("system", body.get("system", ""))
                resend_reason = next(
                    (
                        key
                        for key, value in resend_signals.items()
                        if key.startswith("state_resend_") and value
                    ),
                    None,
                )
                has_answer_anchor = bool(
                    behavior_signals.get("answer_anchor_present", 0)
                )
        else:
            system_body = inject_system_additions(
                body,
                tok_state=session_memory,
                tool_compatible=False,
                pressure=current_pressure,
                runtime_hints=runtime_hints,
                behavior_signals=behavior_signals,
            )
            body["system"] = system_body.get("system", body.get("system", ""))

        prepared_prompt_tokens = session.prepared_prompt_tokens(body)
        baseline_prompt_tokens = session.prepared_prompt_tokens(original_body)
        saved_prompt_tokens = max(
            0, baseline_prompt_tokens - prepared_prompt_tokens
        )
        if saved_prompt_tokens > 0:
            compressed = True
            saved_tokens += saved_prompt_tokens

        session._bump_signals(behavior_signals)
        session._bump_signals(hot_hint_metrics)
        for key, value in resend_signals.items():
            if value:
                session._bump_signals({key: value})

        if has_answer_anchor and answer_ready:
            session._answer_ready_repair_pending = False
            session._late_answer_followthrough_pending = False
            session._late_answer_assembly_repair_pending = False
        elif request.tool_compatible and not session._baseline_only:
            if has_answer_anchor and not answer_ready:
                session._answer_ready_repair_pending = True
            elif answer_ready and not has_answer_anchor:
                session._late_answer_followthrough_pending = True
        session._save_bridge_memory()
    else:
        prepared_prompt_tokens = session.prepared_prompt_tokens(body)
        baseline_prompt_tokens = prepared_prompt_tokens
        saved_prompt_tokens = 0

    return PreparedRuntimeRequest(
        body=body,
        original_body=original_body,
        compressed=compressed,
        input_saved_tokens=saved_tokens,
        type_breakdown=type_breakdown,
        behavior_signals=behavior_signals,
        baseline_prompt_tokens=baseline_prompt_tokens,
        prepared_prompt_tokens=prepared_prompt_tokens,
        saved_prompt_tokens=saved_prompt_tokens,
        hot_hint_tokens_added=hot_hint_metrics.get("hot_hint_tokens_added", 0),
        reacquisition_tokens_avoided_estimate=hot_hint_metrics.get(
            "reacquisition_tokens_avoided_estimate", 0
        ),
        runtime_hints=runtime_hints,
        injected_state_payload=injected_state_payload,
        history_skip_reason=history_skip_reason,
        resend_reason=resend_reason,
        previous_comparable=previous_comparable,
    )
