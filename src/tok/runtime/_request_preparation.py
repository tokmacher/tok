"""Runtime request-preparation helpers extracted from core."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, cast

import tok.runtime.core as _core
from tok.compression import (
    EDIT_LIKE_TOOLS,
    compress_history,
    compress_recent_window,
    compress_tool_results,
    inject_system_additions,
    text_of,
)
from tok.neuro.ir import Instruction

from .config import (
    _SHORT_SESSION_THRESHOLD,
    FILE_READ_DENSITY_THRESHOLD,
    RUNTIME_HINTS_MAX_PER_TURN,
    TOK_FILE_DELIVERY_STALE_TURNS,
    TOK_HOT_COMMAND_MAX_CHARS,
    TOK_HOT_COMMAND_MAX_LINES,
    TOK_HOT_FILE_MAX_CHARS,
    TOK_HOT_FILE_MAX_LINES,
    TOK_HOT_SEARCH_MAX_CHARS,
    TOK_HOT_SEARCH_MAX_LINES,
    TOK_LARGE_FILE_HINT,
    TOK_NEIGHBORHOOD_TRIGGER_ANCHORS,
    TOK_NEIGHBORHOOD_WINDOW_TURNS,
    TOK_REACQUIRE_STUCK_COUNT,
    TOK_REACQUIRE_STUCK_WINDOW_TURNS,
    TOK_REACQUIRE_TRIGGER_COUNT,
    TOK_REACQUIRE_WINDOW_TURNS,
    TOK_READ_PLAN_HINT,
    TOK_REPEAT_COMMAND_SUPPRESSION_HINT,
    TOK_REQUEST_POLICY_STICKY_TURNS,
    TOK_STABLE_RESULT_INFO_HINT,
    TOOL_USE_DENSITY_THRESHOLD,
)
from .core import UniversalTokRuntime, logger
from .memory.bridge_memory import clean_system_context
from .memory.session_helpers import extract_memory_items
from .pipeline.request_preparation import (
    _capture_repeat_target_snapshots,
    _inject_system,
    _is_answer_ready_turn,
    _is_read_only_audit_turn,
    collect_transient_error_snippets,
    mutation_signals,
)
from .pipeline.request_validation import (
    canonicalize_anthropic_bridge_body,
    detect_prompt_bloat,
    has_recoverable_immediate_pairing_failures,
    validate_anthropic_bridge_body,
)
from .pipeline.response_processing import translate_request_results
from .pipeline.tool_processing import (
    _should_skip_history_rewrite,
    build_tool_use_id_to_context,
    collect_behavior_signals,
    count_tokens,
    logical_target_key_from_context,
    normalize_tool_events,
)
from .policy.macro_handling import _jit_context_matches
from .policy.semantic_validation import calculate_invisible_pressure
from .repeat_targets import (
    SEARCH_LIKE_TOOLS,
    HotSummaryRecord,
    RepeatTargetEvent,
    build_summary_for_family,
    evidence_identity_key,
    resolve_evidence_intent,
    search_result_evidence_level,
    stable_digest,
)
from .types import PreparedRuntimeRequest, RuntimeRequest

globals().update(vars(_core))

_RECENT_COMMAND_WINDOW = 10
_DEFAULT_JIT_HIT_THRESHOLD = 3
_DEFAULT_SPECULATIVE_HIT_THRESHOLD = 2
_STRUCTURED_ANSWER_LABEL_RE = re.compile(r"(?<![\w-])(file|verification|related)(?![\w-])\s*[:=]", re.IGNORECASE)


def _env_int_or_default(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _extract_requested_answer_labels(text: str) -> tuple[str, ...]:
    if not text.strip():
        return ()
    labels: list[str] = []
    seen: set[str] = set()
    for match in _STRUCTURED_ANSWER_LABEL_RE.finditer(text):
        label = match.group(1).lower()
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return tuple(labels)


def _record_structured_answer_expectation(
    session: _core.RuntimeSession,
    body: dict[str, Any],
) -> None:
    latest_user_prompt = ""
    messages = body.get("messages", [])
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).strip() != "user":
                continue
            latest_user_prompt = text_of(cast("Any", message.get("content", ""))).strip()
            if latest_user_prompt:
                break
    session._last_user_prompt_text = latest_user_prompt
    session._last_user_prompt_labels = _extract_requested_answer_labels(latest_user_prompt)


def observe_repeat_target_result_impl(
    session_self: _core.RuntimeSession,
    *,
    tool_id: str,
    tool_name: str,
    path: str | None,
    query: str | None,
    command: str | None,
    raw_content: str,
    tool_args: dict[str, Any] | None = None,
    exact_evidence_key: str | None = None,
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
        session_self._observed_tool_result_ids[tool_id] = None

    if not exact_evidence_key:
        exact_evidence_key = evidence_identity_key(
            tool_name,
            path=path,
            query=query,
            command=command,
            args=tool_args,
        )
    evidence_intent = resolve_evidence_intent(tool_name, path=path, query=query, command=command)
    evidence_anchor = evidence_intent.anchor if evidence_intent else ""

    search_like_result = tool_name in SEARCH_LIKE_TOOLS or (
        evidence_intent is not None and evidence_intent.domain == "search"
    )
    if exact_evidence_key and not (search_like_result and search_result_evidence_level(text) == "navigation"):
        session_self._pending_exact_evidence_keys.add(exact_evidence_key)

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
        and current_turn - event.turn_index < TOK_REACQUIRE_STUCK_WINDOW_TURNS
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
        exact_evidence_key=exact_evidence_key or "",
        hot_promotion_turn=record.hot_promotion_turn if record else 0,
        stuck_promotion_turn=record.stuck_promotion_turn if record else 0,
        last_injected_turn=record.last_injected_turn if record else 0,
        repeat_count=repeat_count,
        recent_window_count=repeat_count,
        stuck_window_count=stuck_count,
        unchanged_result_count=(
            (record.unchanged_result_count + 1) if unchanged_result and record else (1 if unchanged_result else 0)
        ),
        evidence_intent=(record.evidence_intent if record else evidence_intent),
    )
    signals: dict[str, int] = {}
    if session_self.is_predictive_cache_hit(family, logical_target):
        signals["predictive_cache_hits"] = 1
    hot_now = repeat_count >= TOK_REACQUIRE_TRIGGER_COUNT
    stuck_now = stuck_count >= TOK_REACQUIRE_STUCK_COUNT or (hot_now and blocker_rediscovery)
    if (hot_now and not updated.hot_promotion_turn) or (
        hot_now and record and record.hot_promotion_turn < current_turn and repeat_count > record.repeat_count
    ):
        updated.hot_promotion_turn = current_turn
        signals["repeat_target_hot"] = 1
    if (stuck_now and not updated.stuck_promotion_turn) or (
        stuck_now and record and record.stuck_promotion_turn < current_turn and stuck_count > record.stuck_window_count
    ):
        updated.stuck_promotion_turn = current_turn
        signals["repeat_target_stuck"] = 1

    session_self._hot_summary_records[key] = updated
    session_self._trim_repeat_target_state()

    if evidence_intent and evidence_anchor:
        novelty_keys = session_self._evidence_anchor_novelty_keys.setdefault(evidence_anchor, set())
        if evidence_intent.novelty_key:
            if evidence_intent.novelty_key in novelty_keys:
                signals["evidence_novelty_missing"] = 1
            else:
                novelty_keys.add(evidence_intent.novelty_key)
        elif novelty_keys:
            signals["evidence_novelty_missing"] = 1

        if signals.get("repeat_target_hot"):
            signals["evidence_anchor_hot"] = 1
        if signals.get("repeat_target_stuck"):
            signals["evidence_anchor_stuck"] = 1

        if evidence_intent.domain == "file_current" and evidence_anchor:
            parent_dir = str(Path(evidence_anchor).parent)
            if parent_dir and parent_dir != ".":
                neighborhood = session_self._evidence_neighborhoods.setdefault(parent_dir, set())
                neighborhood.add(evidence_anchor)
                recent_neighborhood_events = [
                    e
                    for e in session_self._recent_repeat_target_events
                    if e.evidence_anchor in neighborhood and current_turn - e.turn_index < TOK_NEIGHBORHOOD_WINDOW_TURNS
                ]
                if (
                    len(neighborhood) >= TOK_NEIGHBORHOOD_TRIGGER_ANCHORS
                    and len(recent_neighborhood_events) >= TOK_NEIGHBORHOOD_TRIGGER_ANCHORS
                ):
                    signals["evidence_neighborhood_hot"] = 1

    if family == "file_read" and (signals.get("repeat_target_hot") or signals.get("repeat_target_stuck")):
        warm_metrics = session_self.apply_predictive_cache_warming(logical_target)
        for metric_key, metric_value in warm_metrics.items():
            signals[metric_key] = signals.get(metric_key, 0) + metric_value
    return signals


def _message_has_tool_result(message: dict[str, Any]) -> bool:
    if message.get("role") == "tool_result":
        return bool(str(message.get("tool_use_id", "")).strip())
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def _message_has_user_prompt(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text", "")).strip():
            return True
    return False


def _stream_recovery_winnowing_floor_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve minimal causal context after stream recovery retries."""
    if not messages:
        return []

    latest_user_prompt_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "user" and _message_has_user_prompt(msg):
            latest_user_prompt_idx = idx
            break

    assistant_idx = -1
    assistant_tool_ids: set[str] = set()
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        tool_ids = {
            str(block.get("id", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use" and str(block.get("id", "")).strip()
        }
        if tool_ids:
            assistant_idx = idx
            assistant_tool_ids = tool_ids
            break

    paired_user_idx = -1
    if assistant_idx >= 0:
        for idx in range(assistant_idx + 1, len(messages)):
            msg = messages[idx]
            if msg.get("role") not in {
                "user",
                "tool_result",
            } or not _message_has_tool_result(msg):
                continue
            tool_result_ids: set[str]
            if msg.get("role") == "tool_result":
                tool_result_id = str(msg.get("tool_use_id", "")).strip()
                tool_result_ids = {tool_result_id} if tool_result_id else set()
            else:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                tool_result_ids = {
                    str(block.get("tool_use_id", "")).strip()
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and str(block.get("tool_use_id", "")).strip()
                }
            if assistant_tool_ids & tool_result_ids:
                paired_user_idx = idx
                break

    keep_indexes = sorted(idx for idx in {assistant_idx, paired_user_idx, latest_user_prompt_idx} if idx >= 0)
    return [messages[idx] for idx in keep_indexes]


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


def _resolve_effective_tool_compatible(
    request: RuntimeRequest,
    session: _core.RuntimeSession,
    _translated_messages: list[dict[str, Any]],
    _normalized_tool_events: list[Any],
    behavior_signals: dict[str, int],
) -> tuple[bool, list[str]]:
    if request.request_policy == "forced_baseline" or not request.tool_compatible:
        return False, []
    if request.request_policy == "legacy_tool_compatible":
        return True, ["legacy_default"]

    # Short session detection: avoid tok overhead for sessions < threshold turns
    current_turn = session.bridge_memory.turn
    if current_turn < _SHORT_SESSION_THRESHOLD:
        behavior_signals["short_session_baseline_mode"] = 1
        return False, ["short_session"]

    structured_tool_loop = any(
        behavior_signals.get(key, 0) > 0
        for key in (
            "repeat_file_read",
            "repeat_search",
            "repeat_command",
            "repeated_tool_call",
            "stream_recovery_reacquisition_suppressed",
        )
    )
    reasons: list[str] = []
    if session._request_policy_tool_mode_sticky_turns > 0:
        reasons.append("sticky")
    cooldown_remaining = getattr(session, "_stream_recovery_cooldown_remaining", 0)
    if cooldown_remaining > 0:
        session._stream_recovery_cooldown_remaining = max(0, cooldown_remaining - 1)
    if (
        session._stream_recovery_reacquisition_budget > 0
        or session._stream_recovery_history_floor_budget > 0
        or session._request_policy_stream_recovery_watch_turns > 0
    ):
        reasons.append("stream_recovery")
    if session._request_policy_tool_recovery_watch_turns > 0 or session._invalid_tool_history_recovery_count > 0:
        reasons.append("tool_recovery")
    if any(
        session.pending_behavior_signals.get(key, 0) > 0
        for key in (
            "tok_bridge_provider_sensitive_degraded_to_provider_safe",
            "tok_bridge_provider_sensitive_blocked_local",
            "tok_bridge_provider_pairing_risk_detected",
            "tok_bridge_assistant_tool_use_text_interleaving_blocked",
            "fail_open_retry_upstream_pairing_disagreement",
            "tok_history_pairing_safety_degraded",
        )
    ):
        reasons.append("tool_recovery")
    if structured_tool_loop:
        reasons.append("structured_tool_loop")
    return bool(reasons), reasons


def _snapshot_latest_assistant_thinking(
    messages: list[dict[str, Any]],
) -> str | None:
    """
    Return structured snapshot of the latest protected assistant message.

    Contract: this helper is defensive and must never raise on malformed bridge
    request message shapes; it returns ``None`` when the expected structure is
    absent.

    Returns ``None`` when no assistant message contains thinking/redacted_thinking
    blocks.  The caller can later pass the returned string to
    ``_restore_latest_assistant_thinking`` to guarantee the full protected content
    list survives every intermediate deep-copy and canonicalisation step untouched.

    The snapshot is a JSON-structured object containing:
        - full_content: the complete content list from the protected message
        - content_hash: SHA256 hash of the full_content JSON serialization
        - block_types: sequence of block type identifiers (e.g., ["thinking", "text"])
    """
    # Malformed request bodies can reach bridge preflight before canonicalization.
    # This helper must stay non-throwing on unexpected message/content shapes.
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        thinking_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
        ]
        if not thinking_blocks:
            return None

        block_types = [b.get("type") for b in content if isinstance(b, dict)]
        content_json = json.dumps(content, ensure_ascii=False, sort_keys=True)
        content_hash = hashlib.sha256(content_json.encode()).hexdigest()

        snapshot = {
            "full_content": content,
            "content_hash": content_hash,
            "block_types": block_types,
        }
        return json.dumps(snapshot, ensure_ascii=False)
    return None


def _restore_latest_assistant_thinking(
    messages: list[dict[str, Any]],
    snapshot: str | None,
) -> bool:
    """
    Replace the full protected content list in the latest assistant message.

    This is the inverse of ``_snapshot_latest_assistant_thinking``.  It finds
    the latest assistant message with thinking/redacted_thinking blocks and
    replaces its entire content list with the original protected content from
    *snapshot*.

    Restoration only succeeds if the post-restore hash matches the original
    snapshot hash exactly.  Partial or misaligned restoration returns failure.

    Returns ``True`` only when exact hash verification passes.
    """
    if snapshot is None:
        return False
    try:
        snapshot_data = json.loads(snapshot)
        original_content = snapshot_data.get("full_content")
        original_hash = snapshot_data.get("content_hash")
        original_block_types = snapshot_data.get("block_types")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False
    if not original_content or not original_hash or not original_block_types:
        return False

    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if not any(isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"} for b in content):
            continue

        msg["content"] = original_content

        restored_content_json = json.dumps(original_content, ensure_ascii=False, sort_keys=True)
        restored_hash = hashlib.sha256(restored_content_json.encode()).hexdigest()

        return bool(restored_hash == original_hash)
    return False


def prepare_request_impl(
    runtime_self: UniversalTokRuntime,
    request: RuntimeRequest,
    session: _core.RuntimeSession,
    *,
    result_cache: dict[str, Any] | None = None,
) -> PreparedRuntimeRequest:
    session._request_has_tools = bool(request.request_has_tools)
    session._answer_phase_expected_this_turn = False
    session._natural_response_acceptable_this_turn = False

    body: dict[str, Any] = {
        "model": request.model,
        "messages": copy.deepcopy(request.messages),
    }
    if request.system is not None:
        body["system"] = copy.deepcopy(request.system)
    original_body = copy.deepcopy(body)

    _thinking_snapshot = _snapshot_latest_assistant_thinking(request.messages)
    compressed = False

    _pre_existing_session_signals = dict(session.pending_behavior_signals)

    last_user_msg = ""
    if request.messages:
        for m in reversed(request.messages):
            if m.get("role") == "user":
                last_user_msg = text_of(cast("Any", m.get("content", "")))
                break

    if detect_prompt_bloat(body.get("system"), last_user_msg):
        session.pending_behavior_signals["tok_prompt_bloat_detected"] = 1
        current_sys = cast("Any", body.get("system", ""))
        cleaned_sys = clean_system_context(session.bridge_memory, current_sys)
        if cleaned_sys and cleaned_sys != current_sys:
            body["system"] = cleaned_sys
            session.pending_behavior_signals["tok_prompt_optimized"] = 1
            if session.bridge_memory.top_hot_files(1):
                session.pending_behavior_signals["smoothness_prompt_optimization_active_task"] = 1
            compressed = True
            logger.warning(
                "tok_prompt_optimized: system prompt reduced from %d to %d chars",
                len(text_of(current_sys) if isinstance(current_sys, list) else str(current_sys)),
                len(text_of(cleaned_sys) if isinstance(cleaned_sys, list) else str(cleaned_sys)),
            )

    translated_messages = translate_request_results(body.get("messages", []))
    body["messages"] = translated_messages

    rolling_cmds = session.bridge_memory.rolling_cmds
    if rolling_cmds:
        recent_instructions: list[Instruction] = []
        for entry in rolling_cmds[-_RECENT_COMMAND_WINDOW:]:
            parts = entry.value.strip().split()
            if not parts:
                continue
            recent_instructions.append(Instruction(op=parts[0], args=tuple(parts[1:])))

        jit_macro = session.bridge_memory.macro_registry.match_recent_sequence(recent_instructions)
        threshold = _env_int_or_default("TOK_JIT_HIT_THRESHOLD", _DEFAULT_JIT_HIT_THRESHOLD)
        if jit_macro and jit_macro.hit_count >= threshold and _jit_context_matches(jit_macro, session):
            session.pending_behavior_signals["jit_offer_available"] = 1
            session.pending_behavior_signals[f"jit_offer_{jit_macro.name}"] = 1
            session._pending_macro_heal = jit_macro.name
            session._pending_macro_heal_turn = session.bridge_memory.turn
        elif jit_macro and not _jit_context_matches(jit_macro, session):
            session.pending_behavior_signals["jit_offer_context_filtered"] = 1

    _speculative_macro_hint: str | None = None
    if session.bridge_memory.load_global_macros:
        speculative_hit_threshold = _env_int_or_default(
            "TOK_SPECULATIVE_HIT_THRESHOLD",
            _DEFAULT_SPECULATIVE_HIT_THRESHOLD,
        )
        _spec_names = [
            f"@{m.name}"
            for m in session.bridge_memory.macro_registry.macros.values()
            if m.hit_count >= speculative_hit_threshold and _jit_context_matches(m, session)
        ]
        if _spec_names:
            _speculative_macro_hint = (
                "Available macros for current context: " + ", ".join(sorted(_spec_names)) + ". Use @name to invoke."
            )
            session.pending_behavior_signals["speculative_macros_injected"] = len(_spec_names)

    id_to_context = build_tool_use_id_to_context(translated_messages)
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
    runtime_hints: list[str] = []
    injected_state_payload = ""
    history_skip_reason = ""
    should_skip_history = False
    skip_reason = ""
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

    from tok.runtime.smoothness.models import TokMode

    _lossless_mode_active = session.current_tok_mode == TokMode.LOSSLESS_TASK_MODE
    if _lossless_mode_active:
        should_skip_history = True
        skip_reason = "lossless_task_mode"
        history_skip_reason = skip_reason
        behavior_signals["lossless_task_mode_history_skipped"] = 1
        logger.info("LOSSLESS_TASK_MODE: skipping active-workset compression entirely")

    current_pressure = calculate_invisible_pressure(behavior_signals)
    request_policy = request.request_policy
    previous_effective_tool_compatible = session._request_policy_last_effective_tool_compatible
    effective_tool_compatible = request.tool_compatible and request_policy == "legacy_tool_compatible"
    request_policy_reasons: list[str] = ["legacy_default"] if effective_tool_compatible else []
    request_policy_escalated = False

    if translated_messages:
        session.bridge_memory.turn += 1
        if request_policy == "natural_first":
            (
                effective_tool_compatible,
                request_policy_reasons,
            ) = _resolve_effective_tool_compatible(
                request,
                session,
                translated_messages,
                normalized_tool_events,
                behavior_signals,
            )
        elif request_policy == "forced_baseline":
            effective_tool_compatible = False
            request_policy_reasons = []

        fresh_tool_mode_trigger = any(
            reason in {"stream_recovery", "tool_recovery", "structured_tool_loop"} for reason in request_policy_reasons
        )
        fresh_tool_mode_trigger = fresh_tool_mode_trigger and (
            session._stream_recovery_reacquisition_budget > 0
            or session._stream_recovery_history_floor_budget > 0
            or session._invalid_tool_history_recovery_count > 0
            or any(
                session.pending_behavior_signals.get(key, 0) > 0
                for key in (
                    "tok_bridge_provider_sensitive_degraded_to_provider_safe",
                    "tok_bridge_provider_sensitive_blocked_local",
                    "tok_bridge_provider_pairing_risk_detected",
                    "tok_bridge_assistant_tool_use_text_interleaving_blocked",
                    "fail_open_retry_upstream_pairing_disagreement",
                )
            )
            or "structured_tool_loop" in request_policy_reasons
        )
        recovery_sticky_continuation = (
            request_policy == "natural_first"
            and previous_effective_tool_compatible
            and effective_tool_compatible
            and not fresh_tool_mode_trigger
        )
        if request_policy == "natural_first" and effective_tool_compatible:
            if not previous_effective_tool_compatible:
                request_policy_escalated = True
                behavior_signals["request_policy_escalations"] = 1
                for reason in request_policy_reasons:
                    if reason in {
                        "stream_recovery",
                        "tool_recovery",
                        "structured_tool_loop",
                    }:
                        behavior_signals[f"request_policy_escalation_source_{reason}"] = 1
            if fresh_tool_mode_trigger:
                session._request_policy_tool_mode_sticky_turns = max(
                    session._request_policy_tool_mode_sticky_turns,
                    TOK_REQUEST_POLICY_STICKY_TURNS,
                )
        elif request_policy == "natural_first" and previous_effective_tool_compatible and not effective_tool_compatible:
            behavior_signals["request_policy_deescalations"] = 1

        if request_policy == "forced_baseline":
            behavior_signals["request_policy_forced_baseline"] = 1
        elif effective_tool_compatible:
            behavior_signals["request_policy_tool_compatible"] = 1
        else:
            behavior_signals["request_policy_natural_first"] = 1

        for reason in request_policy_reasons:
            behavior_signals[f"request_policy_reason_{reason}"] = 1

        cooldown_suppressed = getattr(session, "_stream_recovery_cooldown_suppressed", False)
        active_recovery_present = (
            session._stream_recovery_reacquisition_budget > 0 or session._stream_recovery_history_floor_budget > 0
        )
        if cooldown_suppressed and not active_recovery_present:
            behavior_signals["request_policy_recovery_cooldown_suppressed"] = 1
        if cooldown_suppressed:
            session._stream_recovery_cooldown_suppressed = False
        if (
            request_policy == "natural_first"
            and session._request_policy_tool_mode_sticky_turns > 0
            and effective_tool_compatible
        ):
            if recovery_sticky_continuation:
                behavior_signals["request_policy_held_by_recovery"] = 1
                behavior_signals["request_policy_recovery_sticky_continuations"] = 1
                session._request_policy_tool_mode_sticky_turns = 0
                session._request_policy_stream_recovery_watch_turns = 0
                session._request_policy_tool_recovery_watch_turns = 0
            else:
                session._request_policy_tool_mode_sticky_turns = max(
                    0, session._request_policy_tool_mode_sticky_turns - 1
                )
        if (
            request_policy == "natural_first"
            and session._request_policy_stream_recovery_watch_turns > 0
            and "stream_recovery" in request_policy_reasons
        ):
            session._request_policy_stream_recovery_watch_turns = max(
                0, session._request_policy_stream_recovery_watch_turns - 1
            )
        if (
            request_policy == "natural_first"
            and session._request_policy_tool_recovery_watch_turns > 0
            and "tool_recovery" in request_policy_reasons
        ):
            session._request_policy_tool_recovery_watch_turns = max(
                0, session._request_policy_tool_recovery_watch_turns - 1
            )
        session._request_policy_last_effective_tool_compatible = effective_tool_compatible
        session._natural_response_acceptable_this_turn = bool(
            request_policy == "natural_first" and request.tool_compatible and not effective_tool_compatible
        )

        runtime_hints = [_speculative_macro_hint] if _speculative_macro_hint else []
        answer_phase_expected = (
            effective_tool_compatible
            and not session._baseline_only
            and (
                session._answer_ready_repair_pending
                or session._late_answer_followthrough_pending
                or session._late_answer_assembly_repair_pending
            )
        )
        if behavior_signals.get("repeat_command_stable_no_change", 0) > 0:
            runtime_hints.append(TOK_REPEAT_COMMAND_SUPPRESSION_HINT)
            behavior_signals["repeat_command_suppression_hint_injected"] = 1
        file_read_count = sum(
            1 for event in normalized_tool_events if getattr(event, "compressibility_class", "") == "file_read"
        )
        if (
            effective_tool_compatible
            and not answer_phase_expected
            and (
                file_read_count >= FILE_READ_DENSITY_THRESHOLD
                or len(normalized_tool_events) >= TOOL_USE_DENSITY_THRESHOLD
            )
        ):
            runtime_hints.append(TOK_READ_PLAN_HINT)
            behavior_signals["read_plan_hint_injected"] = 1
        if effective_tool_compatible and not answer_phase_expected:
            runtime_hints.append(TOK_LARGE_FILE_HINT)
        answer_ready = False
        resend_signals: dict[str, int] = {}
        has_answer_anchor = False
        preserve_exact_search_evidence = False
        read_only_audit_turn = effective_tool_compatible and _is_read_only_audit_turn(translated_messages)
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

        if effective_tool_compatible:
            has_answer_facts = any(
                entry.value.startswith("answer_")
                for bucket in (session.bridge_memory.hot, session.bridge_memory.durable)
                for entry in bucket.get("facts", [])
            )
            seen_exact_evidence = bool(session._first_exact_evidence_seen or session._pending_exact_evidence_keys)
            has_answer_anchor = bool(has_answer_facts or seen_exact_evidence)
            answer_ready_turn = _is_answer_ready_turn(
                translated_messages,
                tool_compatible=effective_tool_compatible,
                has_answer_anchor=has_answer_anchor,
                baseline_only=session._baseline_only,
            )
            if answer_ready_turn:
                answer_phase_expected = True
            preserve_exact_search_evidence = bool(answer_ready_turn and has_answer_anchor)
        session._answer_phase_expected_this_turn = bool(answer_phase_expected)

        session._save_bridge_memory()
        if stream_recovery_history_floor_active:
            body["messages"] = translated_messages
        else:
            body["messages"], type_breakdown = compress_tool_results(
                translated_messages,
                result_cache=(result_cache if result_cache is not None else session.result_cache),
                tool_use_id_to_context=id_to_context,
                compression_level=policy.tool_levels[mode],
                semantic_hash_cache=session.semantic_hash_cache,
                hot_summary_records=session._hot_summary_records,
                session_files_read=session._files_read_this_session,
                files_fully_delivered=session._files_fully_delivered,
                first_exact_evidence_seen=session._first_exact_evidence_seen,
                current_turn=session.bridge_memory.turn,
                keep_turns_window=TOK_FILE_DELIVERY_STALE_TURNS,
                preserve_exact_search_evidence=preserve_exact_search_evidence,
            )
            tool_saved = sum(type_breakdown.values()) // 4
            if tool_saved > 0:
                saved_tokens += tool_saved
                compressed = True
            file_cache_hits = sum(v for k, v in type_breakdown.items() if k.endswith("_cached"))
            if file_cache_hits > 0:
                behavior_signals["tool_result_cache_hit"] = behavior_signals.get("tool_result_cache_hit", 0) + 1
            semantic_dedup_hits = type_breakdown.get("semantic_dedup", 0)
            if semantic_dedup_hits > 0:
                behavior_signals["semantic_dedup_hit"] = behavior_signals.get("semantic_dedup_hit", 0) + 1
                if effective_tool_compatible:
                    from tok.compression import _STABLE_RESULT_EXPLANATION

                    runtime_hints.append(_STABLE_RESULT_EXPLANATION)
                else:
                    runtime_hints.append(TOK_STABLE_RESULT_INFO_HINT)
            if type_breakdown.get("stable_payload_validation_failed", 0) > 0:
                behavior_signals["stable_payload_validation_failed"] = (
                    behavior_signals.get("stable_payload_validation_failed", 0)
                    + type_breakdown["stable_payload_validation_failed"]
                )

        recent: list[dict[str, Any]] = body["messages"]
        tok_state = ""
        session_memory = ""
        keep_turns = session.adaptive_keep_turns()
        if session._tok_memory_snap_triggered:
            logger.info("Memory snap triggered: forcing keep_turns=0")
            keep_turns = 0
            session._tok_memory_snap_triggered = 0

        if stream_recovery_history_floor_active:
            should_skip_history = True
            skip_reason = "stream_recovery_history_floor"
            history_skip_reason = skip_reason
            behavior_signals["stream_recovery_history_floor_applied"] = 1
        elif session.bridge_memory.turn < _SHORT_SESSION_THRESHOLD:
            # Short session: skip ALL compression to avoid overhead
            should_skip_history = True
            skip_reason = "short_session"
            history_skip_reason = skip_reason
            behavior_signals["short_session_history_skipped"] = 1
        else:
            should_skip_history, skip_reason = _should_skip_history_rewrite(
                request.messages,
                normalized_tool_events,
                tool_compatible=effective_tool_compatible,
            )

        if should_skip_history:
            if stream_recovery_history_floor_active:
                floored_recent = _stream_recovery_winnowing_floor_messages(body["messages"])
                if floored_recent:
                    if len(floored_recent) < len(body["messages"]):
                        compressed = True
                    recent = floored_recent
                    body["messages"] = recent
                    behavior_signals["stream_recovery_history_floor_kept_context"] = 1
                else:
                    recent = body["messages"]
                    behavior_signals["stream_recovery_history_floor_noop"] = 1
            else:
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
            bridge_keep_turns = max(keep_turns, 2) if request.adapter_kind == "claude-bridge" else keep_turns
            recent, tok_state = compress_history(
                body["messages"],
                keep_turns=bridge_keep_turns,
                profile=h_profile,
                prune_tool_results=True,
            )
            recent, recent_breakdown = compress_recent_window(
                recent,
                tool_use_id_to_context=id_to_context,
                tool_compatible=effective_tool_compatible,
                first_exact_evidence_seen=session._first_exact_evidence_seen,
                preserve_exact_search_evidence=preserve_exact_search_evidence,
            )
            if request.adapter_kind == "claude-bridge" and not recent and body["messages"]:
                recent, tok_state = compress_history(
                    body["messages"],
                    keep_turns=max(bridge_keep_turns, 2),
                    profile=h_profile,
                    prune_tool_results=True,
                )
                recent, recent_breakdown = compress_recent_window(
                    recent,
                    tool_use_id_to_context=id_to_context,
                    tool_compatible=effective_tool_compatible,
                    first_exact_evidence_seen=session._first_exact_evidence_seen,
                    preserve_exact_search_evidence=preserve_exact_search_evidence,
                )
                behavior_signals["bridge_minimum_tail_preserved"] = 1
            if request.adapter_kind == "claude-bridge" and _messages_contain_tool_material(recent):
                candidate_body = {
                    "model": request.model,
                    "messages": recent,
                    "system": body.get("system", ""),
                }
                pairing_failures = [
                    failure
                    for failure in validate_anthropic_bridge_body(candidate_body)
                    if has_recoverable_immediate_pairing_failures([failure])
                ]
                if pairing_failures:
                    logger.warning(
                        "tok_history_pairing_safety_degraded: rejecting compressed history that breaks immediate tool-result pairing: %s",
                        pairing_failures,
                    )
                    behavior_signals["tok_history_pairing_safety_degraded"] = 1
                    if "assistant_tool_use_missing_next_tool_result" in pairing_failures:
                        behavior_signals["tok_history_pairing_missing_next_tool_result"] = 1
                    if "assistant_tool_use_incomplete_next_tool_result_coverage" in pairing_failures:
                        behavior_signals["tok_history_pairing_incomplete_next_tool_result_coverage"] = 1
                    if "tool_result_not_immediately_after_assistant_tool_use" in pairing_failures:
                        behavior_signals["tok_history_pairing_ordering_failure"] = 1
                    if "user_tool_result_after_text" in pairing_failures:
                        behavior_signals["tok_history_pairing_user_text_before_tool_result"] = 1
                    recent = body["messages"]
                    tok_state = ""
                    recent_breakdown = {}
                    # Immediate escalation: if natural_first and pairing safety degraded,
                    # escalate to tool-compatible mode now rather than waiting for next turn
                    if request_policy == "natural_first" and not effective_tool_compatible:
                        effective_tool_compatible = True
                        if not request_policy_escalated:
                            request_policy_escalated = True
                            behavior_signals["request_policy_escalations"] = 1
                            behavior_signals["request_policy_escalation_source_tool_recovery"] = 1
                        behavior_signals["request_policy_reason_tool_recovery"] = 1
                        behavior_signals["request_policy_tool_compatible"] = (
                            behavior_signals.get("request_policy_natural_first", 0) + 1
                        )
                        behavior_signals["request_policy_natural_first"] = 0
                        # Set recovery watch for sticky continuation
                        session._request_policy_tool_recovery_watch_turns = max(
                            session._request_policy_tool_recovery_watch_turns,
                            TOK_REQUEST_POLICY_STICKY_TURNS,
                        )
            for k, v in recent_breakdown.items():
                type_breakdown[f"recent_{k}"] = type_breakdown.get(f"recent_{k}", 0) + v

        if not should_skip_history:
            if tok_state:
                logger.info(f"HISTORY WINNOWING SUCCESS: msgs {len(body['messages'])} -> {len(recent)}")
                _in_active_tool_loop = any(
                    behavior_signals.get(k, 0) > 0
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
                    should_skip_history = True
                    behavior_signals["smoothness_guarded_history_winnowing_skipped"] = 1
                    logger.info(
                        "GUARDED_TOK: skipping history winnowing in active tool loop (mode=%s)",
                        session.current_tok_mode.value,
                    )
                else:
                    if _in_active_tool_loop:
                        behavior_signals["smoothness_history_winnowing_active_loop"] = 1
                    body["messages"] = recent
                    compressed = True
                    if effective_tool_compatible:
                        behavior_signals["tool_compatible_compression"] = (
                            behavior_signals.get("tool_compatible_compression", 0) + 1
                        )
                    session_memory = session.refresh_hot_memory(tok_state, model=request.model)
            else:
                behavior_signals["tok_history_cut_point_missing"] = 1
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
                    behavior_signals["tok_history_cut_point_missing_with_tools"] = 1
                    behavior_signals["tok_history_cut_blocked_tool_result"] = 1
                session_memory = session.refresh_hot_memory("", model=request.model)
        else:
            session_memory = session.refresh_hot_memory("", model=request.model)

        if effective_tool_compatible:
            if skip_reason == "short_session":
                # Short session: skip ALL Tok additions to avoid overhead
                behavior_signals["short_session_system_additions_skipped"] = 1
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
                )
                answer_phase_now = bool(
                    answer_ready
                    or session._answer_ready_repair_active
                    or session._late_answer_followthrough_active
                    or session._late_answer_assembly_repair_active
                )
                session._answer_phase_expected_this_turn = answer_phase_now
                if answer_phase_now and runtime_hints:
                    runtime_hints = [
                        hint for hint in runtime_hints if hint not in {TOK_READ_PLAN_HINT, TOK_LARGE_FILE_HINT}
                    ]
                if len(runtime_hints) > RUNTIME_HINTS_MAX_PER_TURN:
                    runtime_hints = runtime_hints[:RUNTIME_HINTS_MAX_PER_TURN]
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
                )
            has_answer_anchor = bool(behavior_signals.get("answer_anchor_present", 0))
        elif skip_reason == "short_session":
            # Short session: skip ALL Tok additions to avoid overhead
            behavior_signals["short_session_system_additions_skipped"] = 1
        else:
            if len(runtime_hints) > RUNTIME_HINTS_MAX_PER_TURN:
                runtime_hints = runtime_hints[:RUNTIME_HINTS_MAX_PER_TURN]
            system_body = inject_system_additions(
                body,
                tok_state=session_memory,
                tool_compatible=False,
                pressure=current_pressure,
                runtime_hints=runtime_hints,
                behavior_signals=behavior_signals,
            )
            body["system"] = system_body.get("system", body.get("system", ""))

        _mut_signals = mutation_signals(original_body, body)
        for key, value in _mut_signals.items():
            behavior_signals[key] = behavior_signals.get(key, 0) + value
        if _mut_signals.get("tok_preflight_rejected"):
            body = original_body
            if session._pending_exact_evidence_keys:
                session._first_exact_evidence_seen.update(session._pending_exact_evidence_keys)
                session._pending_exact_evidence_keys.clear()
            session._bump_signals(_mut_signals)
            session._save_bridge_memory()
            _record_structured_answer_expectation(session, body)
            return PreparedRuntimeRequest(
                body=body,
                compressed=False,
                input_saved_tokens=0,
                type_breakdown={},
                behavior_signals=behavior_signals,
                mode=mode,
                request_policy=request_policy,
                effective_tool_compatible=effective_tool_compatible,
                request_policy_escalated=request_policy_escalated,
                normalized_tool_events=normalized_tool_events,
            )

        prepared_prompt_tokens = session.prepared_prompt_tokens(body)
        baseline_prompt_tokens = session.prepared_prompt_tokens(original_body)
        saved_prompt_tokens = max(0, baseline_prompt_tokens - prepared_prompt_tokens)
        if saved_prompt_tokens > 0:
            compressed = True
            saved_tokens += saved_prompt_tokens

        for key, value in session.pending_behavior_signals.items():
            if value and value > _pre_existing_session_signals.get(key, 0):
                behavior_signals[key] = behavior_signals.get(key, 0) + value - _pre_existing_session_signals.get(key, 0)

        for key, value in hot_hint_metrics.items():
            if value:
                behavior_signals[key] = behavior_signals.get(key, 0) + value

        session._bump_signals(behavior_signals)
        session._bump_signals(hot_hint_metrics)
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
    else:
        session._request_policy_last_effective_tool_compatible = effective_tool_compatible
        session._natural_response_acceptable_this_turn = bool(
            request_policy == "natural_first" and request.tool_compatible and not effective_tool_compatible
        )
        prepared_prompt_tokens = session.prepared_prompt_tokens(body)
        baseline_prompt_tokens = prepared_prompt_tokens
        saved_prompt_tokens = 0

    if session._pending_exact_evidence_keys:
        session._first_exact_evidence_seen.update(session._pending_exact_evidence_keys)
        session._pending_exact_evidence_keys.clear()

    canonical_body, canonicalized, canonical_signals = (
        canonicalize_anthropic_bridge_body(body)
        if request.adapter_kind in ("claude-bridge", "orchestrator")
        else (body, False, {})
    )
    if canonicalized:
        body = canonical_body
        for key, value in canonical_signals.items():
            behavior_signals[key] = behavior_signals.get(key, 0) + value

    if _restore_latest_assistant_thinking(body.get("messages", []), _thinking_snapshot):
        logger.debug("thinking_block_restore: restored latest assistant thinking blocks after canonicalization")

    _record_structured_answer_expectation(session, body)
    return PreparedRuntimeRequest(
        body=body,
        compressed=compressed,
        input_saved_tokens=saved_tokens,
        type_breakdown=type_breakdown,
        behavior_signals=behavior_signals,
        mode=mode,
        request_policy=request_policy,
        effective_tool_compatible=effective_tool_compatible,
        request_policy_escalated=request_policy_escalated,
        normalized_tool_events=normalized_tool_events,
        baseline_prompt_tokens=baseline_prompt_tokens,
        prepared_prompt_tokens=prepared_prompt_tokens,
        saved_prompt_tokens=saved_prompt_tokens,
        hot_hint_tokens_added=hot_hint_metrics.get("hot_hint_tokens_added", 0),
        reacquisition_tokens_avoided_estimate=hot_hint_metrics.get("reacquisition_tokens_avoided_estimate", 0),
    )
