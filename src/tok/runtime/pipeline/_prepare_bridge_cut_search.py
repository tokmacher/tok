from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, cast

from tok.compression import (
    compress_history,
    compress_recent_window,
    text_of,
)
from tok.runtime._history_slicing import (
    _bridge_preflight_safe_recent_suffix,
    _messages_contain_tool_material,
)
from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline.request_validation import (
    canonicalize_anthropic_bridge_body,
    validate_anthropic_bridge_body,
)
from tok.runtime.types import RuntimeRequest

_BRIDGE_CUT_SEARCH_MAX_EXTRA_TURNS = 4
_BRIDGE_CUT_SEARCH_MIN_SAVED_TOKENS = 16

_EVIDENCE_PATH_RE = re.compile(
    r"(?<!\w)([\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|yml|sh|txt|css|html|sql|rs|go|rb))(?!\w)"
)


def _exact_search_evidence_keys_in_messages(
    messages: list[dict[str, Any]],
    tool_use_id_to_context: dict[str, dict[str, Any]],
) -> set[str]:
    from tok.runtime.repeat_targets import evidence_identity_key, search_result_evidence_level

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


def _bridge_candidate_body(
    *,
    request: RuntimeRequest,
    messages: list[dict[str, Any]],
    system: Any,
    seen_mutation_pairs: set[tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    candidate_body = {"model": request.model, "messages": messages, "system": system}
    canonical_body, changed, canonical_signals = canonicalize_anthropic_bridge_body(
        candidate_body, seen_mutation_pairs=seen_mutation_pairs
    )
    if changed:
        candidate_body = canonical_body
    return candidate_body, dict(canonical_signals)


def _bridge_history_cut_candidate(
    *,
    session: RuntimeSession,
    request: RuntimeRequest,
    messages: list[dict[str, Any]],
    system: Any,
    history_baseline_prompt_tokens: int,
    seen_mutation_pairs: set[tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], dict[str, int], int] | None:
    candidate_body, canonical_signals = _bridge_candidate_body(
        request=request,
        messages=messages,
        system=system,
        seen_mutation_pairs=seen_mutation_pairs,
    )
    if validate_anthropic_bridge_body(candidate_body):
        return None
    candidate_saved_prompt_tokens = max(
        0,
        history_baseline_prompt_tokens - session.prepared_prompt_tokens(candidate_body),
    )
    return candidate_body, canonical_signals, candidate_saved_prompt_tokens


def _retains_required_exact_search_evidence(
    messages: list[dict[str, Any]],
    id_to_context: dict[str, dict[str, Any]],
    exact_search_evidence_keys_in_request: set[str],
) -> bool:
    if not exact_search_evidence_keys_in_request:
        return True
    retained_exact_keys = _exact_search_evidence_keys_in_messages(messages, id_to_context)
    return exact_search_evidence_keys_in_request.issubset(retained_exact_keys)


@dataclass
class Step7aResult:
    recent: list[dict[str, Any]] = field(default_factory=list)
    tok_state: str = ""
    recent_breakdown: dict[str, int] = field(default_factory=dict)
    bridge_search_success: bool = False
    behavior_signals: dict[str, int] = field(default_factory=dict)


def run_step_7a_bridge_cut_search(
    *,
    session: RuntimeSession,
    request: RuntimeRequest,
    recent: list[dict[str, Any]],
    original_messages: list[dict[str, Any]],
    system: Any,
    id_to_context: dict[str, dict[str, Any]],
    keep_turns: int,
    bridge_keep_turns: int,
    bridge_profile: dict[str, Any],
    history_baseline_prompt_tokens: int,
    seen_mutation_pairs: set[tuple[str, str]] | None,
    preserve_exact_search_evidence: bool,
    exact_search_evidence_keys_in_request: set[str],
    _first_exact_evidence_seen_for_compression: frozenset[str],
    effective_tool_compatible: bool,
) -> Step7aResult:
    if not (request.uses_cut_search and _messages_contain_tool_material(recent)):
        return Step7aResult(recent=recent, bridge_search_success=False)

    behavior_signals: dict[str, int] = {"bridge_cut_search_guard_passed": 1}
    recent_breakdown: dict[str, int] = {}
    tok_state = ""

    bridge_candidate_had_invalid = False
    bridge_min_saved_prompt_tokens = max(
        _BRIDGE_CUT_SEARCH_MIN_SAVED_TOKENS,
        history_baseline_prompt_tokens // 6,
    )
    bridge_search_success = False

    safe_recent = _bridge_preflight_safe_recent_suffix(recent)
    if safe_recent is not None:
        recent = safe_recent
    else:
        bridge_candidate_had_invalid = True

    bridge_candidate = _bridge_history_cut_candidate(
        session=session,
        request=request,
        messages=recent,
        system=system,
        history_baseline_prompt_tokens=history_baseline_prompt_tokens,
        seen_mutation_pairs=seen_mutation_pairs,
    )
    if safe_recent is not None and bridge_candidate is not None:
        candidate_body, canonical_signals, candidate_saved_prompt_tokens = bridge_candidate
        candidate_preserves_exact = _retains_required_exact_search_evidence(
            candidate_body["messages"],
            id_to_context,
            exact_search_evidence_keys_in_request,
        )
        if candidate_preserves_exact and (
            candidate_saved_prompt_tokens >= bridge_min_saved_prompt_tokens or preserve_exact_search_evidence
        ):
            recent = candidate_body["messages"]
            behavior_signals["bridge_history_cut_search_used"] = 1
            for key, value in canonical_signals.items():
                behavior_signals[key] = behavior_signals.get(key, 0) + value
            bridge_search_success = True
    elif safe_recent is not None:
        bridge_candidate_had_invalid = True
    else:
        bridge_candidate_had_invalid = True

    if not bridge_search_success:
        bridge_search_limit = max(1, bridge_keep_turns - _BRIDGE_CUT_SEARCH_MAX_EXTRA_TURNS)
        for candidate_keep_turns in range(bridge_keep_turns - 1, bridge_search_limit - 1, -1):
            candidate_recent, candidate_tok_state, _ = compress_history(
                original_messages,
                keep_turns=candidate_keep_turns,
                profile=bridge_profile,
                prune_tool_results=True,
            )
            candidate_recent, candidate_breakdown = compress_recent_window(
                candidate_recent,
                tool_use_id_to_context=id_to_context,
                tool_compatible=effective_tool_compatible,
                first_exact_evidence_seen=set(_first_exact_evidence_seen_for_compression),
                preserve_exact_search_evidence=preserve_exact_search_evidence,
                session_files_read=session._files_read_this_session,
                model_profile=session.effective_model_profile,
            )
            candidate_recent = _bridge_preflight_safe_recent_suffix(candidate_recent) or []
            if not candidate_recent:
                bridge_candidate_had_invalid = True
                continue
            bridge_candidate = _bridge_history_cut_candidate(
                session=session,
                request=request,
                messages=candidate_recent,
                system=system,
                history_baseline_prompt_tokens=history_baseline_prompt_tokens,
                seen_mutation_pairs=seen_mutation_pairs,
            )
            if bridge_candidate is None:
                bridge_candidate_had_invalid = True
                continue
            candidate_body, canonical_signals, candidate_saved_prompt_tokens = bridge_candidate
            candidate_preserves_exact = _retains_required_exact_search_evidence(
                candidate_body["messages"],
                id_to_context,
                exact_search_evidence_keys_in_request,
            )
            if candidate_preserves_exact and candidate_saved_prompt_tokens >= bridge_min_saved_prompt_tokens:
                recent = candidate_body["messages"]
                tok_state = candidate_tok_state
                recent_breakdown = candidate_breakdown
                behavior_signals["bridge_history_cut_search_used"] = 1
                if candidate_keep_turns != bridge_keep_turns:
                    behavior_signals["bridge_history_cut_search_extended"] = 1
                    behavior_signals["bridge_history_cut_search_extension_turns"] = (
                        bridge_keep_turns - candidate_keep_turns
                    )
                for key, value in canonical_signals.items():
                    behavior_signals[key] = behavior_signals.get(key, 0) + value
                bridge_search_success = True
                break

    if not bridge_search_success:
        if bridge_candidate_had_invalid:
            behavior_signals["tok_history_pairing_safety_degraded"] = 1
        recent = original_messages
        tok_state = ""
        recent_breakdown = {}

    return Step7aResult(
        recent=recent,
        tok_state=tok_state,
        recent_breakdown=recent_breakdown,
        bridge_search_success=bridge_search_success,
        behavior_signals=behavior_signals,
    )
