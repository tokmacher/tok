from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tok.compression import compress_tool_results
from tok.runtime._context_fidelity import compute_fidelity_overrides
from tok.runtime.core import RuntimeSession
from tok.runtime.types import RuntimeRequest

from ._prepare_translate_classify import _exact_search_evidence_keys_in_messages


@dataclass
class Step6Result:
    body: dict[str, Any] = field(default_factory=dict)
    type_breakdown: dict[str, int] = field(default_factory=dict)
    saved_tokens: int = 0
    compressed: bool = False
    current_path: str = ""
    behavior_signals: dict[str, int] = field(default_factory=dict)
    runtime_hints: list[str] = field(default_factory=list)
    compress_tool_results_bypassed: bool = False


def _first_exact_evidence_seen_for_compression(
    session: RuntimeSession,
    preserve_exact_search_evidence: bool,
    exact_search_evidence_keys_in_request: set[str],
) -> set[str]:
    if not preserve_exact_search_evidence:
        return session._first_exact_evidence_seen
    seen = set(session._first_exact_evidence_seen)
    seen.difference_update(exact_search_evidence_keys_in_request)
    return seen


def _retains_required_exact_search_evidence(
    messages: list[dict[str, Any]],
    id_to_context: dict[str, dict[str, Any]],
    preserve_exact_search_evidence: bool,
    exact_search_evidence_keys_in_request: set[str],
) -> bool:
    if not preserve_exact_search_evidence:
        return True
    retained_exact_keys = _exact_search_evidence_keys_in_messages(messages, id_to_context)
    return exact_search_evidence_keys_in_request.issubset(retained_exact_keys)


def run_step_6(
    session: RuntimeSession,
    request: RuntimeRequest,
    body: dict[str, Any],
    translated_messages: list[dict[str, Any]],
    id_to_context: dict[str, dict[str, Any]],
    behavior_signals: dict[str, int],
    effective_tool_compatible: bool,
    preserve_exact_search_evidence: bool,
    broad_audit_batch: bool,
    edit_reacquisition_signals: dict[str, int],
    stream_recovery_history_floor_active: bool,
    plan_finalization_turn: bool,
    mode: str,
    policy: Any,
    exact_search_evidence_keys_in_request: set[str],
    current_pressure: int,
    saved_tokens: int,
    compressed: bool,
    result_cache: dict[str, Any] | None,
) -> Step6Result:
    from tok.runtime.config import TOK_FILE_DELIVERY_STALE_TURNS

    type_breakdown: dict[str, int] = {}
    runtime_hints: list[str] = []

    session._save_bridge_memory()
    fidelity_overrides, current_path = compute_fidelity_overrides(
        id_to_context,
        session._file_reads_by_turn,
        session._last_elevated_path,
        session.bridge_memory.turn,
    )
    if not fidelity_overrides and session._last_elevated_path:
        session._last_elevated_path = ""
    elif fidelity_overrides and current_path:
        session._last_elevated_path = current_path

    if broad_audit_batch:
        body["messages"] = translated_messages
        behavior_signals["broad_audit_tool_result_compression_skipped"] = 1
        behavior_signals["compress_tool_results_bypassed"] = 1
    elif edit_reacquisition_signals:
        body["messages"] = translated_messages
        behavior_signals["evidence_tool_result_compression_skipped"] = 1
        behavior_signals["compress_tool_results_bypassed"] = 1
    elif stream_recovery_history_floor_active:
        body["messages"] = translated_messages
        behavior_signals["compress_tool_results_bypassed"] = 1
    elif plan_finalization_turn:
        body["messages"] = translated_messages
        behavior_signals["plan_finalization_tool_result_compression_skipped"] = 1
        behavior_signals["compress_tool_results_bypassed"] = 1
    else:
        effective_compression_level = policy.tool_levels[mode]
        if session.model_profile.compression_aggressiveness < 0.8:
            aggressive_levels = {"aggressive", "full", "maximum"}
            if effective_compression_level in aggressive_levels:
                effective_compression_level = "balanced"
        body["messages"], type_breakdown = compress_tool_results(
            translated_messages,
            result_cache=(result_cache if result_cache is not None else session.result_cache),
            tool_use_id_to_context=id_to_context,
            compression_level=effective_compression_level,
            semantic_hash_cache=session.semantic_hash_cache,
            hot_summary_records=session._hot_summary_records,
            session_files_read=session._files_read_this_session,
            files_fully_delivered=session._files_fully_delivered,
            first_exact_evidence_seen=_first_exact_evidence_seen_for_compression(
                session, preserve_exact_search_evidence, exact_search_evidence_keys_in_request
            ),
            current_turn=session.bridge_memory.turn,
            keep_turns_window=TOK_FILE_DELIVERY_STALE_TURNS,
            preserve_exact_search_evidence=preserve_exact_search_evidence,
            recently_edited_files=dict(session._recently_edited_files),
            file_heat=dict(session.bridge_memory._file_heat),
            session=session,
            model_profile=session.effective_model_profile,
        )
        tool_saved = sum(type_breakdown.values()) // 4
        if tool_saved > 0:
            saved_tokens += tool_saved
            compressed = True
        file_cache_hits = sum(v for k, v in type_breakdown.items() if k.endswith("_cached"))
        if file_cache_hits > 0:
            behavior_signals["tool_result_cache_hit"] = behavior_signals.get("tool_result_cache_hit", 0) + 1
        command_cache_saved_chars = int(type_breakdown.get("command_cached", 0))
        _cacheable_count = type_breakdown.get("command_cacheable_seen", 0)
        if _cacheable_count > 0:
            behavior_signals["command_result_cacheable_seen"] = (
                behavior_signals.get("command_result_cacheable_seen", 0) + _cacheable_count
            )
        if command_cache_saved_chars > 0:
            behavior_signals["command_result_cache_hit"] = behavior_signals.get("command_result_cache_hit", 0) + 1
            behavior_signals["command_result_cache_saved_tokens"] = (
                behavior_signals.get("command_result_cache_saved_tokens", 0) + command_cache_saved_chars // 4
            )
        for _cache_sig in (
            "command_cache_stored",
            "command_cache_hit",
            "command_cache_refreshed_stale",
            "command_cache_replaced_changed",
            "command_cache_skip_ineligible_cmd",
            "command_cache_first_exact_ineligible",
            "command_cache_first_exact_no_cache",
            "command_cache_reached_apply",
        ):
            _cache_val = type_breakdown.get(_cache_sig, 0)
            if _cache_val > 0:
                behavior_signals[_cache_sig] = behavior_signals.get(_cache_sig, 0) + _cache_val
        semantic_dedup_hits = type_breakdown.get("semantic_dedup", 0)
        if semantic_dedup_hits > 0:
            behavior_signals["semantic_dedup_hit"] = behavior_signals.get("semantic_dedup_hit", 0) + 1
            from tok.compression import _STABLE_RESULT_EXPLANATION

            runtime_hints.append(_STABLE_RESULT_EXPLANATION)
        if type_breakdown.get("stable_payload_validation_failed", 0) > 0:
            behavior_signals["stable_payload_validation_failed"] = (
                behavior_signals.get("stable_payload_validation_failed", 0)
                + type_breakdown["stable_payload_validation_failed"]
            )

    compress_tool_results_bypassed = bool(behavior_signals.get("compress_tool_results_bypassed", 0))

    return Step6Result(
        body=body,
        type_breakdown=type_breakdown,
        saved_tokens=saved_tokens,
        compressed=compressed,
        current_path=current_path,
        behavior_signals=behavior_signals,
        runtime_hints=runtime_hints,
        compress_tool_results_bypassed=compress_tool_results_bypassed,
    )
