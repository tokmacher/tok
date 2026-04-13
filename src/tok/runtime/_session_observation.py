"""Observation and snapshot helpers for runtime sessions."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any

from .config import (
    TOK_HOT_RECENT_MAX_HINTS,
    TOK_NEIGHBORHOOD_THRASH_HINT,
    TOK_NEIGHBORHOOD_TRIGGER_ANCHORS,
    TOK_NEIGHBORHOOD_WINDOW_TURNS,
    TOK_NOVELTY_REQUIRED_HINT,
    TOK_PREDICTIVE_CACHE_TOP_K,
)
from .pipeline.tool_processing import count_tokens
from .repeat_targets import HotSummaryRecord

if TYPE_CHECKING:
    from .core import RuntimeSession


def record_file_snapshot(session: RuntimeSession, path: str, snippet: str) -> bool:
    """Record a file snapshot in bridge memory and persist state."""
    recorded = session.bridge_memory.record_file_snapshot(path, snippet)
    if recorded:
        session._bump_signals({"file_snapshot_recorded": 1})
        session._save_bridge_memory()
    return recorded


def record_search_snapshot(session: RuntimeSession, query: str, snippet: str) -> bool:
    """Record a search snapshot in bridge memory and persist state."""
    recorded = session.bridge_memory.record_search_snapshot(query, snippet)
    if recorded:
        session._bump_signals({"search_snapshot_recorded": 1})
        session._save_bridge_memory()
    return recorded


def record_history_snapshot(session: RuntimeSession, path: str, revision: str, snippet: str) -> bool:
    """Record a git history snapshot in bridge memory and persist state."""
    recorded = session.bridge_memory.record_history_snapshot(path, revision, snippet)
    if recorded:
        session._bump_signals({"history_snapshot_recorded": 1})
        session._save_bridge_memory()
    return recorded


def record_metadata_snapshot(session: RuntimeSession, path: str, subtype: str, snippet: str) -> bool:
    """Record a metadata snapshot in bridge memory and persist state."""
    recorded = session.bridge_memory.record_metadata_snapshot(path, subtype, snippet)
    if recorded:
        session._save_bridge_memory()
    return recorded


def prepared_prompt_tokens(session: RuntimeSession, payload: dict[str, Any]) -> int:
    """Count and cache tokens for a prepared prompt payload."""
    prompt_payload = {
        "system": copy.deepcopy(payload.get("system", "")),
        "messages": copy.deepcopy(payload.get("messages", [])),
    }
    fingerprint = hashlib.sha256(json.dumps(prompt_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    if fingerprint in session._prepared_prompt_token_cache:
        session._bump_signals({"prepared_prompt_token_cache_hit": 1})
        return session._prepared_prompt_token_cache[fingerprint]
    token_count = count_tokens(json.dumps(prompt_payload, sort_keys=True))
    session._prepared_prompt_token_cache[fingerprint] = token_count
    if len(session._prepared_prompt_token_cache) > 32:
        oldest_key = next(iter(session._prepared_prompt_token_cache))
        session._prepared_prompt_token_cache.pop(oldest_key, None)
    return token_count


def _is_eligible_hot_record(
    record: HotSummaryRecord,
    current_turn: int,
    seen_exact_keys: set[str],
) -> bool:
    """Check if a hot summary record is eligible for hint injection."""
    if record.tool_family == "search":
        exact_key = record.exact_evidence_key
        if not exact_key or exact_key not in seen_exact_keys:
            return False
    promoted_turn = max(
        int(record.hot_promotion_turn or 0),
        int(record.stuck_promotion_turn or 0),
    )
    last_injected_turn = int(record.last_injected_turn or 0)
    if not promoted_turn or promoted_turn <= last_injected_turn:
        return False
    return current_turn >= promoted_turn


def _build_hot_hint(record: HotSummaryRecord, current_turn: int) -> tuple[str, dict[str, int]]:
    """
    Build a hot hint string and metrics for a single record.

    Returns (hint_block, record_metrics).
    """
    label = record.display_target
    if record.tool_family == "file_read":
        reminder = f"@hot_recent_file:{label} |> {record.summary}"
    elif record.tool_family == "search":
        reminder = f"@hot_recent_search:{label} |> {record.summary}"
    else:
        reminder = f"@hot_recent_command:{label} |> {record.summary}"
    block = reminder

    metrics: dict[str, int] = {
        "hot_recent_hint_injected": 1,
        "reacquisition_tokens_avoided_estimate": record.token_cost,
    }
    if record.tool_family in {"search", "command"} and record.unchanged_result_count > 0:
        metrics["repeat_tool_collapse_applied"] = 1

    record.last_injected_turn = current_turn
    return block, metrics


def hot_recent_runtime_hints(
    session: RuntimeSession,
    *,
    max_hints: int | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Generate hot recent hints for eligible repeat targets."""
    current_turn = max(1, session.bridge_memory.turn)
    seen_exact_keys = session._first_exact_evidence_seen | session._pending_exact_evidence_keys
    candidates = [
        record
        for record in session._hot_summary_records.values()
        if _is_eligible_hot_record(record, current_turn, seen_exact_keys)
    ]
    candidates.sort(
        key=lambda record: (
            record.stuck_window_count,
            record.last_seen_turn,
            record.token_cost,
        ),
        reverse=True,
    )
    hint_limit = TOK_HOT_RECENT_MAX_HINTS if max_hints is None else max(0, int(max_hints))
    selected = candidates[:hint_limit]
    hints: list[str] = []
    metrics = {
        "repeat_tool_collapse_applied": 0,
        "hot_recent_hint_injected": 0,
        "hot_hint_tokens_added": 0,
        "reacquisition_tokens_avoided_estimate": 0,
    }
    for record in selected:
        block, record_metrics = _build_hot_hint(record, current_turn)
        hints.append(block)
        for key, value in record_metrics.items():
            metrics[key] = metrics.get(key, 0) + value
    if hints:
        metrics["hot_hint_tokens_added"] = count_tokens("\n\n".join(hints))
    return hints, metrics


def evidence_intent_advisories(session: RuntimeSession) -> list[str]:
    """Generate advisories based on evidence intent patterns."""
    current_turn = max(1, session.bridge_memory.turn)
    for record in session._hot_summary_records.values():
        if not record.evidence_intent:
            continue
        if not (record.hot_promotion_turn or record.stuck_promotion_turn):
            continue
        anchor = record.evidence_intent.anchor
        novelty_keys = session._evidence_anchor_novelty_keys.get(anchor)
        if novelty_keys and record.repeat_count > 1:
            return [TOK_NOVELTY_REQUIRED_HINT.format(anchor=record.display_target)]
    for neighborhood, anchors in session._evidence_neighborhoods.items():
        if len(anchors) < TOK_NEIGHBORHOOD_TRIGGER_ANCHORS:
            continue
        recent_count = sum(
            1
            for event in session._recent_repeat_target_events
            if event.evidence_anchor in anchors and current_turn - event.turn_index < TOK_NEIGHBORHOOD_WINDOW_TURNS
        )
        if recent_count >= TOK_NEIGHBORHOOD_TRIGGER_ANCHORS:
            return [TOK_NEIGHBORHOOD_THRASH_HINT.format(neighborhood=neighborhood)]
    return []


def _is_same_directory(file_a: str, file_b: str) -> bool:
    """Check if two file paths share the same parent directory."""
    return str(file_a.rsplit("/", 1)[0]) == str(file_b.rsplit("/", 1)[0])


def _collect_predictive_candidates(
    session: RuntimeSession,
    logical_target: str,
    last_seen_turn: int,
) -> list[str]:
    """Collect candidate keys for predictive cache warming based on directory proximity."""
    candidate_keys: list[str] = []
    for key, other in sorted(
        session._hot_summary_records.items(),
        key=lambda item: item[1].last_seen_turn,
        reverse=True,
    ):
        if key == f"file_read|{logical_target}":
            continue
        if other.tool_family == "file_read" and _is_same_directory(other.logical_target, logical_target):
            candidate_keys.append(key)
        if other.tool_family == "search":
            same_turn = any(
                event.turn_index == last_seen_turn
                and event.tool_family == "search"
                and f"{event.tool_family}|{event.logical_target}" == key
                for event in session._recent_repeat_target_events
            )
            if same_turn:
                candidate_keys.append(key)
    return candidate_keys


def _dedupe_candidates(candidate_keys: list[str]) -> list[str]:
    """Remove duplicate keys while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _warm_cache_keys(session: RuntimeSession, selected: list[str]) -> int:
    """Add selected keys to predictive cache warm set. Returns warmed count."""
    warmed = 0
    for key in selected:
        if key in session._hot_summary_records:
            session._predictive_cache_warm_keys.add(key)
            warmed += 1
    return warmed


def apply_predictive_cache_warming(session: RuntimeSession, logical_target: str) -> dict[str, int]:
    """Apply predictive cache warming for a logical target and return metrics."""
    record = session._hot_summary_records.get(f"file_read|{logical_target}")
    if not record:
        return {}
    candidate_keys = _collect_predictive_candidates(session, logical_target, record.last_seen_turn)
    deduped = _dedupe_candidates(candidate_keys)
    selected = deduped[:TOK_PREDICTIVE_CACHE_TOP_K]
    warmed = _warm_cache_keys(session, selected)
    if not selected:
        return {}
    return {
        "predictive_cache_warm_applied": 1,
        "predictive_cache_candidates": len(selected),
        "predictive_cache_hits": warmed,
    }
