"""Observation and snapshot helpers for runtime sessions."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any

from .config import (
    TOK_HOT_RECENT_MAX_HINTS,
    TOK_NEIGHBORHOOD_TRIGGER_ANCHORS,
    TOK_NEIGHBORHOOD_WINDOW_TURNS,
    TOK_NOVELTY_REQUIRED_HINT,
    TOK_PREDICTIVE_CACHE_TOP_K,
    TOK_NEIGHBORHOOD_THRASH_HINT,
)
from .pipeline.tool_processing import count_tokens

if TYPE_CHECKING:
    from .core import RuntimeSession


def record_file_snapshot(
    session: RuntimeSession, path: str, snippet: str
) -> bool:
    recorded = session.bridge_memory.record_file_snapshot(path, snippet)
    if recorded:
        session._bump_signals({"file_snapshot_recorded": 1})
        session._save_bridge_memory()
    return recorded


def record_search_snapshot(
    session: RuntimeSession, query: str, snippet: str
) -> bool:
    recorded = session.bridge_memory.record_search_snapshot(query, snippet)
    if recorded:
        session._bump_signals({"search_snapshot_recorded": 1})
        session._save_bridge_memory()
    return recorded


def record_history_snapshot(
    session: RuntimeSession, path: str, revision: str, snippet: str
) -> bool:
    recorded = session.bridge_memory.record_history_snapshot(
        path, revision, snippet
    )
    if recorded:
        session._bump_signals({"history_snapshot_recorded": 1})
        session._save_bridge_memory()
    return recorded


def record_metadata_snapshot(
    session: RuntimeSession, path: str, subtype: str, snippet: str
) -> bool:
    recorded = session.bridge_memory.record_metadata_snapshot(
        path, subtype, snippet
    )
    if recorded:
        session._save_bridge_memory()
    return recorded


def prepared_prompt_tokens(
    session: RuntimeSession, payload: dict[str, Any]
) -> int:
    prompt_payload = {
        "system": copy.deepcopy(payload.get("system", "")),
        "messages": copy.deepcopy(payload.get("messages", [])),
    }
    fingerprint = hashlib.sha256(
        json.dumps(prompt_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    if fingerprint in session._prepared_prompt_token_cache:
        session._bump_signals({"prepared_prompt_token_cache_hit": 1})
        return session._prepared_prompt_token_cache[fingerprint]
    token_count = count_tokens(json.dumps(prompt_payload, sort_keys=True))
    session._prepared_prompt_token_cache[fingerprint] = token_count
    if len(session._prepared_prompt_token_cache) > 32:
        oldest_key = next(iter(session._prepared_prompt_token_cache))
        session._prepared_prompt_token_cache.pop(oldest_key, None)
    return token_count


def hot_recent_runtime_hints(
    session: RuntimeSession,
) -> tuple[list[str], dict[str, int]]:
    current_turn = max(1, session.bridge_memory.turn)
    candidates = []
    for record in session._hot_summary_records.values():
        promoted_turn = max(
            record.hot_promotion_turn, record.stuck_promotion_turn
        )
        if not promoted_turn or promoted_turn <= record.last_injected_turn:
            continue
        if current_turn < promoted_turn:
            continue
        candidates.append(record)
    candidates.sort(
        key=lambda record: (
            record.stuck_window_count,
            record.last_seen_turn,
            record.token_cost,
        ),
        reverse=True,
    )
    selected = candidates[:TOK_HOT_RECENT_MAX_HINTS]
    hints: list[str] = []
    metrics = {
        "repeat_tool_collapse_applied": 0,
        "hot_recent_hint_injected": 0,
        "hot_hint_tokens_added": 0,
        "reacquisition_tokens_avoided_estimate": 0,
    }
    for record in selected:
        label = record.display_target
        if record.tool_family == "file_read":
            reminder = f"@hot_recent_file:{label} |> {record.summary}"
        elif record.tool_family == "search":
            reminder = f"@hot_recent_search:{label} |> {record.summary}"
        else:
            reminder = f"@hot_recent_command:{label} |> {record.summary}"
        guidance = (
            "This target is stuck and unchanged. Reuse the cached result and move forward without rereading it."
            if record.tool_family == "file_read"
            and record.stuck_promotion_turn
            else "Reuse this cached result unless you have a concrete reason to reacquire it."
        )
        block = reminder + "\n" + guidance
        hints.append(block)
        metrics["hot_recent_hint_injected"] += 1
        metrics["reacquisition_tokens_avoided_estimate"] += record.token_cost
        if (
            record.tool_family in {"search", "command"}
            and record.unchanged_result_count > 0
        ):
            metrics["repeat_tool_collapse_applied"] += 1
        record.last_injected_turn = current_turn
    if hints:
        metrics["hot_hint_tokens_added"] = count_tokens("\n\n".join(hints))
    return hints, metrics


def evidence_intent_advisories(session: RuntimeSession) -> list[str]:
    current_turn = max(1, session.bridge_memory.turn)
    for record in session._hot_summary_records.values():
        if not record.evidence_intent:
            continue
        if not (record.hot_promotion_turn or record.stuck_promotion_turn):
            continue
        anchor = record.evidence_intent.anchor
        novelty_keys = session._evidence_anchor_novelty_keys.get(anchor)
        if novelty_keys and record.repeat_count > 1:
            return [
                TOK_NOVELTY_REQUIRED_HINT.format(anchor=record.display_target)
            ]
    for neighborhood, anchors in session._evidence_neighborhoods.items():
        if len(anchors) < TOK_NEIGHBORHOOD_TRIGGER_ANCHORS:
            continue
        recent_count = sum(
            1
            for event in session._recent_repeat_target_events
            if event.evidence_anchor in anchors
            and current_turn - event.turn_index < TOK_NEIGHBORHOOD_WINDOW_TURNS
        )
        if recent_count >= TOK_NEIGHBORHOOD_TRIGGER_ANCHORS:
            return [
                TOK_NEIGHBORHOOD_THRASH_HINT.format(neighborhood=neighborhood)
            ]
    return []


def apply_predictive_cache_warming(
    session: RuntimeSession, logical_target: str
) -> dict[str, int]:
    candidate_keys: list[str] = []
    record = session._hot_summary_records.get(f"file_read|{logical_target}")
    if not record:
        return {}
    for key, other in sorted(
        session._hot_summary_records.items(),
        key=lambda item: item[1].last_seen_turn,
        reverse=True,
    ):
        if key == f"file_read|{logical_target}":
            continue
        if other.tool_family == "file_read" and str(
            other.logical_target.rsplit("/", 1)[0]
        ) == str(logical_target.rsplit("/", 1)[0]):
            candidate_keys.append(key)
        if other.tool_family == "search":
            same_turn = any(
                event.turn_index == record.last_seen_turn
                and event.tool_family == "search"
                and f"{event.tool_family}|{event.logical_target}" == key
                for event in session._recent_repeat_target_events
            )
            if same_turn:
                candidate_keys.append(key)
    deduped: list[str] = []
    seen: set[str] = set()
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    warmed = 0
    selected = deduped[:TOK_PREDICTIVE_CACHE_TOP_K]
    for key in selected:
        if key in session._hot_summary_records:
            session._predictive_cache_warm_keys.add(key)
            warmed += 1
    if not selected:
        return {}
    return {
        "predictive_cache_warm_applied": 1,
        "predictive_cache_candidates": len(selected),
        "predictive_cache_hits": warmed,
    }
