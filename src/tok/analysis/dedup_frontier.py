from __future__ import annotations

"""Replay-first investigation tooling for Tok's semantic dedup frontier."""

import ast
import copy
import json
import os
import posixpath
import re
import shlex
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..compression import (
    _SEMANTIC_HASH_MIN_CHARS,
    _compute_semantic_hash,
    compress_tool_results,
)
from ..runtime.core import RuntimeRequest, RuntimeSession, UniversalTokRuntime

MISS_REASON_TAXONOMY = (
    "first_seen",
    "below_min_chars",
    "context_missing",
    "identity_changed",
    "exact_content_changed",
    "volatile_only_change",
    "history_winnowing_blocked",
    "preflight_reverted",
)
_STRUCTURAL_MISS_REASONS = frozenset(
    {"history_winnowing_blocked", "preflight_reverted"}
)
_PARITY_SESSION_HINTS = (
    "parity",
    "orchestrator",
    "alternating_adapters",
    "bridge_vs_orchestrator",
)
_MIN_BELOW_MIN_CHARS_HEADROOM = int(
    os.getenv("TOK_DEDUP_FRONTIER_MIN_SMALL_HEADROOM", "256")
)
_FILE_READ_THRESHOLD_EXPERIMENTS = (64, 96, 128)
_PRIMARY_INCREMENTAL_CLASSES = frozenset(
    {
        "small_file_repeat",
        "small_search_repeat",
        "small_command_repeat",
        "large_exact_repeat",
        "volatile_repeat",
        "alias_miss",
        "structural_cliff",
    }
)

_FILE_TOOL_ALIASES = frozenset(
    {"view", "view_file", "read", "read_file", "cat", "open_file", "get_file"}
)
_SEARCH_TOOL_ALIASES = frozenset({"grep", "grep_search", "search", "rg"})
_COMMAND_TOOL_ALIASES = frozenset({"bash", "sh", "run_terminal", "computer"})
_PATH_ARG_KEYS = (
    "path",
    "file_path",
    "AbsolutePath",
    "TargetFile",
    "search_path",
)
_QUERY_ARG_KEYS = ("query", "pattern", "search", "text")
_COMMAND_ARG_KEYS = ("command", "cmd")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.I,
)
_HEX_ID_RE = re.compile(r"\b(?:0x)?[0-9a-f]{16,}\b", re.I)
_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?\b"
)
_TIME_ONLY_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b")
_WS_RE = re.compile(r"[ \t]+")
_INLINE_JSON_RE = re.compile(r"(\{[^{}\n]+\}|\[[^\[\]\n]+\])")
_TOOL_RESULT_RE = re.compile(r"Tool results: ~\d+ tokens saved (\{.*?\})")


@dataclass(frozen=True)
class DedupOpportunity:
    session_id: str
    source_kind: str
    turn_index: int
    tool_use_id: str
    tool_name: str
    normalized_tool_identity: str
    normalized_args_identity: str
    logical_target_identity: str
    repeat_class: str
    opportunity_class: str | None
    raw_content_length: int
    current_outcome: str
    miss_reason: str | None
    actual_chars_saved: int
    current_strategy_saved_chars: int
    candidate_strategy: str | None
    candidate_strategy_saved_chars: int
    incremental_headroom_chars: int
    estimated_canonicalization_headroom_chars: int
    canonicalization_would_dedup: bool
    context_missing: bool
    history_winnowing_blocked: bool
    preflight_reverted: bool
    exact_repeat_match: bool = False
    canonical_repeat_match: bool = False
    trusted_source: bool = False
    benign_history_cliff: bool = False
    actionable_miss: bool = False
    repeat_opportunity_exists: bool = False
    context_missing_cause: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DedupTurnSummary:
    session_id: str
    source_kind: str
    turn_index: int
    request_messages: int
    compressed: bool
    input_saved_tokens: int
    type_breakdown: dict[str, int]
    history_winnowing_blocked: bool
    preflight_reverted: bool
    behavior_signals: dict[str, int]
    benign_history_cliff: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _ToolResultEvent:
    turn_index: int
    tool_use_id: str
    raw_content: str
    context: dict[str, Any] | None


@dataclass(frozen=True)
class _SourceClassification:
    source_class: str
    trusted_source: bool
    noisy_reasons: tuple[str, ...]


@dataclass
class _AnalysisAccumulator:
    opportunities: list[DedupOpportunity] = field(default_factory=list)
    turn_summaries: list[DedupTurnSummary] = field(default_factory=list)
    source_summaries: list[dict[str, Any]] = field(default_factory=list)
    bridge_log_summaries: list[dict[str, Any]] = field(default_factory=list)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "tool_result":
                parts.append(str(block.get("content", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _normalize_tool_identity(name: str) -> str:
    lowered = str(name or "").strip().lower()
    if lowered in _FILE_TOOL_ALIASES:
        return "file_read"
    if lowered in _SEARCH_TOOL_ALIASES:
        return "search"
    if lowered in _COMMAND_TOOL_ALIASES:
        return "command"
    return lowered or "unknown"


def _normalize_path(
    path: str,
    workspace_root: Path | None,
    *,
    collapse_aliases: bool = True,
) -> str:
    text = str(path or "").strip()
    if not text:
        return text
    if workspace_root:
        root = str(workspace_root.resolve())
        if text.startswith(root):
            text = "$CWD" + text[len(root) :]
    text = text.replace("\\", "/")
    if not collapse_aliases:
        return text
    if text.startswith("$CWD"):
        suffix = text[4:] or "/"
        normalized_suffix = posixpath.normpath(suffix)
        if not normalized_suffix.startswith("/"):
            normalized_suffix = f"/{normalized_suffix}"
        return f"$CWD{normalized_suffix}"
    if text.startswith("/"):
        return posixpath.normpath(text)
    normalized = posixpath.normpath(text)
    return "" if normalized == "." else normalized


def _normalize_args_identity(
    context: dict[str, Any] | None,
    workspace_root: Path | None,
) -> str:
    if not context:
        return "context-missing"
    args = copy.deepcopy(context.get("args") or {})
    if not isinstance(args, dict):
        args = {"value": args}
    normalized: dict[str, Any] = {}
    for key, value in args.items():
        if key in _PATH_ARG_KEYS:
            normalized[key] = _normalize_path(
                str(value), workspace_root, collapse_aliases=False
            )
        elif key in _QUERY_ARG_KEYS or key in _COMMAND_ARG_KEYS:
            normalized[key] = str(value).strip()
        else:
            normalized[key] = value
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _extract_first_arg(
    context: dict[str, Any] | None,
    keys: tuple[str, ...],
) -> str | None:
    if not context:
        return None
    args = context.get("args") or {}
    if not isinstance(args, dict):
        return None
    for key in keys:
        value = args.get(key)
        if value:
            return str(value).strip()
    return None


def _normalize_command_family(command: str) -> str:
    text = " ".join(str(command or "").strip().split())
    if not text:
        return ""
    try:
        parts = shlex.split(text)
    except Exception:
        parts = text.split()
    while parts and Path(parts[0]).name.lower() in {"env", "/usr/bin/env"}:
        parts = parts[1:]
    if (
        len(parts) >= 2
        and Path(parts[0]).name.lower() == "uv"
        and parts[1] == "run"
    ):
        parts = parts[2:]
    if len(parts) >= 3 and Path(parts[0]).name.lower() in {
        "python",
        "python3",
    }:
        if parts[1] == "-m":
            return parts[2].lower()
    if not parts:
        return text.lower()
    return Path(parts[0]).name.lower()


def _logical_target_identity(
    context: dict[str, Any] | None,
    normalized_tool_identity: str,
    workspace_root: Path | None,
) -> str:
    if not context:
        return "context-missing"
    if normalized_tool_identity == "file_read":
        path = _extract_first_arg(context, _PATH_ARG_KEYS)
        return (
            _normalize_path(path or "", workspace_root, collapse_aliases=True)
            or "path-missing"
        )
    if normalized_tool_identity == "search":
        search_path = _normalize_path(
            _extract_first_arg(context, ("search_path", "path", "file_path"))
            or "",
            workspace_root,
            collapse_aliases=True,
        )
        query = _extract_first_arg(context, _QUERY_ARG_KEYS) or ""
        payload = {
            "query": " ".join(query.split()),
            "search_path": search_path,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if normalized_tool_identity == "command":
        family = _normalize_command_family(
            _extract_first_arg(context, _COMMAND_ARG_KEYS) or ""
        )
        return json.dumps(
            {"family": family}, sort_keys=True, separators=(",", ":")
        )
    return _normalize_args_identity(context, workspace_root)


def canonicalize_tool_result_text(
    text: str,
    *,
    workspace_root: Path | None = None,
) -> str:
    """Canonicalize volatile tool-result text for analysis-only headroom estimates."""
    canonical = str(text or "")
    canonical = canonical.replace("\r\n", "\n").replace("\r", "\n")
    canonical = _ANSI_RE.sub("", canonical)
    canonical = _UUID_RE.sub("<uuid>", canonical)
    canonical = _HEX_ID_RE.sub("<hex>", canonical)
    canonical = _TIMESTAMP_RE.sub("<timestamp>", canonical)
    canonical = _TIME_ONLY_RE.sub("<time>", canonical)
    if workspace_root:
        root = str(workspace_root.resolve())
        canonical = canonical.replace(root, "$CWD")
    stripped = canonical.strip()
    if stripped and stripped[0] in "[{":
        try:
            canonical = json.dumps(
                json.loads(stripped), sort_keys=True, separators=(",", ":")
            )
        except Exception:
            pass
    else:
        canonical = _INLINE_JSON_RE.sub(_normalize_inline_json, canonical)
    lines = []
    for line in canonical.splitlines():
        lines.append(_WS_RE.sub(" ", line.rstrip()))
    return "\n".join(lines).strip()


def _normalize_inline_json(match: re.Match[str]) -> str:
    snippet = match.group(1)
    try:
        return json.dumps(
            json.loads(snippet), sort_keys=True, separators=(",", ":")
        )
    except Exception:
        return snippet


def _stable_result_token(content: str) -> str:
    return f"@stable_result(hash:{_compute_semantic_hash(content)})"


def _stable_result_saved_chars(raw_length: int) -> int:
    token_length = len(_stable_result_token("x"))
    return max(0, raw_length - token_length)


def _extract_compressed_content(message: list[dict[str, Any]]) -> str:
    content = message[0].get("content", [])
    if not isinstance(content, list) or not content:
        return ""
    block = content[0]
    if not isinstance(block, dict):
        return str(block)
    return str(block.get("content", ""))


def _classify_outcome(compressed_content: str, actual_chars_saved: int) -> str:
    if compressed_content.startswith("@stable_result(hash:"):
        return "exact_dedup_hit"
    if "|unchanged|cached" in compressed_content:
        return "cache_hit"
    if actual_chars_saved > 0:
        return "diff_compression"
    return "no_compression"


def _opportunity_class_for(
    *,
    normalized_tool_identity: str,
    raw_content_length: int,
    repeat_class: str,
    miss_reason: str | None,
) -> str | None:
    if miss_reason in _STRUCTURAL_MISS_REASONS:
        return "structural_cliff"
    if repeat_class == "canonical_repeat":
        return "volatile_repeat"
    if repeat_class in {"alias_repeat", "cross_tool_same_output"}:
        return "alias_miss"
    if repeat_class == "first_seen":
        return None
    if raw_content_length < _SEMANTIC_HASH_MIN_CHARS:
        if normalized_tool_identity == "file_read":
            return "small_file_repeat"
        if normalized_tool_identity == "search":
            return "small_search_repeat"
        if normalized_tool_identity == "command":
            return "small_command_repeat"
    return "large_exact_repeat"


def _candidate_experiment_savings(
    *,
    normalized_tool_identity: str,
    logical_target_identity: str,
    repeat_class: str,
    raw_content_length: int,
    exact_repeat_match: bool,
    canonical_repeat_match: bool,
    current_strategy_saved_chars: int,
) -> dict[str, int]:
    stable_saved = _stable_result_saved_chars(raw_content_length)
    candidates: dict[str, int] = {}
    if (
        normalized_tool_identity == "file_read"
        and logical_target_identity not in {"context-missing", "path-missing"}
        and exact_repeat_match
    ):
        for threshold in _FILE_READ_THRESHOLD_EXPERIMENTS:
            if raw_content_length >= threshold:
                candidates[f"experiment_a_file_read_threshold_{threshold}"] = (
                    stable_saved
                )
    if (
        normalized_tool_identity in {"file_read", "search"}
        and repeat_class == "alias_repeat"
        and exact_repeat_match
    ):
        candidates["experiment_b_alias_normalized_key"] = stable_saved
    if (
        normalized_tool_identity in {"file_read", "search", "command"}
        and repeat_class == "canonical_repeat"
        and canonical_repeat_match
    ):
        candidates["experiment_c_canonical_hash"] = stable_saved
    if (
        normalized_tool_identity in {"file_read", "search", "command"}
        and exact_repeat_match
        and raw_content_length >= _SEMANTIC_HASH_MIN_CHARS
    ):
        candidates["experiment_d_shorter_strategy"] = max(
            current_strategy_saved_chars,
            stable_saved,
        )
    return candidates


def _is_user_authored_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, list):
        tool_result_blocks = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        return len(tool_result_blocks) != len(content)
    return True


def _is_benign_history_cliff(
    turn_index: int, history_winnowing_blocked: bool
) -> bool:
    return bool(history_winnowing_blocked and turn_index <= 1)


def _classify_source(
    session_id: str,
    source_kind: str,
    events: list[_ToolResultEvent],
    turn_summaries: list[DedupTurnSummary],
) -> _SourceClassification:
    noisy_reasons: list[str] = []
    lowered_id = session_id.lower()
    if source_kind == "replay_fixture":
        if any(event.context is None for event in events):
            noisy_reasons.append("missing_tool_context")
        if any(summary.preflight_reverted for summary in turn_summaries):
            noisy_reasons.append("preflight_incompatible")
        if any(hint in lowered_id for hint in _PARITY_SESSION_HINTS):
            noisy_reasons.append("parity_fixture")
        return _SourceClassification(
            source_class=(
                "clean_replay_fixture"
                if not noisy_reasons
                else "malformed_or_parity_fixture"
            ),
            trusted_source=not noisy_reasons,
            noisy_reasons=tuple(sorted(set(noisy_reasons))),
        )
    if source_kind == "stress_run":
        if any(summary.preflight_reverted for summary in turn_summaries):
            noisy_reasons.append("preflight_incompatible")
        return _SourceClassification(
            source_class="stress_artifact",
            trusted_source=not noisy_reasons,
            noisy_reasons=tuple(sorted(set(noisy_reasons))),
        )
    return _SourceClassification(
        source_class="bridge_log",
        trusted_source=False,
        noisy_reasons=tuple(sorted(set(noisy_reasons))),
    )


def _load_fixture_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _load_fixture_sessions(
    path: Path,
) -> list[tuple[str, list[dict[str, Any]]]]:
    records = _load_fixture_records(path)
    if not records:
        return []
    if all(isinstance(record.get("messages"), list) for record in records):
        sessions: list[tuple[str, list[dict[str, Any]]]] = []
        for idx, record in enumerate(records, start=1):
            session_id = (
                path.stem if len(records) == 1 else f"{path.stem}#{idx}"
            )
            sessions.append((session_id, list(record.get("messages") or [])))
        return sessions
    if all("role" in record and "content" in record for record in records):
        return [(path.stem, records)]
    messages: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record.get("messages"), list):
            messages.extend(record["messages"])
        elif "role" in record and "content" in record:
            messages.append(record)
    return [(path.stem, messages)] if messages else []


def _iter_tool_result_events(
    messages: list[dict[str, Any]],
) -> list[_ToolResultEvent]:
    turn_index = 0
    contexts: dict[str, dict[str, Any]] = {}
    events: list[_ToolResultEvent] = []
    for message in messages:
        if _is_user_authored_message(message):
            turn_index += 1
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if (
                        not isinstance(block, dict)
                        or block.get("type") != "tool_use"
                    ):
                        continue
                    tool_id = str(block.get("id", "")).strip()
                    tool_input = block.get("input", {})
                    if not tool_id or not isinstance(tool_input, dict):
                        continue
                    path = next(
                        (
                            tool_input.get(key)
                            for key in _PATH_ARG_KEYS
                            if tool_input.get(key)
                        ),
                        None,
                    )
                    query = next(
                        (
                            tool_input.get(key)
                            for key in _QUERY_ARG_KEYS
                            if tool_input.get(key)
                        ),
                        None,
                    )
                    contexts[tool_id] = {
                        "name": str(block.get("name", "")),
                        "args": tool_input,
                        "path": str(path).strip() if path else None,
                        "query": str(query).strip() if query else None,
                    }
        if message.get("role") == "tool_result":
            tool_id = str(message.get("tool_use_id", "")).strip()
            events.append(
                _ToolResultEvent(
                    turn_index=turn_index,
                    tool_use_id=tool_id,
                    raw_content=_content_text(message.get("content", "")),
                    context=copy.deepcopy(contexts.get(tool_id)),
                )
            )
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                not isinstance(block, dict)
                or block.get("type") != "tool_result"
            ):
                continue
            tool_id = str(block.get("tool_use_id", "")).strip()
            events.append(
                _ToolResultEvent(
                    turn_index=turn_index,
                    tool_use_id=tool_id,
                    raw_content=str(block.get("content", "")),
                    context=copy.deepcopy(contexts.get(tool_id)),
                )
            )
    return events


def _build_replay_turn_summaries(
    session_id: str,
    messages: list[dict[str, Any]],
    *,
    workspace_root: Path,
) -> list[DedupTurnSummary]:
    runtime = UniversalTokRuntime()
    with tempfile.TemporaryDirectory(prefix="tok-dedup-frontier-") as tmp_dir:
        session = RuntimeSession(memory_dir=Path(tmp_dir))
        summaries: list[DedupTurnSummary] = []
        for index, message in enumerate(messages):
            if not _is_user_authored_message(message):
                continue
            prefix = copy.deepcopy(messages[: index + 1])
            prepared = runtime.prepare_request(
                RuntimeRequest(
                    model="claude-sonnet-4-6",
                    messages=prefix,
                    adapter_kind="claude-bridge",
                    tool_compatible=True,
                ),
                session,
            )
            behavior = dict(prepared.behavior_signals)
            history_winnowing_blocked = bool(
                behavior.get("tok_history_cut_point_missing", 0)
            )
            summaries.append(
                DedupTurnSummary(
                    session_id=session_id,
                    source_kind="replay_fixture",
                    turn_index=len(summaries) + 1,
                    request_messages=len(prefix),
                    compressed=prepared.compressed,
                    input_saved_tokens=prepared.input_saved_tokens,
                    type_breakdown=dict(prepared.type_breakdown),
                    history_winnowing_blocked=history_winnowing_blocked,
                    preflight_reverted=bool(
                        behavior.get("tok_preflight_rejected", 0)
                        or behavior.get("tok_bridge_preflight_rejected", 0)
                    ),
                    behavior_signals=behavior,
                    benign_history_cliff=_is_benign_history_cliff(
                        len(summaries) + 1, history_winnowing_blocked
                    ),
                )
            )
    return summaries


def _build_stress_turn_summaries(
    session_id: str, turns: list[dict[str, Any]]
) -> list[DedupTurnSummary]:
    summaries: list[DedupTurnSummary] = []
    for turn in turns:
        behavior = dict(turn.get("input_behavior_signals") or {})
        turn_index = int(turn.get("turn_index", len(summaries) + 1))
        history_winnowing_blocked = bool(
            behavior.get("tok_history_cut_point_missing", 0)
        )
        summaries.append(
            DedupTurnSummary(
                session_id=session_id,
                source_kind="stress_run",
                turn_index=turn_index,
                request_messages=int(turn.get("request_messages", 0)),
                compressed=int(turn.get("input_saved_tokens", 0)) > 0,
                input_saved_tokens=int(turn.get("input_saved_tokens", 0)),
                type_breakdown={},
                history_winnowing_blocked=history_winnowing_blocked,
                preflight_reverted=bool(
                    behavior.get("tok_preflight_rejected", 0)
                    or behavior.get("tok_bridge_preflight_rejected", 0)
                ),
                behavior_signals=behavior,
                benign_history_cliff=_is_benign_history_cliff(
                    turn_index, history_winnowing_blocked
                ),
            )
        )
    return summaries


def _stress_turns_to_messages(
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in turns:
        prompt = str(turn.get("prompt", "")).strip()
        if prompt:
            messages.append({"role": "user", "content": prompt})
        tool_uses = turn.get("tool_uses") or []
        if tool_uses:
            messages.append(
                {"role": "assistant", "content": copy.deepcopy(tool_uses)}
            )
        tool_results = turn.get("tool_results") or []
        if tool_results:
            content_blocks: list[dict[str, Any]] = []
            for result in tool_results:
                if not isinstance(result, dict):
                    continue
                content_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": result.get("tool_use_id", ""),
                        "content": result.get("content", ""),
                    }
                )
            if content_blocks:
                messages.append({"role": "user", "content": content_blocks})
    return messages


def _analyze_events(
    *,
    session_id: str,
    source_kind: str,
    events: list[_ToolResultEvent],
    turn_summaries: list[DedupTurnSummary],
    workspace_root: Path,
    source_classification: _SourceClassification,
) -> list[DedupOpportunity]:
    result_cache: dict[str, tuple[str, str]] = {}
    semantic_hash_cache: dict[str, str] = {}
    last_by_identity: dict[str, dict[str, Any]] = {}
    last_by_logical_target: dict[str, dict[str, Any]] = {}
    last_by_raw_hash: dict[str, dict[str, Any]] = {}
    canonical_identities_by_tool_hash: dict[tuple[str, str], set[str]] = (
        defaultdict(set)
    )
    turn_index_to_summary = {
        summary.turn_index: summary for summary in turn_summaries
    }
    opportunities: list[DedupOpportunity] = []

    for event in events:
        raw = event.raw_content
        context = copy.deepcopy(event.context)
        normalized_tool_identity = _normalize_tool_identity(
            context.get("name", "") if context else ""
        )
        normalized_args_identity = _normalize_args_identity(
            context, workspace_root
        )
        logical_target_identity = _logical_target_identity(
            context, normalized_tool_identity, workspace_root
        )
        identity_key = f"{normalized_tool_identity}|{normalized_args_identity}"
        logical_key = f"{normalized_tool_identity}|{logical_target_identity}"
        canonical_text = canonicalize_tool_result_text(
            raw, workspace_root=workspace_root
        )
        raw_hash = _compute_semantic_hash(raw)
        canonical_hash = _compute_semantic_hash(canonical_text)
        previous = last_by_identity.get(identity_key)
        previous_same_logical = last_by_logical_target.get(logical_key)
        previous_cross_tool = last_by_raw_hash.get(raw_hash)
        prior_same_canonical_other_identity = (
            canonical_hash
            and normalized_tool_identity
            and any(
                other_identity != identity_key
                for other_identity in canonical_identities_by_tool_hash.get(
                    (normalized_tool_identity, canonical_hash), set()
                )
            )
        )
        turn_summary = turn_index_to_summary.get(event.turn_index)
        context_missing = context is None

        synthesized = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": event.tool_use_id,
                        "content": raw,
                    }
                ],
            }
        ]
        id_to_context = {event.tool_use_id: context} if context else {}
        compressed_messages, breakdown = compress_tool_results(
            synthesized,
            result_cache=result_cache,
            tool_use_id_to_context=id_to_context if context else None,
            semantic_hash_cache=semantic_hash_cache,
        )
        actual_chars_saved = int(sum(breakdown.values()))
        compressed_content = _extract_compressed_content(compressed_messages)
        current_outcome = _classify_outcome(
            compressed_content, actual_chars_saved
        )
        current_strategy_saved_chars = actual_chars_saved

        miss_reason: str | None = None
        canonical_headroom = 0
        canonicalization_would_dedup = False
        stable_savings = _stable_result_saved_chars(len(raw))
        history_winnowing_blocked = bool(
            turn_summary and turn_summary.history_winnowing_blocked
        )
        preflight_reverted = bool(
            turn_summary and turn_summary.preflight_reverted
        )
        benign_history_cliff = bool(
            turn_summary and turn_summary.benign_history_cliff
        )
        exact_repeat_match = bool(
            (previous is not None and previous.get("raw_hash") == raw_hash)
            or (
                previous_same_logical is not None
                and previous_same_logical.get("raw_hash") == raw_hash
            )
            or (
                previous_cross_tool is not None
                and previous_cross_tool.get("normalized_tool_identity")
                != normalized_tool_identity
                and previous_cross_tool.get("raw_hash") == raw_hash
            )
        )
        canonical_repeat_match = bool(
            (
                previous is not None
                and previous.get("canonical_hash") == canonical_hash
            )
            or (
                previous_same_logical is not None
                and previous_same_logical.get("canonical_hash")
                == canonical_hash
            )
        )
        if previous is not None:
            if (
                previous["canonical_hash"] == canonical_hash
                and previous["raw_hash"] != raw_hash
            ):
                repeat_class = "canonical_repeat"
            else:
                repeat_class = "same_identity_repeat"
        elif previous_same_logical is not None:
            if (
                previous_same_logical["canonical_hash"] == canonical_hash
                and previous_same_logical["raw_hash"] != raw_hash
            ):
                repeat_class = "canonical_repeat"
            else:
                repeat_class = "alias_repeat"
        elif (
            previous_cross_tool is not None
            and previous_cross_tool.get("normalized_tool_identity")
            != normalized_tool_identity
        ):
            repeat_class = "cross_tool_same_output"
        else:
            repeat_class = "first_seen"
        repeat_opportunity_exists = bool(
            repeat_class != "first_seen"
            or prior_same_canonical_other_identity
            or current_outcome in {"exact_dedup_hit", "cache_hit"}
        )
        context_missing_cause: str | None = None

        if current_outcome == "exact_dedup_hit":
            miss_reason = None
        elif preflight_reverted and exact_repeat_match:
            miss_reason = "preflight_reverted"
            canonical_headroom = stable_savings
            canonicalization_would_dedup = True
        elif history_winnowing_blocked and exact_repeat_match:
            miss_reason = "history_winnowing_blocked"
            canonical_headroom = stable_savings
            canonicalization_would_dedup = True
        elif context_missing:
            miss_reason = "context_missing"
            context_missing_cause = (
                "source_missing_context"
                if source_kind == "replay_fixture"
                else "runtime_missing_context"
            )
        elif repeat_class == "first_seen":
            miss_reason = (
                "identity_changed"
                if prior_same_canonical_other_identity
                else "first_seen"
            )
            if miss_reason == "identity_changed":
                canonical_headroom = stable_savings
                canonicalization_would_dedup = True
        elif len(raw) < _SEMANTIC_HASH_MIN_CHARS:
            miss_reason = "below_min_chars"
            if exact_repeat_match or canonical_repeat_match:
                canonical_headroom = stable_savings
                canonicalization_would_dedup = True
        elif repeat_class == "canonical_repeat":
            miss_reason = "volatile_only_change"
            canonical_headroom = stable_savings
            canonicalization_would_dedup = True
        elif not exact_repeat_match:
            miss_reason = "exact_content_changed"

        experiment_candidates = _candidate_experiment_savings(
            normalized_tool_identity=normalized_tool_identity,
            logical_target_identity=logical_target_identity,
            repeat_class=repeat_class,
            raw_content_length=len(raw),
            exact_repeat_match=exact_repeat_match,
            canonical_repeat_match=canonical_repeat_match,
            current_strategy_saved_chars=current_strategy_saved_chars,
        )
        candidate_strategy: str | None = None
        candidate_strategy_saved_chars = current_strategy_saved_chars
        for name, candidate_saved in sorted(
            experiment_candidates.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        ):
            if candidate_saved > candidate_strategy_saved_chars:
                candidate_strategy = name
                candidate_strategy_saved_chars = candidate_saved
        incremental_headroom_chars = max(
            0, candidate_strategy_saved_chars - current_strategy_saved_chars
        )
        opportunity_class = _opportunity_class_for(
            normalized_tool_identity=normalized_tool_identity,
            raw_content_length=len(raw),
            repeat_class=repeat_class,
            miss_reason=miss_reason,
        )

        actionable_miss = bool(
            source_classification.trusted_source
            and not benign_history_cliff
            and repeat_opportunity_exists
            and miss_reason is not None
            and miss_reason != "first_seen"
            and (
                incremental_headroom_chars > 0
                or miss_reason in _STRUCTURAL_MISS_REASONS
            )
        )

        opportunities.append(
            DedupOpportunity(
                session_id=session_id,
                source_kind=source_kind,
                turn_index=event.turn_index,
                tool_use_id=event.tool_use_id,
                tool_name=str(context.get("name", "")) if context else "",
                normalized_tool_identity=normalized_tool_identity,
                normalized_args_identity=normalized_args_identity,
                logical_target_identity=logical_target_identity,
                repeat_class=repeat_class,
                opportunity_class=opportunity_class,
                raw_content_length=len(raw),
                current_outcome=current_outcome,
                miss_reason=miss_reason,
                actual_chars_saved=actual_chars_saved,
                current_strategy_saved_chars=current_strategy_saved_chars,
                candidate_strategy=candidate_strategy,
                candidate_strategy_saved_chars=candidate_strategy_saved_chars,
                incremental_headroom_chars=incremental_headroom_chars,
                estimated_canonicalization_headroom_chars=canonical_headroom,
                canonicalization_would_dedup=canonicalization_would_dedup,
                context_missing=context_missing,
                history_winnowing_blocked=history_winnowing_blocked,
                preflight_reverted=preflight_reverted,
                exact_repeat_match=exact_repeat_match,
                canonical_repeat_match=canonical_repeat_match,
                trusted_source=source_classification.trusted_source,
                benign_history_cliff=benign_history_cliff,
                actionable_miss=actionable_miss,
                repeat_opportunity_exists=repeat_opportunity_exists,
                context_missing_cause=context_missing_cause,
            )
        )

        previous_record = {
            "raw_hash": raw_hash,
            "canonical_hash": canonical_hash,
            "identity_key": identity_key,
            "logical_key": logical_key,
            "normalized_tool_identity": normalized_tool_identity,
        }
        last_by_identity[identity_key] = previous_record
        last_by_logical_target[logical_key] = previous_record
        last_by_raw_hash[raw_hash] = {
            **previous_record,
            "logical_target_identity": logical_target_identity,
        }
        canonical_identities_by_tool_hash[
            (normalized_tool_identity, canonical_hash)
        ].add(identity_key)

    return opportunities


def _summarize_source(
    *,
    session_id: str,
    source_kind: str,
    opportunities: list[DedupOpportunity],
    turn_summaries: list[DedupTurnSummary],
    source_classification: _SourceClassification,
) -> dict[str, Any]:
    miss_counts = Counter(
        opportunity.miss_reason
        for opportunity in opportunities
        if opportunity.miss_reason is not None
    )
    return {
        "session_id": session_id,
        "source_kind": source_kind,
        "source_class": source_classification.source_class,
        "trusted_source": source_classification.trusted_source,
        "noisy_reasons": list(source_classification.noisy_reasons),
        "turns": len(turn_summaries),
        "dedup_opportunities": len(opportunities),
        "trusted_dedup_opportunities": sum(
            1
            for opportunity in opportunities
            if opportunity.trusted_source
            and opportunity.repeat_opportunity_exists
        ),
        "dedup_hits": sum(
            1
            for opportunity in opportunities
            if opportunity.current_outcome == "exact_dedup_hit"
        ),
        "trusted_dedup_hits": sum(
            1
            for opportunity in opportunities
            if opportunity.trusted_source
            and opportunity.repeat_opportunity_exists
            and opportunity.current_outcome == "exact_dedup_hit"
        ),
        "cache_hits": sum(
            1
            for opportunity in opportunities
            if opportunity.current_outcome == "cache_hit"
        ),
        "diff_compressions": sum(
            1
            for opportunity in opportunities
            if opportunity.current_outcome == "diff_compression"
        ),
        "canonicalization_headroom_chars": sum(
            opportunity.estimated_canonicalization_headroom_chars
            for opportunity in opportunities
        ),
        "incremental_headroom_chars": sum(
            opportunity.incremental_headroom_chars
            for opportunity in opportunities
        ),
        "trusted_incremental_headroom_chars": sum(
            opportunity.incremental_headroom_chars
            for opportunity in opportunities
            if opportunity.trusted_source
            and opportunity.repeat_opportunity_exists
        ),
        "actionable_headroom_chars": sum(
            opportunity.incremental_headroom_chars
            for opportunity in opportunities
            if opportunity.actionable_miss
        ),
        "benign_first_turn_cut_failures": sum(
            1 for summary in turn_summaries if summary.benign_history_cliff
        ),
        "history_cliff_events": sum(
            1
            for summary in turn_summaries
            if summary.history_winnowing_blocked
            and not summary.benign_history_cliff
        ),
        "preflight_revert_events": sum(
            1 for summary in turn_summaries if summary.preflight_reverted
        ),
        "dedup_misses_by_reason": {
            reason: int(miss_counts.get(reason, 0))
            for reason in MISS_REASON_TAXONOMY
        },
        "opportunity_classes": dict(
            Counter(
                opportunity.opportunity_class
                for opportunity in opportunities
                if opportunity.opportunity_class is not None
            )
        ),
    }


def _parse_bridge_log(path: Path) -> dict[str, Any]:
    text = path.read_text()
    type_breakdown: Counter[str] = Counter()
    benign_first_turn_cut_failures = 0
    history_cliff_events = 0
    for line in text.splitlines():
        if "FAILED TO FIND CUT POINT" not in line:
            continue
        if "detail=cut_index_zero_only" in line:
            benign_first_turn_cut_failures += 1
        else:
            history_cliff_events += 1
    for match in _TOOL_RESULT_RE.finditer(text):
        try:
            payload = ast.literal_eval(match.group(1))
        except Exception:
            continue
        if isinstance(payload, dict):
            for key, value in payload.items():
                type_breakdown[str(key)] += int(value)
    return {
        "session_id": path.stem,
        "source_kind": "bridge_log",
        "dedup_opportunities": 0,
        "dedup_hits": 0,
        "canonicalization_headroom_chars": 0,
        "benign_first_turn_cut_failures": benign_first_turn_cut_failures,
        "history_cliff_events": history_cliff_events,
        "preflight_revert_events": text.count("bridge_preflight_rejected"),
        "tool_result_breakdown": dict(type_breakdown),
    }


def _bucket_from_opportunities(
    opportunities: list[DedupOpportunity], opportunity_class: str
) -> dict[str, Any]:
    affected = [
        item
        for item in opportunities
        if item.opportunity_class == opportunity_class
    ]
    trusted = [
        item
        for item in affected
        if item.trusted_source and item.repeat_opportunity_exists
    ]
    actionable = [item for item in affected if item.actionable_miss]
    structural_evidence = sum(
        1
        for item in actionable
        if item.miss_reason in _STRUCTURAL_MISS_REASONS and item.turn_index > 1
    )
    return {
        "opportunity_class": opportunity_class,
        "count": len(affected),
        "trusted_count": len(trusted),
        "actionable_count": len(actionable),
        "gross_headroom_chars": sum(
            item.estimated_canonicalization_headroom_chars for item in affected
        ),
        "incremental_headroom_chars": sum(
            item.incremental_headroom_chars for item in actionable
        ),
        "structural_evidence_count": structural_evidence,
    }


def _rank_incremental_buckets(
    opportunities: list[DedupOpportunity],
) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for opportunity_class in _PRIMARY_INCREMENTAL_CLASSES:
        bucket = _bucket_from_opportunities(opportunities, opportunity_class)
        if bucket["count"] <= 0 or bucket["trusted_count"] <= 0:
            continue
        if (
            opportunity_class
            in {"small_search_repeat", "small_command_repeat"}
            and bucket["incremental_headroom_chars"]
            < _MIN_BELOW_MIN_CHARS_HEADROOM
        ):
            continue
        if (
            bucket["incremental_headroom_chars"] <= 0
            and bucket["structural_evidence_count"] <= 0
        ):
            continue
        buckets.append(bucket)
    buckets.sort(
        key=lambda item: (
            item["incremental_headroom_chars"],
            item["structural_evidence_count"],
            item["actionable_count"],
            item["trusted_count"],
        ),
        reverse=True,
    )
    return buckets


def _tool_family_incremental_headroom(
    opportunities: list[DedupOpportunity],
) -> dict[str, dict[str, int]]:
    buckets: dict[str, dict[str, int]] = {}
    for opportunity in opportunities:
        if not (
            opportunity.trusted_source
            and opportunity.repeat_opportunity_exists
        ):
            continue
        family = opportunity.normalized_tool_identity
        entry = buckets.setdefault(
            family, {"count": 0, "incremental_headroom_chars": 0}
        )
        entry["count"] += 1
        entry["incremental_headroom_chars"] += (
            opportunity.incremental_headroom_chars
        )
    return dict(
        sorted(
            buckets.items(),
            key=lambda item: item[1]["incremental_headroom_chars"],
            reverse=True,
        )
    )


def _hot_targets(
    opportunities: list[DedupOpportunity],
) -> list[dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for opportunity in opportunities:
        if not (
            opportunity.trusted_source
            and opportunity.repeat_opportunity_exists
        ):
            continue
        target = opportunity.logical_target_identity
        if target in {"context-missing", "path-missing"}:
            continue
        entry = targets.setdefault(
            target,
            {
                "logical_target_identity": target,
                "tool_families": set(),
                "count": 0,
                "incremental_headroom_chars": 0,
            },
        )
        entry["tool_families"].add(opportunity.normalized_tool_identity)
        entry["count"] += 1
        entry["incremental_headroom_chars"] += (
            opportunity.incremental_headroom_chars
        )
    ranked = sorted(
        targets.values(),
        key=lambda item: (item["incremental_headroom_chars"], item["count"]),
        reverse=True,
    )
    return [
        {
            "logical_target_identity": item["logical_target_identity"],
            "tool_families": sorted(item["tool_families"]),
            "count": item["count"],
            "incremental_headroom_chars": item["incremental_headroom_chars"],
        }
        for item in ranked[:10]
        if item["incremental_headroom_chars"] > 0
    ]


def _experiment_matrix(
    opportunities: list[DedupOpportunity],
) -> list[dict[str, Any]]:
    experiments: list[tuple[str, str]] = [
        (
            "experiment_a_file_read_threshold_64",
            "Experiment A: file-read semantic dedup threshold 64",
        ),
        (
            "experiment_a_file_read_threshold_96",
            "Experiment A: file-read semantic dedup threshold 96",
        ),
        (
            "experiment_a_file_read_threshold_128",
            "Experiment A: file-read semantic dedup threshold 128",
        ),
        (
            "experiment_b_alias_normalized_key",
            "Experiment B: alias-normalized semantic keys",
        ),
        (
            "experiment_c_canonical_hash",
            "Experiment C: canonical semantic hashing",
        ),
        (
            "experiment_d_shorter_strategy",
            "Experiment D: choose shorter cache-or-stable strategy",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for experiment_id, label in experiments:
        eligible = 0
        trusted = 0
        incremental_headroom = 0
        for opportunity in opportunities:
            candidate_savings = _candidate_experiment_savings(
                normalized_tool_identity=opportunity.normalized_tool_identity,
                logical_target_identity=opportunity.logical_target_identity,
                repeat_class=opportunity.repeat_class,
                raw_content_length=opportunity.raw_content_length,
                exact_repeat_match=opportunity.exact_repeat_match,
                canonical_repeat_match=opportunity.canonical_repeat_match,
                current_strategy_saved_chars=opportunity.current_strategy_saved_chars,
            ).get(experiment_id, opportunity.current_strategy_saved_chars)
            if candidate_savings == opportunity.current_strategy_saved_chars:
                continue
            eligible += 1
            if (
                opportunity.trusted_source
                and opportunity.repeat_opportunity_exists
            ):
                trusted += 1
                incremental_headroom += max(
                    0,
                    candidate_savings
                    - opportunity.current_strategy_saved_chars,
                )
        rows.append(
            {
                "experiment": experiment_id,
                "label": label,
                "eligible_rows": eligible,
                "trusted_rows": trusted,
                "incremental_headroom_chars": incremental_headroom,
            }
        )
    return rows


def _live_confirmation_candidates(
    incremental_buckets: list[dict[str, Any]],
) -> list[str]:
    return [
        item["opportunity_class"]
        for item in incremental_buckets
        if item["opportunity_class"] != "structural_cliff"
    ][:2]


def _build_summary(accumulator: _AnalysisAccumulator) -> dict[str, Any]:
    miss_counts: Counter[str] = Counter()
    trusted_miss_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    trusted_outcome_counts: Counter[str] = Counter()
    trusted_opportunities = [
        item
        for item in accumulator.opportunities
        if item.trusted_source and item.repeat_opportunity_exists
    ]
    actionable_opportunities = [
        item for item in accumulator.opportunities if item.actionable_miss
    ]

    for opportunity in accumulator.opportunities:
        outcome_counts[opportunity.current_outcome] += 1
        if opportunity.miss_reason is not None:
            miss_counts[opportunity.miss_reason] += 1
            if (
                opportunity.trusted_source
                and opportunity.repeat_opportunity_exists
            ):
                trusted_miss_counts[opportunity.miss_reason] += 1
        if (
            opportunity.trusted_source
            and opportunity.repeat_opportunity_exists
        ):
            trusted_outcome_counts[opportunity.current_outcome] += 1

    benign_first_turn_cut_failures = sum(
        1
        for summary in accumulator.turn_summaries
        if summary.benign_history_cliff
    ) + sum(
        int(item.get("benign_first_turn_cut_failures", 0))
        for item in accumulator.bridge_log_summaries
    )
    history_cliff_events = sum(
        1
        for summary in accumulator.turn_summaries
        if summary.history_winnowing_blocked
        and not summary.benign_history_cliff
    ) + sum(
        item.get("history_cliff_events", 0)
        for item in accumulator.bridge_log_summaries
    )
    preflight_revert_events = sum(
        1
        for summary in accumulator.turn_summaries
        if summary.preflight_reverted
    ) + sum(
        item.get("preflight_revert_events", 0)
        for item in accumulator.bridge_log_summaries
    )
    noisy_sources = [
        item
        for item in accumulator.source_summaries
        if item["source_kind"] == "replay_fixture"
        and not item["trusted_source"]
    ]
    incremental_buckets = _rank_incremental_buckets(accumulator.opportunities)
    tool_family_incremental = _tool_family_incremental_headroom(
        accumulator.opportunities
    )
    hot_targets = _hot_targets(accumulator.opportunities)
    experiment_matrix = _experiment_matrix(accumulator.opportunities)
    live_confirmation_candidates = _live_confirmation_candidates(
        incremental_buckets
    )

    patch_queue: list[dict[str, Any]] = []
    suggestion_map = {
        "small_file_repeat": "Experiment A: test file-read semantic dedup below the global threshold on exact logical-target repeats.",
        "alias_miss": "Experiment B: normalize file/search semantic keys by logical target and tool family.",
        "volatile_repeat": "Experiment C: test canonical semantic hashing only for trusted volatility patterns.",
        "large_exact_repeat": "Experiment D: choose the shorter of cache stub and stable token on large exact repeats.",
        "structural_cliff": "Fix trusted cut-point or preflight cliffs before counting on additional dedup wins.",
    }
    for bucket in incremental_buckets:
        suggestion = suggestion_map.get(bucket["opportunity_class"])
        if suggestion:
            patch_queue.append(
                {
                    "opportunity_class": bucket["opportunity_class"],
                    "gross_headroom_chars": bucket["gross_headroom_chars"],
                    "incremental_headroom_chars": bucket[
                        "incremental_headroom_chars"
                    ],
                    "actionable_count": bucket["actionable_count"],
                    "suggested_fix": suggestion,
                }
            )
        if len(patch_queue) >= 3:
            break

    return {
        "dedup_opportunities": len(accumulator.opportunities),
        "trusted_dedup_opportunities": len(trusted_opportunities),
        "dedup_hits": int(outcome_counts.get("exact_dedup_hit", 0)),
        "trusted_dedup_hits": int(
            trusted_outcome_counts.get("exact_dedup_hit", 0)
        ),
        "outcome_counts": dict(outcome_counts),
        "trusted_outcome_counts": dict(trusted_outcome_counts),
        "dedup_misses_by_reason": {
            reason: int(miss_counts.get(reason, 0))
            for reason in MISS_REASON_TAXONOMY
        },
        "trusted_dedup_misses_by_reason": {
            reason: int(trusted_miss_counts.get(reason, 0))
            for reason in MISS_REASON_TAXONOMY
        },
        "canonicalization_headroom_chars": sum(
            item.estimated_canonicalization_headroom_chars
            for item in accumulator.opportunities
        ),
        "incremental_headroom_chars": sum(
            item.incremental_headroom_chars
            for item in accumulator.opportunities
        ),
        "trusted_incremental_headroom_chars": sum(
            item.incremental_headroom_chars for item in trusted_opportunities
        ),
        "actionable_headroom_chars": sum(
            item.incremental_headroom_chars
            for item in actionable_opportunities
        ),
        "benign_first_turn_cut_failures": benign_first_turn_cut_failures,
        "history_cliff_events": history_cliff_events,
        "preflight_revert_events": preflight_revert_events,
        "noisy_fixture_count": len(noisy_sources),
        "noisy_sources": noisy_sources,
        "turn_summaries": [
            item.to_dict() for item in accumulator.turn_summaries
        ],
        "source_summaries": accumulator.source_summaries,
        "bridge_log_summaries": accumulator.bridge_log_summaries,
        "incremental_top_buckets": incremental_buckets[:5],
        "top_miss_buckets": incremental_buckets[:5],
        "tool_family_incremental_headroom": tool_family_incremental,
        "hot_targets": hot_targets,
        "experiment_matrix": experiment_matrix,
        "live_confirmation_candidates": live_confirmation_candidates,
        "patch_queue": patch_queue,
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Dedup Frontier Report",
        "",
        "## All-Source Summary",
        "",
        f"- Dedup opportunities: `{summary['dedup_opportunities']}`",
        f"- Exact dedup hits: `{summary['dedup_hits']}`",
        f"- Canonicalization headroom (chars): `{summary['canonicalization_headroom_chars']}`",
        f"- Incremental headroom (chars): `{summary['incremental_headroom_chars']}`",
        f"- Trusted incremental headroom (chars): `{summary['trusted_incremental_headroom_chars']}`",
        f"- Actionable headroom (chars): `{summary['actionable_headroom_chars']}`",
        f"- Benign first-turn cut failures: `{summary['benign_first_turn_cut_failures']}`",
        f"- History cliff events: `{summary['history_cliff_events']}`",
        f"- Preflight revert events: `{summary['preflight_revert_events']}`",
        f"- Noisy replay fixtures excluded from trusted ranking: `{summary['noisy_fixture_count']}`",
        "",
        "## Trusted Corpus Summary",
        "",
        f"- Trusted dedup opportunities: `{summary['trusted_dedup_opportunities']}`",
        f"- Trusted dedup hits: `{summary['trusted_dedup_hits']}`",
        "",
        "## Outcome Counts",
        "",
    ]
    for outcome, count in sorted(summary.get("outcome_counts", {}).items()):
        lines.append(f"- `{outcome}`: `{count}`")

    lines.extend(["", "## Trusted Outcome Counts", ""])
    for outcome, count in sorted(
        summary.get("trusted_outcome_counts", {}).items()
    ):
        lines.append(f"- `{outcome}`: `{count}`")

    lines.extend(["", "## Incremental Top Buckets (Trusted Rows)", ""])
    ranked = summary.get("incremental_top_buckets", [])
    if ranked:
        for item in ranked:
            lines.append(
                f"- `{item['opportunity_class']}`: count=`{item['count']}` "
                f"trusted=`{item['trusted_count']}` "
                f"actionable=`{item['actionable_count']}` "
                f"incremental_headroom=`{item['incremental_headroom_chars']}`"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Tool Family Incremental Headroom", ""])
    tool_families = summary.get("tool_family_incremental_headroom", {})
    if tool_families:
        for family, payload in tool_families.items():
            lines.append(
                f"- `{family}`: count=`{payload['count']}` "
                f"incremental_headroom=`{payload['incremental_headroom_chars']}`"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Hot Targets", ""])
    hot_targets = summary.get("hot_targets", [])
    if hot_targets:
        for item in hot_targets:
            lines.append(
                f"- `{item['logical_target_identity']}`: "
                f"families=`{', '.join(item['tool_families'])}` "
                f"count=`{item['count']}` "
                f"incremental_headroom=`{item['incremental_headroom_chars']}`"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Experiment Matrix", ""])
    experiment_matrix = summary.get("experiment_matrix", [])
    if experiment_matrix:
        for item in experiment_matrix:
            lines.append(
                f"- `{item['experiment']}`: trusted_rows=`{item['trusted_rows']}` "
                f"incremental_headroom=`{item['incremental_headroom_chars']}`"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Live Confirmation Candidates", ""])
    live_candidates = summary.get("live_confirmation_candidates", [])
    if live_candidates:
        for item in live_candidates:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Patch Queue (Trusted Rows)", ""])
    patch_queue = summary.get("patch_queue", [])
    if patch_queue:
        for item in patch_queue:
            lines.append(
                f"- `{item['opportunity_class']}`: {item['suggested_fix']} "
                f"(incremental headroom `{item['incremental_headroom_chars']}` chars)"
            )
    else:
        lines.append("- none")

    if summary.get("noisy_sources"):
        lines.extend(["", "## Noisy Source Exclusions", ""])
        for item in summary["noisy_sources"]:
            lines.append(
                f"- `{item['session_id']}` ({item['source_class']}): "
                f"reasons=`{', '.join(item['noisy_reasons']) or 'none'}`"
            )

    lines.extend(["", "## Source Summaries", ""])
    for item in summary.get("source_summaries", []):
        lines.append(
            f"- `{item['session_id']}` ({item['source_class']}): "
            f"opportunities=`{item['dedup_opportunities']}` "
            f"trusted_opportunities=`{item['trusted_dedup_opportunities']}` "
            f"hits=`{item['dedup_hits']}` "
            f"trusted_hits=`{item['trusted_dedup_hits']}` "
            f"trusted_incremental_headroom=`{item['trusted_incremental_headroom_chars']}` "
            f"benign_first_turn_cut_failures=`{item['benign_first_turn_cut_failures']}` "
            f"history_cliffs=`{item['history_cliff_events']}` "
            f"preflight_reverts=`{item['preflight_revert_events']}`"
        )

    if summary.get("bridge_log_summaries"):
        lines.extend(["", "## Bridge Logs", ""])
        for item in summary["bridge_log_summaries"]:
            lines.append(
                f"- `{item['session_id']}`: history_cliffs=`{item['history_cliff_events']}` "
                f"preflight_reverts=`{item['preflight_revert_events']}`"
            )

    lines.append("")
    return "\n".join(lines)


def run_dedup_frontier(
    *,
    output_dir: Path,
    fixtures_dir: Path | None = None,
    fixture_paths: list[Path] | None = None,
    stress_run_paths: list[Path] | None = None,
    bridge_log_paths: list[Path] | None = None,
    workspace_root: Path | None = None,
) -> dict[str, Path]:
    """Run the replay-first dedup frontier investigation and write artifacts."""
    workspace_root = (workspace_root or Path.cwd()).resolve()
    accumulator = _AnalysisAccumulator()

    selected_fixtures: list[Path] = []
    if fixtures_dir and fixtures_dir.exists():
        selected_fixtures.extend(
            path
            for path in sorted(fixtures_dir.rglob("*.jsonl"))
            if not path.name.endswith(".meta.json")
        )
    if fixture_paths:
        selected_fixtures.extend(
            path for path in fixture_paths if path.exists()
        )

    for fixture_path in sorted({path.resolve() for path in selected_fixtures}):
        for session_id, messages in _load_fixture_sessions(fixture_path):
            if not messages:
                continue
            turn_summaries = _build_replay_turn_summaries(
                session_id, messages, workspace_root=workspace_root
            )
            events = _iter_tool_result_events(messages)
            source_classification = _classify_source(
                session_id, "replay_fixture", events, turn_summaries
            )
            opportunities = _analyze_events(
                session_id=session_id,
                source_kind="replay_fixture",
                events=events,
                turn_summaries=turn_summaries,
                workspace_root=workspace_root,
                source_classification=source_classification,
            )
            accumulator.turn_summaries.extend(turn_summaries)
            accumulator.opportunities.extend(opportunities)
            accumulator.source_summaries.append(
                _summarize_source(
                    session_id=session_id,
                    source_kind="replay_fixture",
                    opportunities=opportunities,
                    turn_summaries=turn_summaries,
                    source_classification=source_classification,
                )
            )

    selected_stress_runs: list[Path] = []
    if stress_run_paths:
        selected_stress_runs.extend(
            path for path in stress_run_paths if path.exists()
        )
    else:
        default_stress_dir = workspace_root / "tests/fixtures/stress"
        if default_stress_dir.exists():
            selected_stress_runs.extend(
                sorted(default_stress_dir.rglob("*.json"))
            )

    for stress_path in sorted(
        {path.resolve() for path in selected_stress_runs}
    ):
        if not stress_path.exists():
            continue
        data = json.loads(stress_path.read_text())
        turns = list(data.get("turns") or [])
        session_id = stress_path.stem
        turn_summaries = _build_stress_turn_summaries(session_id, turns)
        messages = _stress_turns_to_messages(turns)
        events = _iter_tool_result_events(messages)
        source_classification = _classify_source(
            session_id, "stress_run", events, turn_summaries
        )
        opportunities = _analyze_events(
            session_id=session_id,
            source_kind="stress_run",
            events=events,
            turn_summaries=turn_summaries,
            workspace_root=workspace_root,
            source_classification=source_classification,
        )
        accumulator.turn_summaries.extend(turn_summaries)
        accumulator.opportunities.extend(opportunities)
        accumulator.source_summaries.append(
            _summarize_source(
                session_id=session_id,
                source_kind="stress_run",
                opportunities=opportunities,
                turn_summaries=turn_summaries,
                source_classification=source_classification,
            )
        )

    for log_path in bridge_log_paths or []:
        if not log_path.exists():
            continue
        accumulator.bridge_log_summaries.append(_parse_bridge_log(log_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "dedup_frontier.jsonl"
    summary_path = output_dir / "dedup_frontier_summary.json"
    report_path = output_dir / "dedup_frontier_report.md"

    with ledger_path.open("w") as handle:
        for item in accumulator.opportunities:
            handle.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")

    summary = _build_summary(accumulator)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    report_path.write_text(_render_report(summary))

    return {
        "ledger": ledger_path,
        "summary": summary_path,
        "report": report_path,
    }


__all__ = [
    "MISS_REASON_TAXONOMY",
    "canonicalize_tool_result_text",
    "run_dedup_frontier",
]
