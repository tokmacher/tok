"""History- and request-side compression orchestration helpers."""

from __future__ import annotations

import contextlib
import copy
import difflib
import re
from collections.abc import MutableMapping
from typing import Any

from tok.runtime.config import (
    RESULT_CACHE_TTL_SECONDS,
    TOK_ENABLE_FILE_OVERLAP_DELTA,
    TOK_ENABLE_FILE_REREAD_DIFF,
    TOK_ENABLE_PYTEST_FAIL_COMPRESSION,
    TOK_ENABLE_SEARCH_OVERLAP_DELTA,
    TOK_ENABLE_STACK_REPEAT_DELTA,
)
from tok.runtime.repeat_targets import (
    SEARCH_LIKE_TOOLS,
    build_file_skeleton,
    build_file_summary,
    evidence_identity_key,
    normalize_path_target,
    search_result_evidence_level,
)
from tok.utils.event_logging import log_semantic_dedup

from . import (
    _CUT_REJECTION_REASONS,
    _SEMANTIC_HASH_MIN_CHARS,
    EDIT_LIKE_TOOLS,
    FILE_LIKE_TOOLS,
    QUESTION_PREFIXES,
    RECENT_WINDOW_EVIDENCE_THRESHOLD,
    RECENT_WINDOW_THRESHOLD,
    STOP_WORDS,
    _apply_result_cache,
    _compute_semantic_hash,
    _make_semantic_cache_key,
    _should_include_tok_state,
    _summarize_causal_failures,
    _summarize_decision_hypotheses,
    classify_cut_eligibility,
    logger,
    text_of,
)
from ._registry import Compressor
from ._tool_result_codecs import (
    _compress_file_read,
    _compress_git_diff,
    _compress_grep,
    _compress_grep_context,
    _compress_install,
    _compress_ls,
    _compress_pytest,
    _compress_search_results,
    _compress_stack_traces,
)
from ._tool_result_pipeline import (
    compress_git_log_impl as _compress_git_log_impl_fn,
)
from ._tool_result_pipeline import (
    detect_tool_content_type_impl as _detect_tool_content_type_impl_fn,
)
from ._tool_result_pipeline import (
    tok_tool_result_impl as _tok_tool_result_impl,
)

__all__ = [
    "RECENT_WINDOW_THRESHOLD",
    "TOOL_COMPRESS_THRESHOLD",
    "_compress_git_log_impl",
    "_detect_tool_content_type_impl",
    "compress_history_impl",
    "compress_recent_window_impl",
    "compress_tool_results_impl",
    "inject_system_additions_impl",
    "tok_tool_result_impl",
]


def _detect_tool_content_type_impl(text: str) -> str:
    return _detect_tool_content_type_impl_fn(text)


def _compress_git_log_impl(text: str) -> str:
    return _compress_git_log_impl_fn(text)


def _extract_normalized_path(context: dict[str, Any] | None) -> str:
    """Extract normalized file path from tool context."""
    if not context:
        return ""
    raw_args = context.get("args")
    args = raw_args if isinstance(raw_args, dict) else None
    if not args:
        return ""
    path = str(args.get("path") or args.get("file_path") or args.get("AbsolutePath") or args.get("TargetFile") or "")
    return path.lower().strip()


TOOL_COMPRESS_THRESHOLD = 0


def _is_tool_result_only_user_message(message: dict[str, Any]) -> bool:
    if str(message.get("role", "")).strip() != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def _cut_splits_tool_pair(messages: list[dict[str, Any]], cut_index: int) -> bool:
    prefix_use_ids: set[str] = set()
    for idx in range(cut_index):
        msg = messages[idx]
        if not isinstance(msg, dict) or str(msg.get("role", "")).strip() != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tid = str(block.get("id", "")).strip()
                if tid:
                    prefix_use_ids.add(tid)
    if not prefix_use_ids:
        return False
    suffix_result_ids: set[str] = set()
    for idx in range(cut_index, len(messages)):
        msg = messages[idx]
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).strip() == "tool_result":
            tid = str(msg.get("tool_use_id", "")).strip()
            if tid:
                suffix_result_ids.add(tid)
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = str(block.get("tool_use_id", "")).strip()
                    if tid:
                        suffix_result_ids.add(tid)
    return bool(prefix_use_ids & suffix_result_ids)


def _advance_cut_index_past_tool_result_only_users(
    messages: list[dict[str, Any]],
    cut_index: int,
) -> int | None:
    """
    Move the cut boundary to the next plain user message when a candidate lands on a
    tool_result-only user turn.

    The retained recent suffix must still begin with a user message for Anthropic
    bridge validity, so a tool_result-only candidate is only useful as a marker that
    the next plain user turn can be a safe split point.
    """
    if cut_index >= len(messages) or not isinstance(messages[cut_index], dict):
        return None
    if not _is_tool_result_only_user_message(messages[cut_index]):
        if _cut_splits_tool_pair(messages, cut_index):
            logger.info(
                "compress_history: cut at index %d rejected to prevent tool_use/tool_result split",
                cut_index,
            )
            return None
        return cut_index

    for next_index in range(cut_index + 1, len(messages)):
        next_message = messages[next_index]
        if not isinstance(next_message, dict):
            continue
        if str(next_message.get("role", "")).strip() != "user":
            continue
        if _is_tool_result_only_user_message(next_message):
            continue
        return next_index
    return None


def compress_history_impl(
    messages: list[dict[str, Any]],
    keep_turns: int = 2,
    profile: dict[str, int | list[str]] | None = None,
    prune_tool_results: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Split messages into old (to compress) + recent (to keep verbatim)."""
    if keep_turns == 0:
        old = copy.deepcopy(messages)
        recent: list[dict[str, Any]] = []
    else:
        turns_seen = 0
        cut_index = None
        rejection_counts: dict[str, int] = {}
        eligible_indices: list[int] = []
        bridge_cut_search = bool(profile and profile.get("_bridge_cut_search"))
        for i in range(len(messages)):
            cls = classify_cut_eligibility(messages[i])
            if cls.eligible:
                eligible_indices.append(i)
            elif cls.reason in _CUT_REJECTION_REASONS:
                if bridge_cut_search and cls.reason == "user_contains_tool_result_block":
                    eligible_indices.append(i)
                else:
                    rejection_counts[cls.reason] = rejection_counts.get(cls.reason, 0) + 1
        for i in reversed(eligible_indices):
            adjusted_cut_index = i if bridge_cut_search else _advance_cut_index_past_tool_result_only_users(messages, i)
            if adjusted_cut_index is None:
                continue
            if _cut_splits_tool_pair(messages, adjusted_cut_index):
                rejection_counts["tool_pair_split_prevented"] = rejection_counts.get("tool_pair_split_prevented", 0) + 1
                logger.info(
                    "compress_history: cut candidate at index %d rejected to prevent tool_use/tool_result split",
                    adjusted_cut_index,
                )
                continue
            turns_seen += 1
            if turns_seen == keep_turns:
                cut_index = adjusted_cut_index
                break

        if cut_index is None:
            user_msgs = sum(1 for m in messages if m.get("role") == "user")
            if eligible_indices == [0]:
                failure_detail = "cut_index_zero_only"
            elif not eligible_indices:
                failure_detail = "no_candidates"
            else:
                failure_detail = "insufficient_candidates"
            logger.warning(
                "compress_history: FAILED TO FIND CUT POINT "
                "detail=%s msgs=%d keep=%d user_msgs=%d "
                "eligible=%d rejections=%s",
                failure_detail,
                len(messages),
                keep_turns,
                user_msgs,
                len(eligible_indices),
                rejection_counts,
            )

            return messages, ""

        old = messages[:cut_index]
        recent = copy.deepcopy(messages[cut_index:])

    if prune_tool_results:
        for i in range(len(recent) - 1):
            msg = recent[i]
            if msg.get("role") == "tool_result":
                content = msg.get("content", "")
                if isinstance(content, str) and ("PASSED" in content or "SUCCESS" in content or "DONE" in content):
                    is_completed_scenario = False
                    for j in range(i + 1, len(recent)):
                        if recent[j].get("role") == "user":
                            is_completed_scenario = True
                            break
                    if is_completed_scenario:
                        msg["content"] = "[Pruned: SUCCESS]"

    user_turns = 0
    topic_snippets: list[str] = []
    facts: dict[str, str] = {}
    file_scores: dict[str, int] = {}
    cmd_scores: dict[str, int] = {}
    error_scores: dict[str, int] = {}
    test_scores: dict[str, int] = {}
    constraint_scores: dict[str, int] = {}
    question_scores: dict[str, int] = {}
    blocker_scores: dict[str, int] = {}
    outcome_events: list[dict[str, str | int]] = []

    def _extract_test_anchor(text: str) -> str:
        test_case = re.search(r"\b(tests?/[\w./-]+::[\w./-]+)\b", text, re.IGNORECASE)
        if test_case:
            return test_case.group(1).lower()
        test_file = re.search(r"\b(tests?/[\w./-]+\.(?:py|ts|tsx|js|jsx|go|rb|rs))\b", text, re.IGNORECASE)
        if test_file:
            return test_file.group(1).lower()
        source_file = re.search(r"\b(src/[\w./-]+\.(?:py|ts|tsx|js|jsx|go|rb|rs))\b", text, re.IGNORECASE)
        if source_file:
            return source_file.group(1).lower()
        return ""

    def _line_outcome_type(line: str) -> str:
        lowered_line = line.lower()
        if re.search(r"\b\d+\s+passed\b", lowered_line) or " passed" in lowered_line or "passed " in lowered_line:
            return "pass"
        if (
            re.search(r"\b\d+\s+failed\b", lowered_line)
            or " failed" in lowered_line
            or "failed " in lowered_line
            or "error:" in lowered_line
            or "assertionerror" in lowered_line
            or "exception" in lowered_line
            or "traceback" in lowered_line
        ):
            return "fail"
        return ""

    def _record_outcome_event(line: str, recency_score: int) -> None:
        outcome = _line_outcome_type(line)
        if not outcome:
            return
        anchor = _extract_test_anchor(line)
        if not anchor:
            return
        outcome_events.append(
            {
                "anchor": anchor,
                "outcome": outcome,
                "line": _norm(line, 96),
                "recency_score": recency_score,
            }
        )

    def _is_placeholder_fact_value(value: str) -> bool:
        lowered_value = value.lower()
        return any(
            marker in lowered_value
            for marker in (
                "<the ",
                "<the file",
                "<the command",
                "<the result",
                "<the primary",
                "<the function",
                "<the class",
                "<the specific",
            )
        )

    def _norm(value: str, max_len: int) -> str:
        s = value.strip()
        if len(s) <= max_len:
            return s

        lowered = s.lower()
        if "error" in lowered or "fail" in lowered or "parse_error" in lowered or "exception" in lowered:
            lines = s.splitlines()
            if len(lines) > 1:
                first_line = lines[0].strip()
                if len(first_line) <= max_len:
                    return first_line
                return first_line[:max_len].strip()
            for signal in [
                "parse_error",
                "error",
                "fail",
                "exception",
                "traceback",
            ]:
                idx = lowered.find(signal)
                if idx != -1:
                    start = max(0, idx - 10)
                    end = min(len(s), start + max_len)
                    return ("..." if start > 0 else "") + s[start:end].strip()

        return s[:max_len].strip()

    def _bump(bucket: dict[str, int], value: str, score: int, max_len: int) -> None:
        cleaned = _norm(value, max_len)
        if cleaned:
            bucket[cleaned] = bucket.get(cleaned, 0) + score

    def _top_items(bucket: dict[str, int], limit: int) -> list[str]:
        return [item for item, _score in sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]

    next_scores: dict[str, int] = {}

    def _extract_next(text: str, role: str) -> None:
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if role == "assistant" and any(
                lowered.startswith(prefix)
                for prefix in (
                    "next",
                    "i will",
                    "i'll",
                    "plan",
                    "then",
                    "going to",
                )
            ):
                _bump(next_scores, stripped[:48], 2, 48)

    def _extract_blockers(text: str, recency: int) -> None:
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if not stripped:
                continue
            if any(
                phrase in lowered
                for phrase in (
                    "blocked on ",
                    "blocked by ",
                    "waiting on ",
                    "can't continue",
                    "cannot continue",
                    "still failing",
                    "failing test",
                    "fails with",
                )
            ):
                _bump(error_scores, stripped[:60], 4 + recency, 60)
                if "blocked" in lowered or "waiting on" in lowered:
                    _bump(next_scores, stripped[:48], 3 + recency, 48)
                _bump(blocker_scores, stripped[:60], 4 + recency, 60)

    def _extract_questions(text: str, role: str, recency: int) -> None:
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if not stripped:
                continue
            if "?" in stripped or (role == "user" and lowered.startswith(QUESTION_PREFIXES)):
                _bump(question_scores, stripped[:60], 2 + recency, 60)

    for idx, msg in enumerate(old):
        text = text_of(msg.get("content", ""))
        role = msg.get("role", "")
        # Newer evidence should score higher than older evidence.
        recency = idx + 1

        if role == "user":
            user_turns += 1
            snippet = text.split("\n")[0][:60].strip()
            if snippet:
                topic_snippets.append(snippet)
            for line in text.splitlines():
                stripped = line.strip()
                lowered = stripped.lower()
                if any(
                    phrase in lowered
                    for phrase in (
                        "avoid ",
                        "don't ",
                        "do not ",
                        "no ",
                        "read only",
                        "read-only",
                        "without writing",
                        "no longer read only",
                        "always invert",
                    )
                ):
                    _bump(constraint_scores, stripped[:48], 3 + recency, 48)

        for match in re.finditer(
            r"(?<!\w)([\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|yml|sh|txt|css|html|sql|rs|go|rb))(?!\w)",
            text,
        ):
            _bump(file_scores, match.group(1), 2 + recency, 48)

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            _record_outcome_event(stripped, recency)
            cmd_match: re.Match[str] | None = re.search(
                r"(?:^|\s|>|#|run|exec)\s*(?:sudo\s+)?(pytest|python|python3|uv|npm|pnpm|yarn|cargo|go|git|rg|grep|sed|cat|ls|find|make|bash|sh|pip|docker|kubectl|gcloud|az|aws|gh|code|vi|vim|nano|emacs|test|build|install|update|delete|create|start|stop|restart|status|log|diff|mv|cp|rm|mkdir|rmdir|chmod|chown|pwd|cd|echo|print|export|unset|source|env|which|whereis|type|alias|unalias|history|jobs|fg|bg|kill|ps|top|htop|df|du|free|netstat|ss|curl|wget|ping|traceroute|dig|nslookup|ssh|scp|rsync|tar|zip|unzip|gzip|gunzip|bzip2|bunzip2|xz|unxz|7z|un7z|apt|yum|dnf|pacman|brew|choco|winget|snap|flatpak|gem|bundle|rake|mvn|gradle|cmake)\b",
                stripped,
            )
            if cmd_match:
                cmd_part = stripped[cmd_match.start(1) :].strip()
                _bump(cmd_scores, cmd_part[:60], 2 + recency, 60)

        content = msg.get("content")
        if role == "assistant" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_name = str(block.get("name", "")).lower()
                tool_input = block.get("input", {})
                if not isinstance(tool_input, dict):
                    tool_input = {}
                path = str(
                    tool_input.get("path")
                    or tool_input.get("file_path")
                    or tool_input.get("AbsolutePath")
                    or tool_input.get("TargetFile")
                    or ""
                ).strip()
                if path and tool_name in EDIT_LIKE_TOOLS:
                    _bump(file_scores, path, 6 + recency, 64)
                elif path and tool_name in FILE_LIKE_TOOLS:
                    _bump(file_scores, path, 3 + recency, 64)
                command = str(tool_input.get("command") or tool_input.get("cmd") or "").strip()
                if not command:
                    command = f"{tool_name} {path}".strip()
                if command:
                    _bump(cmd_scores, command[:60], 2 + recency, 60)

        lowered = text.lower()
        if any(
            token in lowered
            for token in (
                "traceback",
                "exception",
                "error:",
                "failed",
                "assertionerror",
                "enoent",
                "syntaxerror",
            )
        ):
            first = text.splitlines()[0] if text.splitlines() else text
            _bump(error_scores, first[:60], 3 + recency, 60)
        for match in re.finditer(r"\b(\d+\s+(?:passed|failed|errors?))\b", lowered):
            _bump(test_scores, match.group(1), 2 + recency, 24)
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("verification:"):
                _bump(test_scores, stripped[:96], 5 + recency, 96)
                continue
            if "FAILED" in stripped or "PASSED" in stripped:
                _bump(test_scores, stripped[:48], 2 + recency, 48)

        _extract_blockers(text, recency)
        _extract_questions(text, role, recency)
        _extract_next(text, role)

        for match in re.finditer(
            r"\b([a-zA-Z_][a-zA-Z0-9_]{1,20})\s*[:=]\s*([^\n,;|]{1,30})",
            text,
        ):
            key = match.group(1).lower()
            value = match.group(2).strip()
            if key not in STOP_WORDS and len(key) > 2 and not _is_placeholder_fact_value(value):
                facts[key] = _norm(value[:25], 25)

    _summarize_causal_failures(old, error_scores, blocker_scores)
    _summarize_decision_hypotheses(old, next_scores, question_scores)

    suppressed_failure_markers: set[str] = set()
    if outcome_events:
        latest_event_by_anchor: dict[str, dict[str, str | int]] = {}
        for event in outcome_events:
            anchor = str(event["anchor"])
            latest_event_by_anchor[anchor] = event
        for event in latest_event_by_anchor.values():
            if event.get("outcome") == "pass":
                anchor = str(event.get("anchor", "")).strip()
                if not anchor:
                    continue
                suppressed_failure_markers.add(anchor)
                anchor_basename = anchor.rsplit("/", 1)[-1]
                if anchor_basename:
                    suppressed_failure_markers.add(anchor_basename)

    profile = profile or {}

    def _get_int(d: dict[str, Any], key: str, default: int) -> int:
        val = d.get(key, default)
        if isinstance(val, int):
            return val
        return default

    state_map: dict[str, str] = {"turns": str(user_turns)}
    if topic_snippets:
        goal = _norm(topic_snippets[-1][:36], 36)
        state_map["goal"] = goal
    top_files = _top_items(file_scores, _get_int(profile, "files", 3))
    top_cmds = _top_items(cmd_scores, _get_int(profile, "cmds", 2))
    top_tests = _top_items(test_scores, _get_int(profile, "tests", 2))
    top_errors = _top_items(error_scores, _get_int(profile, "errs", 2))

    def _mentions_suppressed_failure(item: str) -> bool:
        lowered_item = item.lower()
        if not any(fail_word in lowered_item for fail_word in ("failed", "error", "exception", "traceback")):
            return False
        return any(marker and marker in lowered_item for marker in suppressed_failure_markers)

    if suppressed_failure_markers:
        top_tests = [item for item in top_tests if not _mentions_suppressed_failure(item)]
        top_errors = [item for item in top_errors if not _mentions_suppressed_failure(item)]

    top_constraints = _top_items(constraint_scores, _get_int(profile, "constraints", 2))
    top_questions = _top_items(question_scores, _get_int(profile, "questions", 2))
    top_blockers = _top_items(blocker_scores, _get_int(profile, "blockers", 2))
    top_next = _top_items(next_scores, _get_int(profile, "next", 2))
    if top_files:
        state_map["files"] = ",".join(top_files)
    if top_cmds:
        state_map["cmds"] = ",".join(top_cmds)
    if top_tests:
        state_map["tests"] = ",".join(top_tests)
    if top_errors:
        state_map["errs"] = ",".join(top_errors)
    if top_constraints:
        state_map["constraints"] = ",".join(top_constraints)
    if top_questions:
        state_map["questions"] = ",".join(top_questions)
    if top_blockers:
        state_map["blockers"] = ",".join(top_blockers)
    if top_next:
        state_map["next"] = ",".join(top_next)
    if facts:
        compact_facts = [f"{key}:{value}" for key, value in sorted(facts.items())[: _get_int(profile, "facts", 3)]]
        if compact_facts:
            state_map["facts"] = ",".join(compact_facts)

    ordered_keys = [
        "turns",
        "goal",
        "files",
        "cmds",
        "tests",
        "errs",
        "constraints",
        "questions",
        "blockers",
        "next",
        "facts",
    ]
    payload = "|".join(f"{key}:{state_map[key]}" for key in ordered_keys if key in state_map)
    return recent, f">>> {payload}" if payload else ""


def tok_tool_result_impl(
    content: str,
    compression_level: str = "balanced",
    tool_context: dict[str, Any] | None = None,
    session: Any | None = None,
) -> str:
    return _tok_tool_result_impl(
        content,
        tool_compress_threshold=TOOL_COMPRESS_THRESHOLD,
        compression_level=compression_level,
        tool_context=tool_context,
        session=session,
    )


def compress_tool_results_impl(
    messages: list[dict[str, Any]],
    result_cache: MutableMapping[str, dict[str, object] | tuple[str, str, float] | tuple[str, str] | tuple[str]]
    | None = None,
    tool_use_id_to_context: dict[str, dict[str, Any]] | None = None,
    compression_level: str = "balanced",
    semantic_hash_cache: dict[str, str] | None = None,
    bypass_result_cache: bool = False,
    hot_summary_records: dict[str, Any] | None = None,
    session_files_read: set[str] | None = None,
    files_fully_delivered: dict[str, int] | None = None,
    first_exact_evidence_seen: set[str] | None = None,
    current_turn: int | None = None,
    keep_turns_window: int | None = None,
    preserve_exact_search_evidence: bool = False,
    recently_edited_files: dict[str, int] | None = None,
    file_heat: dict[str, float] | None = None,
    session: Any | None = None,
    model_profile: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    breakdown: dict[str, int] = {}
    precision_ranges_by_path: dict[str, list[tuple[int, int]]] = {}
    last_full_file_by_path: dict[str, str] = {}
    search_seen_matches: dict[str, set[str]] = {}
    stack_prev_by_signature: dict[str, str] = {}
    feature_telemetry: dict[str, dict[str, int]] = {}
    _skip_stable_result = model_profile is not None and not getattr(model_profile, "stable_result_enabled", True)
    _skip_file_skeleton = model_profile is not None and not getattr(model_profile, "skeletonize_files", True)

    def _record_feature_telemetry(
        feature: str,
        outcome: str,
        chars_saved: int = 0,
    ) -> None:
        metrics = feature_telemetry.setdefault(
            feature,
            {"attempted": 0, "applied": 0, "skipped": 0, "fallback": 0, "chars_saved": 0},
        )
        if outcome == "attempted":
            metrics["attempted"] += 1
        else:
            if outcome not in metrics:
                metrics[outcome] = 0
            metrics[outcome] += 1
        if chars_saved > 0:
            metrics["chars_saved"] += chars_saved

    def _extract_precision_range(
        context: dict[str, Any] | None,
        raw: str,
    ) -> tuple[str, int, int, list[str]] | None:
        if not _is_precision_read_context(context):
            return None
        if not isinstance(context, dict):
            return None
        args = context.get("args")
        if not isinstance(args, dict):
            return None
        norm_path = _extract_normalized_path(context)
        if not norm_path:
            return None
        lines = raw.splitlines()
        if not lines:
            return None
        start_raw = args.get("offset", args.get("start"))
        if start_raw is None:
            return None
        try:
            start = int(start_raw)
        except (TypeError, ValueError):
            return None
        if start < 0:
            return None
        end: int | None = None
        if args.get("end") is not None:
            try:
                end = int(args["end"])
            except (TypeError, ValueError):
                end = None
        if end is None and args.get("limit") is not None:
            try:
                end = start + int(args["limit"])
            except (TypeError, ValueError):
                end = None
        if end is None:
            end = start + len(lines)
        if end < start:
            return None
        expected_len = end - start
        if expected_len <= 0:
            expected_len = len(lines)
            end = start + expected_len
        # Keep alignment stable with the actual payload length.
        if len(lines) != expected_len:
            end = start + len(lines)
        return norm_path, start, end, lines

    def _mark_precision_range(path: str, start: int, end: int) -> None:
        if start >= end:
            return
        ranges = precision_ranges_by_path.setdefault(path, [])
        merged_start = start
        merged_end = end
        remaining: list[tuple[int, int]] = []
        for existing_start, existing_end in ranges:
            if existing_end < merged_start or existing_start > merged_end:
                remaining.append((existing_start, existing_end))
                continue
            merged_start = min(merged_start, existing_start)
            merged_end = max(merged_end, existing_end)
        remaining.append((merged_start, merged_end))
        remaining.sort(key=lambda pair: pair[0])
        precision_ranges_by_path[path] = remaining

    def _is_index_covered(path: str, index: int) -> bool:
        for range_start, range_end in precision_ranges_by_path.get(path, []):
            if range_start <= index < range_end:
                return True
        return False

    def _build_precision_overlap_delta(
        path: str,
        start: int,
        end: int,
        lines: list[str],
    ) -> str | None:
        unseen: list[str] = []
        overlap_count = 0
        for idx, line in enumerate(lines):
            abs_index = start + idx
            if _is_index_covered(path, abs_index):
                overlap_count += 1
                continue
            unseen.append(f"{abs_index + 1}: {line}")
        if overlap_count == 0:
            return None
        header = (
            f">>> tool:file_read_overlap_delta|path:{path}|range:{start + 1}-{end}"
            f"|new_lines:{len(unseen)}|overlap_lines:{overlap_count}"
        )
        body = "\n".join(unseen) if unseen else "no new lines (all overlap with prior precision reads)"
        return header + "\n" + body

    def _count_changed_diff_lines(diff_lines: list[str]) -> int:
        return sum(1 for line in diff_lines if line.startswith("+") or line.startswith("-")) - sum(
            1 for line in diff_lines if line.startswith("+++") or line.startswith("---")
        )

    def _build_file_reread_diff(path: str, previous: str, current: str) -> str | None:
        if previous == current:
            return None
        diff_lines = list(
            difflib.unified_diff(
                previous.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
                lineterm="",
            )
        )
        if not diff_lines:
            return None
        changed_lines = max(1, _count_changed_diff_lines(diff_lines))
        diff_text = "".join(diff_lines)
        return f">>> tool:file_reread_diff|path:{path}|changed_lines:{changed_lines}\n{diff_text}"

    def _extract_stack_frames(trace_text: str) -> tuple[list[str], str]:
        lines = trace_text.splitlines()
        frames: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('File "') or stripped.startswith("at "):
                frames.append(stripped)
        exception_line = ""
        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                exception_line = stripped
                break
        return frames, exception_line

    def _is_file_fully_delivered(norm_path: str) -> bool:
        if recently_edited_files and norm_path in recently_edited_files:
            edit_step = recently_edited_files[norm_path]
            if current_turn is not None and (current_turn - edit_step) < 2:
                return False
        if files_fully_delivered is None:
            return False  # No tracking = not delivered; protects cold reads
        delivery_turn = files_fully_delivered.get(norm_path)
        if delivery_turn is None:
            return False
        if current_turn is None or keep_turns_window is None:
            return True
        return (current_turn - delivery_turn) < keep_turns_window

    def _mark_file_fully_delivered(norm_path: str) -> None:
        if files_fully_delivered is not None and norm_path:
            files_fully_delivered[norm_path] = current_turn or 0

    def _should_bypass_cache(context: dict[str, Any] | None) -> bool:
        """Check if tok_bypass_cache flag is set in tool context args."""
        if not context:
            return False
        args = context.get("args")
        if isinstance(args, dict):
            return bool(args.get("tok_bypass_cache"))
        return False

    def _is_zero_heat(context: dict[str, Any] | None) -> bool:
        """Check if file has zero heat (never been read before in this session)."""
        if not file_heat or not context:
            return False
        args = context.get("args") if isinstance(context, dict) else None
        if not isinstance(args, dict):
            return False
        path = str(
            args.get("path") or args.get("file_path") or args.get("AbsolutePath") or args.get("TargetFile") or ""
        )
        if not path:
            return False
        # Normalize path for lookup
        norm_path = path.lower().strip()
        if session_files_read is not None and norm_path in session_files_read:
            return False
        heat = file_heat.get(norm_path, 0.0)
        return heat == 0.0

    def _cache_semantic_hash(
        context: dict[str, Any] | None,
        raw: str,
        cache: dict[str, str],
    ) -> None:
        cache_key = _make_semantic_cache_key(context, raw)
        if cache_key is None:
            return
        args = context.get("args") if isinstance(context, dict) else None
        if isinstance(args, dict) and any(k in args for k in ("offset", "limit", "start", "end")):
            return
        cache[cache_key] = _compute_semantic_hash(raw)

    def _is_precision_read_context(context: dict[str, Any] | None) -> bool:
        if not context:
            return False
        tool_name = str(context.get("name", "")).lower()
        if tool_name not in FILE_LIKE_TOOLS:
            return False
        args = context.get("args")
        if not isinstance(args, dict):
            return False
        return any(k in args for k in ("offset", "limit", "start", "end"))

    def _first_exact_guard(context: dict[str, Any] | None, raw: str) -> bool:
        """Guard first exact observation from compression.

        Preserves content if:
        1. Never seen before in conversation (not in first_exact_evidence_seen), OR
        2. First read in current session (not in session_files_read)
        """
        if not context:
            return False
        tool_name = str(context.get("name", "")).lower()
        if tool_name in SEARCH_LIKE_TOOLS and search_result_evidence_level(raw) == "navigation":
            return False
        key = evidence_identity_key(
            str(context.get("name", "")),
            path=_extract_normalized_path(context),
            query=str(context.get("query") or "").strip() or None,
            command=str(
                (context.get("args") or {}).get("command") or (context.get("args") or {}).get("cmd") or ""
            ).strip()
            or None,
            args=context.get("args") if isinstance(context.get("args"), dict) else None,
        )
        if not key:
            return False

        # Check if first time ever in conversation
        is_first_ever = first_exact_evidence_seen is not None and key not in first_exact_evidence_seen

        # Check if first time in current session
        norm_path = _extract_normalized_path(context)
        is_first_session = session_files_read is not None and norm_path and norm_path not in session_files_read

        # Preserve if either first ever OR first in this session
        if is_first_ever or is_first_session:
            if is_first_ever and first_exact_evidence_seen is not None:
                first_exact_evidence_seen.add(key)
            return True

        return False

    def _truncate_stable_snippet(text: str, limit: int) -> str:
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        if limit <= 1:
            return cleaned[:limit]
        return cleaned[: limit - 1].rstrip() + "…"

    def _preserve_first_exact_observation(
        context: dict[str, Any] | None,
        raw: str,
        norm_path: str = "",
    ) -> bool:
        if not _first_exact_guard(context, raw):
            return False
        if context:
            tool_name = str(context.get("name", "")).lower()
            if tool_name in FILE_LIKE_TOOLS:
                args = context.get("args")
                if isinstance(args, dict) and any(k in args for k in ("offset", "limit", "start", "end")):
                    return False
        if session_files_read is not None and norm_path and norm_path not in session_files_read:
            session_files_read.add(norm_path)
            if semantic_hash_cache is not None and len(raw) >= _SEMANTIC_HASH_MIN_CHARS:
                _cache_semantic_hash(context, raw, semantic_hash_cache)
        if norm_path:
            _mark_file_fully_delivered(norm_path)
        return True

    def _should_preserve_exact_search_observation(
        context: dict[str, Any] | None,
        raw: str,
    ) -> bool:
        if not preserve_exact_search_evidence or not context:
            return False
        tool_name = str(context.get("name", "")).lower()
        evidence_level = search_result_evidence_level(raw)
        if tool_name not in SEARCH_LIKE_TOOLS or evidence_level != "exact_content":
            return False
        # Only preserve if this is the FIRST observation (key not yet seen)
        # Repeats should be compressed, not preserved
        key = evidence_identity_key(
            tool_name,
            path=_extract_normalized_path(context),
            query=str(context.get("query") or "").strip() or None,
            args=context.get("args") if isinstance(context.get("args"), dict) else None,
        )
        if key and first_exact_evidence_seen is not None and key in first_exact_evidence_seen:
            # This is a repeat - don't preserve, let compression happen
            return False
        # First observation - preserve and track
        if key and first_exact_evidence_seen is not None:
            first_exact_evidence_seen.add(key)
        return True

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            if (
                msg.get("role") == "tool_result"
                and isinstance(content, str)
                and result_cache is not None
                and tool_use_id_to_context is not None
            ):
                tool_id = msg.get("tool_use_id", "")
                context = tool_use_id_to_context.get(tool_id)
                if context:
                    tool_name = str(context.get("name", "")).lower()
                    is_file_like_tool = tool_name in FILE_LIKE_TOOLS
                    norm_path = _extract_normalized_path(context)
                    # First: check if this is a first exact observation (must be before last_file_read_ids check)
                    if _preserve_first_exact_observation(context, content, norm_path):
                        msg["content"] = content
                        continue
                    if _should_preserve_exact_search_observation(context, content):
                        msg["content"] = content
                        continue
                    # Check for explicit bypass flag
                    if _should_bypass_cache(context):
                        breakdown["bypass_reacquire"] = breakdown.get("bypass_reacquire", 0) + 1
                        msg["content"] = content
                        continue
                # Remaining processing for non-bypassed and non-first-exact content.
                if context:
                    norm_path = _extract_normalized_path(context)
                    if (
                        is_file_like_tool
                        and session_files_read is not None
                        and norm_path
                        and norm_path not in session_files_read
                    ):
                        session_files_read.add(norm_path)
                        if semantic_hash_cache is not None and len(content) >= _SEMANTIC_HASH_MIN_CHARS:
                            _cache_semantic_hash(context, content, semantic_hash_cache)
                        _mark_file_fully_delivered(norm_path)
                        continue
                    compressed, saved = _apply_result_cache(
                        content,
                        context,
                        result_cache,
                        compression_level=compression_level,
                        bypass_cache=bypass_result_cache,
                        ttl_seconds=RESULT_CACHE_TTL_SECONDS,
                        preserve_exact_search_evidence=preserve_exact_search_evidence,
                    )
                    if "stable_payload_validation_failed" in compressed:
                        breakdown["stable_payload_validation_failed"] = (
                            breakdown.get("stable_payload_validation_failed", 0) + 1
                        )
                    if saved > 0:
                        kind = _detect_tool_content_type_impl(content)
                        key = f"{kind}_cached" if "|unchanged|" in compressed else f"{kind}_diff"
                        breakdown[key] = breakdown.get(key, 0) + saved
                    msg["content"] = compressed
            continue
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue

            raw = block.get("content", "")
            if not isinstance(raw, str):
                continue

            tool_id = block.get("tool_use_id", "")
            ctx: dict[str, Any] | None = None
            if tool_use_id_to_context is not None:
                ctx = tool_use_id_to_context.get(tool_id)

            if ctx:
                norm_path = _extract_normalized_path(ctx)
                if _should_preserve_exact_search_observation(ctx, raw):
                    block["content"] = raw
                    continue
                if _preserve_first_exact_observation(ctx, raw, norm_path):
                    block["content"] = raw
                    continue
                # Check for explicit bypass flag
                if _should_bypass_cache(ctx):
                    breakdown["bypass_reacquire"] = breakdown.get("bypass_reacquire", 0) + 1
                    block["content"] = raw
                    continue
            else:
                norm_path = ""

            # Search-specific repeat compression path
            # Search tools use query+scope identity, not norm_path
            tool_name = str(ctx.get("name", "")).lower() if ctx else ""
            if (
                TOK_ENABLE_SEARCH_OVERLAP_DELTA
                and ctx
                and tool_name in SEARCH_LIKE_TOOLS
                and not preserve_exact_search_evidence
                and search_result_evidence_level(raw) != "navigation"
            ):
                search_scope = str(ctx.get("path") or "").strip().lower()
                search_key = f"{tool_name}|{search_scope or 'global'}"
                current_lines = [line for line in raw.splitlines() if line.strip()]
                if current_lines:
                    previous_lines = search_seen_matches.get(search_key)
                    if previous_lines is None:
                        search_seen_matches[search_key] = set(current_lines)
                    else:
                        _record_feature_telemetry("search_overlap_delta", "attempted")
                        new_lines = [line for line in current_lines if line not in previous_lines]
                        omitted_count = len(current_lines) - len(new_lines)
                        candidate = (
                            f">>> tool:search_overlap_delta|scope:{search_scope or 'global'}"
                            f"|new_matches:{len(new_lines)}|omitted_seen:{max(0, omitted_count)}\n"
                        )
                        candidate += "\n".join(new_lines) if new_lines else "no new matches"
                        saved = len(raw) - len(candidate)
                        if saved > 0:
                            breakdown["search_overlap_delta"] = breakdown.get("search_overlap_delta", 0) + saved
                            block["content"] = candidate
                            _record_feature_telemetry("search_overlap_delta", "applied", chars_saved=saved)
                            previous_lines.update(current_lines)
                            continue
                        _record_feature_telemetry("search_overlap_delta", "skipped")
                        previous_lines.update(current_lines)
            if ctx and tool_name in SEARCH_LIKE_TOOLS and first_exact_evidence_seen is not None:
                search_key = evidence_identity_key(
                    tool_name,
                    path=str(ctx.get("path") or "").strip() or None,
                    query=str(ctx.get("query") or "").strip() or None,
                    args=ctx.get("args") if isinstance(ctx.get("args"), dict) else None,
                )
                # Only compress if this is a repeat (key already seen)
                if search_key and search_key in first_exact_evidence_seen:
                    # Skip navigation-only results - they stay raw
                    if search_result_evidence_level(raw) == "navigation":
                        block["content"] = raw
                        continue
                    # Apply result cache compression for repeat search
                    if result_cache is not None and not bypass_result_cache:
                        compressed, saved = _apply_result_cache(
                            raw,
                            ctx,
                            result_cache,
                            compression_level=compression_level,
                            bypass_cache=bypass_result_cache,
                            ttl_seconds=RESULT_CACHE_TTL_SECONDS,
                            preserve_exact_search_evidence=preserve_exact_search_evidence,
                        )
                        if saved > 0:
                            breakdown["search_repeat_cached"] = breakdown.get("search_repeat_cached", 0) + saved
                        block["content"] = compressed
                        continue
                    # Apply semantic hash compression for repeat search
                    if (
                        not _skip_stable_result
                        and semantic_hash_cache is not None
                        and len(raw) >= _SEMANTIC_HASH_MIN_CHARS
                    ):
                        cache_key = _make_semantic_cache_key(ctx, raw)
                        if cache_key is not None:
                            content_hash = _compute_semantic_hash(raw)
                            prev_hash = semantic_hash_cache.get(cache_key)
                            if prev_hash == content_hash:
                                # Build compressed representation
                                summary = ""
                                if hot_summary_records is not None:
                                    record_key = f"search|{search_key}"
                                    record = hot_summary_records.get(record_key)
                                    if record and hasattr(record, "summary"):
                                        summary = record.summary
                                if not summary:
                                    from tok.runtime.repeat_targets import (
                                        build_search_summary,
                                    )

                                    with contextlib.suppress(Exception):
                                        summary = build_search_summary(
                                            raw,
                                            max_chars=280,
                                            max_lines=12,
                                        )
                                if not summary:
                                    summary = _truncate_stable_snippet(raw, 280)
                                token = f"@stable_result(hash:{content_hash})\n@stable_summary |> {summary}"
                                saved = len(raw) - len(token)
                                if saved > 0:
                                    breakdown["search_repeat_dedup"] = breakdown.get("search_repeat_dedup", 0) + saved
                                    block["content"] = token
                                    continue
                            else:
                                semantic_hash_cache[cache_key] = content_hash

            if _is_precision_read_context(ctx):
                if TOK_ENABLE_FILE_OVERLAP_DELTA:
                    precision_window = _extract_precision_range(ctx, raw)
                    if precision_window is not None:
                        path, start, end, raw_lines = precision_window
                        _record_feature_telemetry("file_overlap_delta", "attempted")
                        overlap_candidate = _build_precision_overlap_delta(path, start, end, raw_lines)
                        if overlap_candidate:
                            overlap_saved = len(raw) - len(overlap_candidate)
                            if overlap_saved > 0:
                                block["content"] = overlap_candidate
                                breakdown["file_overlap_delta"] = breakdown.get("file_overlap_delta", 0) + overlap_saved
                                _mark_precision_range(path, start, end)
                                _mark_file_fully_delivered(path)
                                _record_feature_telemetry("file_overlap_delta", "applied", chars_saved=overlap_saved)
                                continue
                            _record_feature_telemetry("file_overlap_delta", "fallback")
                        else:
                            _record_feature_telemetry("file_overlap_delta", "skipped")
                        _mark_precision_range(path, start, end)
                norm_path = _extract_normalized_path(ctx) if ctx else ""
                if session_files_read is not None and norm_path and norm_path not in session_files_read:
                    session_files_read.add(norm_path)
                    if semantic_hash_cache is not None and len(raw) >= _SEMANTIC_HASH_MIN_CHARS:
                        _cache_semantic_hash(ctx, raw, semantic_hash_cache)
                if result_cache is not None and not bypass_result_cache:
                    if ctx is None:
                        continue
                    compressed, _saved = _apply_result_cache(
                        raw,
                        ctx,
                        result_cache,
                        compression_level=compression_level,
                        bypass_cache=bypass_result_cache,
                        ttl_seconds=RESULT_CACHE_TTL_SECONDS,
                        preserve_exact_search_evidence=preserve_exact_search_evidence,
                    )
                    if "stable_payload_validation_failed" in compressed:
                        breakdown["stable_payload_validation_failed"] = (
                            breakdown.get("stable_payload_validation_failed", 0) + 1
                        )
                    block["content"] = compressed
                else:
                    block["content"] = raw
                _mark_file_fully_delivered(norm_path)
                continue

            if (
                TOK_ENABLE_FILE_REREAD_DIFF
                and ctx
                and tool_name in FILE_LIKE_TOOLS
                and norm_path
                and not _is_precision_read_context(ctx)
            ):
                previous_full = last_full_file_by_path.get(norm_path)
                if previous_full:
                    _record_feature_telemetry("file_reread_diff", "attempted")
                    reread_candidate = _build_file_reread_diff(norm_path, previous_full, raw)
                    if reread_candidate is not None:
                        reread_saved = len(raw) - len(reread_candidate)
                        if reread_saved > 0:
                            breakdown["file_reread_diff"] = breakdown.get("file_reread_diff", 0) + reread_saved
                            block["content"] = reread_candidate
                            last_full_file_by_path[norm_path] = raw
                            _mark_file_fully_delivered(norm_path)
                            _record_feature_telemetry("file_reread_diff", "applied", chars_saved=reread_saved)
                            continue
                        _record_feature_telemetry("file_reread_diff", "fallback")
                    else:
                        _record_feature_telemetry("file_reread_diff", "skipped")
                last_full_file_by_path[norm_path] = raw

            norm_path = _extract_normalized_path(ctx) if ctx else ""
            # First-session preservation: None means fresh session, preserve all file reads
            if (
                tool_name in FILE_LIKE_TOOLS
                and session_files_read is not None
                and norm_path
                and norm_path not in session_files_read
            ):
                if session_files_read is not None:
                    session_files_read.add(norm_path)
                    if semantic_hash_cache is not None and len(raw) >= _SEMANTIC_HASH_MIN_CHARS:
                        _cache_semantic_hash(ctx, raw, semantic_hash_cache)
                _mark_file_fully_delivered(norm_path)
                continue

            # Only apply semantic dedup if this is a repeat read in the current session
            # First reads must be preserved verbatim (not compressed)
            if (
                not _skip_stable_result
                and tool_name in FILE_LIKE_TOOLS
                and _is_file_fully_delivered(norm_path)
                and (session_files_read is None or norm_path in session_files_read)
                and semantic_hash_cache is not None
                and len(raw) >= _SEMANTIC_HASH_MIN_CHARS
                and tool_use_id_to_context is not None
            ):
                # Zero-heat check: never compress files that haven't been read before
                if tool_name in FILE_LIKE_TOOLS and _is_zero_heat(ctx):
                    block["content"] = raw
                    continue
                cache_key = _make_semantic_cache_key(ctx, raw)
                ctx_args = ctx.get("args") if isinstance(ctx, dict) else None
                if isinstance(ctx_args, dict) and any(k in ctx_args for k in ("offset", "limit", "start", "end")):
                    cache_key = None
                if cache_key is not None:
                    content_hash = _compute_semantic_hash(raw)
                    prev_hash = semantic_hash_cache.get(cache_key)
                    if prev_hash == content_hash:
                        if _skip_file_skeleton:
                            summary = ""
                            if hot_summary_records is not None:
                                record_key = f"file_read|{norm_path}"
                                record = hot_summary_records.get(record_key)
                                if record and hasattr(record, "summary"):
                                    summary = record.summary
                            if summary:
                                token = f"@stable_result(hash:{content_hash})\n@stable_summary |> {_truncate_stable_snippet(summary, 280)}"
                                saved = len(raw) - len(token)
                                if saved > 0:
                                    breakdown["semantic_dedup"] = breakdown.get("semantic_dedup", 0) + max(0, saved)
                                    log_semantic_dedup(cache_key, saved)
                                    block["content"] = token
                                    continue
                            continue
                        compressed = _compress_file_read(raw, tool_context=ctx, session=session)
                        if len(compressed) < len(raw):
                            saved = len(raw) - len(compressed)
                            breakdown["semantic_dedup"] = breakdown.get("semantic_dedup", 0) + max(0, saved)
                            log_semantic_dedup(cache_key, saved)
                            block["content"] = compressed
                            continue
                        # Small files return unchanged — preserve verbatim, don't build skeleton
                        if compressed == raw:
                            continue
                        summary = ""
                        if hot_summary_records is not None and ctx is not None:
                            path = ctx.get("path")
                            if path:
                                normalized_path = normalize_path_target(path)
                                record_key = f"file_read|{normalized_path}"
                                record = hot_summary_records.get(record_key)
                                if record and hasattr(record, "summary"):
                                    summary = record.summary
                        if not summary and ctx is not None:
                            path = ctx.get("path")
                            if path and len(raw) >= 100:
                                with contextlib.suppress(Exception):
                                    summary = build_file_summary(
                                        raw,
                                        max_chars=280,
                                        max_lines=12,
                                    )
                        if not summary:
                            summary = _truncate_stable_snippet(raw, 280)
                        skeleton = ""
                        if ctx is not None:
                            path = ctx.get("path")
                            if path and len(raw) >= 100:
                                with contextlib.suppress(Exception):
                                    skeleton = build_file_skeleton(
                                        raw,
                                        max_chars=280,
                                        max_lines=14,
                                    )
                        lines = [f"@stable_result(hash:{content_hash})"]
                        if summary:
                            lines.append(f"@stable_summary |> {_truncate_stable_snippet(summary, 280)}")
                        if skeleton:
                            lines.append(f"@stable_skeleton |> {_truncate_stable_snippet(skeleton, 280)}")
                        token = "\n".join(lines)
                        saved = len(raw) - len(token)
                        if saved > 0:
                            breakdown["semantic_dedup"] = breakdown.get("semantic_dedup", 0) + max(0, saved)
                            log_semantic_dedup(cache_key, saved)
                            block["content"] = token
                            continue
                    else:
                        semantic_hash_cache[cache_key] = content_hash

            if (
                tool_name in FILE_LIKE_TOOLS
                and result_cache is not None
                and tool_use_id_to_context is not None
                and not bypass_result_cache
            ):
                context = ctx
                if context:
                    if _preserve_first_exact_observation(context, raw, norm_path):
                        block["content"] = raw
                        continue
                    compressed, saved = _apply_result_cache(
                        raw,
                        context,
                        result_cache,
                        compression_level=compression_level,
                        bypass_cache=bypass_result_cache,
                        ttl_seconds=RESULT_CACHE_TTL_SECONDS,
                        preserve_exact_search_evidence=preserve_exact_search_evidence,
                    )
                    if "stable_payload_validation_failed" in compressed:
                        breakdown["stable_payload_validation_failed"] = (
                            breakdown.get("stable_payload_validation_failed", 0) + 1
                        )
                    if saved > 0:
                        kind = _detect_tool_content_type_impl(raw)
                        key = f"{kind}_cached" if "|unchanged|" in compressed else f"{kind}_diff"
                        breakdown[key] = breakdown.get(key, 0) + saved
                    block["content"] = compressed
                    continue

            # Zero-heat check: never compress files that haven't been read before
            if tool_name in FILE_LIKE_TOOLS and _is_zero_heat(ctx):
                block["content"] = raw
                continue

            compressed = tok_tool_result_impl(
                raw,
                compression_level=compression_level,
                tool_context=ctx,
                session=session,
            )
            if TOK_ENABLE_STACK_REPEAT_DELTA:
                kind = _detect_tool_content_type_impl(raw)
                if kind == "stack_trace":
                    frames, exception_line = _extract_stack_frames(raw)
                    signature = exception_line or "unknown_exception"
                    previous_trace = stack_prev_by_signature.get(signature)
                    if previous_trace:
                        _record_feature_telemetry("stack_repeat_delta", "attempted")
                        previous_frames, _prev_exception = _extract_stack_frames(previous_trace)
                        if (
                            len(frames) >= 2
                            and len(previous_frames) >= 2
                            and frames[1:] == previous_frames[1:]
                            and frames[0] != previous_frames[0]
                        ):
                            candidate = (
                                ">>> tool:stack_trace_delta|baseline:previous|changed_top_frames:1\n"
                                f"{frames[0]}\n{exception_line}"
                            )
                            saved = len(raw) - len(candidate)
                            if saved > 0:
                                breakdown["stack_repeat_delta"] = breakdown.get("stack_repeat_delta", 0) + saved
                                block["content"] = candidate
                                stack_prev_by_signature[signature] = raw
                                _record_feature_telemetry("stack_repeat_delta", "applied", chars_saved=saved)
                                continue
                            _record_feature_telemetry("stack_repeat_delta", "fallback")
                        else:
                            _record_feature_telemetry("stack_repeat_delta", "skipped")
                    stack_prev_by_signature[signature] = raw
            saved = len(raw) - len(compressed)
            if saved > 0:
                kind = _detect_tool_content_type_impl(raw)
                breakdown[kind] = breakdown.get(kind, 0) + saved
                block["content"] = compressed
    if feature_telemetry:
        logger.debug("compression_feature_telemetry=%s", feature_telemetry)
    return messages, breakdown


def inject_system_additions_impl(
    body: dict[str, Any],
    tok_state: str | None = None,
    tool_compatible: bool = False,
    grammar: str | None = None,
    todo: str | None = None,
    deltas: str | None = None,
    pressure: int = 0,
    runtime_hints: list[str] | None = None,
    behavior_signals: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Inject dynamic state into system prompt."""
    output_directive = ""
    if runtime_hints:
        output_directive = "\n".join(str(hint).strip() for hint in runtime_hints if str(hint).strip())

    include_tok_state = _should_include_tok_state(tok_state, tool_compatible=tool_compatible)

    if tool_compatible and not include_tok_state and not grammar and not todo and not deltas:
        current_sys_prompt = body.get("system", "")
        if isinstance(current_sys_prompt, str):
            body["system"] = current_sys_prompt + "\n\n" + output_directive if current_sys_prompt else output_directive
        elif isinstance(current_sys_prompt, list):
            body["system"] = [
                *current_sys_prompt,
                {"type": "text", "text": output_directive},
            ]
        else:
            body["system"] = output_directive
        return body

    dynamic_blocks = []
    if grammar:
        dynamic_blocks.append(f"@grammar\n{grammar}")
    if include_tok_state:
        if not tool_compatible and (grammar or deltas or todo):
            dynamic_blocks.append(f"@state\n{tok_state}")
        else:
            dynamic_blocks.append(f">>>\n{tok_state}")
    if deltas:
        dynamic_blocks.append(f"@delta\n{deltas}")
    if todo:
        dynamic_blocks.append(f"@todo\n{todo}")

    dynamic_state = "\n\n".join(dynamic_blocks)
    current_sys_prompt = body.get("system", "")
    if isinstance(current_sys_prompt, str):
        additions = [output_directive]
        if dynamic_state:
            additions.append(dynamic_state)
        addition = "\n\n".join(additions)
        body["system"] = current_sys_prompt + "\n\n" + addition if current_sys_prompt else addition
    elif isinstance(current_sys_prompt, list):
        new_blocks = [*current_sys_prompt]
        if output_directive.strip():
            new_blocks.append({"type": "text", "text": output_directive})
        if dynamic_state:
            new_blocks.append({"type": "text", "text": dynamic_state})
        body["system"] = new_blocks
    else:
        additions = [output_directive]
        if dynamic_state:
            additions.append(dynamic_state)
        body["system"] = "\n\n".join(additions)

    logger.debug(
        "System prompt injected (tok_state=%s, orchestrator=%s)",
        "yes" if include_tok_state else "no",
        "yes" if grammar or todo or deltas else "no",
    )
    return body


def compress_recent_window_impl(
    messages: list[dict[str, Any]],
    tool_use_id_to_context: dict[str, dict[str, Any]] | None = None,
    threshold: int = RECENT_WINDOW_THRESHOLD,
    tool_compatible: bool = False,
    first_exact_evidence_seen: set[str] | None = None,
    preserve_exact_search_evidence: bool = False,
    session_files_read: set[str] | None = None,
    file_heat: dict[str, float] | None = None,
    model_profile: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply content-aware compression to recent window messages."""

    def _is_precision_read_context(context: dict[str, Any] | None) -> bool:
        if not context:
            return False
        tool_name = str(context.get("name", "")).lower()
        if tool_name not in FILE_LIKE_TOOLS:
            return False
        args = context.get("args")
        if not isinstance(args, dict):
            return False
        return any(k in args for k in ("offset", "limit", "start", "end"))

    def _preserve_first_exact_observation(
        context: dict[str, Any] | None,
    ) -> bool:
        """Preserve first exact observation and track it in session."""
        if first_exact_evidence_seen is None or not context:
            return False
        key = evidence_identity_key(
            str(context.get("name", "")),
            path=str(context.get("path") or "").strip() or None,
            query=str(context.get("query") or "").strip() or None,
            command=str(
                (context.get("args") or {}).get("command") or (context.get("args") or {}).get("cmd") or ""
            ).strip()
            or None,
            args=context.get("args") if isinstance(context.get("args"), dict) else None,
        )
        if not key or key in first_exact_evidence_seen:
            return False
        first_exact_evidence_seen.add(key)
        return True

    def _first_exact_guard(
        context: dict[str, Any] | None,
        _raw: str,
    ) -> bool:
        """Guard first exact observation from compression.

        Preserves content if:
        1. Never seen before in conversation (not in first_exact_evidence_seen), OR
        2. First read in current session (not in session_files_read)
        """
        if not context:
            return False
        key = evidence_identity_key(
            str(context.get("name", "")),
            path=str(context.get("path") or "").strip() or None,
            query=str(context.get("query") or "").strip() or None,
            command=str(
                (context.get("args") or {}).get("command") or (context.get("args") or {}).get("cmd") or ""
            ).strip()
            or None,
            args=context.get("args") if isinstance(context.get("args"), dict) else None,
        )
        if not key:
            return False

        # Check if first time ever in conversation
        is_first_ever = first_exact_evidence_seen is not None and key not in first_exact_evidence_seen

        # Check if first time in current session
        norm_path = _extract_normalized_path(context)
        is_first_session = session_files_read is not None and norm_path and norm_path not in session_files_read

        # Preserve if either first ever OR first in this session
        if is_first_ever or is_first_session:
            if is_first_ever and first_exact_evidence_seen is not None:
                first_exact_evidence_seen.add(key)
            return True

        return False

    def _should_preserve_exact_search_observation(
        context: dict[str, Any] | None,
        raw: str,
    ) -> bool:
        if not preserve_exact_search_evidence or not context:
            return False
        tool_name = str(context.get("name", "")).lower()
        if tool_name not in SEARCH_LIKE_TOOLS or search_result_evidence_level(raw) != "exact_content":
            return False
        # Only preserve if this is the FIRST observation (key not yet seen)
        key = evidence_identity_key(
            tool_name,
            path=_extract_normalized_path(context),
            query=str(context.get("query") or "").strip() or None,
            args=context.get("args") if isinstance(context.get("args"), dict) else None,
        )
        if key and first_exact_evidence_seen is not None and key in first_exact_evidence_seen:
            # This is a repeat - don't preserve, let compression happen
            return False
        # First observation - preserve and track
        if key and first_exact_evidence_seen is not None:
            first_exact_evidence_seen.add(key)
        return True

    def _is_zero_heat(context: dict[str, Any] | None) -> bool:
        """Check if file has zero heat (never been read before in this session)."""
        if not file_heat or not context:
            return False
        args = context.get("args") if isinstance(context, dict) else None
        if not isinstance(args, dict):
            return False
        path = str(
            args.get("path") or args.get("file_path") or args.get("AbsolutePath") or args.get("TargetFile") or ""
        )
        if not path:
            return False
        norm_path = path.lower().strip()
        if session_files_read is not None and norm_path in session_files_read:
            return False
        heat = file_heat.get(norm_path, 0.0)
        return heat == 0.0

    breakdown: dict[str, int] = {}
    _skip_file_skeleton = model_profile is not None and not getattr(model_profile, "skeletonize_files", True)

    def _file_compressor(text: str) -> str:
        return _compress_file_read(text, tool_context={"_model_profile": model_profile} if model_profile else None)

    compressors: dict[str, Compressor] = {
        "file": _file_compressor,
        "grep": _compress_grep,
        "grep_context": _compress_grep_context,
        "stack_trace": _compress_stack_traces,
        "search_results": _compress_search_results,
        "pytest": _compress_pytest,
        "git_diff": _compress_git_diff,
        "ls": _compress_ls,
        "install": _compress_install,
        "git_log": _compress_git_log_impl,
    }

    for msg in messages:
        content = msg.get("content")
        if msg.get("role") == "tool_result" and isinstance(content, str):
            tool_id = str(msg.get("tool_use_id", ""))
            ctx = (tool_use_id_to_context or {}).get(tool_id, {})
            if _is_precision_read_context(ctx):
                continue
            tool_name = str(ctx.get("name", "")).lower()
            if tool_name in SEARCH_LIKE_TOOLS and search_result_evidence_level(content) == "navigation":
                continue
            if _should_preserve_exact_search_observation(ctx, content):
                continue
            if _first_exact_guard(ctx, content):
                msg["content"] = content
                continue
            # First-session preservation: None means fresh session, preserve all file reads
            if tool_name in FILE_LIKE_TOOLS:
                norm_path = _extract_normalized_path(ctx)
                if bool(norm_path) and (session_files_read is None or norm_path not in session_files_read):
                    continue
                # Zero-heat check: never compress files that haven't been read before
                if _is_zero_heat(ctx):
                    continue
            kind = _detect_tool_content_type_impl(content)
            if tool_name in FILE_LIKE_TOOLS:
                kind = "file"
            elif tool_name in SEARCH_LIKE_TOOLS and kind == "file":
                kind = "grep"
            elif kind == "raw":
                continue
            if kind == "pytest" and " FAILED" in content and not TOK_ENABLE_PYTEST_FAIL_COMPRESSION:
                continue
            effective_threshold = (
                RECENT_WINDOW_EVIDENCE_THRESHOLD
                if tool_compatible and kind in {"file", "grep", "grep_context", "search_results"}
                else threshold
            )
            if len(content) <= effective_threshold:
                continue
            compressor = compressors.get(kind)
            if compressor is None:
                continue
            if kind == "file" and _skip_file_skeleton:
                continue
            compressed: str = compressor(content)
            saved = len(content) - len(compressed)
            if saved <= 0:
                continue
            breakdown[kind] = breakdown.get(kind, 0) + saved
            msg["content"] = compressed
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            raw = block.get("content", "")
            if not isinstance(raw, str):
                continue

            tool_id = block.get("tool_use_id", "")
            tool_ctx: dict[str, Any] | None = None
            if tool_use_id_to_context is not None:
                tool_ctx = tool_use_id_to_context.get(tool_id)

            if _should_preserve_exact_search_observation(tool_ctx, raw):
                continue
            tool_name = str(tool_ctx.get("name", "")).lower() if tool_ctx else ""
            if tool_name in SEARCH_LIKE_TOOLS and search_result_evidence_level(raw) == "navigation":
                continue
            # First-read preservation: check both first-ever and first-in-session
            if tool_ctx:
                key = evidence_identity_key(
                    str(tool_ctx.get("name", "")),
                    path=str(tool_ctx.get("path") or "").strip() or None,
                    query=str(tool_ctx.get("query") or "").strip() or None,
                    command=str(
                        (tool_ctx.get("args") or {}).get("command") or (tool_ctx.get("args") or {}).get("cmd") or ""
                    ).strip()
                    or None,
                    args=tool_ctx.get("args") if isinstance(tool_ctx.get("args"), dict) else None,
                )
                norm_path = _extract_normalized_path(tool_ctx)
                is_first_ever = first_exact_evidence_seen is not None and key and key not in first_exact_evidence_seen
                # None means "fresh session" for file reads only — search tools use first_exact_evidence_seen
                is_first_session = (
                    tool_name in FILE_LIKE_TOOLS
                    and bool(norm_path)
                    and (session_files_read is None or norm_path not in session_files_read)
                )

                if is_first_ever or is_first_session:
                    if is_first_ever and first_exact_evidence_seen is not None:
                        first_exact_evidence_seen.add(key)
                    block["content"] = raw
                    continue

            kind = _detect_tool_content_type_impl(raw)
            if kind == "raw" and tool_name in SEARCH_LIKE_TOOLS:
                kind = "search_results"
            if kind == "pytest" and " FAILED" in raw and not TOK_ENABLE_PYTEST_FAIL_COMPRESSION:
                continue
            if kind in {"raw", "file"}:
                context = tool_ctx or {}
                if _is_precision_read_context(context):
                    continue
                tool_name = str(context.get("name", "")).lower()
                if tool_name in FILE_LIKE_TOOLS:
                    kind = "file"
                elif tool_name in SEARCH_LIKE_TOOLS and kind == "file":
                    kind = "grep"
                elif kind == "raw":
                    continue
            effective_threshold = (
                RECENT_WINDOW_EVIDENCE_THRESHOLD
                if tool_compatible and kind in {"file", "grep", "grep_context", "search_results"}
                else threshold
            )
            if len(raw) <= effective_threshold:
                continue
            compressor = compressors.get(kind)
            if compressor is None:
                continue
            if kind == "file" and _skip_file_skeleton:
                continue
            compressed_block: str = compressor(raw)
            saved = len(raw) - len(compressed_block)
            if saved <= 0:
                continue
            breakdown[kind] = breakdown.get(kind, 0) + saved
            block["content"] = compressed_block

    return messages, breakdown
