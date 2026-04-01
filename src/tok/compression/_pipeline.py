from __future__ import annotations

"""Compression pipeline helpers behind the public facade."""

import copy
import json
import os
import posixpath
import re
from typing import Any

import tok.compression as _compression
from . import (
    COMMAND_LIKE_TOOLS,
    EDIT_LIKE_TOOLS,
    FILE_LIKE_TOOLS,
    QUESTION_PREFIXES,
    RECENT_WINDOW_EVIDENCE_THRESHOLD,
    RECENT_WINDOW_THRESHOLD,
    STOP_WORDS,
    TOK_FRESHNESS_SIGNALS_EXPLANATION,
    TOK_OUTPUT_DIRECTIVE_MINIMAL,
    TOK_OUTPUT_DIRECTIVE_REINFORCED,
    TOK_PROTOCOL_LAW,
    TOK_TOOL_COMPAT_DIRECTIVE,
    _CUT_REJECTION_REASONS,
    _SEMANTIC_HASH_MIN_CHARS,
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
from ..runtime.config import RESULT_CACHE_TTL_SECONDS
from ..runtime.repeat_targets import (
    build_file_summary,
    build_file_skeleton,
    normalize_path_target,
)

globals().update(vars(_compression))


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
        for i in range(len(messages)):
            cls = classify_cut_eligibility(messages[i])
            if cls.eligible:
                eligible_indices.append(i)
            elif cls.reason in _CUT_REJECTION_REASONS:
                rejection_counts[cls.reason] = (
                    rejection_counts.get(cls.reason, 0) + 1
                )

        for i in reversed(eligible_indices):
            turns_seen += 1
            if turns_seen == keep_turns:
                cut_index = i
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
                if isinstance(content, str) and (
                    "PASSED" in content
                    or "SUCCESS" in content
                    or "DONE" in content
                ):
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

    def _norm(value: str, max_len: int) -> str:
        s = value.strip()
        if len(s) <= max_len:
            return s

        lowered = s.lower()
        if (
            "error" in lowered
            or "fail" in lowered
            or "parse_error" in lowered
            or "exception" in lowered
        ):
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

    def _bump(
        bucket: dict[str, int], value: str, score: int, max_len: int
    ) -> None:
        cleaned = _norm(value, max_len)
        if cleaned:
            bucket[cleaned] = bucket.get(cleaned, 0) + score

    def _top_items(bucket: dict[str, int], limit: int) -> list[str]:
        return [
            item
            for item, _score in sorted(
                bucket.items(), key=lambda kv: (-kv[1], kv[0])
            )[:limit]
        ]

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
            if "?" in stripped or (
                role == "user" and lowered.startswith(QUESTION_PREFIXES)
            ):
                _bump(question_scores, stripped[:60], 2 + recency, 60)

    total_old = len(old)
    for idx, msg in enumerate(old):
        text = text_of(msg.get("content", ""))
        role = msg.get("role", "")
        recency = total_old - idx

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
            cmd_match: re.Match[str] | None = re.search(
                r"(?:^|\s|>|#|run|exec)\s*(?:sudo\s+)?(pytest|python|python3|uv|npm|pnpm|yarn|cargo|go|git|rg|grep|sed|cat|ls|find|make|bash|sh|pip|docker|kubectl|gcloud|az|aws|gh|code|vi|vim|nano|emacs|test|build|install|update|delete|create|start|stop|restart|status|log|diff|mv|cp|rm|mkdir|rmdir|chmod|chown|pwd|cd|echo|print|export|unset|source|env|which|whereis|type|alias|unalias|history|jobs|fg|bg|kill|ps|top|htop|df|du|free|netstat|ss|curl|wget|ping|traceroute|dig|nslookup|ssh|scp|rsync|tar|zip|unzip|gzip|gunzip|bzip2|bunzip2|xz|unxz|7z|un7z|apt|yum|dnf|pacman|brew|choco|winget|snap|flatpak|gem|bundle|rake|mvn|gradle|cmake|make|configure|./configure|./build|./install|./run|./test|./start|./stop|./restart|./status|./log|./diff|./mv|./cp|\/rm|\/mkdir|\/rmdir|\/chmod|\/chown|\/pwd|\/cd|\/echo|\/print|\/export|\/unset|\/source|\/env|\/which|\/whereis|\/type|\/alias|\/unalias|\/history|\/jobs|\/fg|\/bg|\/kill|\/ps|\/top|\/htop|\/df|\/du|\/free|\/netstat|\/ss|\/curl|\/wget|\/ping|\/traceroute|\/dig|\/nslookup|\/ssh|\/scp|\/rsync|\/tar|\/zip|\/unzip|\/gzip|\/gunzip|\/bzip2|\/bunzip2|xz|\/unxz|7z|\/un7z|apt|yum|dnf|pacman|brew|choco|winget|snap|flatpak|gem|bundle|rake|mvn|gradle|cmake)\b",
                stripped,
            )
            if cmd_match:
                cmd_part = stripped[cmd_match.start(1) :].strip()
                _bump(cmd_scores, cmd_part[:60], 2 + recency, 60)

        content = msg.get("content")
        if role == "assistant" and isinstance(content, list):
            for block in content:
                if (
                    not isinstance(block, dict)
                    or block.get("type") != "tool_use"
                ):
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
                command = str(
                    tool_input.get("command") or tool_input.get("cmd") or ""
                ).strip()
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
        for match in re.finditer(
            r"\b(\d+\s+(?:passed|failed|errors?))\b", lowered
        ):
            _bump(test_scores, match.group(1), 2 + recency, 24)
        for line in text.splitlines():
            stripped = line.strip()
            if "FAILED" in stripped or "PASSED" in stripped:
                _bump(test_scores, stripped[:48], 2 + recency, 48)

        _extract_blockers(text, recency)
        _extract_questions(text, role, recency)
        _extract_next(text, role)

        for m in re.finditer(
            r"\b([a-zA-Z_][a-zA-Z0-9_]{1,20})\s*[:=]\s*([^\n,;|]{1,30})",
            text,
        ):
            k = m.group(1).lower()
            v = m.group(2).strip()
            if k not in STOP_WORDS and len(k) > 2:
                facts[k] = _norm(v[:25], 25)

    _summarize_causal_failures(old, error_scores, blocker_scores)
    _summarize_decision_hypotheses(old, next_scores, question_scores)

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
    top_constraints = _top_items(
        constraint_scores, _get_int(profile, "constraints", 2)
    )
    top_questions = _top_items(
        question_scores, _get_int(profile, "questions", 2)
    )
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
        compact_facts = [
            f"{k}:{v}"
            for k, v in sorted(facts.items())[: _get_int(profile, "facts", 3)]
        ]
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
    payload = "|".join(
        f"{key}:{state_map[key]}" for key in ordered_keys if key in state_map
    )
    return recent, f">>> {payload}" if payload else ""


def _detect_tool_content_type_impl(text: str) -> str:
    """Detect the content type of a tool result."""
    return _detect_tool_content_type(text)


def _compress_git_log_impl(text: str) -> str:
    """Compress verbose git log to compact table."""
    return _compress_git_log(text)


def tok_tool_result_impl(
    content: str, compression_level: str = "balanced"
) -> str:
    """Convert large tool result to dense tok representation."""
    if len(content) <= TOOL_COMPRESS_THRESHOLD:
        return content

    kind = _detect_tool_content_type_impl(content)
    original_chars = len(content)

    if kind == "pytest":
        compressed = _compress_pytest(content)
    elif kind == "grep":
        compressed = _compress_grep(content)
    elif kind == "git_diff":
        compressed = _compress_git_diff(content)
    elif kind == "ls":
        compressed = _compress_ls(content)
    elif kind == "install":
        compressed = _compress_install(content)
    elif kind == "git_log":
        compressed = _compress_git_log_impl(content)
    elif kind == "repetitive":
        compressed = _compress_repetitive(content)
    elif kind == "file":
        compressed = _compress_file_read(content)
    elif kind == "search_results":
        compressed = _compress_search_results(content)
    elif kind == "stack_trace":
        compressed = _compress_stack_traces(content)
    elif kind == "ps_output":
        compressed = _compress_env_ps(content, kind)
    elif kind == "env_output":
        compressed = _compress_env_ps(content, kind)
    elif kind == "grep_context":
        compressed = _compress_grep_context(content)
    elif kind in {"config_json", "json_skeleton"}:
        compressed = _compress_config_json(content)
    else:
        compressed = content

    compressed = _tighten_compressed_output(
        kind, compressed, compression_level
    )
    compressed = truncate_large_result(compressed)

    saved = original_chars - len(compressed)
    if saved <= 0:
        return content

    if not compressed.startswith(">>>") and saved > 0:
        compressed = (
            f">>> tok_compressed:tool_result|type:{kind}"
            f"|original_chars:{original_chars}|saved_chars:{saved}\n"
            + compressed
        )

    return compressed


def compress_tool_results_impl(
    messages: list[dict[str, Any]],
    result_cache: dict[str, tuple[str, str, float]] | None = None,
    tool_use_id_to_context: dict[str, dict[str, Any]] | None = None,
    compression_level: str = "balanced",
    semantic_hash_cache: dict[str, str] | None = None,
    bypass_result_cache: bool = False,
    hot_summary_records: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Walk messages, apply caching and tok_tool_result() to large tool_result blocks."""
    breakdown: dict[str, int] = {}

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

    def _truncate_stable_snippet(text: str, limit: int) -> str:
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        if limit <= 1:
            return cleaned[:limit]
        return cleaned[: limit - 1].rstrip() + "…"

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
                if (
                    context
                    and isinstance(context.get("args"), dict)
                    and context["args"].get("tok_bypass_cache")
                ):
                    breakdown["tok_bypass_cache_applied"] = (
                        breakdown.get("tok_bypass_cache_applied", 0) + 1
                    )
                    continue
                if context:
                    compressed, saved = _apply_result_cache(
                        content,
                        context,
                        result_cache,
                        compression_level=compression_level,
                        bypass_cache=bypass_result_cache,
                        ttl_seconds=RESULT_CACHE_TTL_SECONDS,
                    )
                    if "stable_payload_validation_failed" in compressed:
                        breakdown["stable_payload_validation_failed"] = (
                            breakdown.get(
                                "stable_payload_validation_failed", 0
                            )
                            + 1
                        )
                    if saved > 0:
                        kind = _detect_tool_content_type_impl(content)
                        key = (
                            f"{kind}_cached"
                            if "|unchanged|" in compressed
                            else f"{kind}_diff"
                        )
                        breakdown[key] = breakdown.get(key, 0) + saved
                    msg["content"] = compressed
            continue
        for block in content:
            if not (
                isinstance(block, dict) and block.get("type") == "tool_result"
            ):
                continue

            raw = block.get("content", "")
            if not isinstance(raw, str):
                continue

            tool_id = ""
            ctx: dict[str, Any] | None = None
            if tool_use_id_to_context is not None:
                tool_id = block.get("tool_use_id", "")
                ctx = tool_use_id_to_context.get(tool_id)
                if (
                    ctx
                    and isinstance(ctx.get("args"), dict)
                    and ctx["args"].get("tok_bypass_cache")
                ):
                    breakdown["tok_bypass_cache_applied"] = (
                        breakdown.get("tok_bypass_cache_applied", 0) + 1
                    )
                    continue

            # Precision file reads (offset/limit/start/end) must remain verbatim.
            # Skip semantic dedup, result-cache stubbing, and tok_tool_result compression.
            if _is_precision_read_context(ctx):
                if (
                    result_cache is not None
                    and ctx is not None
                    and tool_use_id_to_context is not None
                    and not bypass_result_cache
                ):
                    compressed, _saved = _apply_result_cache(
                        raw,
                        ctx,
                        result_cache,
                        compression_level=compression_level,
                        bypass_cache=bypass_result_cache,
                        ttl_seconds=RESULT_CACHE_TTL_SECONDS,
                    )
                    if "stable_payload_validation_failed" in compressed:
                        breakdown["stable_payload_validation_failed"] = (
                            breakdown.get(
                                "stable_payload_validation_failed", 0
                            )
                            + 1
                        )
                    block["content"] = compressed
                else:
                    block["content"] = raw
                continue

            if (
                semantic_hash_cache is not None
                and len(raw) >= _SEMANTIC_HASH_MIN_CHARS
                and tool_use_id_to_context is not None
            ):
                cache_key = _make_semantic_cache_key(ctx, raw)
                ctx_args = ctx.get("args") if isinstance(ctx, dict) else None
                if isinstance(ctx_args, dict) and any(
                    k in ctx_args for k in ("offset", "limit", "start", "end")
                ):
                    cache_key = None
                if cache_key is not None:
                    content_hash = _compute_semantic_hash(raw)
                    prev_hash = semantic_hash_cache.get(cache_key)
                    if prev_hash == content_hash:
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
                                try:
                                    summary = build_file_summary(
                                        raw,
                                        max_chars=280,
                                        max_lines=12,
                                    )
                                except Exception:
                                    pass
                        if not summary:
                            summary = _truncate_stable_snippet(raw, 280)

                        skeleton = ""
                        if ctx is not None:
                            path = ctx.get("path")
                            if path and len(raw) >= 100:
                                try:
                                    skeleton = build_file_skeleton(
                                        raw,
                                        max_chars=280,
                                        max_lines=14,
                                    )
                                except Exception:
                                    pass

                        lines = [f"@stable_result(hash:{content_hash})"]
                        if summary:
                            lines.append(
                                f"@stable_summary |> "
                                f"{_truncate_stable_snippet(summary, 280)}"
                            )
                        if skeleton:
                            lines.append(
                                f"@stable_skeleton |> "
                                f"{_truncate_stable_snippet(skeleton, 280)}"
                            )
                        token = "\n".join(lines)
                        saved = len(raw) - len(token)
                        if saved > 0:
                            breakdown["semantic_dedup"] = breakdown.get(
                                "semantic_dedup", 0
                            ) + max(0, saved)
                            block["content"] = token
                            continue
                    else:
                        semantic_hash_cache[cache_key] = content_hash

            if (
                result_cache is not None
                and tool_use_id_to_context is not None
                and not bypass_result_cache
            ):
                context = ctx

                if context:
                    compressed, saved = _apply_result_cache(
                        raw,
                        context,
                        result_cache,
                        compression_level=compression_level,
                        bypass_cache=bypass_result_cache,
                        ttl_seconds=RESULT_CACHE_TTL_SECONDS,
                    )
                    if "stable_payload_validation_failed" in compressed:
                        breakdown["stable_payload_validation_failed"] = (
                            breakdown.get(
                                "stable_payload_validation_failed", 0
                            )
                            + 1
                        )
                    if saved > 0:
                        kind = _detect_tool_content_type_impl(raw)
                        key = (
                            f"{kind}_cached"
                            if "|unchanged|" in compressed
                            else f"{kind}_diff"
                        )
                        breakdown[key] = breakdown.get(key, 0) + saved
                    block["content"] = compressed
                    continue

            compressed = tok_tool_result_impl(
                raw, compression_level=compression_level
            )
            saved = len(raw) - len(compressed)
            if saved > 0:
                kind = _detect_tool_content_type_impl(raw)
                breakdown[kind] = breakdown.get(kind, 0) + saved
                block["content"] = compressed
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
    """Inject the Tok output directive into every request."""
    sys_prompt = body.get("system", "")

    # Add Tok freshness signals explanation for Claude to understand system metadata
    if not tool_compatible and sys_prompt:
        # Only add if not already present to avoid duplication
        if isinstance(sys_prompt, str):
            if "[Tok File Freshness System]" not in sys_prompt:
                sys_prompt = (
                    sys_prompt + "\n\n" + TOK_FRESHNESS_SIGNALS_EXPLANATION
                )
                body["system"] = sys_prompt
        elif isinstance(sys_prompt, list):
            # Check if any text block already contains the freshness explanation
            has_freshness = any(
                isinstance(block, dict)
                and block.get("type") == "text"
                and "[Tok File Freshness System]" in block.get("text", "")
                for block in sys_prompt
            )
            if not has_freshness:
                sys_prompt = sys_prompt + [
                    {"type": "text", "text": TOK_FRESHNESS_SIGNALS_EXPLANATION}
                ]
                body["system"] = sys_prompt

    directive_parts = []

    if not tool_compatible:
        directive_parts.append("=== MODE: TOK-NATIVE ===")

    if not tool_compatible and (pressure > 1):
        directive_parts.append(TOK_PROTOCOL_LAW)

    if tool_compatible:
        base_directive = TOK_TOOL_COMPAT_DIRECTIVE
    elif pressure > 50 or (
        behavior_signals and behavior_signals.get("semantic_drift_detected")
    ):
        base_directive = TOK_OUTPUT_DIRECTIVE_REINFORCED
    else:
        base_directive = TOK_OUTPUT_DIRECTIVE_MINIMAL

    directive_parts.append(base_directive)
    if runtime_hints:
        directive_parts.append(
            "\n".join(
                str(hint).strip()
                for hint in runtime_hints
                if str(hint).strip()
            )
        )
    output_directive = "\n\n".join(directive_parts)

    if (
        tool_compatible
        and not _should_include_tok_state(tok_state, tool_compatible=True)
        and not grammar
        and not todo
        and not deltas
    ):
        current_sys_prompt = body.get("system", "")
        if isinstance(current_sys_prompt, str):
            body["system"] = (
                current_sys_prompt + "\n\n" + output_directive
                if current_sys_prompt
                else output_directive
            )
        elif isinstance(current_sys_prompt, list):
            body["system"] = current_sys_prompt + [
                {"type": "text", "text": output_directive}
            ]
        else:
            body["system"] = output_directive
        return body
    include_tok_state = _should_include_tok_state(
        tok_state, tool_compatible=tool_compatible
    )

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
        body["system"] = (
            current_sys_prompt + "\n\n" + addition
            if current_sys_prompt
            else addition
        )
    elif isinstance(current_sys_prompt, list):
        body["system"] = current_sys_prompt + [
            {"type": "text", "text": output_directive}
        ]
        if dynamic_state:
            body["system"].append({"type": "text", "text": dynamic_state})
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
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply content-aware compression to tool_result blocks in the recent window."""

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

    breakdown: dict[str, int] = {}
    compressors = {
        "file": _compress_file_read,
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
            kind = _detect_tool_content_type_impl(content)
            if tool_name in FILE_LIKE_TOOLS:
                kind = "file"
            elif kind == "raw":
                continue
            effective_threshold = (
                RECENT_WINDOW_EVIDENCE_THRESHOLD
                if tool_compatible
                and kind in {"file", "grep", "grep_context", "search_results"}
                else threshold
            )
            if len(content) <= effective_threshold:
                continue
            compressor = compressors.get(kind)
            if compressor is None:
                continue
            compressed = compressor(content)
            saved = len(content) - len(compressed)
            if saved <= 0:
                continue
            breakdown[kind] = breakdown.get(kind, 0) + saved
            msg["content"] = compressed
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not (
                isinstance(block, dict) and block.get("type") == "tool_result"
            ):
                continue
            raw = block.get("content", "")
            if not isinstance(raw, str):
                continue

            kind = _detect_tool_content_type_impl(raw)
            if kind in {"raw", "file"}:
                tool_id = block.get("tool_use_id", "")
                ctx = (tool_use_id_to_context or {}).get(tool_id, {})
                if _is_precision_read_context(ctx):
                    continue
                tool_name = str(ctx.get("name", "")).lower()
                if tool_name in FILE_LIKE_TOOLS:
                    kind = "file"
                elif kind == "raw":
                    continue
            effective_threshold = (
                RECENT_WINDOW_EVIDENCE_THRESHOLD
                if tool_compatible
                and kind in {"file", "grep", "grep_context", "search_results"}
                else threshold
            )
            if len(raw) <= effective_threshold:
                continue

            compressor = compressors.get(kind)
            if compressor is None:
                continue

            compressed = compressor(raw)
            saved = len(raw) - len(compressed)
            if saved <= 0:
                continue

            breakdown[kind] = breakdown.get(kind, 0) + saved
            block["content"] = compressed

    return messages, breakdown


# TOOL_COMPRESS_THRESHOLD is imported from __init__.py via globals().update(vars(_compression))

# Heuristic: source code indicators
_CODE_PATTERNS = re.compile(
    r"\bdef \b|\bclass \b|\bimport \b|\basync def \b|\bfunction \b"
)


def _detect_tool_content_type(text: str) -> str:
    """Detect the content type of a tool result."""
    # Stack trace: looks like a Python or JS traceback
    if "Traceback (most recent call last):" in text or "at new " in text:
        return "stack_trace"

    # Pytest: has PASSED/FAILED lines + summary
    if re.search(r"\b(PASSED|FAILED)\b", text) and re.search(
        r"\d+ (passed|failed)( in | ,)", text
    ):
        return "pytest"

    # Git diff: starts with 'diff --git' or has '--- a/' + '+++ b/' lines
    if re.search(r"^diff --git ", text, re.MULTILINE) or (
        re.search(r"^--- a/", text, re.MULTILINE)
        and re.search(r"^\+\+\+ b/", text, re.MULTILINE)
    ):
        return "git_diff"

    # ps aux and env
    if (
        re.match(r"^(USER\s+PID\s+%CPU|UID\s+PID\s+PPID)", text)
        or "COMMAND" in text[:200]
    ):
        return "ps_output"
    if (
        re.match(r"^(HOME|PATH|SHELL|USER|LANG)=", text, re.MULTILINE)
        and "=" in text
    ):
        return "env_output"

    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]

    # Grep context: path-line-context
    if len(non_empty) >= 4:
        grep_c_matches = sum(
            1 for l in non_empty if re.match(r"^[^\s-][^-]*-(\d+)-", l)
        )
        if grep_c_matches / len(non_empty) > 0.6:
            return "grep_context"

    # JSON output
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return "json_skeleton"

    # Git log: verbose (commit <sha40>) or oneline format
    if len(non_empty) >= 2:
        if sum(1 for l in non_empty if _GIT_LOG_COMMIT_RE.match(l)) >= 2:
            return "git_log"
        oneline_matches = sum(
            1 for l in non_empty if _GIT_LOG_ONELINE_RE.match(l.strip())
        )
        if oneline_matches >= 4 and oneline_matches / len(non_empty) > 0.4:
            return "git_log"

    # Directory listing: ≥8 lines of filenames or ls -la output
    if len(non_empty) >= 8:
        la_lines = sum(1 for l in non_empty if re.match(r"^[dl-][rwx-]{9}", l))
        plain_file_lines = sum(
            1
            for l in non_empty
            if re.match(r"^\S+\.\w{1,6}$", l.strip())
            or re.match(r"^\S+/$", l.strip())
        )
        # Claude Code / glob style: simple list of absolute paths or JSON-like
        glob_lines = sum(
            1 for l in non_empty if re.match(r"^(/[^/ ]+)+$", l.strip())
        )
        if (
            la_lines >= 6
            or plain_file_lines / len(non_empty) > 0.7
            or glob_lines / len(non_empty) > 0.7
        ):
            return "ls"

    # Package install output: >5 progress lines
    if len(non_empty) >= 6:
        install_lines = sum(
            1 for l in non_empty if _INSTALL_PROGRESS_RE.match(l)
        )
        if install_lines >= 5:
            return "install"

    # Grep: majority of lines match path:linenum:content or path:content
    if len(non_empty) >= 3:
        grep_matches = sum(
            1
            for l in non_empty
            if re.match(r"^[^\s:][^:]*:\d+:", l)
            or re.match(r"^[^\s:][^:]*:[^\n]+$", l)
        )
        if grep_matches / len(non_empty) > 0.7:
            return "grep"

    # File read: large + looks like source code — check before repetitive to
    # avoid misfire
    if len(text) > 1000 and _CODE_PATTERNS.search(text):
        return "file"

    # Repetitive: ≥5 consecutive lines sharing same prefix
    if len(lines) >= 5:
        for i in range(len(lines) - 4):
            prefix = re.split(r"[/: ]", lines[i].rstrip())[0]
            if prefix and all(
                lines[i + j].rstrip().startswith(prefix) for j in range(1, 5)
            ):
                return "repetitive"

    # Search results: JSON-like list of uniform dictionaries
    if text.strip().startswith("[") and text.strip().endswith("]"):
        try:
            data = json.loads(text)
            if isinstance(data, list) and len(data) >= 3:
                if all(isinstance(x, dict) for x in data[:3]):
                    return "search_results"
        except Exception:
            pass

    # Config JSON: flat-ish JSON object
    if text.strip().startswith("{") and text.strip().endswith("}"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and len(data) >= 5:
                # Heuristic for config: many keys, not mostly long text
                return "config_json"
        except Exception:
            pass

    return "raw"


def _compress_pytest(text: str) -> str:
    """Compress pytest output: strip PASSED lines, keep FAILED + tracebacks + summary."""
    lines = text.splitlines()
    result: list[str] = []
    in_failure = False
    passed = 0
    failed = 0

    for line in lines:
        # Summary line
        if re.match(r"=+\s+\d+.*\s+=+\s*$", line):
            result.append(line)
            in_failure = False
            continue

        if " PASSED" in line or line.endswith(" PASSED"):
            passed += 1
            in_failure = False
            continue

        if " FAILED" in line or line.endswith(" FAILED"):
            failed += 1
            in_failure = True
            result.append(line)
            continue

        # Section headers
        if line.startswith("=") or line.startswith("_"):
            in_failure = (
                line.startswith("_ FAILURES") or "FAILED" in line or in_failure
            )
            result.append(line)
            continue

        if in_failure:
            result.append(line)
        elif (
            line.startswith("collected ")
            or line.startswith("platform ")
            or line.startswith("rootdir")
        ):
            result.append(line)

    # Build tok header
    header = f">>> tool:pytest|passed:{passed}|failed:{failed}"
    return header + "\n" + "\n".join(result)


def _compress_grep(text: str) -> str:
    """Compress grep output: group matches by file path."""
    lines = text.splitlines()
    by_file: dict[str, list[str]] = {}
    order: list[str] = []

    for line in lines:
        m = re.match(r"^([^\s:][^:]*):(\d+):(.*)", line)
        if m:
            path, _lnum, snippet = m.group(1), m.group(2), m.group(3)
        else:
            m2 = re.match(r"^([^\s:][^:]*):(.+)", line)
            if m2:
                path, snippet = m2.group(1), m2.group(2)
            else:
                path, snippet = "", line
        key = path or "__other__"
        if key not in by_file:
            by_file[key] = []
            order.append(key)
        by_file[key].append(snippet.strip())

    total = sum(len(v) for v in by_file.values())
    if total <= 3:
        return text  # keep verbatim

    result = [
        f">>> tool:grep|matches:{total}|files:{len([k for k in order if k != '__other__'])}"
    ]
    for key in order:
        snippets = by_file[key]
        first = snippets[0][:80]
        suffix = f" ({len(snippets)} matches)" if len(snippets) > 1 else ""
        result.append(
            f"{key}: {len(snippets)} match{'es' if len(snippets) > 1 else ''} — {first}{suffix}"
        )

    return "\n".join(result)


def _compress_repetitive(text: str) -> str:
    """Compress repetitive bash output using run-length grouping."""
    lines = text.splitlines()
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        parts = re.split(r"[/: ]", line.rstrip())
        # For lines starting with '/' the first part is empty; use second segment
        prefix = next((p for p in parts if p), "")

        if prefix:
            # Count run
            j = i + 1
            while j < len(lines) and lines[j].rstrip().startswith(prefix):
                j += 1
            run_len = j - i
            if run_len >= 5:
                result.append(f"[{prefix}...]: {run_len} lines")
                i = j
                continue

        result.append(line)
        i += 1

    if len(result) >= len(lines):
        return text  # no compression achieved

    header = f">>> tool:bash|original_lines:{len(lines)}|compressed_lines:{len(result)}"
    return header + "\n" + "\n".join(result)


def _compress_file_read(text: str) -> str:
    """Compress source file reads to structural skeleton."""
    lines = text.splitlines()
    result: list[str] = []
    i = 0
    in_body = False
    body_line_count = 0

    SIGNATURE_RE = re.compile(
        r"^(import |from |class |def |async def |[A-Z_][A-Z0-9_]+ =|\s*def |\s*async def |\s*class )"
    )
    INDENT_RE = re.compile(r"^(\s+)")

    # Determine base indentation level (for top-level detection)
    def _indent(l: str) -> int:
        m = INDENT_RE.match(l)
        return len(m.group(1)) if m else 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            result.append("")
            i += 1
            continue

        _indent(line)

        # Top-level imports, class/def signatures → always keep
        if SIGNATURE_RE.match(line):
            if in_body and body_line_count > 0:
                result.append(f"  |> [{body_line_count} lines]")
            in_body = False
            body_line_count = 0
            result.append(line)
            i += 1
            continue

        # Indented def/class inside a class → keep signature
        if re.match(r"^\s+(def |async def |class )", line):
            if in_body and body_line_count > 0:
                result.append(f"  |> [{body_line_count} lines]")
            in_body = False
            body_line_count = 0
            result.append(line)
            i += 1
            continue

        # Body line
        if in_body:
            body_line_count += 1
        else:
            in_body = True
            body_line_count = 1
        i += 1

    if in_body and body_line_count > 0:
        result.append(f"  |> [{body_line_count} lines]")

    if len(result) >= len(lines):
        return text  # skeleton not smaller

    trimmed_result = result
    if len(result) > 32:
        head_count = 18
        tail_count = 8
        omitted = max(0, len(result) - head_count - tail_count)
        trimmed_result = list(result[:head_count])
        if omitted:
            trimmed_result.append(f"  |> [{omitted} skeleton lines omitted]")
        trimmed_result.extend(result[-tail_count:])

    original_chars = len(text)
    compressed = "\n".join(trimmed_result)
    header = (
        f">>> tool:file_read|original_chars:{original_chars}|"
        f"skeleton_lines:{len(result)}|retained_skeleton_lines:{len(trimmed_result)}"
    )
    return header + "\n" + compressed


def _compress_git_diff(text: str) -> str:
    """Strip context lines from git diff output; keep only actual changes."""
    lines = text.splitlines()
    result: list[str] = []
    files = 0
    insertions = 0
    deletions = 0

    for line in lines:
        # Always keep diff headers, hunk headers, change lines
        if line.startswith("diff --git") or line.startswith("index "):
            if line.startswith("diff --git"):
                files += 1
            result.append(line)
        elif line.startswith("---") or line.startswith("+++"):
            result.append(line)
        elif line.startswith("@@"):
            result.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            insertions += 1
            result.append(line)
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
            result.append(line)
        elif not line.strip():
            result.append(line)
        # else: context line — drop

    if len(result) >= len(lines):
        return text  # no gain

    header = f">>> tool:git_diff|files:{files}|insertions:{insertions}|deletions:{deletions}"
    return header + "\n" + "\n".join(result)


def _compress_ls(text: str) -> str:
    """Compress directory listing to extension groups."""
    lines = [l for l in text.splitlines() if l.strip()]

    # Detect ls -la style (starts with permissions or 'total')
    is_la = any(re.match(r"^(total\s+\d+|[dl-][rwx-]{9})", l) for l in lines)

    names: list[str] = []
    dirs: list[str] = []

    for line in lines:
        if is_la:
            # Strip permission/ownership/date columns — last token is name
            parts = line.split()
            if not parts:
                continue
            if parts[0].startswith("total"):
                continue
            name = parts[-1]
            if line.startswith("d"):
                dirs.append(name)
            else:
                names.append(name)
        else:
            names.append(line.strip())

    # Group by extension
    ext_counts: dict[str, int] = {}
    unusual: list[str] = []
    for name in names:
        if "." in name and not name.startswith("."):
            ext = name.rsplit(".", 1)[1].lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        else:
            unusual.append(name)

    result_lines = [
        f">>> tool:ls|total:{len(names) + len(dirs)}|dirs:{len(dirs)}"
    ]
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        result_lines.append(f"  .{ext}: {count}")
    if dirs:
        result_lines.append(
            f"  dirs: {', '.join(dirs[:10])}"
            + (" ..." if len(dirs) > 10 else "")
        )
    if unusual:
        result_lines.append(
            f"  other: {', '.join(unusual[:10])}"
            + (" ..." if len(unusual) > 10 else "")
        )

    result = "\n".join(result_lines)
    if len(result) >= len(text):
        return text
    return result


_INSTALL_PROGRESS_RE = re.compile(
    r"^\s*(Downloading|Installing|Resolving|Fetching|Installed|Resolved|Locked"
    r"|Preparing|Collecting|Obtaining|Already satisfied|Using cached"
    r"|Requirement already|Building|Running|Successfully installed"
    r"|Prepared|Uninstalled|Built)",
    re.IGNORECASE,
)
_INSTALL_ERROR_RE = re.compile(
    r"\b(error|warning|failed|conflict)\b", re.IGNORECASE
)
_INSTALL_SUMMARY_RE = re.compile(
    r"(Successfully installed|installed \d+|added \d+|in \d+\.\d+s|\d+ packages?)",
    re.IGNORECASE,
)


def _compress_install(text: str) -> str:
    """Compress npm/pip/uv/cargo install output."""
    lines = text.splitlines()
    kept: list[str] = []
    summary_line = ""
    packages = 0
    duration = ""

    for line in lines:
        # Check summary first (takes priority over progress pattern)
        if _INSTALL_SUMMARY_RE.search(line):
            summary_line = line
            m = re.search(r"in\s+(\d+\.\d+s)", line)
            if m:
                duration = m.group(1)
            continue
        if _INSTALL_ERROR_RE.search(line):
            kept.append(line)
            continue
        if _INSTALL_PROGRESS_RE.match(line):
            packages += 1
            continue  # drop progress line
        kept.append(line)

    if summary_line:
        kept.append(summary_line)

    header = f">>> tool:install|packages:{packages}|duration:{duration or 'unknown'}"
    result = header + "\n" + "\n".join(kept)
    if len(result) >= len(text):
        return text
    return result


_GIT_LOG_COMMIT_RE = re.compile(r"^commit ([0-9a-f]{40})$")
_GIT_LOG_ONELINE_RE = re.compile(r"^([0-9a-f]{7,40})\s+(.+)")


def _compress_git_log(text: str) -> str:
    """Compress verbose git log to compact table."""
    lines = text.splitlines()

    # Detect oneline format
    oneline = all(
        not l.strip() or _GIT_LOG_ONELINE_RE.match(l)
        for l in lines
        if l.strip()
    )
    if oneline:
        entries: list[str] = []
        for line in lines:
            m = _GIT_LOG_ONELINE_RE.match(line.strip())
            if m:
                entries.append(f"{m.group(1)[:8]} {m.group(2)[:80]}")
        if not entries:
            return text
        header = f">>> tool:git_log|commits:{len(entries)}"
        result = header + "\n" + "\n".join(entries)
        if len(result) >= len(text):
            return text
        return result

    # Verbose format — parse commit/author/date/subject blocks
    entries = []
    current: dict[str, str] = {}
    in_body = False

    for line in lines:
        m = _GIT_LOG_COMMIT_RE.match(line)
        if m:
            if current.get("hash"):
                entries.append(
                    f"{current.get('hash', '')} {current.get('subject', '')[:40]}"
                )
            current = {
                "hash": m.group(1)[:8],
                "author": "",
                "date": "",
                "subject": "",
            }
            in_body = False
            continue
        if line.startswith("Author:"):
            parts = line[7:].strip().split("<")[0].strip().split()
            current["author"] = parts[0] if parts else ""
            in_body = False
            continue
        if line.startswith("Date:"):
            current["date"] = line[5:].strip()[:20]
            in_body = False
            continue
        stripped = line.strip()
        if (
            stripped
            and not in_body
            and current.get("hash")
            and not current["subject"]
        ):
            current["subject"] = stripped[:72]
            in_body = True

    if current.get("hash"):
        entries.append(
            f"{current.get('hash', '')} {current.get('subject', '')[:40]}"
        )

    if not entries:
        return text

    result_lines = [f">>> tool:git_log|commits:{len(entries)}"]
    for e in entries:
        result_lines.append(e)

    result = "\n".join(result_lines)
    if len(result) >= len(text):
        return text
    return result


def _compress_search_results(text: str) -> str:
    """Compress JSON search results into a dense tabular format."""
    try:
        data = json.loads(text)
        if not isinstance(data, list) or not data:
            return text

        # Identify common keys
        sample = data[0]
        if not isinstance(sample, dict):
            return text

        common_keys = [
            k for k in sample.keys() if all(k in x for x in data[:5])
        ]
        # Prioritize interesting keys for the header
        header_keys = [
            k
            for k in ("path", "file", "name", "title", "line", "id")
            if k in common_keys
        ]
        if not header_keys:
            header_keys = common_keys[:3]

        result = [
            f">>> tool:search_results|count:{len(data)}|keys:{','.join(header_keys)}"
        ]
        for item in data:
            vals = [
                str(item.get(k, ""))[:50].replace("\n", " ")
                for k in header_keys
            ]
            result.append(" | ".join(vals))

        return "\n".join(result)
    except Exception:
        return text


def _compress_stack_traces(text: str) -> str:
    """Normalize and densify stack traces; filter out internal library frames."""
    lines = text.splitlines()
    result = []

    # Common library path patterns to hide
    LIB_PATTERNS = re.compile(
        r"(node_modules|site-packages|dist-packages|/lib/python|/usr/lib|/usr/include|/Library/Frameworks|/usr/local/Cellar)"
    )

    # Heuristic: find common absolute path prefix
    paths = re.findall(r'File "([^"]+)"', text)
    common_prefix = ""
    if len(paths) >= 2:
        common_prefix = (
            os.path.commonpath(paths) if hasattr(os, "commonpath") else ""
        )
        if common_prefix and len(common_prefix) < 10:
            common_prefix = ""

    hidden_count = 0
    for line in lines:
        # Python: File "/path/to/file.py", line 123, in function
        m = re.search(r'File "(.+)", line (\d+), in (\w+)', line)
        if m:
            path, line_num, func = m.group(1), m.group(2), m.group(3)
            if LIB_PATTERNS.search(path):
                hidden_count += 1
                continue
            if common_prefix and path.startswith(common_prefix):
                path = "..." + path[len(common_prefix) :]
            result.append(f"  at {func} ({path}:{line_num})")
            continue

        # JS: at function (/path/to/file.js:123:45)
        m = re.search(r"at (\w+) \((.+):(\d+):(\d+)\)", line)
        if m:
            func, path, lnum, _col = (
                m.group(1),
                m.group(2),
                m.group(3),
                m.group(4),
            )
            if LIB_PATTERNS.search(path):
                hidden_count += 1
                continue
            if common_prefix and path.startswith(common_prefix):
                path = "..." + path[len(common_prefix) :]
            result.append(f"  at {func} ({path}:{lnum})")
            continue

        result.append(line)

    if hidden_count > 0:
        result.insert(0, f"  [... filtered {hidden_count} library frames]")

    header = (
        f">>> tool:stack_trace|lines:{len(lines)}|hidden_frames:{hidden_count}"
    )
    return header + "\n" + "\n".join(result)


def _compress_json_response(data: Any, depth: int = 0) -> Any:
    """Recursively skeletonize large JSON objects."""
    if isinstance(data, dict):
        if len(data) > 20 and depth > 1:
            return f"{{... {len(data)} keys}}"
        res = {}
        for k, v in data.items():
            res[k] = _compress_json_response(v, depth + 1)
        return res
    if isinstance(data, list):
        if len(data) > 10:
            return [
                _compress_json_response(data[0], depth + 1),
                f"... {len(data) - 1} more items",
            ]
        return [_compress_json_response(x, depth + 1) for x in data]
    if isinstance(data, str) and len(data) > 200:
        return data[:197] + "..."
    return data


def _compress_grep_context(text: str) -> str:
    """Compress grep context output into unified blocks."""
    lines = text.splitlines()
    if not lines:
        return text

    # Pattern: path-line-context OR path:line:match
    # We focus on the '-' version (context)
    result = []
    current_file = None
    current_block: list[str] = []
    last_line_num = -1

    for line in lines:
        m = re.match(r"^([^\s-][^-]*)-(\d+)-(.*)", line)
        if m:
            path, lnum, content = m.group(1), int(m.group(2)), m.group(3)
            if path != current_file:
                if current_block:
                    result.append(f"  [{last_line_num}]")
                current_file = path
                result.append(f"file://{path}:")
                result.append(f"  [{lnum}] {content}")
                last_line_num = lnum
            else:
                if lnum > last_line_num + 1:
                    result.append("  ...")
                result.append(f"  [{lnum}] {content}")
                last_line_num = lnum
            continue

        # Fallback for search matches (:)
        m2 = re.match(r"^([^\s:][^:]*):(\d+):(.*)", line)
        if m2:
            path, lnum, content = m2.group(1), int(m2.group(2)), m2.group(3)
            if path != current_file:
                current_file = path
                result.append(f"file://{path}:")
            result.append(f"  [{lnum}]* {content}")
            last_line_num = lnum
            continue

        result.append(line)

    header = f">>> tool:grep_context|lines:{len(lines)}"
    return header + "\n" + "\n".join(result)


def _compress_env_ps(text: str, kind: str) -> str:
    """Extract targeted info from env or ps output."""
    lines = text.splitlines()
    # We don't have the original request here, but we can look for "interesting" items
    # or just show a high-level summary if too large.

    if kind == "ps_output":
        # Keep header + any non-system processes (heuristic: not in /System or /usr/lib)
        kept = [lines[0]] if lines else []
        for line in lines[1:]:
            if (
                "/System/" in line
                or "/usr/libexec/" in line
                or "kernel_task" in line
            ):
                continue
            kept.append(line)

        if len(kept) > 20:
            kept = kept[:20] + [f"... {len(kept) - 20} more active processes"]

        header = (
            f">>> tool:ps|total_lines:{len(lines)}|interesting:{len(kept) - 1}"
        )
        return header + "\n" + "\n".join(kept)

    if kind == "env_output":
        # Keep only commonly useful vars
        INTERESTING = {
            "PATH",
            "HOME",
            "USER",
            "SHELL",
            "EDITOR",
            "LANG",
            "PWD",
            "VIRTUAL_ENV",
        }
        kept = []
        for line in lines:
            if "=" in line:
                k = line.split("=", 1)[0]
                if (
                    k in INTERESTING
                    or "API" in k
                    or "TOKEN" in k
                    or "URL" in k
                    or "PORT" in k
                ):
                    kept.append(line)

        header = f">>> tool:env|total_vars:{len(lines)}|displayed:{len(kept)}"
        return header + "\n" + "\n".join(kept)

    return text


def _compress_config_json(text: str) -> str:
    """Convert JSON to sparse key:val format with skeletonization."""
    try:
        data = json.loads(text)
        skeleton = _compress_json_response(data)
        compressed = json.dumps(skeleton, indent=2)

        header = f">>> tool:json_skeleton|original_chars:{len(text)}|saved_chars:{len(text) - len(compressed)}"
        return header + "\n" + compressed
    except Exception:
        return text


def _tighten_compressed_output(
    kind: str, compressed: str, compression_level: str
) -> str:
    if compression_level != "aggressive":
        return compressed
    if kind not in {
        "grep",
        "grep_context",
        "ls",
        "install",
        "repetitive",
        "search_results",
    }:
        return compressed
    lines = compressed.splitlines()
    if len(lines) <= 4:
        return compressed
    header = lines[0]
    body = lines[1:]
    limit = 4
    if len(body) <= limit:
        return compressed
    trimmed = (
        [header]
        + body[:limit]
        + [f"... {len(body) - limit} more lines omitted"]
    )
    candidate = "\n".join(trimmed)
    return candidate if len(candidate) < len(compressed) else compressed


def truncate_large_result(text: str, limit: int = 1200) -> str:
    """Head-tail truncation for extremely large results to prevent context flooding.
    Now signal-aware: scans the middle part for 'interesting' lines (errors, etc.)
    and preserves the first one found.
    """
    if len(text) <= int(limit * 1.5):
        return text

    signals = re.compile(
        r"\b(error|fail|exception|traceback|parse_error|collision|conflict|issue|bug|diff|warning)\b",
        re.IGNORECASE,
    )

    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    middle = text[limit // 2 : -limit // 2]

    important_line = ""
    for line in middle.splitlines():
        if signals.search(line):
            important_line = f"\n... [SIGNAL FOUND] {line.strip()[:100]} ..."
            break

    omitted = len(text) - (limit // 2 * 2)
    return (
        f"{head}\n... [TRUNCATED {omitted} CHARS] ...{important_line}\n{tail}"
    )
