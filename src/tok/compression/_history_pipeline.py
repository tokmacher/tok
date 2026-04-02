from __future__ import annotations

"""History- and request-side compression orchestration helpers."""

import copy
import re
from typing import Any

from . import (
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
from ._registry import build_default_registry
from ._tool_result_codecs import (
    _compress_config_json,
    _compress_env_ps,
    _compress_file_read,
    _compress_git_diff,
    _compress_git_log,
    _compress_grep,
    _compress_grep_context,
    _compress_install,
    _compress_ls,
    _compress_pytest,
    _compress_repetitive,
    _compress_search_results,
    _compress_stack_traces,
    _detect_tool_content_type,
    _tighten_compressed_output,
    truncate_large_result,
)
from ..runtime.config import RESULT_CACHE_TTL_SECONDS
from ..runtime.repeat_targets import (
    build_file_skeleton,
    build_file_summary,
    normalize_path_target,
)

__all__ = [
    "TOOL_COMPRESS_THRESHOLD",
    "_compress_git_log_impl",
    "_detect_tool_content_type_impl",
    "compress_history_impl",
    "compress_recent_window_impl",
    "compress_tool_results_impl",
    "inject_system_additions_impl",
    "tok_tool_result_impl",
]

TOOL_COMPRESS_THRESHOLD = 0


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
                r"(?:^|\s|>|#|run|exec)\s*(?:sudo\s+)?(pytest|python|python3|uv|npm|pnpm|yarn|cargo|go|git|rg|grep|sed|cat|ls|find|make|bash|sh|pip|docker|kubectl|gcloud|az|aws|gh|code|vi|vim|nano|emacs|test|build|install|update|delete|create|start|stop|restart|status|log|diff|mv|cp|rm|mkdir|rmdir|chmod|chown|pwd|cd|echo|print|export|unset|source|env|which|whereis|type|alias|unalias|history|jobs|fg|bg|kill|ps|top|htop|df|du|free|netstat|ss|curl|wget|ping|traceroute|dig|nslookup|ssh|scp|rsync|tar|zip|unzip|gzip|gunzip|bzip2|bunzip2|xz|unxz|7z|un7z|apt|yum|dnf|pacman|brew|choco|winget|snap|flatpak|gem|bundle|rake|mvn|gradle|cmake)\b",
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
            if key not in STOP_WORDS and len(key) > 2:
                facts[key] = _norm(value[:25], 25)

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
            f"{key}:{value}"
            for key, value in sorted(facts.items())[
                : _get_int(profile, "facts", 3)
            ]
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
    return _detect_tool_content_type(text)


def _compress_git_log_impl(text: str) -> str:
    return _compress_git_log(text)


def _tool_command_hint(tool_context: dict[str, Any] | None) -> str:
    if not isinstance(tool_context, dict):
        return ""
    args = tool_context.get("args")
    if isinstance(args, dict):
        for key in ("command", "cmd"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value
    for key in ("command", "cmd"):
        value = tool_context.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def tok_tool_result_impl(
    content: str,
    compression_level: str = "balanced",
    tool_context: dict[str, Any] | None = None,
) -> str:
    if len(content) <= TOOL_COMPRESS_THRESHOLD:
        return content

    kind = _detect_tool_content_type_impl(content)
    original_chars = len(content)
    registry = build_default_registry(
        compress_pytest=lambda text: _compress_pytest(
            text, command=_tool_command_hint(tool_context)
        ),
        compress_grep=_compress_grep,
        compress_git_diff=_compress_git_diff,
        compress_ls=_compress_ls,
        compress_install=_compress_install,
        compress_git_log=_compress_git_log_impl,
        compress_repetitive=_compress_repetitive,
        compress_file_read=_compress_file_read,
        compress_search_results=_compress_search_results,
        compress_stack_traces=_compress_stack_traces,
        compress_grep_context=_compress_grep_context,
        compress_config_json=_compress_config_json,
        compress_ps_output=lambda text: _compress_env_ps(text, "ps_output"),
        compress_env_output=lambda text: _compress_env_ps(text, "env_output"),
    )
    compressor = registry.get(kind)
    compressed = compressor(content) if compressor else content

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
                raw,
                compression_level=compression_level,
                tool_context=ctx,
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
    sys_prompt = body.get("system", "")
    if not tool_compatible and sys_prompt:
        if isinstance(sys_prompt, str):
            if "[Tok File Freshness System]" not in sys_prompt:
                sys_prompt = (
                    sys_prompt + "\n\n" + TOK_FRESHNESS_SIGNALS_EXPLANATION
                )
                body["system"] = sys_prompt
        elif isinstance(sys_prompt, list):
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
