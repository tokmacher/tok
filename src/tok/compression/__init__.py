"""Input-side compression: compresses old message history into a Tok rolling state."""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import threading
import time as time_module
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from tok.utils.event_logging import log_delta_compress

if TYPE_CHECKING:
    from collections.abc import MutableMapping

__all__ = [
    "TOOL_COMPRESS_THRESHOLD",
    "compress_user_prompt",
]

# Threshold in characters for when to compress tool results
# Set to 0 to always compress if we have a strategy
TOOL_COMPRESS_THRESHOLD = 0

logger = logging.getLogger("tok.compression")

_HARNESS_INJECTION_RE = re.compile(r"\n?<system-reminder>.*?</system-reminder>\n?", re.DOTALL)


def _strip_harness_injections(text: str) -> str:
    """Remove harness-injected <system-reminder> blocks before hashing or caching.

    Consumes the optional newline immediately before and after the tag so removal
    doesn't leave stray blank lines.  If no reminder is present the text is
    returned unchanged so callers that pass clean content see no side effects.
    """
    cleaned = _HARNESS_INJECTION_RE.sub("", text)
    return text if cleaned == text else cleaned


@dataclass(frozen=True)
class CutEligibility:
    eligible: bool
    reason: str


# Type alias for result cache entries (supports legacy formats)
# Format: (content_hash, raw_content, timestamp) or legacy 2-tuple/1-tuple
ResultCacheEntry: TypeAlias = dict[str, object] | tuple[str, str, float] | tuple[str, str] | tuple[str]

# Maximum number of entries in the result cache to bound memory usage
RESULT_CACHE_MAX_SIZE = 256

# Lock for thread-safe cache operations (callers must use this for external synchronization)
_result_cache_lock = threading.Lock()

_CUT_REJECTION_REASONS = frozenset({"non_user", "top_level_tool_result", "user_contains_tool_result_block"})


def classify_cut_eligibility(msg: dict[str, Any]) -> CutEligibility:
    if msg.get("role") != "user":
        return CutEligibility(False, "non_user")
    if msg.get("tool_use_id"):
        return CutEligibility(False, "top_level_tool_result")
    content = msg.get("content", "")
    if isinstance(content, str):
        return CutEligibility(True, "eligible")
    if isinstance(content, list):
        if not content:
            return CutEligibility(True, "eligible")
        all_tool_results = all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
        has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
        if all_tool_results:
            return CutEligibility(True, "eligible")
        if has_tool_result:
            return CutEligibility(False, "user_contains_tool_result_block")
        return CutEligibility(True, "eligible")
    return CutEligibility(True, "eligible")


FILE_LIKE_TOOLS = frozenset(
    {
        "view",
        "view_file",
        "read",
        "read_file",
        "cat",
        "open_file",
        "get_file",
    }
)

COMMAND_LIKE_TOOLS = frozenset(
    {
        "bash",
        "run_terminal",
        "run",
        "shell",
        "sh",
        "zsh",
        "bash_script",
        "execute_command",
        "cmd",
        "terminal",
        "exec",
    }
)

EDIT_LIKE_TOOLS = frozenset(
    {
        "edit",
        "write",
        "edit_file",
        "write_file",
        "apply_patch",
        "str_replace_based_edit_tool",
    }
)

_ERROR_EQUIVALENCE_PATTERNS = [
    (
        re.compile(r"no such file|file not found|does not exist|enoent", re.IGNORECASE),
        "enoent",
    ),
    (
        re.compile(r"permission denied|access denied|eacces", re.IGNORECASE),
        "eacces",
    ),
    (
        re.compile(r"not found|cannot find|could not find", re.IGNORECASE),
        "not_found",
    ),
    (
        re.compile(r"regex.*error|error.*regex|invalid regex|bad regex", re.IGNORECASE),
        "regex_error",
    ),
    (
        re.compile(
            r"importerror|modulenotfounderror|no module named|import error",
            re.IGNORECASE,
        ),
        "import_error",
    ),
    (
        re.compile(
            r"syntaxerror|\bsyntax\s+error\b|\bparse\s+error\b|\bparse-error\b",
            re.IGNORECASE,
        ),
        "syntax_error",
    ),
    (
        re.compile(
            r"attributeerror|has no attribute|no attribute|attribute error",
            re.IGNORECASE,
        ),
        "attr_error",
    ),
    (
        re.compile(r"typeerror|incompatible type|type error", re.IGNORECASE),
        "type_error",
    ),
    (
        re.compile(r"valueerror|invalid value|value error", re.IGNORECASE),
        "value_error",
    ),
    (
        re.compile(r"timeout|timed out|deadline exceeded", re.IGNORECASE),
        "timeout",
    ),
    (
        re.compile(
            r"connection.*refused|connection.*reset|network.*error",
            re.IGNORECASE,
        ),
        "network_error",
    ),
    (
        re.compile(r"command failed|exit code [0-9]+|non-zero exit", re.IGNORECASE),
        "command_failed",
    ),
    (
        re.compile(r"already exists|file exists", re.IGNORECASE),
        "already_exists",
    ),
    (
        re.compile(r"empty|zero bytes|no such", re.IGNORECASE),
        "empty_or_missing",
    ),
]


def _normalize_error_content(raw: str) -> str | None:
    """
    Extract canonical error type from raw error content.

    Returns normalized error string like '|err:enoent|' or None if no pattern matches.
    """
    for pattern, error_type in _ERROR_EQUIVALENCE_PATTERNS:
        if pattern.search(raw):
            return f"|err:{error_type}|"
    return None


def _error_stub(tool: str, reason: str, fallback: str = "verbatim") -> str:
    """Emit a structured, readable error stub. Never swallow failures silently."""
    return f">>> tool:{tool}|status:error|reason:{reason}|fallback:{fallback}"


def _compute_confidence(
    content_hash: str | None,
    cached_hash: str | None,
    is_heuristic: bool = False,
) -> tuple[str, str]:
    """Return (confidence_level, reason) for stub metadata.

    Levels:
    - high: hash match confirmed
    - medium: hash available but no cached reference
    - low: heuristic applied (size/age threshold)
    """
    if content_hash and cached_hash and content_hash == cached_hash:
        return "high", "hash-match"
    if content_hash:
        return "medium", "hash-available"
    if is_heuristic:
        return "low", "heuristic"
    return "low", "default"


_SOURCE_EVIDENCE_LINE_RE = re.compile(
    r"(?m)^\s*(?:\.?/)?[A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|go|rs|rb|java|c|cc|cpp|h):\d+(?::|\s|$)"
)
_SOURCE_EVIDENCE_SEARCH_TOOLS = frozenset(
    {
        "grep",
        "grep_search",
        "search",
        "search_files",
        "ripgrep",
        "rg",
    }
)


def _looks_like_source_evidence(raw: str) -> bool:
    if not raw.strip():
        return False
    return bool(_SOURCE_EVIDENCE_LINE_RE.search(raw))


def _should_preserve_source_evidence_for_error_stub(
    raw_text: str,
    *,
    tool_name: str | None,
    context: dict[str, Any] | None = None,
) -> bool:
    normalized_tool_name = str(tool_name or "").lower().strip()
    if not normalized_tool_name and isinstance(context, dict):
        normalized_tool_name = str(context.get("name", "")).lower().strip()
    # Always preserve search tool output - never convert it to error stubs
    # even if it contains error-like patterns (e.g., "ValueError" in source code)
    if normalized_tool_name in _SOURCE_EVIDENCE_SEARCH_TOOLS:
        return True
    return _looks_like_source_evidence(raw_text)


CANONICAL_MEMORY_FIELDS = (
    "turns",
    "goal",
    "files",
    "edited",
    "cmds",
    "tests",
    "errs",
    "constraints",
    "next",
)

TOK_FIELD_ALIAS = {
    "turns": "t",
    "goal": "g",
    "next": "n",
    "files": "f",
    "cmds": "c",
    "errs": "e",
    "tests": "s",
    "blockers": "b",
    "facts": "x",
    "questions": "q",
    "constraints": "k",
    "episodes": "p",
    "edited": "i",
}

TOK_REVERSE_ALIAS = {v: k for k, v in TOK_FIELD_ALIAS.items()}

STOP_WORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "this",
        "that",
        "with",
        "have",
        "from",
        "are",
        "was",
        "not",
        "but",
        "they",
        "you",
        "what",
        "can",
        "will",
        "your",
        "just",
        "use",
        "also",
        "more",
        "its",
        "their",
    ]
)

QUESTION_PREFIXES = (
    "why ",
    "what ",
    "which ",
    "who ",
    "where ",
    "when ",
    "how ",
    "is ",
    "are ",
    "should ",
    "can ",
    "could ",
    "would ",
    "do ",
    "does ",
    "did ",
)

TOK_PROTOCOL_LAW = "[Tok law] No JSON, no prose. Use @Tool name=: body. SNAP: SNAP\n"

TOK_OUTPUT_DIRECTIVE = """\
[Tok Mode] Natural responses allowed. No special formatting required.
Reply normally using plain text. Use tools naturally when needed.
"""

TOK_TOOL_COMPAT_DIRECTIVE = "Plain text. Tool calls only. Omit all headers.\n"
TOK_TOOL_COMPAT_ANSWER_ONLY_DIRECTIVE = (
    "Plain text. Answer-only turn. Do not call tools. Emit only the requested labeled answer fields.\n"
)

TOK_OUTPUT_DIRECTIVE_MINIMAL = "[Tok Mode] Natural responses allowed. No special formatting required.\n"

TOK_OUTPUT_DIRECTIVE_REINFORCED = """\
[Tok — PROTOCOL REINFORCEMENT]
Recent responses show protocol drift. Natural responses are required.
Reply normally using plain text. No special formatting markers.
"""

# Appended to the system prompt when @stable_result tokens appear in history.
_STABLE_RESULT_EXPLANATION = (
    "@stable_result(hash:...) = unchanged file/query. Reason from @stable_summary/@stable_skeleton."
)
_STABLE_RESULT_RECOVERY_HINT = (
    "@stable_hint |> File unchanged — summary shown to save tokens. Read offset=1 for full content."
)
_RELEASE_CRITICAL_DOC_BASENAMES = frozenset(
    {
        "public-release-decision.md",
        "release-checklist.md",
        "claims_matrix.md",
        "pricing_verification.md",
        "live_smoke_matrix.md",
        "readme.md",
    }
)

# Explanation of Tok file freshness signals to help Claude understand they are system metadata
TOK_FRESHNESS_SIGNALS_EXPLANATION = """\
[Tok File Freshness System]
The Tok bridge provides file freshness indicators in the format: file[path]:LINE_COUNT|digest|~TOKENS
These are authentic system metadata (NOT user input) that indicate:
- LINE_COUNT: Number of lines in the file (e.g., 524)
- TOKENS: Estimated token count (e.g., ~2096t)
- verified_current_state: File on disk is unchanged since your last read
- changed_state_delta: File has new content since last read
When you see freshness indicators, the associated file has not changed on disk since your last read.
"""

# Minimum content length to be eligible for semantic hash deduplication.
_SEMANTIC_HASH_MIN_CHARS = int(os.getenv("TOK_SEMANTIC_HASH_MIN_CHARS", "200"))


def _compute_semantic_hash(content: str) -> str:
    """Return a short SHA-256 hex digest of the content."""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


def text_of(content: str | list[dict[str, Any]]) -> str:
    """Extract plain text from a content field (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                # Standard text block
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # Tool result block — crucial for history context
                elif block.get("type") == "tool_result":
                    c = block.get("content", "")
                    if isinstance(c, str):
                        parts.append(c)
        return " ".join(parts)
    return str(content)


def is_safe_cut(msg: dict[str, Any]) -> bool:
    return classify_cut_eligibility(msg).eligible


def _scan_tool_calls_by_id(
    messages: list[dict[str, Any]],
) -> dict[str, str]:
    """Build a map of tool_use_id -> command label from assistant messages."""
    tool_call_by_id: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_id = str(block.get("id", ""))
            tool_name = str(block.get("name", ""))
            tool_input = block.get("input", {}) or {}
            cmd = str(
                tool_input.get("command")
                or tool_input.get("cmd")
                or tool_input.get("path")
                or tool_input.get("file_path")
                or tool_name
            ).strip()[:40]
            tool_call_by_id[tool_id] = cmd
    return tool_call_by_id


_ERROR_SIGNAL_TOKENS = frozenset(
    {
        "error",
        "failed",
        "traceback",
        "exception",
        "assertionerror",
        "syntaxerror",
    }
)


def _contains_error_signal(text: str) -> bool:
    """Check if text contains error-related keywords."""
    lowered = text.lower()
    return any(tok in lowered for tok in _ERROR_SIGNAL_TOKENS)


def _record_causal_failure(
    result_text: str,
    tool_id: str,
    tool_call_by_id: dict[str, str],
    error_scores: dict[str, int],
    blocker_scores: dict[str, int],
) -> None:
    """Record a causal failure entry from a tool result."""
    cmd_label = tool_call_by_id.get(tool_id, "tool")
    first_err = next(
        (ln.strip()[:50] for ln in result_text.splitlines() if ln.strip()),
        "",
    )
    causal = re.sub(r"\s+", "_", f"{cmd_label}\u2192{first_err}")[:60]
    blocker_scores[causal] = blocker_scores.get(causal, 0) + 3
    error_scores[causal] = error_scores.get(causal, 0) + 3


def _summarize_causal_failures(
    messages: list[dict[str, Any]],
    error_scores: dict[str, int],
    blocker_scores: dict[str, int],
) -> None:
    """
    Augment error/blocker scores with causal context from tool_result pairs.

    For each tool_use/tool_result pair whose result contains an error signal,
    records "cmd→error" so errs/blockers reflect *why* the failure happened.
    """
    tool_call_by_id = _scan_tool_calls_by_id(messages)

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_id = str(block.get("tool_use_id", ""))
            raw = block.get("content", "")
            result_text = raw if isinstance(raw, str) else text_of(raw)
            if not _contains_error_signal(result_text):
                continue
            _record_causal_failure(
                result_text,
                tool_id,
                tool_call_by_id,
                error_scores,
                blocker_scores,
            )


def _summarize_decision_hypotheses(
    messages: list[dict[str, Any]],
    next_scores: dict[str, int],
    _question_scores: dict[str, int],
) -> None:
    """
    Augment next/questions scores with decision+rationale snippets.

    Looks for assistant lines that state a reason for the planned action
    (e.g., "next step because …") and promotes them in next_scores.
    """
    rationale_triggers = (
        "because ",
        "so that ",
        "in order to ",
        "reason:",
        "rationale:",
    )
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        msg_text = text_of(msg.get("content", ""))
        for line in msg_text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if any(lowered.startswith(p) for p in ("next", "i will", "i'll", "plan", "then")):
                for trigger in rationale_triggers:
                    if trigger in lowered:
                        snippet = re.sub(r"\s+", "_", stripped[:60])
                        next_scores[snippet] = next_scores.get(snippet, 0) + 3
                        break


def compress_history(
    messages: list[dict[str, Any]],
    keep_turns: int = 2,
    profile: dict[str, int | list[str]] | None = None,
    prune_tool_results: bool = False,
) -> tuple[list[dict[str, Any]], str, set[str]]:
    """Split messages into old (to compress) + recent (to keep verbatim)."""
    from ._pipeline import compress_history_impl

    return compress_history_impl(
        messages,
        keep_turns=keep_turns,
        profile=profile,
        prune_tool_results=prune_tool_results,
    )


def _detect_tool_content_type(text: str) -> str:
    """Detect the content type of a tool result."""
    from ._pipeline import _detect_tool_content_type_impl

    return _detect_tool_content_type_impl(text)


def _compress_git_log(text: str) -> str:
    """Compress verbose git log to compact table."""
    from ._pipeline import _compress_git_log_impl

    return _compress_git_log_impl(text)


def _compress_git_diff(text: str) -> str:
    """Compress git diff output while preserving the public facade."""
    from ._pipeline import _compress_git_diff

    return _compress_git_diff(text)


def _compress_ls(text: str) -> str:
    """Compress ls-style output while preserving the public facade."""
    from ._pipeline import _compress_ls

    return _compress_ls(text)


def _compress_install(text: str) -> str:
    """Compress install output while preserving the public facade."""
    from ._pipeline import _compress_install

    return _compress_install(text)


def truncate_large_result(text: str, limit: int = 1200, *, already_compressed: bool = False) -> str:
    """Head-tail truncation for extremely large results to prevent context flooding."""
    from ._pipeline import truncate_large_result as _truncate_large_result

    return _truncate_large_result(text, limit=limit, already_compressed=already_compressed)


def tok_tool_result(
    content: str,
    compression_level: str = "balanced",
    tool_context: dict[str, Any] | None = None,
    session: Any | None = None,
) -> str:
    """Convert large tool result to dense tok representation."""
    from ._pipeline import tok_tool_result_impl

    return tok_tool_result_impl(
        content,
        compression_level=compression_level,
        tool_context=tool_context,
        session=session,
    )


def _apply_result_cache(
    raw: str,
    context: dict[str, Any],
    result_cache: MutableMapping[str, ResultCacheEntry],
    compression_level: str = "balanced",
    bypass_cache: bool = False,
    ttl_seconds: int = 1800,
    preserve_exact_search_evidence: bool = False,
) -> tuple[str, int]:
    """
    Apply general result cache dedup for any tool result.

    Returns (compressed_text, chars_saved).
    """
    tool_name = context.get("name")
    normalized_tool_name = str(tool_name or "").lower()
    args = context.get("args") if isinstance(context.get("args"), dict) else {}
    is_precision_read = (
        normalized_tool_name in FILE_LIKE_TOOLS
        and isinstance(args, dict)
        and any(k in args for k in ("offset", "limit", "start", "end"))
    )
    is_file_like = normalized_tool_name in FILE_LIKE_TOOLS

    if bypass_cache or normalized_tool_name in COMMAND_LIKE_TOOLS:
        compressed = tok_tool_result(
            raw,
            compression_level=compression_level,
            tool_context=context,
        )
        return compressed, len(raw) - len(compressed)

    cache_key = _build_cache_key(tool_name, context)
    raw_text = _strip_harness_injections(str(raw or ""))
    raw = raw_text  # use stripped content for both hashing and storage
    cached_entry = result_cache.get(cache_key)

    if cached_entry is None:
        logger.debug("result_cache_miss: key=%s tool=%s", cache_key, tool_name)
        return _store_cache_entry(
            result_cache,
            cache_key,
            raw_text,
            raw,
            is_file_like,
            context,
            tool_name,
            compression_level,
            prefer_normalized_error=True,
        )

    cached_hash, cached_raw, timestamp, first_read_complete, entry_length = _unpack_cache_entry(cached_entry)

    # Check staleness for entries with timestamps (3-tuple) or legacy entries (1/2-tuple)
    if _is_cache_entry_stale(timestamp, ttl_seconds):
        return _store_cache_entry(
            result_cache,
            cache_key,
            raw_text,
            raw,
            is_file_like,
            context,
            tool_name,
            compression_level,
            prefer_normalized_error=False,
        )

    if _is_file_mtime_changed(context, timestamp):
        return _store_cache_entry(
            result_cache,
            cache_key,
            raw_text,
            raw,
            is_file_like,
            context,
            tool_name,
            compression_level,
            prefer_normalized_error=False,
        )

    cached_raw_text = str(cached_raw or "")
    logger.debug("result_cache_hit: key=%s tool=%s", cache_key, tool_name)
    return _process_cache_hit(
        raw,
        raw_text,
        context,
        tool_name,
        normalized_tool_name,
        is_precision_read,
        is_file_like,
        result_cache,
        cache_key,
        entry_length,
        cached_hash,
        cached_raw,
        cached_raw_text,
        compression_level,
        preserve_exact_search_evidence=preserve_exact_search_evidence,
        first_read_complete=first_read_complete,
    )


def _context_path(context: dict[str, Any]) -> str:
    raw_args = context.get("args")
    raw_path = (
        context.get("path")
        or (raw_args.get("path") if isinstance(raw_args, dict) else None)
        or (raw_args.get("file_path") if isinstance(raw_args, dict) else None)
        or (raw_args.get("AbsolutePath") if isinstance(raw_args, dict) else None)
        or (raw_args.get("TargetFile") if isinstance(raw_args, dict) else None)
    )
    return str(raw_path or "").strip()


def _is_release_critical_doc_context(context: dict[str, Any]) -> bool:
    path = _context_path(context)
    if not path:
        return False
    normalized = path.replace("\\", "/").lower()
    basename = normalized.rsplit("/", maxsplit=1)[-1]
    if basename not in _RELEASE_CRITICAL_DOC_BASENAMES:
        return False
    return "/docs/" in normalized or normalized.startswith("docs/") or basename == "readme.md"


def _process_cache_hit(
    raw: str,
    raw_text: str,
    context: dict[str, Any],
    tool_name: str | None,
    normalized_tool_name: str,
    is_precision_read: bool,
    is_file_like: bool,
    result_cache: MutableMapping[str, ResultCacheEntry],
    cache_key: str,
    entry_length: int,
    cached_hash: str,
    cached_raw: str,
    cached_raw_text: str,
    compression_level: str,
    preserve_exact_search_evidence: bool = False,
    first_read_complete: bool = True,
) -> tuple[str, int]:
    """Process a cache hit and return compressed result."""
    host_stub_replayed = _should_replay_host_stub(
        is_file_like,
        cached_raw_text,
        raw_text,
        raw_text,
    )
    if is_file_like and _is_release_critical_doc_context(context):
        if host_stub_replayed and cached_raw:
            return cached_raw, 0
        return raw_text, 0
    if host_stub_replayed:
        _update_cache_after_hit(
            result_cache,
            cache_key,
            host_stub_replayed=True,
            entry_length=entry_length,
            cached_hash=cached_hash,
            cached_raw=cached_raw,
            content_hash=cached_hash,
            raw=cached_raw,
        )
        return _serve_cached_content_hash_match(
            raw,
            raw_text,
            context,
            tool_name,
            normalized_tool_name,
            is_precision_read,
            is_file_like,
            result_cache,
            cache_key,
            entry_length,
            cached_hash,
            cached_raw,
            host_stub_replayed=True,
            _compression_level=compression_level,
            preserve_exact_search_evidence=preserve_exact_search_evidence,
            first_read_complete=first_read_complete,
        )

    if _is_content_hash_match(raw_text, cached_raw_text):
        _update_cache_after_hit(
            result_cache,
            cache_key,
            host_stub_replayed=False,
            entry_length=entry_length,
            cached_hash=cached_hash,
            cached_raw=cached_raw,
            content_hash=cached_hash,
            raw=raw,
        )
        return _serve_cached_content_hash_match(
            raw,
            raw_text,
            context,
            tool_name,
            normalized_tool_name,
            is_precision_read,
            is_file_like,
            result_cache,
            cache_key,
            entry_length,
            cached_hash,
            cached_raw,
            host_stub_replayed=False,
            _compression_level=compression_level,
            preserve_exact_search_evidence=preserve_exact_search_evidence,
            first_read_complete=first_read_complete,
        )

    # Content changed - need to compute diff
    diff_lines = _compute_diff_lines(cached_raw_text, raw_text)
    if diff_lines is None or len(diff_lines) > 1000:
        return _handle_diff_result(
            raw_text,
            cached_raw_text,
            hashlib.sha256(raw_text.encode()).hexdigest()[:8],
            cached_hash,
            _normalize_error_content(raw_text),
            tool_name,
            compression_level,
            context,
        )
    return _build_diff_result(
        raw_text,
        tool_name,
        normalized_tool_name,
        diff_lines,
    )


def _build_cache_key(tool_name: str | None, context: dict[str, Any]) -> str:
    serialized_args = context.get("args")
    if isinstance(serialized_args, dict):
        from tok.runtime.repeat_targets import normalize_path_target

        path_key = next(
            (k for k in ("path", "file_path", "AbsolutePath", "TargetFile") if k in serialized_args),
            None,
        )
        if path_key:
            serialized_args = {**serialized_args, path_key: normalize_path_target(str(serialized_args[path_key]))}
    args_str = json.dumps(serialized_args, sort_keys=True, default=str)
    raw_cache_key = f"{tool_name or ''}:{args_str}"
    return hashlib.sha256(raw_cache_key.encode()).hexdigest()[:12]


def _store_cache_entry(
    result_cache: MutableMapping[str, ResultCacheEntry],
    cache_key: str,
    raw_text: str,
    raw: str,
    is_file_like: bool,
    context: dict[str, Any],
    tool_name: str | None,
    compression_level: str,
    prefer_normalized_error: bool,
) -> tuple[str, int]:
    normalized_error = _normalize_error_content(raw_text)
    content_hash = hashlib.sha256(raw_text.encode()).hexdigest()[:8]
    with _result_cache_lock:
        result_cache[cache_key] = {
            "hash": content_hash,
            "raw": raw,
            "timestamp": time_module.time(),
            "first_read_complete": False,
        }
        while len(result_cache) > RESULT_CACHE_MAX_SIZE:
            try:
                oldest = next(iter(result_cache))
                logger.debug("result_cache_evict: key=%s size=%d", oldest, len(result_cache))
                del result_cache[oldest]
            except StopIteration:
                break
    if is_file_like:
        return raw, 0
    if (
        prefer_normalized_error
        and normalized_error
        and not _should_preserve_source_evidence_for_error_stub(raw_text, tool_name=tool_name, context=context)
    ):
        return normalized_error, len(raw_text) - len(normalized_error)
    compressed = tok_tool_result(
        raw_text,
        compression_level=compression_level,
        tool_context=context,
    )
    return compressed, len(raw_text) - len(compressed)


def _unpack_cache_entry(
    entry: tuple[Any, ...] | dict[str, Any],
) -> tuple[str, str, float | None, bool, int]:
    """Unpack cache entry, handling both old tuple format and new dict format."""
    # Handle new dict format
    if isinstance(entry, dict):
        return (
            str(entry.get("hash", "")),
            str(entry.get("raw", "")),
            cast("float | None", entry.get("timestamp")),
            bool(entry.get("first_read_complete", False)),  # Default False for safety (first-read guarantee)
            4,  # Dict format version
        )
    # Handle legacy tuple format (backwards compatibility)
    entry_length = len(entry)
    if entry_length >= 3:
        return (
            str(entry[0]),
            str(entry[1]),
            cast("float | None", entry[2]),
            False,  # Legacy entries must require a verbatim first read
            entry_length,
        )
    if entry_length == 2:
        return str(entry[0]), str(entry[1]), None, False, entry_length
    if entry_length == 1:
        return str(entry[0]), "", None, True, entry_length
    return "", "", None, True, entry_length


def _is_cache_entry_stale(timestamp: float | None, ttl_seconds: int) -> bool:
    if timestamp is None:
        return True
    return time_module.time() - timestamp > ttl_seconds


def _is_file_mtime_changed(context: dict[str, Any], cached_timestamp: float | None) -> bool:
    """Check if a file's mtime changed since the cache entry was stored."""
    if cached_timestamp is None:
        return False
    args = context.get("args")
    if not isinstance(args, dict):
        return False
    tool_name = str(context.get("name", "")).lower()
    if tool_name not in FILE_LIKE_TOOLS:
        return False
    path = str(args.get("path") or args.get("file_path") or args.get("AbsolutePath") or args.get("TargetFile") or "")
    if not path:
        return False
    try:
        mtime = os.path.getmtime(path)
        return mtime > cached_timestamp
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, OSError) and path and (path.startswith("/") or "/" in path):
            logger.debug("result_cache_mtime_check_failed: path=%s error=%s", path, exc)
        return False


def _is_content_hash_match(text_a: str, text_b: str) -> bool:
    """Check if two texts have identical content hashes."""
    if not text_a and not text_b:
        return True
    if not text_a or not text_b:
        return False
    hash_a = hashlib.sha256(text_a.encode()).hexdigest()[:16]
    hash_b = hashlib.sha256(text_b.encode()).hexdigest()[:16]
    return hash_a == hash_b


def _update_cache_after_hit(
    result_cache: MutableMapping[str, ResultCacheEntry],
    cache_key: str,
    host_stub_replayed: bool,
    entry_length: int,
    cached_hash: str,
    cached_raw: str,
    content_hash: str,
    raw: str,
) -> None:
    """Update the cache entry after a cache hit."""
    with _result_cache_lock:
        if host_stub_replayed and entry_length == 3:
            result_cache[cache_key] = {
                "hash": cached_hash,
                "raw": cached_raw,
                "timestamp": time_module.time(),
                "first_read_complete": True,
            }
        else:
            result_cache[cache_key] = {
                "hash": content_hash,
                "raw": raw,
                "timestamp": time_module.time(),
                "first_read_complete": True,
            }


def _count_changed_lines(diff_lines: list[str]) -> int:
    """Count the number of changed lines in a unified diff."""
    return sum(1 for line in diff_lines if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))


def _compute_diff_lines(old_text: str, new_text: str) -> list[str] | None:
    """Compute unified diff between two texts."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old",
            tofile="new",
            lineterm="",
        )
    )
    return diff if diff else None


def _should_strip_diff_whitespace(normalized_tool_name: str) -> bool:
    """Check if diff whitespace should be stripped for this tool type."""
    return normalized_tool_name in (
        "view_file",
        "read_file",
        "cat",
        "bash",
        "run_terminal",
        "computer",
        "sh",
        "edit_file",
        "write_file",
    )


def _handle_hash_mismatch_result(
    raw_text: str,
    _cached_raw_text: str,
    _content_hash: str,
    _cached_hash: str,
    normalized_error: str | None,
    tool_name: str | None,
    compression_level: str,
    context: dict[str, Any],
) -> tuple[str, int]:
    """Handle the case when content hash doesn't match (content changed)."""
    if normalized_error and not _should_preserve_source_evidence_for_error_stub(
        raw_text,
        tool_name=tool_name,
        context=context,
    ):
        error_type = normalized_error[5:-1] if normalized_error else "unknown"
        stub = _error_stub(tool=str(tool_name or "unknown"), reason=error_type)
        saved = len(raw_text) - len(stub)
        if saved < 0:
            return _error_stub(tool=str(tool_name or "unknown"), reason=error_type) + "\n", 0
        return stub + "\n", saved
    compressed = tok_tool_result(
        raw_text,
        compression_level=compression_level,
        tool_context=context,
    )
    confidence, reason = _compute_confidence(None, None, is_heuristic=True)
    header = f">>> tool:{tool_name}|delta|changed|confidence:{confidence}|reason:{reason}\n"
    return header + compressed, len(raw_text) - (len(header) + len(compressed))


def _handle_diff_result(
    raw_text: str,
    _cached_raw_text: str,
    _content_hash: str,
    _cached_hash: str,
    normalized_error: str | None,
    tool_name: str | None,
    compression_level: str,
    context: dict[str, Any],
) -> tuple[str, int]:
    """Handle the case when diff is empty or too large."""
    # Content differs but diff is empty (edge case with trailing newlines)
    if normalized_error and not _should_preserve_source_evidence_for_error_stub(
        raw_text,
        tool_name=tool_name,
        context=context,
    ):
        error_type = normalized_error[5:-1] if normalized_error else "unknown"
        stub = _error_stub(tool=str(tool_name or "unknown"), reason=error_type)
        saved = len(raw_text) - len(stub)
        if saved < 0:
            return _error_stub(tool=str(tool_name or "unknown"), reason=error_type) + "\n", 0
        return stub + "\n", saved
    compressed = tok_tool_result(
        raw_text,
        compression_level=compression_level,
        tool_context=context,
    )
    confidence, reason = _compute_confidence(None, None, is_heuristic=True)
    header = f">>> tool:{tool_name}|delta|changed|confidence:{confidence}|reason:{reason}\n"
    return header + compressed, len(raw_text) - (len(header) + len(compressed))


def _build_diff_result(
    raw_text: str,
    tool_name: str | None,
    normalized_tool_name: str,
    diff_lines: list[str],
) -> tuple[str, int]:
    """Build the result with diff output."""
    changed_lines = _count_changed_lines(diff_lines)
    confidence, reason = _compute_confidence(None, None, is_heuristic=True)
    header = f">>> tool:{tool_name}|delta|changed_lines:{changed_lines}|confidence:{confidence}|reason:{reason}"
    diff_text = "".join(diff_lines)
    if _should_strip_diff_whitespace(normalized_tool_name):
        stripped_diff = [line for line in diff_lines if not line.startswith(" ")]
        diff_text = "".join(stripped_diff)
    result = header + "\n" + diff_text
    saved = len(raw_text) - len(result)
    if saved > 0:
        log_delta_compress(str(tool_name), len(raw_text), len(result))
    return result, saved


def _should_replay_host_stub(
    is_file_like: bool,
    cached_raw_text: str,
    stub_text: str,
    raw_text: str,
) -> bool:
    if not is_file_like or not cached_raw_text:
        return False
    if not stub_text and cached_raw_text:
        return True
    if "unchanged since last read" in stub_text.lower():
        return True
    if len(raw_text) < 80 and len(cached_raw_text) > 200:
        return _is_content_hash_match(raw_text, cached_raw_text)
    return False


def _serve_cached_content_hash_match(
    raw: str,
    raw_text: str,
    context: dict[str, Any],
    tool_name: str | None,
    normalized_tool_name: str,
    is_precision_read: bool,
    _is_file_like: bool,
    result_cache: MutableMapping[str, ResultCacheEntry],
    cache_key: str,
    _entry_length: int,
    cached_hash: str,
    cached_raw: str,
    host_stub_replayed: bool,
    _compression_level: str,
    preserve_exact_search_evidence: bool = False,
    first_read_complete: bool = True,
) -> tuple[str, int]:
    from tok.runtime.repeat_targets import SEARCH_LIKE_TOOLS

    current_time = time_module.time()
    if is_precision_read:
        content_to_return = cached_raw if host_stub_replayed else raw
        with _result_cache_lock:
            result_cache[cache_key] = {
                "hash": cached_hash,
                "raw": content_to_return,
                "timestamp": current_time,
                "first_read_complete": True,
            }
        return content_to_return, 0

    if normalized_tool_name in FILE_LIKE_TOOLS:
        if not first_read_complete:
            with _result_cache_lock:
                result_cache[cache_key] = {
                    "hash": cached_hash,
                    "raw": cached_raw,
                    "timestamp": current_time,
                    "first_read_complete": True,
                }
            return (cached_raw if host_stub_replayed else raw_text), 0
        confidence, reason = _compute_confidence(cached_hash, cached_hash)
        raw_path = _context_path(context)
        stub_parts = [f">>> tool:{tool_name}|unchanged|cached|confidence:{confidence}|reason:{reason}"]
        if raw_path:
            stub_parts[0] += f"|path:{raw_path}"
        stub_parts.append(_STABLE_RESULT_RECOVERY_HINT)
        if len(cached_raw) >= 100:
            try:
                from tok.runtime.repeat_targets import build_file_skeleton

                skeleton = build_file_skeleton(cached_raw, max_chars=280, max_lines=14)
                summary = " ".join(cached_raw.splitlines()[:3])
                if len(summary) > 280:
                    summary = summary[:279] + "…"
                if skeleton:
                    stub_parts[0] += "|fidelity:summary|lossy:true"
                    stub_parts.append(f"@stable_summary |> {summary}")
                    stub_parts.append(f"@stable_skeleton |> {skeleton}")
            except Exception:
                logger.debug("Failed to build file skeleton", exc_info=True)
        stub = "\n".join(stub_parts)
        if len(stub) >= len(raw_text):
            return raw_text, 0
        return stub, len(raw_text) - len(stub)

    # Handle search-like tools: compress repeat searches instead of bare stub
    if normalized_tool_name in SEARCH_LIKE_TOOLS:
        from tok.runtime.repeat_targets import search_result_evidence_level

        evidence = search_result_evidence_level(cached_raw)
        if evidence == "exact_content" and first_read_complete:
            return raw_text, 0
        from ._tool_result_codecs import _compress_grep

        compressed_search = _compress_grep(raw_text)
        if len(compressed_search) < len(raw_text):
            confidence, reason = _compute_confidence(cached_hash, cached_hash)
            return (
                f">>> tool:{tool_name}|matches_compressed|cached|confidence:{confidence}|reason:{reason}\n"
                + compressed_search,
                len(raw_text) - len(compressed_search),
            )

    confidence, reason = _compute_confidence(cached_hash, cached_hash)
    stub = f">>> tool:{tool_name}|unchanged|cached|confidence:{confidence}|reason:{reason}"
    raw_path = _context_path(context)
    if raw_path:
        stub += f"|path:{raw_path}"
    return stub, len(raw_text) - len(stub)


def _apply_file_cache(
    raw: str,
    path: str,
    file_cache: MutableMapping[str, ResultCacheEntry],
) -> tuple[str, int]:
    """Compatibility wrapper for old file cache tests."""
    context = {"name": "view_file", "path": path, "args": {"path": path}}
    return _apply_result_cache(raw, context, file_cache)


def _make_semantic_cache_key(context: dict[str, Any] | None, _raw: str) -> str | None:
    """Return a stable cache key for (tool_name, args) if context is available."""
    if not context:
        return None
    tool_name = context.get("name", "")
    raw_args = context.get("args") or context.get("path") or ""
    # Return None if context has no meaningful content
    if not tool_name and not raw_args:
        return None
    args_for_hash: dict[str, Any]
    if isinstance(raw_args, dict):
        args_for_hash = {
            k: v
            for k, v in raw_args.items()
            if k
            not in (
                "offset",
                "limit",
                "start",
                "end",
                "AbsolutePath",
                "TargetFile",
                "file_path",
                # Exclude "path" from args so we can include a normalized path
                # separately without volatile formatting differences.
                "path",
            )
        }
    else:
        args_for_hash = {"args": raw_args}

    raw_path = (
        context.get("path")
        or (raw_args.get("path") if isinstance(raw_args, dict) else None)
        or (raw_args.get("file_path") if isinstance(raw_args, dict) else None)
        or (raw_args.get("AbsolutePath") if isinstance(raw_args, dict) else None)
        or (raw_args.get("TargetFile") if isinstance(raw_args, dict) else None)
    )
    raw_query = (
        context.get("query")
        or (raw_args.get("query") if isinstance(raw_args, dict) else None)
        or (raw_args.get("pattern") if isinstance(raw_args, dict) else None)
        or (raw_args.get("search") if isinstance(raw_args, dict) else None)
        or (raw_args.get("text") if isinstance(raw_args, dict) else None)
    )

    normalized_path = ""
    if raw_path:
        try:
            from tok.runtime.repeat_targets import normalize_path_target

            normalized_path = normalize_path_target(str(raw_path))
        except Exception:
            normalized_path = str(raw_path).strip()

    normalized_query = " ".join(str(raw_query or "").split())
    payload = {
        "path": normalized_path,
        "query": normalized_query,
        "args": args_for_hash,
    }
    args_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8", errors="replace")
    ).hexdigest()[:16]
    return f"{tool_name}:{args_hash}"


def compress_tool_results(
    messages: list[dict[str, Any]],
    result_cache: (
        MutableMapping[str, ResultCacheEntry] | dict[str, tuple[str, str, float] | tuple[str, str] | tuple[str]] | None
    ) = None,
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
    model_profile: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Walk messages, apply caching and tok_tool_result() to large tool_result blocks."""
    from ._pipeline import compress_tool_results_impl

    return compress_tool_results_impl(
        messages,
        result_cache=cast("MutableMapping[str, ResultCacheEntry] | None", result_cache),
        tool_use_id_to_context=tool_use_id_to_context,
        compression_level=compression_level,
        semantic_hash_cache=semantic_hash_cache,
        bypass_result_cache=bypass_result_cache,
        hot_summary_records=hot_summary_records,
        session_files_read=session_files_read,
        files_fully_delivered=files_fully_delivered,
        first_exact_evidence_seen=first_exact_evidence_seen,
        current_turn=current_turn,
        keep_turns_window=keep_turns_window,
        preserve_exact_search_evidence=preserve_exact_search_evidence,
        recently_edited_files=recently_edited_files,
        file_heat=file_heat,
        model_profile=model_profile,
    )


def inject_system_additions(
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
    from ._pipeline import inject_system_additions_impl

    return inject_system_additions_impl(
        body,
        tok_state=tok_state,
        tool_compatible=tool_compatible,
        grammar=grammar,
        todo=todo,
        deltas=deltas,
        pressure=pressure,
        runtime_hints=runtime_hints,
        behavior_signals=behavior_signals,
    )


def _should_include_tok_state(tok_state: str | None, *, tool_compatible: bool) -> bool:
    if not tok_state:
        return False
    stripped = tok_state.strip()
    if not stripped:
        return False
    if not tool_compatible:
        return True
    # In native-tool sessions, a bare marker or empty wire state adds protocol noise
    # without preserving any useful working memory.
    if stripped in {">>>", ">>>|"}:
        return False
    if stripped == ">>> ":
        return False
    return stripped.startswith(">>> ") and len(stripped) > len(">>> ")


RECENT_WINDOW_THRESHOLD = 8_000  # chars — compress recent-window results larger than this
RECENT_WINDOW_EVIDENCE_THRESHOLD = 1_200  # chars — file/search-like recent evidence should compress much sooner


def compress_recent_window(
    messages: list[dict[str, Any]],
    tool_use_id_to_context: dict[str, dict[str, Any]] | None = None,
    threshold: int = RECENT_WINDOW_THRESHOLD,
    tool_compatible: bool = False,
    first_exact_evidence_seen: set[str] | None = None,
    preserve_exact_search_evidence: bool = False,
    session_files_read: set[str] | None = None,
    model_profile: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply content-aware compression to tool_result blocks in the recent window."""
    from ._pipeline import compress_recent_window_impl

    return compress_recent_window_impl(
        messages,
        tool_use_id_to_context=tool_use_id_to_context,
        threshold=threshold,
        tool_compatible=tool_compatible,
        first_exact_evidence_seen=first_exact_evidence_seen,
        preserve_exact_search_evidence=preserve_exact_search_evidence,
        session_files_read=session_files_read,
        model_profile=model_profile,
    )


from ._prompt_compression import compress_user_prompt  # noqa: E402
