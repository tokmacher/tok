"""Input-side compression: compresses old message history into a Tok rolling state."""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("tok.compression")


@dataclass(frozen=True)
class CutEligibility:
    eligible: bool
    reason: str


_CUT_REJECTION_REASONS = frozenset(
    {"non_user", "top_level_tool_result", "user_contains_tool_result_block"}
)


def classify_cut_eligibility(msg: dict[str, Any]) -> CutEligibility:
    if msg.get("role") != "user":
        return CutEligibility(False, "non_user")
    if msg.get("tool_use_id"):
        return CutEligibility(False, "top_level_tool_result")
    content = msg.get("content", "")
    if isinstance(content, str):
        return CutEligibility(True, "eligible")
    if isinstance(content, list):
        if any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        ):
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
        re.compile(r"no such file|file not found|does not exist|enoent", re.I),
        "enoent",
    ),
    (re.compile(r"permission denied|access denied|eacces", re.I), "eacces"),
    (re.compile(r"not found|cannot find|could not find", re.I), "not_found"),
    (
        re.compile(r"regex.*error|error.*regex|invalid regex|bad regex", re.I),
        "regex_error",
    ),
    (
        re.compile(
            r"importerror|modulenotfounderror|no module named|import error",
            re.I,
        ),
        "import_error",
    ),
    (
        re.compile(r"syntaxerror|parse.?error|parse error", re.I),
        "syntax_error",
    ),
    (
        re.compile(
            r"attributeerror|has no attribute|no attribute|attribute error",
            re.I,
        ),
        "attr_error",
    ),
    (
        re.compile(r"typeerror|incompatible type|type error", re.I),
        "type_error",
    ),
    (re.compile(r"valueerror|invalid value|value error", re.I), "value_error"),
    (re.compile(r"timeout|timed out|deadline exceeded", re.I), "timeout"),
    (
        re.compile(
            r"connection.*refused|connection.*reset|network.*error", re.I
        ),
        "network_error",
    ),
    (
        re.compile(r"command failed|exit code [0-9]+|non-zero exit", re.I),
        "command_failed",
    ),
    (re.compile(r"already exists|file exists", re.I), "already_exists"),
    (re.compile(r"empty|zero bytes|no such", re.I), "empty_or_missing"),
]


def _normalize_error_content(raw: str) -> str | None:
    """Extract canonical error type from raw error content.

    Returns normalized error string like '|err:enoent|' or None if no pattern matches.
    """
    for pattern, error_type in _ERROR_EQUIVALENCE_PATTERNS:
        if pattern.search(raw):
            return f"|err:{error_type}|"
    return None


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

TOK_PROTOCOL_LAW = (
    "[Tok law] No JSON, no prose. Use @Tool name= |> body. SNAP: |> SNAP\n"
)

TOK_OUTPUT_DIRECTIVE = """\
[Tok Mode] >>> t:N|usr:X|agt:Y|state:Z
@msg role:assistant |> Reply
@Tool name=... |> args...
Turn 2+: Omit header and @msg if mode unchanged. No prose.
"""

TOK_TOOL_COMPAT_DIRECTIVE = (
    "Native tools only. Plain text. Omit all headers.\n"
)

TOK_OUTPUT_DIRECTIVE_MINIMAL = (
    ">>> t:N|usr:X|agt:Y|state:Z\n@msg role:assistant |> Reply\n"
)

TOK_OUTPUT_DIRECTIVE_REINFORCED = """\
[Tok — PROTOCOL REINFORCEMENT]
Recent responses show protocol drift. Strict compliance required:
1. Start every response with the >>> line.
2. Use @msg role:assistant |> for text.
3. Use @Tool name=... |> for tool args.
4. No raw markdown headers. No raw JSON tool blocks.
"""

# Appended to the system prompt when @stable_result tokens appear in history.
_STABLE_RESULT_EXPLANATION = (
    "@stable_result(hash:...) means the tool output is identical to a previous turn."
    " Treat it as: the state is unchanged, no new information."
)

# Minimum content length to be eligible for semantic hash deduplication.
_SEMANTIC_HASH_MIN_CHARS = int(os.getenv("TOK_SEMANTIC_HASH_MIN_CHARS", "200"))


def _compute_semantic_hash(content: str) -> str:
    """Return a short SHA-256 hex digest of the content."""
    return hashlib.sha256(
        content.encode("utf-8", errors="replace")
    ).hexdigest()[:16]


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


def _summarize_causal_failures(
    messages: list[dict[str, Any]],
    error_scores: dict[str, int],
    blocker_scores: dict[str, int],
) -> None:
    """Augment error/blocker scores with causal context from tool_result pairs.

    For each tool_use/tool_result pair whose result contains an error signal,
    records "cmd→error" so errs/blockers reflect *why* the failure happened.
    """
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

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                not isinstance(block, dict)
                or block.get("type") != "tool_result"
            ):
                continue
            tool_id = str(block.get("tool_use_id", ""))
            raw = block.get("content", "")
            result_text = raw if isinstance(raw, str) else text_of(raw)
            lowered = result_text.lower()
            if not any(
                tok in lowered
                for tok in (
                    "error",
                    "failed",
                    "traceback",
                    "exception",
                    "assertionerror",
                    "syntaxerror",
                )
            ):
                continue
            cmd_label = tool_call_by_id.get(tool_id, "tool")
            first_err = next(
                (
                    ln.strip()[:50]
                    for ln in result_text.splitlines()
                    if ln.strip()
                ),
                "",
            )
            causal = re.sub(r"\s+", "_", f"{cmd_label}\u2192{first_err}")[:60]
            blocker_scores[causal] = blocker_scores.get(causal, 0) + 3
            error_scores[causal] = error_scores.get(causal, 0) + 3


def _summarize_decision_hypotheses(
    messages: list[dict[str, Any]],
    next_scores: dict[str, int],
    question_scores: dict[str, int],
) -> None:
    """Augment next/questions scores with decision+rationale snippets.

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
            if any(
                lowered.startswith(p)
                for p in ("next", "i will", "i'll", "plan", "then")
            ):
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
) -> tuple[list[dict[str, Any]], str]:
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


def truncate_large_result(text: str, limit: int = 1200) -> str:
    """Head-tail truncation for extremely large results to prevent context flooding."""
    from ._pipeline import truncate_large_result as _truncate_large_result

    return _truncate_large_result(text, limit=limit)


def tok_tool_result(content: str, compression_level: str = "balanced") -> str:
    """Convert large tool result to dense tok representation."""
    from ._pipeline import tok_tool_result_impl

    return tok_tool_result_impl(content, compression_level=compression_level)


def _apply_result_cache(
    raw: str,
    context: dict[str, Any],
    result_cache: dict[str, tuple[str, str]],
    compression_level: str = "balanced",
) -> tuple[str, int]:
    """Apply general result cache dedup for any tool result.

    Returns (compressed_text, chars_saved).
    """
    tool_name = context.get("name")
    normalized_tool_name = str(tool_name or "").lower()
    args_str = json.dumps(context.get("args"), sort_keys=True)
    raw_cache_key = f"{tool_name}:{args_str}"
    cache_key = hashlib.sha256(raw_cache_key.encode()).hexdigest()[:12]

    normalized_error = _normalize_error_content(raw)
    content_hash = hashlib.sha256(raw.encode()).hexdigest()[:8]

    if cache_key not in result_cache:
        result_cache[cache_key] = (content_hash, raw)
        if len(result_cache) > 256:
            oldest = next(iter(result_cache))
            del result_cache[oldest]
        if normalized_tool_name in FILE_LIKE_TOOLS:
            return raw, 0
        if normalized_error:
            compressed = normalized_error
        else:
            compressed = tok_tool_result(
                raw, compression_level=compression_level
            )
        return compressed, len(raw) - len(compressed)

    cached_hash, cached_raw = result_cache[cache_key]

    if normalized_error:
        cached_normalized = _normalize_error_content(cached_raw)
        if cached_normalized == normalized_error:
            stub = f">>> tool:{tool_name}|err:{normalized_error[5:-1]}|cached"
            return stub, len(raw) - len(stub)

    if content_hash == cached_hash:
        stub = f">>> tool:{tool_name}|unchanged|cached"
        if context.get("path"):
            stub += f"|path:{context['path']}"
        return stub, len(raw) - len(stub)

    old_lines = cached_raw.splitlines(keepends=True)
    new_lines = raw.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(old_lines, new_lines))
    diff_text = "".join(diff_lines)

    result_cache[cache_key] = (content_hash, raw)

    if not diff_lines or len(diff_text) >= 0.7 * len(raw):
        if normalized_error:
            stub = f">>> tool:{tool_name}|delta|err:{normalized_error[5:-1]}\n"
            saved = len(raw) - len(stub)
            if saved < 0:
                return normalized_error + "\n", 0
            return stub, saved
        compressed = tok_tool_result(raw, compression_level=compression_level)
        header = f">>> tool:{tool_name}|delta|changed\n"
        return header + compressed, len(raw) - (len(header) + len(compressed))

    # Emit diff stub
    changed_lines = sum(
        1
        for l in diff_lines
        if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))
    )
    header = f">>> tool:{tool_name}|delta|changed_lines:{changed_lines}"

    # For many tools, a raw diff is fine; for file_read, we skip context lines
    if normalized_tool_name in (
        "view_file",
        "read_file",
        "cat",
        "bash",
        "run_terminal",
        "computer",
        "sh",
        "edit_file",
        "write_file",
    ):
        stripped_diff = [
            l
            for l in diff_lines
            if not l.startswith(" ") or l.startswith(("---", "+++"))
        ]
        diff_text = "".join(stripped_diff)

    result = header + "\n" + diff_text
    return result, len(raw) - len(result)


def _apply_file_cache(
    raw: str,
    path: str,
    file_cache: dict[str, tuple[str, str]],
) -> tuple[str, int]:
    """Compatibility wrapper for old file cache tests."""
    context = {"name": "view_file", "path": path, "args": {"path": path}}
    return _apply_result_cache(raw, context, file_cache)


def _make_semantic_cache_key(
    context: dict[str, Any] | None, raw: str
) -> str | None:
    """Return a stable cache key for (tool_name, args) if context is available."""
    if not context:
        return None
    tool_name = context.get("name", "")
    # Use input args dict if present, otherwise fall back to the path field.
    args = context.get("args") or context.get("path") or ""
    args_hash = hashlib.sha256(
        json.dumps(args, sort_keys=True).encode("utf-8", errors="replace")
    ).hexdigest()[:16]
    return f"{tool_name}:{args_hash}"


def compress_tool_results(
    messages: list[dict[str, Any]],
    result_cache: dict[str, tuple[str, str]] | None = None,
    tool_use_id_to_context: dict[str, dict[str, Any]] | None = None,
    compression_level: str = "balanced",
    semantic_hash_cache: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Walk messages, apply caching and tok_tool_result() to large tool_result blocks."""
    from ._pipeline import compress_tool_results_impl

    return compress_tool_results_impl(
        messages,
        result_cache=result_cache,
        tool_use_id_to_context=tool_use_id_to_context,
        compression_level=compression_level,
        semantic_hash_cache=semantic_hash_cache,
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


def _should_include_tok_state(
    tok_state: str | None, *, tool_compatible: bool
) -> bool:
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


RECENT_WINDOW_THRESHOLD = (
    8_000  # chars — compress recent-window results larger than this
)
RECENT_WINDOW_EVIDENCE_THRESHOLD = 1_200  # chars — file/search-like recent evidence should compress much sooner


def compress_recent_window(
    messages: list[dict[str, Any]],
    tool_use_id_to_context: dict[str, dict[str, Any]] | None = None,
    threshold: int = RECENT_WINDOW_THRESHOLD,
    tool_compatible: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply content-aware compression to tool_result blocks in the recent window."""
    from ._pipeline import compress_recent_window_impl

    return compress_recent_window_impl(
        messages,
        tool_use_id_to_context=tool_use_id_to_context,
        threshold=threshold,
        tool_compatible=tool_compatible,
    )


def compress_user_prompt(prompt: str) -> str:
    """Extract tasks, requirements, and constraints from a verbose prompt."""
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    filtered_lines: list[str] = []
    for line in lines:
        lower = line.lower()
        if (
            line.startswith(">>>")
            or "optimized task context" in lower
            or any(
                line.startswith(prefix)
                for prefix in ("goal:", "files:", "constraints:")
            )
        ):
            continue
        filtered_lines.append(line)

    if not filtered_lines and "optimized task context" in prompt.lower():
        return re.sub(
            r"### Optimized Task Context\n", "", prompt, flags=re.I
        ).strip()

    goals: list[str] = []
    constraints: list[str] = []
    files: set[str] = set()

    for line in filtered_lines:
        lower = line.lower()
        if any(
            prefix in lower
            for prefix in (
                "task:",
                "goal:",
                "requirement:",
                "implement ",
                "add ",
            )
        ):
            goals.append(line[:60])
        elif line.startswith(("- ", "* ", "1. ")) and any(
            keyword in lower
            for keyword in ("should", "must", "need to", "implement")
        ):
            goals.append(re.sub(r"^[-*1.\s]+", "", line)[:60])

        if any(
            keyword in lower
            for keyword in ("avoid", "don't", "do not", "never", "only")
        ):
            constraints.append(line[:60])

        for match in re.finditer(
            r"\b([\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|sh|txt|css|html|sql|rs|go|rb))\b",
            line,
        ):
            files.add(match.group(1))

    parts: list[str] = []
    if goals:
        parts.append(f"goal:{','.join(goals[:2])}")
    if files:
        parts.append(f"files:{','.join(list(files)[:3])}")
    if constraints:
        parts.append(f"constraints:{','.join(constraints[:2])}")

    if not parts:
        for line in filtered_lines:
            if len(line) > 10:
                return f"goal:{line[:100].strip()}"
        return f"goal:{prompt[:100].strip()}"

    return "|".join(parts)
