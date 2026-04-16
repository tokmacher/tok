"""
Anthropic-specific request/response optimizations for the Claude bridge.

These translations are ONLY applied when traffic flows through the Claude bridge
adapter. Canonical Tok syntax remains unchanged for all other adapters.

Vectors:
  1. System prompt cache-control splitting (Anthropic prompt caching)
  2. Tool-result stdout sifting (compress raw tool output before forwarding)
  3. BPE-aligned wire format (token-cheaper serialization for Anthropic's tokenizer)
  4. Native thinking passthrough (preserve Anthropic <thinking> blocks as-is)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from tok.compression._tool_result_codecs import (
    _CODE_PATTERNS,
    _compress_file_read,
    _compress_git_log,
    _compress_grep,
    _compress_grep_context,
    _compress_install,
    _compress_ls,
    _compress_pytest,
    _compress_stack_traces,
    _detect_tool_content_type,
    truncate_large_result,
)

logger = logging.getLogger("tok.gateway.anthropic")

_DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"

_CACHE_MIN_CHARS = 800


def split_system_for_caching(body: dict[str, Any]) -> dict[str, Any]:
    """
    Vector 1: Convert system string to Anthropic cacheable array format.

    If the system prompt contains the claw-code dynamic boundary sentinel,
    split into a static block (with cache_control) and a dynamic block.
    Otherwise, place cache_control on the entire system prompt when it's
    large enough to meet the Anthropic minimum cacheable prefix.
    """
    system = body.get("system")
    if system is None:
        return body
    if isinstance(system, list):
        return body
    if not isinstance(system, str):
        return body

    if _DYNAMIC_BOUNDARY in system:
        parts = system.split(_DYNAMIC_BOUNDARY, 1)
        static_text = parts[0].rstrip()
        dynamic_text = parts[1].lstrip("\n")
        if len(static_text) >= _CACHE_MIN_CHARS:
            body["system"] = [
                {
                    "type": "text",
                    "text": static_text,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": dynamic_text},
            ]
            logger.info(
                "anthropic_opt: split system prompt for caching (static=%d chars, dynamic=%d chars)",
                len(static_text),
                len(dynamic_text),
            )
        return body

    if len(system) >= _CACHE_MIN_CHARS:
        body["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            },
        ]
        logger.info(
            "anthropic_opt: wrapped system prompt for caching (%d chars)",
            len(system),
        )

    return body


def sift_tool_results(body: dict[str, Any]) -> dict[str, Any]:
    """
    Vector 2: Compress tool_result stdout before forwarding to Anthropic.

    Walks all user messages, finds tool_result blocks, and applies content-type
    aware compression to raw stdout. This strips visual noise (ANSI codes,
    human-readable formatting, redundant headers) that wastes tokens.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body

    total_saved_chars = 0

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            inner = block.get("content")
            if isinstance(inner, list):
                for sub in inner:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        original_len = len(sub.get("text", ""))
                        compressed = _sift_stdout(sub.get("text", ""))
                        if len(compressed) < original_len:
                            sub["text"] = compressed
                            total_saved_chars += original_len - len(compressed)
            elif isinstance(inner, str):
                original_len = len(inner)
                compressed = _sift_stdout(inner)
                if len(compressed) < original_len:
                    block["content"] = compressed
                    total_saved_chars += original_len - len(compressed)

    if total_saved_chars > 0:
        logger.info(
            "anthropic_opt: sifted tool results, saved ~%d chars (~%d tokens)",
            total_saved_chars,
            total_saved_chars // 4,
        )

    return body


_SIFT_MIN_CHARS = 80
_SIFT_SMALL_FILE_MAX_LINES = 100
_SIFT_SMALL_FILE_MAX_CHARS = 3000


def _sift_stdout(text: str) -> str:
    if not text or len(text) < _SIFT_MIN_CHARS:
        return text
    # Skip already-compressed Tok content (repeat search, stable_result, etc.)
    if text.startswith(">>> "):
        return text
    # Small files: never truncate — token savings negligible, friction high
    line_count = text.count("\n") + 1
    if line_count <= _SIFT_SMALL_FILE_MAX_LINES and len(text) <= _SIFT_SMALL_FILE_MAX_CHARS:
        return text
    content_type = _detect_tool_content_type(text)
    if content_type == "ls":
        return _compress_ls(text)
    if content_type == "git_log":
        return _compress_git_log(text)
    if content_type == "grep_context":
        return _compress_grep_context(text)
    if content_type == "grep":
        return _compress_grep(text)
    if content_type == "pytest":
        return _compress_pytest(text)
    if content_type == "stack_trace":
        return _compress_stack_traces(text)
    if content_type == "install":
        return _compress_install(text)
    if len(text) > 1200:
        # For code-like content, use skeletonizer instead of blunt truncation
        if content_type == "file" or _CODE_PATTERNS.search(text):
            skeletonized = _compress_file_read(text)
            if len(skeletonized) < len(text):
                return skeletonized
        return truncate_large_result(text, limit=1200)
    return text


_BPE_TRANSLATIONS = [
    (" | ", ","),
    ("|> ", "> "),
    ("|>", ">"),
]

_STATE_LINE_RE = re.compile(r"^(>>> )(.+)$", re.MULTILINE)
_PIPE_ATTR_RE = re.compile(r"\|([a-z_]+:)")

_TOK_BLOCK_RE = re.compile(r"^( {2,})@(msg|thought|Tool|result|meta|Delegate)\b", re.MULTILINE)
_BLANK_BETWEEN_BLOCKS_RE = re.compile(r"\n{3,}")


def _translate_bpe(text: str) -> str:
    """
    Vector 3: Translate canonical Tok syntax to Anthropic BPE-cheaper wire format.

    Transformations (request-side only, on compressed state heading to api.anthropic.com):
      - >>>|  delimiters in state lines -> commas
      - |> verbatim prefix -> > prefix
      - Leading 2+ space indent on @-blocks -> removed
      - Triple blank lines -> single newline
    """
    if not text or len(text) < 10:
        return text

    def _translate_state_line(m: re.Match[str]) -> str:
        prefix = m.group(1)
        rest = m.group(2)
        rest = _PIPE_ATTR_RE.sub(r",\1", rest)
        return prefix + rest

    result = _STATE_LINE_RE.sub(_translate_state_line, text)

    for old, new in _BPE_TRANSLATIONS:
        result = result.replace(old, new)

    result = _TOK_BLOCK_RE.sub(lambda m: f"@{m.group(2)}", result)

    return _BLANK_BETWEEN_BLOCKS_RE.sub("\n", result)


def bpe_translate_request(body: dict[str, Any]) -> dict[str, Any]:
    """
    Apply BPE translation to all Tok-formatted text in a request body.

    Walks message text blocks and translates canonical Tok syntax to
    Anthropic-optimized wire format. Only touches text that contains
    Tok markers (>>>, @, |>).
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body

    total_saved_chars = 0

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            if ">>>" in content or "|>" in content or " @msg" in content:
                translated = _translate_bpe(content)
                total_saved_chars += len(content) - len(translated)
                msg["content"] = translated
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            raw = block.get("text", "")
            if ">>>" in raw or "|>" in raw or "@msg" in raw:
                translated = _translate_bpe(raw)
                total_saved_chars += len(raw) - len(translated)
                block["text"] = translated

    system = body.get("system")
    if isinstance(system, str):
        if ">>>" in system or "|>" in system:
            translated = _translate_bpe(system)
            total_saved_chars += len(system) - len(translated)
            body["system"] = translated
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                raw = block.get("text", "")
                if ">>>" in raw or "|>" in raw:
                    translated = _translate_bpe(raw)
                    total_saved_chars += len(raw) - len(translated)
                    block["text"] = translated

    if total_saved_chars > 0:
        logger.info(
            "anthropic_opt: BPE translation saved ~%d chars (~%d tokens)",
            total_saved_chars,
            total_saved_chars // 4,
        )

    return body


def apply_anthropic_optimizations(
    body: dict[str, Any],
    *,
    is_claude_bridge: bool = True,
) -> dict[str, Any]:
    """
    Apply all Anthropic-specific optimizations to a request body.

    This is the single entry point called from the gateway bridge handler.
    Only applies when is_claude_bridge is True (i.e., traffic is going to
    api.anthropic.com via the Claude bridge adapter).
    """
    if not is_claude_bridge:
        return body
    try:
        body = split_system_for_caching(body)
        body = sift_tool_results(body)
        body = bpe_translate_request(body)
    except Exception as exc:
        logger.debug("anthropic_opt: skipping due to error: %s", exc)
    return body
