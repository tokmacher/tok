"""Output-side translation: Tok grammar -> readable English."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("tok.translator")

IS_TOK = re.compile(r"(^>>>|^@[A-Za-z_]|\s+\|>)", re.MULTILINE)

_MD_STRIP = [
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"),
    (re.compile(r"\*(.+?)\*", re.DOTALL), r"\1"),
    (re.compile(r"^---+\s*$", re.MULTILINE), ""),
    (re.compile(r"^_{3,}\s*$", re.MULTILINE), ""),
    (re.compile(r"\n{3,}"), "\n\n"),
]


_BLOCK_HEADER_RE = re.compile(r"^@[A-Za-z_][A-Za-z0-9_]*")
_PIPE_LINE_RE = re.compile(r"^\s*\|[#\d]?>")


@dataclass
class _TokReadableState:
    in_thought: bool = False
    in_msg_assistant: bool = False
    has_seen_any_at_block: bool = False


def _handle_block_header(stripped: str, state: _TokReadableState) -> bool:
    if not _BLOCK_HEADER_RE.match(stripped):
        return False
    state.has_seen_any_at_block = True
    if stripped.startswith("@thought"):
        state.in_thought = True
        state.in_msg_assistant = False
    elif stripped.startswith("@msg") and "role:assistant" in stripped:
        state.in_thought = False
        state.in_msg_assistant = True
    else:
        state.in_thought = False
        state.in_msg_assistant = False
    return True


def _handle_pipe_line(
    line: str, state: _TokReadableState, output: list[str]
) -> bool:
    if not _PIPE_LINE_RE.match(line):
        return False
    if not state.in_thought and not state.in_msg_assistant:
        state.in_msg_assistant = True
    if state.in_thought:
        return True
    if state.in_msg_assistant:
        content = re.sub(r"^\s*\|[#\d]?>\s?", "", line)
        output.append(content)
    return True


def _should_capture_plain_line(
    stripped: str, state: _TokReadableState
) -> bool:
    if state.in_thought:
        return False
    if state.in_msg_assistant:
        return True
    return bool(stripped and not state.has_seen_any_at_block)


def strip_markdown_fallback(text: str) -> str:
    """Fallback: strip markdown boilerplate when response isn't Tok-formatted."""
    code_blocks: list[str] = []
    placeholder = "\x00CODE{}\x00"

    def _save(m: re.Match[str]) -> str:
        code_blocks.append(m.group(0))
        return placeholder.format(len(code_blocks) - 1)

    text = re.sub(r"```[\s\S]*?```", _save, text)
    text = re.sub(r"`[^`]+`", _save, text)
    for pat, rep in _MD_STRIP:
        text = pat.sub(rep, text)
    for i, block in enumerate(code_blocks):
        text = text.replace(placeholder.format(i), block)
    return text.strip()


def tok_to_readable(text: str) -> str:
    """Parse Tok-grammar response and extract user-visible content.

    - >>> lines: stripped (internal state)
    - @thought blocks: stripped entirely (internal reasoning)
    - @msg role:assistant blocks: |> content extracted as output
    - All other @blocks: stripped
    - Non-Tok lines inside @msg blocks (e.g. code fences): passed through
    - Lazy Tok: assume @msg role:assistant if |> is found without a preceding @block.
    """
    lines = text.splitlines()
    output: list[str] = []
    state = _TokReadableState()

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(">>>"):
            state.in_thought = False
            state.in_msg_assistant = False
            continue

        if _handle_block_header(stripped, state):
            continue

        if _handle_pipe_line(line, state, output):
            continue

        if _should_capture_plain_line(stripped, state):
            output.append(line)

    result = "\n".join(output).strip()
    return result


def postprocess_response(text: str) -> tuple[str, str]:
    """Process Claude's response text.

    Returns (processed_text, mode) where mode is 'tok-native', 'tok-empty', 'tok', or 'markdown'.
    """
    if IS_TOK.search(text):
        readable = tok_to_readable(text)
        if readable:
            return readable, "tok-native"
        # Check if there's any content at all (even if not readable)
        has_content = bool(text.strip())
        if has_content:
            return strip_markdown_fallback(text), "tok-empty"
        return strip_markdown_fallback(text), "tok"
    return strip_markdown_fallback(text), "markdown"
