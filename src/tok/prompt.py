"""Backward-compatible facade for the Tok system prompt."""

from __future__ import annotations

from .analysis.prompt import (
    NAKED_TOK_SYSTEM_PROMPT as NAKED_TOK_SYSTEM_PROMPT,
    TOK_EXPLORE_PROMPT as TOK_EXPLORE_PROMPT,
    TOK_SYSTEM_PROMPT as TOK_SYSTEM_PROMPT,
    MINIMAL_PULSE_PROMPT as MINIMAL_PULSE_PROMPT,
    get_grammar_snippet as get_grammar_snippet,
)

__all__ = [
    "NAKED_TOK_SYSTEM_PROMPT",
    "TOK_EXPLORE_PROMPT",
    "TOK_SYSTEM_PROMPT",
    "MINIMAL_PULSE_PROMPT",
    "get_grammar_snippet",
]
