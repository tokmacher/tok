"""Prompt-size heuristics used by runtime request validation."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_PROMPT_BLOAT_THRESHOLD_CHARS = 2000
DEFAULT_PROMPT_OPTIMIZE_LIMIT_CHARS = 2500
USER_PROMPT_LEAK_MIN_CHARS = 200
USER_PROMPT_LEAK_SNIPPET_CHARS = 100


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def detect_prompt_bloat(system_prompt: str | list[dict[str, Any]] | None, user_prompt: str = "") -> bool:
    """
    Identify when system prompts are unusually large or contain leaked user content.

    Returns True when system prompt length crosses TOK_PROMPT_BLOAT_THRESHOLD
    or when a substantial prefix of the current user prompt appears in system text.
    """
    if system_prompt is None:
        return False

    bloat_threshold = _int_env("TOK_PROMPT_BLOAT_THRESHOLD", DEFAULT_PROMPT_BLOAT_THRESHOLD_CHARS)

    system_text = ""
    if isinstance(system_prompt, list):
        system_text = " ".join(
            str(block.get("text", ""))
            for block in system_prompt
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        system_text = str(system_prompt)

    if len(system_text) > bloat_threshold:
        return True

    if user_prompt and len(user_prompt) > USER_PROMPT_LEAK_MIN_CHARS:
        snippet = user_prompt[:USER_PROMPT_LEAK_SNIPPET_CHARS].strip()
        if snippet and snippet in system_text:
            return True

    return False


def should_optimize_prompts(
    system_prompt: str | list[dict[str, Any]] | None,
    session_metrics: dict[str, int],
) -> bool:
    """Check whether prompt optimization should run for current request payload."""
    size_limit = _int_env("TOK_PROMPT_OPTIMIZE_LIMIT", DEFAULT_PROMPT_OPTIMIZE_LIMIT_CHARS)

    system_text = ""
    if isinstance(system_prompt, list):
        system_text = " ".join(
            str(block.get("text", ""))
            for block in system_prompt
            if isinstance(block, dict) and block.get("type") == "text"
        )
    elif system_prompt:
        system_text = str(system_prompt)

    if len(system_text) > size_limit:
        return True

    if session_metrics.get("tok_prompt_growth_high"):
        return True

    return detect_prompt_bloat(system_prompt)
