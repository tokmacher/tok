from __future__ import annotations

import json
import re
from typing import Any

from tok.runtime.core import count_tokens


def _system_to_messages(
    system: str | list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    if not system:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    messages: list[dict[str, str]] = []
    for block in system:
        if isinstance(block, dict):
            messages.append({"role": "system", "content": block.get("text", "")})
    return messages


def _estimate_tokens(value: str | dict[str, Any] | list[Any] | None) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return count_tokens(value)
    return count_tokens(json.dumps(value, sort_keys=True))


def _sum_warning_signals(signals: dict[str, int]) -> int:
    return sum(
        int(signals.get(key, 0))
        for key in (
            "non_tok_response",
            "fail_open_compat_response",
            "malformed_tok_response",
            "tok_drift_healed",
        )
    )


def _success_term_matches(term: str, search_space: str) -> bool:
    escaped_term = re.escape(term.lower())
    pattern = rf"(?<![A-Za-z0-9_]){escaped_term}(?![A-Za-z0-9_])"
    return bool(re.search(pattern, search_space))


def _content_text(content: str | dict[str, Any] | list[Any]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_text(item) for item in content)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return _content_text(content["content"])
    return str(content)
