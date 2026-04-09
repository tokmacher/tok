"""
LLM client interfaces and implementations.

This module provides protocol definitions and concrete implementations
for interacting with various Large Language Model providers.
"""

from __future__ import annotations

from typing import Protocol


class ChatLLM(Protocol):
    """
    Protocol for chat-based LLM interactions.

    Defines the interface for chat-based language models that can
    process system and user messages to generate responses.
    """

    def chat(self, *, system: str, user: str) -> str: ...


class StubClient:
    """Offline stub: always returns a placeholder function."""

    def chat(self, *, system: str, user: str) -> str:
        _ = system, user
        if "extract a concise, reusable rule" in system:
            return "Rule: define knowledge as power"
        if "Multiple Choice" in system or "choices" in user.lower():
            return "A"
        return "```python\ndef solve(x):\n    raise NotImplementedError\n```"
