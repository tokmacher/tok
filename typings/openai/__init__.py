from __future__ import annotations

from typing import Any, Protocol


class ChatMessage(Protocol):
    role: str
    content: str | None


class _Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _Choice:
    message: ChatMessage
    delta: ChatMessage | None


class _ChatResponse:
    def __init__(self) -> None:
        self.choices: list[_Choice] = []
        self.usage: _Usage | None = None

    def __iter__(self) -> Any:
        return iter([])


class _Completions:
    def create(self, *args: Any, **kwargs: Any) -> _ChatResponse:
        return _ChatResponse()


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class _Models:
    def retrieve(self, model: str) -> Any:
        return type("Model", (), {"pricing": {}})()


class OpenAI:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.chat = _Chat()
        self.models = _Models()


__all__ = ["OpenAI"]
