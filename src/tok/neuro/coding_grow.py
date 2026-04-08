"""Stub module for experimental neuro features."""

from typing import Any


class CodingTask:
    """Stub for CodingTask class."""

    def __init__(self, prompt: str, tests: list[tuple[Any, Any]]) -> None:
        self.prompt = prompt
        self.tests = tests


class TokifyResult:
    """Result of tokify_code with tokens list."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens


def tokify_code(_code: str) -> TokifyResult:
    """Stub for tokify_code function."""
