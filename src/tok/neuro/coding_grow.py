"""Stub module for experimental neuro features."""

from typing import Any


class CodingTask:
    """Stub for CodingTask class."""

    def __init__(self, prompt: str, tests: list[tuple[Any, Any]]) -> None:
        self.prompt = prompt
        self.tests = tests


def tokify_code(code: str) -> Any:
    """Stub for tokify_code function."""
    pass
