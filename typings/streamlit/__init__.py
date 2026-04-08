from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def cache_data(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


def columns(count: int) -> list[Any]:
    return [object() for _ in range(count)]


def metric(label: str, value: Any) -> None: ...


def title(*args: Any, **kwargs: Any) -> None: ...


def markdown(*args: Any, **kwargs: Any) -> None: ...


def subheader(*args: Any, **kwargs: Any) -> None: ...


def warning(*args: Any, **kwargs: Any) -> None: ...


def text(*args: Any, **kwargs: Any) -> None: ...


def button(*args: Any, **kwargs: Any) -> bool:
    return False


def json(*args: Any, **kwargs: Any) -> None: ...


def dataframe(*args: Any, **kwargs: Any) -> None: ...


def set_page_config(*args: Any, **kwargs: Any) -> None: ...


def code(*args: Any, **kwargs: Any) -> None: ...


def tabs(labels: Iterable[str]) -> list[Any]:
    return [object() for _ in labels]


def rerun() -> None: ...


__all__ = [
    "cache_data",
    "columns",
    "metric",
    "title",
    "markdown",
    "subheader",
    "warning",
    "text",
    "button",
    "json",
    "dataframe",
    "set_page_config",
    "code",
    "tabs",
    "rerun",
]
