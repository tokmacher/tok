from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any


class Request:  # pragma: no cover - type stub
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    async def body(self) -> bytes: ...

    @property
    def headers(self) -> dict[str, str]: ...

    @property
    def url(self) -> Any: ...


class Response:  # pragma: no cover - type stub
    def __init__(
        self,
        content: Any = None,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
    ) -> None: ...


class BackgroundTasks:
    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None: ...


class FastAPI:  # pragma: no cover - type stub
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def api_route(
        self,
        path: str,
        *,
        methods: Iterable[str] | None = None,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            return func

        return decorator

    def on_event(self, event_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def post(self, path: str, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def get(self, path: str, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


__all__ = ["FastAPI", "Request", "Response", "BackgroundTasks"]
