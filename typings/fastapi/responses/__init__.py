from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any


class StreamingResponse:  # pragma: no cover - type stub
    def __init__(
        self,
        content: AsyncIterator[bytes] | Awaitable[bytes] | Callable[..., Any] | Any,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
    ) -> None: ...


__all__ = ["StreamingResponse"]
