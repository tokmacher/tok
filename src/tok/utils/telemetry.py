"""Telemetry emission logic for Tok."""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
from typing import Any, TypedDict

import httpx

logger = logging.getLogger("tok.telemetry")


def _default_collector_url() -> str:
    host = os.getenv("TOK_COLLECTOR_HOST", "localhost")
    port = os.getenv("TOK_COLLECTOR_PORT", "8000")
    return f"http://{host}:{port}/ingest"


DEFAULT_COLLECTOR_URL = _default_collector_url()


class TokEvent(TypedDict):
    event_type: str
    timestamp: str
    request_id: str
    model: str
    payload: dict[str, Any]


_CLIENT: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.AsyncClient(timeout=2.0)
    return _CLIENT


async def emit_event(
    event_type: str,
    payload: dict[str, Any],
    model: str = "unknown",
    request_id: str | None = None,
) -> None:
    """Emit a telemetry event to the central collector (fire-and-forget)."""
    collector_url = os.getenv("TOK_TELEMETRY_URL", DEFAULT_COLLECTOR_URL)
    if not collector_url:
        return

    event: TokEvent = {
        "event_type": event_type,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "request_id": request_id or "unknown",
        "model": model,
        "payload": payload,
    }

    try:
        client = get_client()
        # Fire and forget: we don't await the response in a blocking way
        # for the caller if they use asyncio.create_task.
        # But here we provide a standard async function.
        await client.post(collector_url, json=event)
    except Exception as exc:
        logger.debug("Telemetry emission failed: %s", exc)


def emit_event_sync(
    event_type: str,
    payload: dict[str, Any],
    model: str = "unknown",
    request_id: str | None = None,
) -> None:
    """Sync wrapper for telemetry emission. Uses sync httpx if no loop is running."""
    collector_url = os.getenv("TOK_TELEMETRY_URL", DEFAULT_COLLECTOR_URL)
    if not collector_url:
        return

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            loop.create_task(
                emit_event(event_type, payload, model, request_id)
            )
            return
    except RuntimeError:
        pass

    # No loop running, use sync httpx
    event: TokEvent = {
        "event_type": event_type,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "request_id": request_id or "unknown",
        "model": model,
        "payload": payload,
    }

    try:
        # Use a short timeout to avoid hanging sync callers
        with httpx.Client(timeout=1.0) as client:
            client.post(collector_url, json=event)
    except Exception as exc:
        logger.debug("Sync telemetry emission failed: %s", exc)
