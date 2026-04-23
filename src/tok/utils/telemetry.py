"""Telemetry emission logic for Tok."""

from __future__ import annotations

import asyncio
import atexit
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
_CLIENT_LOCK = asyncio.Lock()
_CLEANUP_COMPLETED = False


def get_client() -> httpx.AsyncClient:
    """Get or create the async HTTP client with proper lifecycle management."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.AsyncClient(
            timeout=2.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _CLIENT


async def cleanup_telemetry() -> None:
    """
    Cleanup telemetry resources. Call this on application shutdown.
    This function is idempotent and can be called multiple times safely.
    """
    global _CLIENT, _CLEANUP_COMPLETED

    if _CLEANUP_COMPLETED or _CLIENT is None:
        # Already cleaned up or nothing to clean
        return

    try:
        await _CLIENT.aclose()
        logger.debug("Telemetry client closed")
    except Exception as exc:
        logger.debug("Error closing telemetry client: %s", exc)
    finally:
        _CLIENT = None
        _CLEANUP_COMPLETED = True


def _sync_cleanup() -> None:
    """
    Synchronous cleanup for atexit handler.
    Python 3.10+ compatible version that handles loop detection properly.
    """
    global _CLEANUP_COMPLETED

    if _CLEANUP_COMPLETED:
        # Already cleaned up
        return

    try:
        # Python 3.10+ compatible way to handle event loops
        try:
            # Try to get the current running loop (Python 3.7+)
            loop = asyncio.get_running_loop()
            # If we get here, there's a running loop
            loop.create_task(cleanup_telemetry())
        except RuntimeError:
            # No running loop, create a new one for cleanup
            try:
                # Try to get any existing loop (Python 3.10+ compatible)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(cleanup_telemetry())
                else:
                    asyncio.run(cleanup_telemetry())
            except RuntimeError:
                # No loop exists, create one and run cleanup
                asyncio.run(cleanup_telemetry())
    except Exception as exc:
        logger.debug("Error in sync telemetry cleanup: %s", exc)


# Register cleanup on exit
atexit.register(_sync_cleanup)


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
            loop.create_task(emit_event(event_type, payload, model, request_id))
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
            response = client.post(collector_url, json=event)
            response.raise_for_status()
    except Exception as exc:
        logger.debug("Sync telemetry emission failed: %s", exc)
