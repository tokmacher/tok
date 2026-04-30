"""Opt-in live Tok Trace sidecar emission."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from .trace_v0_1 import TRACE_VERSION, canonical_payload_digest

_TRACE_START = time.strftime("%Y%m%d_%H%M%S")
_STEP_LOCK = threading.Lock()
_LAST_STEP_BY_SESSION: dict[str, int] = {}


def trace_enabled() -> bool:
    """Return whether live trace sidecar emission is enabled."""
    return os.getenv("TOK_TRACE", "0").strip().lower() in {"1", "true", "yes", "on"}


def trace_artifact_capture_enabled() -> bool:
    """Return whether sanitized live trace metadata artifacts should be captured."""
    return os.getenv("TOK_TRACE_CAPTURE_ARTIFACTS", "0").strip().lower() in {"1", "true", "yes", "on"}


def emit_live_trace(
    session: Any,
    event: str,
    *,
    trace_class: str,
    action: str,
    result: str,
    expectation: str,
    reason: str,
    direction: str = "request",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one live trace block, swallowing all trace failures."""
    if not trace_enabled():
        return

    try:
        path = live_trace_path(session)
        block = build_live_trace_block(
            session,
            event,
            trace_class=trace_class,
            action=action,
            result=result,
            expectation=expectation,
            reason=reason,
            direction=direction,
            metadata=metadata or {},
            trace_file=path,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(block, sort_keys=True, separators=(",", ":")) + "\n")
    except Exception:
        _emit_trace_warning(session, event)


def build_live_trace_block(
    session: Any,
    event: str,
    *,
    trace_class: str,
    action: str,
    result: str,
    expectation: str,
    reason: str,
    direction: str,
    metadata: dict[str, Any],
    trace_file: Path | None = None,
) -> dict[str, Any]:
    """Build a Tok Trace block for live bridge auditing."""
    session_id = _session_id(session)
    runtime_session = getattr(session, "runtime_session", None)
    bridge_memory = getattr(runtime_session, "bridge_memory", None)
    turn = int(getattr(bridge_memory, "turn", 0) or 0)
    clean_metadata = _json_safe(metadata)
    payload_bytes = json.dumps(clean_metadata, sort_keys=True, default=str).encode("utf-8")
    artifact_uri = _write_metadata_artifact(trace_file, event, payload_bytes)
    block = {
        "envelope": {
            "trace_version": TRACE_VERSION,
            "block_id": f"live-{uuid.uuid4().hex}",
            "session_id": session_id,
            "turn": max(0, turn),
            "step": _next_trace_step(session_id),
            "direction": direction,
            "payload_digest": "draft-uncomputed",
        },
        "observation": {
            "class": trace_class,
            "key": f"live:{event}",
            "action": action,
            "result": result,
        },
        "content": {
            "exact": False,
            "hash": "sha256:" + sha256(payload_bytes).hexdigest(),
            "size_bytes": len(payload_bytes),
        },
        "audit": {
            "resolver_state": "available_local" if artifact_uri is not None else "missing_identifiable",
            "expectation": expectation,
            "reason": reason if artifact_uri is None else f"{reason}; sanitized metadata artifact captured",
        },
        "extensions": {
            "tok.live": {
                "event": event,
                "metadata": clean_metadata,
            }
        },
    }
    if artifact_uri is not None:
        cast(dict[str, Any], block["content"])["resolver_uri"] = artifact_uri
    cast(dict[str, Any], block["envelope"])["payload_digest"] = canonical_payload_digest(block)
    return block


def _next_trace_step(session_id: str) -> int:
    raw_step = time.time_ns()
    with _STEP_LOCK:
        previous = _LAST_STEP_BY_SESSION.get(session_id, -1)
        step = raw_step if raw_step > previous else previous + 1
        _LAST_STEP_BY_SESSION[session_id] = step
        return step


def live_trace_path(session: Any) -> Path:
    """Return the JSONL path for a session's live trace sidecar."""
    configured = os.getenv("TOK_TRACE_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()

    memory_dir = getattr(session, "memory_dir", None) or Path.home() / ".tok"
    session_id = _session_id(session)
    suffix = _safe_name(session_id)
    return Path(memory_dir) / "traces" / f"{_TRACE_START}_{suffix}.jsonl"


def _emit_trace_warning(session: Any, event: str) -> None:
    try:
        logger = __import__("logging").getLogger("tok.trace")
        logger.debug("tok_trace_emit_failed: event=%s session=%s", event, _session_id(session), exc_info=True)
    except Exception:
        return


def _session_id(session: Any) -> str:
    key = str(getattr(session, "_active_session_key", "") or "default")
    return "live:" + sha256(key.encode("utf-8")).hexdigest()[:24]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "default"


def _write_metadata_artifact(trace_file: Path | None, event: str, payload_bytes: bytes) -> str | None:
    if trace_file is None or not trace_artifact_capture_enabled():
        return None
    artifact_dir = trace_file.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256(payload_bytes).hexdigest()
    artifact_name = f"{_safe_name(event)}_{digest[:16]}.json"
    artifact_path = artifact_dir / artifact_name
    artifact_path.write_bytes(payload_bytes)
    return f"artifacts/{artifact_name}"


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [_json_safe(v) for v in value]
        return str(value)
