"""Tok Bridge Gateway — sits between Claude Code and api.anthropic.com.

Two-sided compression:
  INPUT:  Compresses old message history into a Tok rolling state (>>> ...)
  OUTPUT: Forces Claude to respond in Tok grammar, then translates back to
          readable English before returning to Claude Code.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import uvicorn
from typing import Any, cast
from collections.abc import AsyncIterator
from fastapi import FastAPI

from ..runtime.memory.bridge_memory import BridgeMemoryState
from ..runtime.smoothness import SmoothnessTracker
from ..stats import SavingsTracker
from ..universal_runtime import (
    RuntimeSession,
    UniversalTokRuntime,
    build_tool_use_id_to_context,
    collect_behavior_signals,
    response_contract_for_mode,
)
from ..runtime.pipeline.request_validation import (
    normalize_tool_use_blocks,
    summarize_message_structure,
)

# Internal gateway support modules

from ._fingerprint import (
    _request_body_fingerprint,
)
from ._bridge_comparison import _request_fingerprint_diff

# PID file handling for foreground mode
TOK_DIR = Path.home() / ".tok"
PID_FILE = TOK_DIR / "bridge.pid"

logger = logging.getLogger("tok.gateway")

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _tok_count(text: str) -> int:
        return len(_ENC.encode(text))

except Exception:

    def _tok_count(text: str) -> int:
        return len(text) // 4


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANTHROPIC_API_BASE = "https://api.anthropic.com"

_REQUEST_POLICY_ALIASES = {
    "legacy": "legacy_tool_compatible",
    "legacy_tool_compatible": "legacy_tool_compatible",
    "tool_compatible": "legacy_tool_compatible",
    "tool-compatible": "legacy_tool_compatible",
    "natural": "natural_first",
    "natural_first": "natural_first",
    "natural-first": "natural_first",
    "baseline": "forced_baseline",
    "forced_baseline": "forced_baseline",
    "forced-baseline": "forced_baseline",
}


def _normalize_request_policy(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return ""
    return _REQUEST_POLICY_ALIASES.get(normalized, "")


def _default_request_policy() -> str:
    tok_mode = os.getenv("TOK_MODE", "tool-compatible").strip().lower()
    if tok_mode == "baseline":
        return "forced_baseline"
    explicit_policy = _normalize_request_policy(
        os.getenv("TOK_REQUEST_POLICY", "")
    )
    if explicit_policy:
        return explicit_policy
    return "natural_first"


def _request_policy_mode_label(policy: str) -> str:
    if policy == "forced_baseline":
        return "baseline"
    return "tool-compatible"


def _log_bridge_body_structure(
    event: str,
    *,
    body: dict[str, Any] | None = None,
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
    original_body: dict[str, Any] | None = None,
    original_content: bytes | None = None,
    compressed_request: bool | None = None,
    canonicalized_changed: bool | None = None,
    strict_failures: list[str] | None = None,
    reverted_to_original: bool | None = None,
) -> None:
    summary: str | dict[str, Any]
    parsed_body: dict[str, Any] | None = None
    parsed_original_body: dict[str, Any] | None = (
        original_body if isinstance(original_body, dict) else None
    )

    if body is None and content is not None:
        try:
            parsed = json.loads(content)
        except Exception as exc:
            summary = {
                "message_count": 0,
                "role_sequence": [],
                "message_block_types": [],
                "payload_parse_error": exc.__class__.__name__,
                "payload_bytes": len(content),
            }
        else:
            if isinstance(parsed, dict):
                parsed_body = parsed
            summary = (
                summarize_message_structure(parsed.get("messages", []))
                if isinstance(parsed, dict)
                else {
                    "message_count": 0,
                    "role_sequence": [],
                    "message_block_types": [],
                    "payload_shape": type(parsed).__name__,
                }
            )
    elif isinstance(body, dict):
        parsed_body = body
        summary = summarize_message_structure(body.get("messages", []))
    else:
        summary = {
            "message_count": 0,
            "role_sequence": [],
            "message_block_types": [],
        }

    if parsed_original_body is None and original_content is not None:
        try:
            parsed_original = json.loads(original_content)
        except Exception:
            parsed_original_body = None
        else:
            if isinstance(parsed_original, dict):
                parsed_original_body = parsed_original

    fingerprint: dict[str, object] = {}
    if parsed_body is not None:
        if parsed_original_body is not None:
            fingerprint = _request_fingerprint_diff(
                headers or {}, parsed_body, parsed_original_body
            )
        else:
            fingerprint = _request_body_fingerprint(headers or {}, parsed_body)

    log = (
        logger.warning
        if strict_failures or reverted_to_original
        else logger.info
    )
    log(
        "Bridge request structure | event=%s compressed=%s canonicalized_changed=%s reverted_to_original=%s strict_failures=%s summary=%s fingerprint=%s",
        event,
        compressed_request,
        canonicalized_changed,
        reverted_to_original,
        strict_failures or [],
        summary,
        fingerprint,
    )


# ---------------------------------------------------------------------------
# Tool Translation
# ---------------------------------------------------------------------------

_RUNTIME = UniversalTokRuntime()
_build_tool_use_id_to_context = build_tool_use_id_to_context
_collect_behavior_signals = collect_behavior_signals


@dataclass(frozen=True)
class ResponseContract:
    content_blocks: list[dict[str, Any]]
    behavior_signals: dict[str, int]
    mode: str


def _record_fallback_once(
    session: BridgeSession, request_state: dict[str, bool]
) -> None:
    """Record at most one fallback threshold step per client request."""
    if request_state.get("fallback_recorded", False):
        return
    session.runtime_session.record_fallback_event()
    request_state["fallback_recorded"] = True


def _has_visible_content_block(content_blocks: list[dict[str, Any]]) -> bool:
    for block in content_blocks:
        if block.get("type") == "tool_use":
            return True
        if block.get("type") == "text" and str(block.get("text", "")).strip():
            return True
    return False


def _response_contract(text: str) -> ResponseContract:
    return _response_contract_for_mode(text, tool_compatible=False)


def _response_contract_for_mode(
    text: str, *, tool_compatible: bool
) -> ResponseContract:
    processed = response_contract_for_mode(
        text, tool_compatible=tool_compatible
    )
    return ResponseContract(
        content_blocks=processed.content_blocks,
        behavior_signals=processed.behavior_signals,
        mode=processed.mode,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class BridgeSession:
    """HTTP gateway configuration and telemetry (delegates runtime state to RuntimeSession)."""

    port: int = int(
        os.getenv("TOK_BRIDGE_PORT", os.getenv("TOK_PROXY_PORT", "9090"))
    )
    keep_turns: int = int(
        os.getenv("TOK_KEEP_TURNS", os.getenv("TOK_PROXY_KEEP_TURNS", "2"))
    )
    debug: bool = os.getenv("TOK_DEBUG", "0") == "1"
    fail_open: bool = os.getenv("TOK_FAIL_OPEN", "1") == "1"
    capture: bool = os.getenv("TOK_CAPTURE", "0") == "1"
    rate_limit_retry_max_attempts: int = int(
        os.getenv("TOK_RATE_LIMIT_RETRY_MAX_ATTEMPTS", "2")
    )
    rate_limit_backoff_base_ms: int = int(
        os.getenv("TOK_RATE_LIMIT_BACKOFF_BASE_MS", "150")
    )
    rate_limit_backoff_cap_ms: int = int(
        os.getenv("TOK_RATE_LIMIT_BACKOFF_CAP_MS", "1000")
    )
    rate_limit_throttle_threshold: int = int(
        os.getenv("TOK_RATE_LIMIT_THROTTLE_THRESHOLD", "4")
    )
    rate_limit_throttle_cooldown_sec: int = int(
        os.getenv("TOK_RATE_LIMIT_THROTTLE_COOLDOWN_SEC", "20")
    )
    rate_limit_throttle_window_sec: int = int(
        os.getenv("TOK_RATE_LIMIT_THROTTLE_WINDOW_SEC", "30")
    )
    # TOK_MODE=baseline still forces the conservative baseline path.
    # Otherwise, TOK_REQUEST_POLICY can explicitly override the stable
    # tool-compatible request policy default.
    request_policy_default: str = field(
        default_factory=_default_request_policy
    )
    # Kept for compatibility with existing gateway/reporting code.
    tool_compatible_default: bool = True
    memory_dir: Path | None = None
    tracker: SavingsTracker = field(default_factory=SavingsTracker)
    # Canonical runtime state: delegates to this
    runtime_session: RuntimeSession = field(default_factory=RuntimeSession)
    # Smoothness tracking for interaction quality
    smoothness_tracker: SmoothnessTracker = field(
        default_factory=SmoothnessTracker
    )

    def __post_init__(self) -> None:
        self.request_policy_default = (
            _normalize_request_policy(self.request_policy_default)
            or _default_request_policy()
        )
        self.tool_compatible_default = (
            self.request_policy_default != "forced_baseline"
        )
        explicit_memory_dir = self.memory_dir is not None
        if self.memory_dir is None:
            project_dir = os.getenv("TOK_PROJECT_DIR", "")
            if project_dir:
                self.memory_dir = Path(project_dir) / ".tok"
            else:
                self.memory_dir = Path.home() / ".tok"
        self._capture_file: Path | None = None
        if self.capture:
            import datetime

            sessions_dir = (
                self.memory_dir or Path.home() / ".tok"
            ) / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._capture_file = sessions_dir / f"{ts}.jsonl"
        if explicit_memory_dir:
            self.runtime_session = RuntimeSession(
                memory_dir=self.memory_dir,
                keep_turns=self.keep_turns,
            )
        # Reset session stats so each bridge run starts with a clean slate
        if os.getenv("TOK_RESET_SESSION", "0") == "1":
            self.tracker.reset_session_stats()

    def capture_request(self, body: dict[str, Any]) -> None:
        """Append raw request body to capture file (strips sensitive headers)."""
        if not self.capture or self._capture_file is None:
            return
        import datetime

        record = {
            "ts": datetime.datetime.now().isoformat(),
            "messages": body.get("messages", []),
            "system": body.get("system", ""),
        }
        try:
            with open(self._capture_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.debug("Capture write error: %s", exc)

    def capture_event(self, record: dict[str, Any]) -> None:
        """Append a structured capture event to the session file."""
        if not self.capture or self._capture_file is None:
            return
        try:
            with open(self._capture_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.debug("Capture write error: %s", exc)

    def write_memory(self, text: str) -> str:
        """Delegate to runtime session."""
        return self.runtime_session.write_memory(text)

    def policy_snapshot(self, model: str) -> tuple[str, object]:
        """Delegate to runtime session."""
        return self.runtime_session.policy_snapshot(model)

    def load_memory(self, model: str = "") -> str:
        """Delegate to runtime session."""
        return self.runtime_session.load_memory(model)

    def refresh_hot_memory(self, tok_state: str, model: str = "") -> str:
        """Delegate to runtime session."""
        return self.runtime_session.refresh_hot_memory(tok_state, model)

    def update_family_mode(self, model: str, signals: dict[str, int]) -> str:
        """Delegate to runtime session."""
        return self.runtime_session.update_family_mode(model, signals)

    def consume_behavior_signals(self) -> dict[str, int]:
        """Delegate to runtime session."""
        return self.runtime_session.consume_behavior_signals()

    def _bump_signals(self, signals: dict[str, int]) -> None:
        """Delegate internal runtime signal updates."""
        self.runtime_session._bump_signals(signals)

    def _save_bridge_memory(self) -> None:
        """Delegate bridge memory persistence to runtime session."""
        self.runtime_session._save_bridge_memory()

    @property
    def result_cache(self) -> dict[str, tuple[str, str]]:
        """Delegate to runtime session."""
        return cast(
            "dict[str, tuple[str, str]]", self.runtime_session.result_cache
        )

    @property
    def bridge_memory(self) -> BridgeMemoryState:
        """Delegate to runtime session."""
        return self.runtime_session.bridge_memory

    @bridge_memory.setter
    def bridge_memory(self, value: BridgeMemoryState) -> None:
        """Delegate to runtime session."""
        self.runtime_session.bridge_memory = value


# ---------------------------------------------------------------------------
# SSE buffering + re-streaming
# ---------------------------------------------------------------------------


async def _buffer_strip_restream(
    session: BridgeSession,
    client: httpx.AsyncClient,
    response: httpx.Response,
    input_saved_tokens: int = 0,
    type_breakdown: dict[str, int] | None = None,
    behavior_signals: dict[str, int] | None = None,
    prompt_metrics: dict[str, int] | None = None,
    tool_compatible: bool = False,
    request_method: str = "POST",
    request_url: str = "",
    request_headers: dict[str, str] | None = None,
    request_content: bytes | None = None,
    request_state: dict[str, bool] | None = None,
) -> AsyncIterator[bytes]:
    """Buffer the full SSE stream, translate Tok -> readable English/tool_use, re-emit."""
    from ._app_factory import buffer_strip_restream_impl

    async for chunk in buffer_strip_restream_impl(
        session,
        client,
        response,
        input_saved_tokens=input_saved_tokens,
        type_breakdown=type_breakdown,
        behavior_signals=behavior_signals,
        prompt_metrics=prompt_metrics,
        tool_compatible=tool_compatible,
        request_method=request_method,
        request_url=request_url,
        request_headers=request_headers,
        request_content=request_content,
        request_state=request_state,
    ):
        yield chunk


def _materialize_stream_tool_blocks(
    stream_blocks: dict[int, dict[str, Any]], stream_order: list[int]
) -> list[dict[str, Any]]:
    tool_blocks: list[dict[str, Any]] = []
    for index in stream_order:
        block = stream_blocks.get(index, {})
        if block.get("type") != "tool_use":
            continue
        tool_input = (
            dict(block.get("input", {}))
            if isinstance(block.get("input", {}), dict)
            else {}
        )
        partial_json = "".join(
            part
            for part in block.get("partial_json", [])
            if isinstance(part, str)
        )
        if partial_json.strip():
            try:
                parsed = json.loads(partial_json)
                if isinstance(parsed, dict):
                    tool_input.update(parsed)
            except json.JSONDecodeError:
                logger.debug(
                    "Tool JSON delta parse error: %s", partial_json[:120]
                )
                continue
        tool_blocks.append(
            {
                "type": "tool_use",
                "id": block.get("id", ""),
                "name": block.get("name", "unknown"),
                "input": tool_input,
            }
        )
    normalized_tool_blocks, _ = normalize_tool_use_blocks(
        tool_blocks, seed_prefix="toolu_stream"
    )
    return normalized_tool_blocks


# FastAPI app factory
# ---------------------------------------------------------------------------


def create_app(session: BridgeSession | None = None) -> FastAPI:
    """Create the bridge FastAPI application."""
    from ._app_factory import create_app_impl

    return create_app_impl(session)


def run_bridge(
    port: int | None = None,
    keep_turns: int | None = None,
    debug: bool | None = None,
    fail_open: bool | None = None,
    _foreground: bool = True,
    _api_base: str = "https://api.anthropic.com",
) -> None:
    """Start the bridge server."""
    _port_env: str = os.getenv(
        "TOK_BRIDGE_PORT", os.getenv("TOK_PROXY_PORT", "9090")
    )
    port = int(port if port is not None else _port_env)
    _keep_turns_env: str = os.getenv(
        "TOK_KEEP_TURNS", os.getenv("TOK_PROXY_KEEP_TURNS", "2")
    )
    keep_turns = int(keep_turns if keep_turns is not None else _keep_turns_env)
    debug = debug if debug is not None else os.getenv("TOK_DEBUG", "0") == "1"
    fail_open = (
        fail_open
        if fail_open is not None
        else os.getenv("TOK_FAIL_OPEN", "1") == "1"
    )

    try:
        TOK_DIR.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))
    except OSError as exc:
        logger.error(
            "Failed to write PID file %s: %s. "
            "Check permissions on %s or set TOK_DIR to a writable location.",
            PID_FILE,
            exc,
            TOK_DIR,
        )
        raise

    from rich.logging import RichHandler

    log_level = "debug" if debug else "warning"

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    try:
        session = BridgeSession(
            port=port,
            keep_turns=keep_turns,
            debug=debug,
            fail_open=fail_open,
        )
    except Exception as exc:
        logger.error("Failed to create bridge session: %s", exc)
        raise

    atexit.register(session.tracker.merge_session_to_ledger)
    atexit.register(lambda: PID_FILE.unlink(missing_ok=True))

    try:
        app = create_app(session)
    except Exception as exc:
        logger.error("Failed to create bridge application: %s", exc)
        raise

    host_display = os.getenv("TOK_BRIDGE_HOST", "localhost")
    logger.info("Listening on http://%s:%d", host_display, port)
    logger.info("Keeping last %d human turns verbatim", keep_turns)
    logger.info("Fail-open: %s", "enabled" if fail_open else "disabled")
    logger.info(
        "Default Claude bridge mode: %s (request_policy=%s, TOK_MODE=%s, TOK_REQUEST_POLICY=%s)",
        _request_policy_mode_label(session.request_policy_default),
        session.request_policy_default,
        os.getenv("TOK_MODE", "tool-compatible"),
        os.getenv("TOK_REQUEST_POLICY", "<unset>"),
    )

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level=log_level)
    except Exception as exc:
        logger.error(
            "Bridge server exited unexpectedly on port %d: %s", port, exc
        )
        raise


if __name__ == "__main__":
    run_bridge()
