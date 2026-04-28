"""
Tok Bridge Gateway — sits between Claude Code and api.anthropic.com.

Two-sided compression:
  INPUT:  Compresses old message history into a Tok rolling state (>>> ...)
  OUTPUT: Forces Claude to respond in Tok grammar, then translates back to
          readable English before returning to Claude Code.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import uvicorn

from tok.runtime.pipeline.request_validation import (
    normalize_tool_use_blocks,
    summarize_message_structure,
)
from tok.runtime.smoothness import SmoothnessTracker
from tok.stats import SavingsTracker
from tok.universal_runtime import (
    RuntimeSession,
    UniversalTokRuntime,
    build_tool_use_id_to_context,
    collect_behavior_signals,
    response_contract_for_mode,
)

from ._bridge_comparison import _request_fingerprint_diff

# Internal gateway support modules
from ._fingerprint import _request_body_fingerprint
from ._request_policy import (
    default_request_policy as _default_request_policy,
)
from ._request_policy import (
    normalize_request_policy as _normalize_request_policy,
)
from ._request_policy import (
    request_policy_mode_label as _request_policy_mode_label,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    import httpx
    from fastapi import FastAPI

    from tok.runtime.memory.bridge_memory import BridgeMemoryState

# PID file handling for foreground mode
TOK_DIR = Path.home() / ".tok"
PID_FILE = TOK_DIR / "bridge.pid"

logger = logging.getLogger("tok.gateway")

ANTHROPIC_API_BASE = "https://api.anthropic.com"
_CAPTURE_SENSITIVE_KEYS = {
    "authorization",
    "x_api_key",
    "api_key",
    "access_token",
    "refresh_token",
    "bearer_token",
    "openai_api_key",
    "openrouter_api_key",
    "anthropic_api_key",
}
_CAPTURE_INLINE_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]+\b", re.IGNORECASE), "Bearer <redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9._\-]+\b"), "sk-<redacted>"),
)


def _default_api_base() -> str:
    configured = os.getenv("TOK_API_BASE", ANTHROPIC_API_BASE).strip()
    return configured or ANTHROPIC_API_BASE


def _default_bind_host() -> str:
    configured = os.getenv("TOK_BRIDGE_BIND_HOST", "127.0.0.1").strip()
    return configured or "127.0.0.1"


def _env_int(name: str, fallback: int, *, legacy_name: str | None = None) -> int:
    raw = os.getenv(name)
    if raw is None and legacy_name is not None:
        raw = os.getenv(legacy_name)
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def _env_bool(name: str, fallback: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    return raw == "1"


def _is_sensitive_capture_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized in _CAPTURE_SENSITIVE_KEYS or normalized.endswith("_api_key")


def _redact_capture_string(value: str) -> str:
    redacted = value
    for pattern, replacement in _CAPTURE_INLINE_SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _sanitize_capture_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _is_sensitive_capture_key(key):
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = _sanitize_capture_payload(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_capture_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_capture_payload(item) for item in value]
    if isinstance(value, str):
        return _redact_capture_string(value)
    return value


def _parse_request_body(
    body: dict[str, Any] | None,
    content: bytes | None,
) -> tuple[dict[str, Any] | None, str | dict[str, Any]]:
    """Parse request body from dict or bytes, returning (parsed_body, summary)."""
    parsed_body: dict[str, Any] | None = None
    summary: str | dict[str, Any]

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

    return parsed_body, summary


def _parse_original_body(
    original_body: dict[str, Any] | None,
    original_content: bytes | None,
) -> dict[str, Any] | None:
    """Parse original request body from dict or bytes."""
    if isinstance(original_body, dict):
        return original_body

    if original_content is not None:
        try:
            parsed_original = json.loads(original_content)
        except Exception:
            return None
        else:
            if isinstance(parsed_original, dict):
                return parsed_original
    return None


def _build_request_fingerprint(
    parsed_body: dict[str, Any] | None,
    parsed_original_body: dict[str, Any] | None,
    headers: dict[str, str] | None,
) -> dict[str, object]:
    """Build fingerprint from parsed bodies."""
    if parsed_body is None:
        return {}
    if parsed_original_body is not None:
        return _request_fingerprint_diff(headers or {}, parsed_body, parsed_original_body)
    return _request_body_fingerprint(headers or {}, parsed_body)


def _select_log_level(
    strict_failures: list[str] | None,
    reverted_to_original: bool | None,
) -> Callable[..., None]:
    """Select appropriate log level based on failure state."""
    if strict_failures or reverted_to_original:
        return logger.warning
    return logger.info


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
    parsed_body, summary = _parse_request_body(body, content)
    parsed_original_body = _parse_original_body(original_body, original_content)
    fingerprint = _build_request_fingerprint(parsed_body, parsed_original_body, headers)
    log = _select_log_level(strict_failures, reverted_to_original)
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


_RUNTIME = UniversalTokRuntime()
_build_tool_use_id_to_context = build_tool_use_id_to_context
_collect_behavior_signals = collect_behavior_signals


@dataclass(frozen=True)
class ResponseContract:
    content_blocks: list[dict[str, Any]]
    behavior_signals: dict[str, int]
    mode: str


def _record_fallback_once(session: BridgeSession, request_state: dict[str, bool]) -> None:
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


def _response_contract_for_mode(text: str, *, tool_compatible: bool) -> ResponseContract:
    processed = response_contract_for_mode(text, tool_compatible=tool_compatible)
    return ResponseContract(
        content_blocks=processed.content_blocks,
        behavior_signals=processed.behavior_signals,
        mode=processed.mode,
    )


@dataclass
class BridgeSession:
    """HTTP gateway configuration and telemetry (delegates runtime state to RuntimeSession)."""

    port: int = field(default_factory=lambda: _env_int("TOK_BRIDGE_PORT", 9090, legacy_name="TOK_PROXY_PORT"))
    keep_turns: int = field(default_factory=lambda: _env_int("TOK_KEEP_TURNS", 2, legacy_name="TOK_PROXY_KEEP_TURNS"))
    debug: bool = field(default_factory=lambda: _env_bool("TOK_DEBUG", False))
    fail_open: bool = field(default_factory=lambda: _env_bool("TOK_FAIL_OPEN", True))
    capture: bool = field(default_factory=lambda: _env_bool("TOK_CAPTURE", False))
    api_base: str = field(default_factory=_default_api_base)
    rate_limit_retry_max_attempts: int = field(default_factory=lambda: _env_int("TOK_RATE_LIMIT_RETRY_MAX_ATTEMPTS", 2))
    rate_limit_backoff_base_ms: int = field(default_factory=lambda: _env_int("TOK_RATE_LIMIT_BACKOFF_BASE_MS", 150))
    rate_limit_backoff_cap_ms: int = field(default_factory=lambda: _env_int("TOK_RATE_LIMIT_BACKOFF_CAP_MS", 1000))
    rate_limit_throttle_threshold: int = field(default_factory=lambda: _env_int("TOK_RATE_LIMIT_THROTTLE_THRESHOLD", 4))
    rate_limit_throttle_cooldown_sec: int = field(
        default_factory=lambda: _env_int("TOK_RATE_LIMIT_THROTTLE_COOLDOWN_SEC", 20)
    )
    rate_limit_throttle_window_sec: int = field(
        default_factory=lambda: _env_int("TOK_RATE_LIMIT_THROTTLE_WINDOW_SEC", 30)
    )
    _rate_limit_throttle_until: float = 0.0
    _rate_limit_429_history: list[float] = field(default_factory=list)
    _rate_limit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _rate_limit_retry_owner: bool = False
    # TOK_MODE=baseline still forces the conservative baseline path.
    # Otherwise, TOK_REQUEST_POLICY can explicitly override the stable
    # tool-compatible request policy default.
    request_policy_default: str = field(default_factory=_default_request_policy)
    # Kept for compatibility with existing gateway/reporting code.
    tool_compatible_default: bool = True
    memory_dir: Path | None = None
    tracker: SavingsTracker = field(default_factory=SavingsTracker)
    # Canonical runtime state: delegates to this
    runtime_session: RuntimeSession = field(default_factory=RuntimeSession)
    # Smoothness tracking for interaction quality
    smoothness_tracker: SmoothnessTracker = field(default_factory=SmoothnessTracker)

    def __post_init__(self) -> None:
        self.api_base = self.api_base.strip() or ANTHROPIC_API_BASE
        self.request_policy_default = (
            _normalize_request_policy(self.request_policy_default) or _default_request_policy()
        )
        self.tool_compatible_default = self.request_policy_default != "forced_baseline"
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

            sessions_dir = (self.memory_dir or Path.home() / ".tok") / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._capture_file = sessions_dir / f"{ts}.jsonl"
        if explicit_memory_dir:
            self.runtime_session = RuntimeSession(
                memory_dir=self.memory_dir,
                keep_turns=self.keep_turns,
            )
        else:
            self.runtime_session.keep_turns = self.keep_turns
        self.runtime_session._keep_turns_explicit = True
        # Reset session stats so each bridge run starts with a clean slate
        if os.getenv("TOK_RESET_SESSION", "0") == "1":
            self.tracker.reset_session_stats()
            self.runtime_session.reset_session()

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
            sanitized_record = _sanitize_capture_payload(record)
            with open(self._capture_file, "a") as f:
                f.write(json.dumps(sanitized_record) + "\n")
        except Exception as exc:
            logger.debug("Capture write error: %s", exc)

    def capture_event(self, record: dict[str, Any]) -> None:
        """Append a structured capture event to the session file."""
        if not self.capture or self._capture_file is None:
            return
        try:
            sanitized_record = _sanitize_capture_payload(record)
            with open(self._capture_file, "a") as f:
                f.write(json.dumps(sanitized_record) + "\n")
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
        return cast("dict[str, tuple[str, str]]", self.runtime_session.result_cache)

    @property
    def bridge_memory(self) -> BridgeMemoryState:
        """Delegate to runtime session."""
        return self.runtime_session.bridge_memory

    @bridge_memory.setter
    def bridge_memory(self, value: BridgeMemoryState) -> None:
        """Delegate to runtime session."""
        self.runtime_session.bridge_memory = value


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
    client_owned: bool = False,
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
        client_owned=client_owned,
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
        tool_input = dict(block.get("input", {})) if isinstance(block.get("input", {}), dict) else {}
        partial_json = "".join(part for part in block.get("partial_json", []) if isinstance(part, str))
        if partial_json.strip():
            try:
                parsed = json.loads(partial_json)
                if isinstance(parsed, dict):
                    tool_input.update(parsed)
            except json.JSONDecodeError:
                logger.debug("Tool JSON delta parse error: %s", partial_json[:120])
        tool_blocks.append(
            {
                "type": "tool_use",
                "id": block.get("id", ""),
                "name": block.get("name", "unknown"),
                "input": tool_input,
            }
        )
    normalized_tool_blocks, _ = normalize_tool_use_blocks(tool_blocks, seed_prefix="toolu_stream")
    return normalized_tool_blocks


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
    _api_base: str | None = None,
) -> None:
    """Start the bridge server."""
    _port_env: str = os.getenv("TOK_BRIDGE_PORT", os.getenv("TOK_PROXY_PORT", "9090"))
    port = int(port if port is not None else _port_env)
    _keep_turns_env: str = os.getenv("TOK_KEEP_TURNS", os.getenv("TOK_PROXY_KEEP_TURNS", "2"))
    keep_turns = int(keep_turns if keep_turns is not None else _keep_turns_env)
    debug = debug if debug is not None else os.getenv("TOK_DEBUG", "0") == "1"
    fail_open = fail_open if fail_open is not None else os.getenv("TOK_FAIL_OPEN", "1") == "1"
    api_base = (
        _api_base if _api_base is not None else os.getenv("TOK_API_BASE", ANTHROPIC_API_BASE)
    ).strip() or ANTHROPIC_API_BASE

    try:
        TOK_DIR.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))
    except OSError as exc:
        logger.exception(
            "Failed to write PID file %s: %s. Check permissions on %s or set TOK_DIR to a writable location.",
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
            api_base=api_base,
        )
    except Exception as exc:
        logger.exception("Failed to create bridge session: %s", exc)
        raise

    atexit.register(session.tracker.merge_session_to_ledger)
    atexit.register(lambda: PID_FILE.unlink(missing_ok=True))

    try:
        app = create_app(session)
    except Exception as exc:
        logger.exception("Failed to create bridge application: %s", exc)
        raise

    bind_host = _default_bind_host()
    logger.info("Listening on http://%s:%d", bind_host, port)
    logger.info("Keeping last %d human turns verbatim", keep_turns)
    logger.info("Fail-open: %s", "enabled" if fail_open else "disabled")
    logger.info("Upstream API base: %s", session.api_base)
    logger.info(
        "Default Claude bridge mode: %s (request_policy=%s, TOK_MODE=%s, TOK_REQUEST_POLICY=%s)",
        _request_policy_mode_label(session.request_policy_default),
        session.request_policy_default,
        os.getenv("TOK_MODE", "tool-compatible"),
        os.getenv("TOK_REQUEST_POLICY", "<unset>"),
    )

    try:
        uvicorn.run(app, host=bind_host, port=port, log_level=log_level)
    except Exception as exc:
        logger.exception("Bridge server exited unexpectedly on port %d: %s", port, exc)
        raise


if __name__ == "__main__":
    run_bridge()
