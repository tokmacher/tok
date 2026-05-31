"""Gated local proxy path for non-Claude client adapter proofs.

This module is intentionally separate from the defended Claude Code bridge. It proves
the adapter -> canonical runtime -> provider shape without adding Claude-specific
branches to the existing bridge server.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response

from tok.adapters.adapter_config import get_adapter_probe, parse_or_fail_open
from tok.runtime.core import RuntimeSession, UniversalTokRuntime

ProviderForwarder = Callable[[dict[str, Any]], bytes | dict[str, Any]]


@dataclass
class AdapterProxyResult:
    status_code: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)
    fallback: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


def handle_adapter_proxy_request(
    raw_body: bytes,
    *,
    adapter_name: str = "codex-cli",
    provider_forwarder: ProviderForwarder | None = None,
    session: RuntimeSession | None = None,
    runtime: UniversalTokRuntime | None = None,
) -> AdapterProxyResult:
    """Parse client bytes, route through Tok runtime, and build adapter response.

    If parsing or runtime preparation fails, this returns the original body with
    fallback diagnostics so callers can pass through unchanged.
    """
    adapter = get_adapter_probe(adapter_name)
    request = parse_or_fail_open(adapter, raw_body)
    if request is None:
        return AdapterProxyResult(
            status_code=200,
            body=raw_body,
            fallback=True,
            diagnostics={
                "adapter": adapter_name,
                "fallback": True,
                "fallback_reason": "adapter_parse_failed",
                "execution_path": "raw_passthrough",
            },
        )

    active_session = session or RuntimeSession()
    active_runtime = runtime or UniversalTokRuntime()
    try:
        prepared = active_runtime.prepare_request(request, active_session)
    except Exception:
        return AdapterProxyResult(
            status_code=200,
            body=raw_body,
            fallback=True,
            diagnostics={
                "adapter": adapter_name,
                "fallback": True,
                "fallback_reason": "runtime_prepare_failed",
                "execution_path": "raw_passthrough",
            },
        )

    provider_body = provider_forwarder(prepared.body) if provider_forwarder else _default_provider_echo(prepared.body)
    provider_payload = _decode_provider_payload(provider_body)
    response_text = _extract_response_text(provider_payload)
    adapter_signals = _adapter_diagnostic_signals(request.todo)
    processed = active_runtime.process_response(
        response_text,
        model=request.model,
        session=active_session,
        behavior_signals={
            **prepared.behavior_signals,
            **adapter_signals,
            "adapter_proxy_request": 1,
        },
        tool_compatible=prepared.effective_tool_compatible,
    )
    body = adapter.build_outbound_response(processed, conversation_id=_conversation_id(raw_body))
    diagnostics = {
        "adapter": adapter_name,
        "fallback": False,
        "execution_path": "adapter-proxy",
        "surface_runtime": request.surface_runtime,
        "surface_adapter": request.surface_adapter,
        "request_policy": prepared.request_policy,
        "compressed": prepared.compressed,
        "input_saved_tokens": prepared.input_saved_tokens,
        "behavior_signals": dict(processed.behavior_signals),
    }
    return AdapterProxyResult(status_code=200, body=body, fallback=False, diagnostics=diagnostics)


def create_adapter_proxy_app(
    *,
    adapter_name: str = "codex-cli",
    provider_url: str | None = None,
) -> FastAPI:
    """Create a local-only FastAPI app for the gated local adapter proof."""
    app = FastAPI(title="Tok adapter proxy")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy(path: str, request: Request) -> Response:
        raw_body = await request.body()
        forwarder = _http_forwarder(provider_url, path, dict(request.headers)) if provider_url else None
        result = handle_adapter_proxy_request(
            raw_body,
            adapter_name=adapter_name,
            provider_forwarder=forwarder,
        )
        headers = dict(result.headers)
        headers["x-tok-adapter"] = adapter_name
        headers["x-tok-fallback"] = "1" if result.fallback else "0"
        return Response(
            content=result.body, status_code=result.status_code, headers=headers, media_type="application/json"
        )

    return app


def _http_forwarder(provider_url: str | None, path: str, headers: dict[str, str]) -> ProviderForwarder | None:
    if not provider_url:
        return None

    def _forward(body: dict[str, Any]) -> bytes:
        url = provider_url.rstrip("/") + "/" + path.lstrip("/")
        filtered_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in {"host", "content-length", "transfer-encoding"}
        }
        response = httpx.post(url, json=body, headers=filtered_headers, timeout=60.0)
        return response.content

    return _forward


def _default_provider_echo(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages", [])
    if isinstance(messages, list) and messages:
        text = str(messages[-1].get("content", "")) if isinstance(messages[-1], dict) else ""
    else:
        text = ""
    return {"choices": [{"message": {"content": text or "ok"}}]}


def _decode_provider_payload(payload: bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except Exception:
        return {"output": payload.decode("utf-8", errors="replace")}
    return decoded if isinstance(decoded, dict) else {"output": str(decoded)}


def _extract_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                return str(message.get("content", ""))
            return str(first.get("text", ""))
    output = payload.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = [str(item.get("content", "")) for item in output if isinstance(item, dict)]
        return "\n".join(part for part in parts if part)
    return ""


def _adapter_diagnostic_signals(todo: str | None) -> dict[str, int]:
    if not todo:
        return {}
    try:
        payload = json.loads(todo)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    raw_signals = payload.get("diagnostic_signals", {})
    if not isinstance(raw_signals, dict):
        return {}
    return {str(key): int(value) for key, value in raw_signals.items() if isinstance(value, int | float | str)}


def _conversation_id(raw_body: bytes) -> str:
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return ""
    return str(payload.get("conversation_id", "")) if isinstance(payload, dict) else ""


def unstable_adapters_enabled() -> bool:
    return os.getenv("TOK_UNSTABLE_ADAPTERS", "").strip() == "1"


__all__ = [
    "AdapterProxyResult",
    "create_adapter_proxy_app",
    "handle_adapter_proxy_request",
    "unstable_adapters_enabled",
]
