"""Fixture-only probe adapters. Require TOK_UNSTABLE_ADAPTERS=1.

These adapters prove the adapter boundary can map non-Claude JSON shapes
into Tok runtime types. They are not live integrations.
"""

from __future__ import annotations

import json
from typing import Any

from tok.adapters._utils import _render_text
from tok.runtime.core import ProcessedRuntimeResponse, RuntimeRequest
from tok.runtime.types import SurfaceMetadata


class OpenCodeAdapter:
    """Research-only OpenCode probe implementing AdapterProtocol."""

    adapter_kind = "opencode-probe"

    @property
    def surface(self) -> SurfaceMetadata:
        return SurfaceMetadata(
            runtime="opencode",
            adapter=self.adapter_kind,
            input_shape="opencode_fixture_json",
            output_shape="opencode_fixture_json",
            supports_tool_pairs=True,
        )

    def identify_runtime(self) -> str:
        return "opencode"

    def parse_inbound_request(self, raw_bytes: bytes) -> RuntimeRequest:
        data = json.loads(raw_bytes.decode())
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        return RuntimeRequest(
            model=str(data.get("model", "")),
            messages=messages,
            adapter_kind=self.adapter_kind,
            surface=self.surface,
            request_has_tools=bool(data.get("tools")),
            tool_compatible=bool(data.get("tools")),
        )

    def build_outbound_response(
        self,
        processed: ProcessedRuntimeResponse,
        *,
        response_id: str = "",
    ) -> bytes:
        text = _render_text(processed.content_blocks)
        return json.dumps(
            {
                "id": response_id,
                "message": {"role": "assistant", "content": text},
                "tok": {
                    "adapter": self.adapter_kind,
                    "runtime": "opencode",
                    "fixture_only": True,
                },
            },
            sort_keys=True,
        ).encode()

    def supported_capabilities(self) -> frozenset[str]:
        return frozenset({"tool-pair", "stdio", "fixture-only"})

    def transport_boundary(self) -> str:
        return "stdio"


class CodexAdapter:
    """Research-only Codex CLI probe implementing AdapterProtocol."""

    adapter_kind = "codex-cli-probe"

    @property
    def surface(self) -> SurfaceMetadata:
        return SurfaceMetadata(
            runtime="codex-cli",
            adapter=self.adapter_kind,
            input_shape="codex_cli_fixture_json",
            output_shape="codex_cli_fixture_json",
            supports_tool_pairs=True,
        )

    def identify_runtime(self) -> str:
        return "codex-cli"

    def parse_inbound_request(self, raw_bytes: bytes) -> RuntimeRequest:
        data = json.loads(raw_bytes.decode())
        messages = data.get("input", data.get("messages", []))
        if not isinstance(messages, list):
            messages = []
        normalized_messages = [_with_evidence_metadata(message) for message in messages if isinstance(message, dict)]
        requested_capabilities = _requested_capabilities(data)
        unsupported_capabilities = sorted(requested_capabilities - self.supported_capabilities())
        fallback_required = bool(unsupported_capabilities or _has_unknown_tool(data))
        if fallback_required:
            normalized_messages.append(
                {
                    "role": "system",
                    "content": "Tok Codex probe fallback: pass through without compression.",
                    "tok_probe": {
                        "fallback": True,
                        "unsupported_capabilities": unsupported_capabilities,
                    },
                }
            )
        diagnostic_signals = {
            "adapter_probe_parsed": 1,
            "adapter_probe_fallback": int(fallback_required),
            "adapter_probe_unsupported_capability": len(unsupported_capabilities),
        }
        return RuntimeRequest(
            model=str(data.get("model", "")),
            messages=normalized_messages,
            system=_extract_system(data),
            adapter_kind=self.adapter_kind,
            surface=self.surface,
            request_has_tools=bool(data.get("tools")),
            tool_compatible=bool(data.get("tools")) and not fallback_required,
            request_policy="forced_baseline" if fallback_required else "legacy_tool_compatible",
            todo=json.dumps({"diagnostic_signals": diagnostic_signals}, sort_keys=True),
        )

    def build_outbound_response(
        self,
        processed: ProcessedRuntimeResponse,
        *,
        conversation_id: str = "",
    ) -> bytes:
        text = _render_text(processed.content_blocks)
        return json.dumps(
            {
                "conversation_id": conversation_id,
                "output": [{"role": "assistant", "content": text}],
                "tok": {
                    "adapter": self.adapter_kind,
                    "behavior_signals": processed.behavior_signals,
                    "runtime": "codex-cli",
                    "fixture_only": True,
                },
            },
            sort_keys=True,
        ).encode()

    def supported_capabilities(self) -> frozenset[str]:
        return frozenset({"tool-pair", "stdio", "http-proxy", "fixture-only"})

    def transport_boundary(self) -> str:
        return "http-proxy"


def _extract_system(data: dict[str, Any]) -> str | list[dict[str, Any]] | None:
    system = data.get("system")
    if isinstance(system, str | list):
        return system
    messages = data.get("messages", data.get("input", []))
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            return str(message.get("content", ""))
    return None


def _with_evidence_metadata(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    tool_name = str(normalized.get("tool") or normalized.get("tool_name") or normalized.get("name") or "").lower()
    content = str(normalized.get("content", ""))
    evidence_form = normalized.get("evidence_form")
    if evidence_form not in {"exact", "summary", "skeleton", "reference"}:
        evidence_form = _classify_evidence_form(tool_name=tool_name, content=content, message=normalized)
    tok_probe = dict(normalized.get("tok_probe") or {})
    tok_probe["evidence_form"] = evidence_form
    normalized["tok_probe"] = tok_probe
    return normalized


def _classify_evidence_form(*, tool_name: str, content: str, message: dict[str, Any]) -> str:
    if message.get("reference_id") or content.startswith("tok://"):
        return "reference"
    if tool_name in {"read_file", "read", "file_read"} or message.get("path"):
        return "exact"
    lowered = content.lower()
    if tool_name in {"search", "grep", "rg"} or "matches found" in lowered:
        return "summary"
    if tool_name in {"outline", "symbols"} or "class " in content or "def " in content:
        return "skeleton"
    return "summary"


def _requested_capabilities(data: dict[str, Any]) -> frozenset[str]:
    raw_capabilities = data.get("capabilities", [])
    if not isinstance(raw_capabilities, list):
        return frozenset()
    return frozenset(str(value) for value in raw_capabilities)


def _has_unknown_tool(data: dict[str, Any]) -> bool:
    tools = data.get("tools", [])
    if not isinstance(tools, list):
        return False
    known_tools = {"shell", "read", "read_file", "search", "grep", "rg", "outline", "symbols"}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name", "")).lower()
        if name and name not in known_tools:
            return True
    return False


__all__ = ["CodexAdapter", "OpenCodeAdapter"]
