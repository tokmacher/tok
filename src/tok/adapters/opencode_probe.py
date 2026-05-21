"""Fixture-only OpenCode transport boundary probe.

This module is not a live OpenCode integration. It exists to prove the adapter
boundary can map a simple OpenCode-like JSON shape into Tok runtime types.
"""

from __future__ import annotations

import json
from typing import Any

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


def _render_text(content_blocks: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(block.get("text", "")).strip()
        for block in content_blocks
        if block.get("type") == "text" and str(block.get("text", "")).strip()
    )


__all__ = ["OpenCodeAdapter"]
