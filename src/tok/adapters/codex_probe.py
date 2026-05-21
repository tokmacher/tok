"""Fixture-only Codex CLI transport boundary probe.

This is not a live Codex CLI integration. It validates Tok's adapter boundary
against a small, local, constructed fixture shape.
"""

from __future__ import annotations

import json
from typing import Any

from tok.runtime.core import ProcessedRuntimeResponse, RuntimeRequest
from tok.runtime.types import SurfaceMetadata


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
        messages = data.get("input", [])
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
        conversation_id: str = "",
    ) -> bytes:
        text = _render_text(processed.content_blocks)
        return json.dumps(
            {
                "conversation_id": conversation_id,
                "output": [{"role": "assistant", "content": text}],
                "tok": {
                    "adapter": self.adapter_kind,
                    "runtime": "codex-cli",
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


__all__ = ["CodexAdapter"]
