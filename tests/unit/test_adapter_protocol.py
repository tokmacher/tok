"""Phase 0.2.5: adapter boundary protocol tests."""

from __future__ import annotations

import json


def test_mock_adapter_satisfies_protocol() -> None:
    from tok.adapters.adapter_protocol import AdapterProtocol
    from tok.runtime.types import ProcessedRuntimeResponse, RuntimeRequest

    class MockAdapter:
        def identify_runtime(self) -> str:
            return "mock-runtime"

        def parse_inbound_request(self, raw_bytes: bytes) -> RuntimeRequest:
            data = json.loads(raw_bytes)
            return RuntimeRequest(model=data["model"], messages=data["messages"], adapter_kind="mock")

        def build_outbound_response(self, processed: ProcessedRuntimeResponse) -> bytes:
            return json.dumps({"content": processed.content_blocks}).encode()

        def supported_capabilities(self) -> frozenset[str]:
            return frozenset({"tool-pair"})

        def transport_boundary(self) -> str:
            return "stdio"

    adapter: AdapterProtocol = MockAdapter()
    request = adapter.parse_inbound_request(b'{"model":"m","messages":[]}')
    assert request.model == "m"
    assert adapter.identify_runtime() == "mock-runtime"
    assert adapter.supported_capabilities() == frozenset({"tool-pair"})


def test_claude_bridge_adapter_implements_protocol() -> None:
    from tok.adapters import ClaudeBridgeAdapter
    from tok.adapters.adapter_protocol import AdapterProtocol

    adapter: AdapterProtocol = ClaudeBridgeAdapter()
    assert adapter.identify_runtime() == "claude-code"
    assert adapter.transport_boundary() == "http-proxy"
    assert "tool-pair" in adapter.supported_capabilities()


def test_claude_bridge_adapter_parses_anthropic_request() -> None:
    from tok.adapters import ClaudeBridgeAdapter

    adapter = ClaudeBridgeAdapter()
    request = adapter.parse_inbound_request(
        json.dumps(
            {
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hello"}],
                "system": "sys",
                "tools": [{"name": "Read"}],
            }
        ).encode()
    )
    assert request.model == "claude-test"
    assert request.surface_runtime == "claude-code"
    assert request.surface_adapter == "claude-bridge"
    assert request.request_has_tools is True


def test_claude_bridge_adapter_builds_anthropic_response_bytes() -> None:
    from tok.adapters import ClaudeBridgeAdapter
    from tok.runtime.types import ProcessedRuntimeResponse

    adapter = ClaudeBridgeAdapter()
    payload = adapter.build_outbound_response(
        ProcessedRuntimeResponse(
            content_blocks=[{"type": "text", "text": "hi"}],
            output_saved_tokens=0,
            behavior_signals={},
            mode="tok",
            family_mode="tok",
            updated_memory="",
        )
    )
    assert json.loads(payload)["content"] == [{"type": "text", "text": "hi"}]


def test_adapter_surfaces_are_consistent() -> None:
    from tok.adapters import ClaudeBridgeAdapter, OpenAIChatAdapter, OrchestratorAdapter, TextLoopAdapter

    assert ClaudeBridgeAdapter().surface.runtime == "claude-code"
    assert ClaudeBridgeAdapter().surface.adapter == "claude-bridge"
    assert OpenAIChatAdapter().surface.adapter == "openai-chat"
    assert TextLoopAdapter().surface.adapter == "text-loop"
    assert OrchestratorAdapter().surface.adapter == "orchestrator"
