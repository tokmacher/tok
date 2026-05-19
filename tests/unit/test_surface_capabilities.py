"""Surface capability contract tests for runtime-neutral adapter integration."""

from __future__ import annotations

import pytest

from tok.gateway import BridgeSession
from tok.gateway._bridge_runtime_pipeline import prepare_bridge_payload
from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline._prepare_bridge_cut_search import Step7aResult, run_step_7a_bridge_cut_search
from tok.runtime.types import RuntimeRequest, SignalPacket, SurfaceMetadata


def test_claude_bridge_surface_declares_uses_cut_search() -> None:
    """claude_bridge() surface must advertise uses_cut_search=True."""
    assert SurfaceMetadata.claude_bridge().uses_cut_search is True
    assert SurfaceMetadata.claude_bridge().uses_plan_finalization_guard is True
    assert SurfaceMetadata.claude_bridge().uses_first_turn_broad_audit_guard is True


def test_non_bridge_surfaces_do_not_declare_uses_cut_search() -> None:
    """Every other adapter kind must have uses_cut_search=False."""
    for adapter in ("openai-chat", "orchestrator", "text-loop", "unknown", ""):
        surface = SurfaceMetadata.from_adapter_kind(adapter)
        assert surface.uses_cut_search is False, f"expected False for adapter={adapter!r}"
        assert surface.uses_plan_finalization_guard is False
        assert surface.uses_first_turn_broad_audit_guard is False


def test_explicit_custom_surface_defaults_uses_cut_search_to_false() -> None:
    """Manually constructed surfaces default to uses_cut_search=False."""
    custom = SurfaceMetadata(runtime="my-rt", adapter="my-adapter")
    assert custom.uses_cut_search is False


def test_explicit_custom_surface_can_opt_in_to_cut_search() -> None:
    """A custom surface may set uses_cut_search=True explicitly."""
    custom = SurfaceMetadata(runtime="my-rt", adapter="my-adapter", uses_cut_search=True)
    assert custom.uses_cut_search is True


def test_runtime_request_exposes_uses_cut_search_from_bridge_surface() -> None:
    req = RuntimeRequest(
        model="m",
        messages=[],
        surface=SurfaceMetadata.claude_bridge(),
    )
    assert req.uses_cut_search is True


def test_runtime_request_exposes_uses_cut_search_false_for_non_bridge() -> None:
    req = RuntimeRequest(
        model="m",
        messages=[],
        surface=SurfaceMetadata.from_adapter_kind("orchestrator"),
    )
    assert req.uses_cut_search is False


def test_runtime_request_uses_cut_search_follows_custom_surface() -> None:
    custom = SurfaceMetadata(runtime="x", adapter="x", uses_cut_search=True)
    req = RuntimeRequest(model="m", messages=[], surface=custom)
    assert req.uses_cut_search is True


def test_codex_cli_like_surface_can_enter_core_without_claude_bridge_capabilities() -> None:
    """A future Codex CLI adapter should not need to masquerade as Claude Code."""
    surface = SurfaceMetadata(
        runtime="codex-cli",
        adapter="codex-cli",
        input_shape="codex_cli_transcript",
        output_shape="codex_cli_response",
    )
    request = RuntimeRequest(model="gpt-5.5-codex", messages=[], surface=surface)
    packet = SignalPacket.from_request(request)

    assert request.surface_runtime == "codex-cli"
    assert request.surface_adapter == "codex-cli"
    assert request.uses_bridge_profile is False
    assert request.supports_tool_pairs is False
    assert request.requires_provider_canonicalization is False
    assert request.uses_cut_search is False
    assert request.uses_plan_finalization_guard is False
    assert request.uses_first_turn_broad_audit_guard is False
    assert packet.observability["surface_input_shape"] == "codex_cli_transcript"


def test_cut_search_guard_respects_uses_cut_search_not_adapter_string(tmp_path: pytest.TempdirFactory) -> None:
    """A surface with uses_cut_search=True but adapter!='claude-bridge' must pass the guard.

    The guard should follow request.uses_cut_search, so this custom surface is
    allowed to reach the cut-search logic without using the Claude adapter name.
    """
    custom_surface = SurfaceMetadata(
        runtime="custom-rt",
        adapter="custom-not-bridge",
        supports_tool_pairs=True,
        uses_bridge_profile=True,
        requires_provider_canonicalization=True,
        uses_cut_search=True,
    )
    recent: list[dict] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Bash",
                    "input": {"command": "cat README.md"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "# My project\nThis is a README.",
                }
            ],
        },
    ]
    request = RuntimeRequest(
        model="claude-sonnet-4",
        messages=recent,
        surface=custom_surface,
    )
    session = RuntimeSession(memory_dir=tmp_path / "cut-search-cap")

    result: Step7aResult = run_step_7a_bridge_cut_search(
        session=session,
        request=request,
        recent=recent,
        original_messages=recent,
        system=None,
        id_to_context={},
        keep_turns=4,
        bridge_keep_turns=4,
        bridge_profile={"_bridge_cut_search": 1},
        history_baseline_prompt_tokens=0,
        seen_mutation_pairs=None,
        preserve_exact_search_evidence=False,
        exact_search_evidence_keys_in_request=set(),
        _first_exact_evidence_seen_for_compression=frozenset(),
        effective_tool_compatible=True,
    )

    # Guard passed → cut-search was at least attempted (entry signal set)
    # The cut-search may still find no savings (bridge_search_success can be False)
    # but the guard must NOT have been the reason for an immediate empty return.
    assert result.behavior_signals.get("bridge_cut_search_guard_passed", 0) == 1


def test_non_cut_search_surface_is_still_blocked_by_guard(tmp_path: pytest.TempdirFactory) -> None:
    """A surface with uses_cut_search=False should still get the early return."""
    orchestrator_surface = SurfaceMetadata.from_adapter_kind("orchestrator")
    assert orchestrator_surface.uses_cut_search is False

    recent: list[dict] = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "/f"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "data"}],
        },
    ]
    request = RuntimeRequest(
        model="claude-sonnet-4",
        messages=recent,
        surface=orchestrator_surface,
    )
    session = RuntimeSession(memory_dir=tmp_path / "orchestrator")

    result: Step7aResult = run_step_7a_bridge_cut_search(
        session=session,
        request=request,
        recent=recent,
        original_messages=recent,
        system=None,
        id_to_context={},
        keep_turns=4,
        bridge_keep_turns=4,
        bridge_profile={},
        history_baseline_prompt_tokens=0,
        seen_mutation_pairs=None,
        preserve_exact_search_evidence=False,
        exact_search_evidence_keys_in_request=set(),
        _first_exact_evidence_seen_for_compression=frozenset(),
        effective_tool_compatible=True,
    )

    # Guard blocked → no entry signal
    assert result.behavior_signals.get("bridge_cut_search_guard_passed", 0) == 0
    assert result.bridge_search_success is False


def test_streaming_and_non_streaming_produce_same_safety_signals(tmp_path: pytest.TempdirFactory) -> None:
    """prepare_bridge_payload must reach the same safety decision regardless of stream flag.

    This is a contract test that documents the invariant and will catch future
    regressions where gateway streaming branches are added before safety checks.
    """
    base_body = {
        "model": "claude-sonnet-4",
        "max_tokens": 4096,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Read the file foo.py and summarize it."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "Read",
                        "input": {"file_path": "foo.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": "def main(): pass",
                    }
                ],
            },
            {"role": "user", "content": "Now summarize it."},
        ],
    }

    session_stream = BridgeSession()
    session_stream.runtime_session.memory_dir = tmp_path / "stream"
    streaming_body = {**base_body, "stream": True}
    payload_stream, _ = prepare_bridge_payload(
        session=session_stream,
        body=streaming_body,
        headers={},
        path="v1/messages",
    )

    session_nostream = BridgeSession()
    session_nostream.runtime_session.memory_dir = tmp_path / "nostream"
    nonstreaming_body = {**base_body, "stream": False}
    payload_nostream, _ = prepare_bridge_payload(
        session=session_nostream,
        body=nonstreaming_body,
        headers={},
        path="v1/messages",
    )

    # Safety-relevant signals must be identical regardless of stream flag
    _SAFETY_SIGNAL_KEYS = {
        "plan_finalization_turn",
        "plan_finalization_passthrough",
        "tok_bridge_preflight_rejected",
        "tok_fallback_zero_compression_revert",
    }
    stream_safety = {k: payload_stream.behavior_signals.get(k, 0) for k in _SAFETY_SIGNAL_KEYS}
    nostream_safety = {k: payload_nostream.behavior_signals.get(k, 0) for k in _SAFETY_SIGNAL_KEYS}
    assert stream_safety == nostream_safety, (
        f"Safety signals differ: stream={stream_safety}, non-stream={nostream_safety}"
    )


def test_bridge_payload_carries_surface_runtime_field(tmp_path: pytest.TempdirFactory) -> None:
    """BridgePreparedPayload must expose surface_runtime for request-level diagnostics.

    Early preflight exits and fully prepared requests should both carry this field.
    """
    session = BridgeSession()
    session.runtime_session.memory_dir = tmp_path / "trace"
    body = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload, _ = prepare_bridge_payload(
        session=session,
        body=body,
        headers={},
        path="v1/messages",
    )
    assert payload.surface_runtime == "claude-code"


def test_bridge_payload_carries_surface_adapter_field(tmp_path: pytest.TempdirFactory) -> None:
    """BridgePreparedPayload must expose surface_adapter for request-level diagnostics."""
    session = BridgeSession()
    session.runtime_session.memory_dir = tmp_path / "trace2"
    body = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload, _ = prepare_bridge_payload(
        session=session,
        body=body,
        headers={},
        path="v1/messages",
    )
    assert payload.surface_adapter == "claude-bridge"


def test_non_message_bridge_payload_still_carries_surface_fields(tmp_path: pytest.TempdirFactory) -> None:
    """Diagnostics should retain surface fields even before runtime preparation runs."""
    session = BridgeSession()
    session.runtime_session.memory_dir = tmp_path / "count-tokens"
    body = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload, _ = prepare_bridge_payload(
        session=session,
        body=body,
        headers={},
        path="v1/messages/count_tokens",
    )

    assert payload.surface_runtime == "claude-code"
    assert payload.surface_adapter == "claude-bridge"


def test_signal_packet_observability_includes_surface_runtime_and_adapter() -> None:
    """SignalPacket.from_request must include surface_runtime and surface_adapter in observability."""
    request = RuntimeRequest(
        model="m",
        messages=[],
        surface=SurfaceMetadata.claude_bridge(),
    )
    packet = SignalPacket.from_request(request)
    assert packet.observability["surface_runtime"] == "claude-code"
    assert packet.observability["surface_adapter"] == "claude-bridge"
