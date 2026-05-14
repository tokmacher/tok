from tok.adapters import (
    OpenAIChatAdapter,
    OrchestratorAdapter,
    TextLoopAdapter,
)
from tok.runtime.core import RuntimeSession, UniversalTokRuntime
from tok.runtime.types import RuntimeRequest, SignalPacket, SurfaceMetadata


def test_openai_chat_adapter_builds_system_and_user_messages() -> None:
    adapter = OpenAIChatAdapter()
    adapter.session.bridge_memory.turn = 10

    messages, prepared = adapter.build_chat_messages(
        model="claude-sonnet-4",
        user_text="Fix the gateway",
        system_prompt="Existing system prompt",
    )

    assert messages[0]["role"] == "system"
    assert "Existing system prompt" in str(messages[0]["content"])
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Fix the gateway"
    assert prepared.body["messages"][-1]["content"] == "Fix the gateway"


def test_text_loop_adapter_finalizes_runtime_response() -> None:
    adapter = TextLoopAdapter()
    _, _prepared = adapter.prepare_messages(
        model="google/gemini-2.0-flash",
        messages=[{"role": "user", "content": "@msg role:user\n  |> hello"}],
        system_prompt="tok system",
    )

    processed = adapter.finalize(
        text=">>> turns:1|goal:hello\n@msg role:assistant\n  |> ok",
        model="google/gemini-2.0-flash",
    )

    assert processed.mode == "tok-native"
    assert processed.content_blocks == [{"type": "text", "text": "ok"}]
    assert processed.updated_memory == ">>> turns:1|goal:hello"


def test_orchestrator_adapter_prepares_dynamic_messages_with_runtime_contract() -> None:
    adapter = OrchestratorAdapter()

    messages, _prepared = adapter.prepare_turn(
        model="google/gemini-2.0-flash-lite-001",
        system_prompt="orchestrator system",
        dynamic_messages=[
            {"role": "system", "content": "dynamic state"},
            {"role": "user", "content": "audit the codebase"},
        ],
    )

    assert messages[0]["role"] == "system"
    assert "orchestrator system" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"][-1]["text"] == "audit the codebase"
    assert messages[-1]["content"][-1]["type"] == "text"


def test_orchestrator_and_text_loop_finalize_with_same_runtime_contract() -> None:
    orchestrator = OrchestratorAdapter()
    text_loop = TextLoopAdapter()
    payload = ">>> turns:1|goal:hello\n@msg role:assistant\n  |> ok"

    orchestrated = orchestrator.finalize(
        text=payload,
        model="google/gemini-2.0-flash-lite-001",
        behavior_signals={"repeat_file_read": 1},
    )
    looped = text_loop.finalize(
        text=payload,
        model="google/gemini-2.0-flash-lite-001",
        behavior_signals={"repeat_file_read": 1},
    )

    assert orchestrated.mode == looped.mode == "tok-native"
    assert orchestrated.content_blocks == looped.content_blocks
    assert orchestrated.updated_memory == looped.updated_memory
    assert orchestrated.behavior_signals == looped.behavior_signals


def test_fake_surface_enters_same_signal_packet_core_path(tmp_path) -> None:
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / "fake-surface")
    request = RuntimeRequest(
        model="fake-frontier-runtime",
        messages=[{"role": "user", "content": "Summarize the signal path."}],
        system="fake runtime shell",
        adapter_kind="fake-agent-shell",
        surface=SurfaceMetadata(
            runtime="fake-agent-runtime",
            adapter="fake-agent-shell",
            input_shape="fake_messages",
            output_shape="fake_messages",
        ),
    )
    packet = SignalPacket.from_request(request)

    prepared = runtime.prepare_signal_packet(packet, session)

    assert prepared.surface.runtime == "fake-agent-runtime"
    assert prepared.surface.adapter == "fake-agent-shell"
    assert prepared.surface.input_shape == "fake_messages"
    assert prepared.body["messages"][-1]["content"] == "Summarize the signal path."
    assert "claude" not in prepared.surface.runtime
    assert packet.observability["core_path"] == "runtime.prepare_request"
    assert packet.observability["surface_runtime"] == "fake-agent-runtime"
