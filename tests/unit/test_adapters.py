from tok.adapters import (
    OpenAIChatAdapter,
    OrchestratorAdapter,
    TextLoopAdapter,
)


def test_openai_chat_adapter_builds_system_and_user_messages() -> None:
    adapter = OpenAIChatAdapter()
    adapter.session.bridge_memory.turn = 10

    messages, prepared = adapter.build_chat_messages(
        model="claude-sonnet-4",
        user_text="Fix the gateway",
        system_prompt="Existing system prompt",
    )

    assert messages[0]["role"] == "system"
    assert "TOK" in messages[0]["content"]
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
