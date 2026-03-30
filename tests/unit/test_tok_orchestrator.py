from types import SimpleNamespace

from tok.adapters import OrchestratorAdapter
from tok.tok_orchestrator import TokOrchestrator


def test_orchestrator_adapter_prepare_turn_delegates_to_base_prepare(
    monkeypatch,
):
    adapter = OrchestratorAdapter()
    captured = {}

    def fake_prepare(self, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            body={"system": "sys", "messages": kwargs["messages"]}
        )

    monkeypatch.setattr(type(adapter), "prepare", fake_prepare)

    messages, prepared = adapter.prepare_turn(
        model="google/gemini-2.0-flash-lite-001",
        system_prompt="orchestrator system",
        dynamic_messages=[{"role": "user", "content": "audit the codebase"}],
        grammar="grammar",
        todo="[ ] audit",
        deltas=">>> turns:1|goal:audit",
    )

    assert captured["model"] == "google/gemini-2.0-flash-lite-001"
    assert captured["system"] == "orchestrator system"
    assert captured["messages"] == [
        {"role": "user", "content": "audit the codebase"}
    ]
    assert captured["grammar"] == "grammar"
    assert captured["todo"] == "[ ] audit"
    assert captured["deltas"] == ">>> turns:1|goal:audit"
    assert messages[0]["role"] == "system"
    assert prepared.body["messages"][-1]["content"] == "audit the codebase"


def test_orchestrator_chat_passes_prepared_behavior_signals_to_finalize(
    monkeypatch,
):
    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_kwargs):
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content="plain assistant reply"
                                )
                            )
                        ]
                    )

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(
        "tok.tok_orchestrator.fetch_model_pricing", lambda *_: (0.1, 0.2)
    )
    monkeypatch.setattr(
        "tok.tok_orchestrator.OpenAI", lambda **_kwargs: FakeClient()
    )

    orchestrator = TokOrchestrator(model="openai/gpt-4.1-mini")
    finalize_calls = {}

    def fake_prepare_turn(**kwargs):
        return (
            [{"role": "user", "content": "audit"}],
            SimpleNamespace(
                behavior_signals={"repeat_file_read": 1},
                input_saved_tokens=0,
                type_breakdown={},
            ),
        )

    def fake_finalize(*, text, model, behavior_signals=None):
        finalize_calls["text"] = text
        finalize_calls["model"] = model
        finalize_calls["behavior_signals"] = behavior_signals
        return SimpleNamespace(
            mode="tool-compatible",
            output_saved_tokens=0,
            behavior_signals={},
            content_blocks=[{"type": "text", "text": "ok"}],
            updated_memory="",
        )

    class FakeExecutor:
        @staticmethod
        def execute_normalized_tool(_event):
            return {"status": "SUCCESS", "message": "unused"}

        @staticmethod
        def get_pending_deltas():
            return []

        @staticmethod
        def clear_pending_deltas():
            return None

    monkeypatch.setattr(orchestrator, "_load_grammar", lambda: "")
    monkeypatch.setattr(
        orchestrator, "_get_pulse_prompt", lambda: "tok system"
    )
    monkeypatch.setattr(
        orchestrator.adapter, "prepare_turn", fake_prepare_turn
    )
    monkeypatch.setattr(orchestrator.adapter, "finalize", fake_finalize)
    orchestrator.tracker = SimpleNamespace(record_call=lambda **_kwargs: None)
    monkeypatch.setattr(
        "tok.runtime_tools.get_default_executor", lambda: FakeExecutor()
    )

    result = orchestrator.chat("audit the codebase", verbose=False)

    assert result == "plain assistant reply"
    assert finalize_calls["text"] == "plain assistant reply"
    assert finalize_calls["model"] == "openai/gpt-4.1-mini"
    assert finalize_calls["behavior_signals"] == {"repeat_file_read": 1}


def test_orchestrator_chat_updates_savings_tracker_session(
    monkeypatch, tmp_path
):
    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_kwargs):
                    return SimpleNamespace(
                        usage=SimpleNamespace(
                            prompt_tokens=180,
                            completion_tokens=20,
                        ),
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content="plain assistant reply"
                                )
                            )
                        ],
                    )

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))
    monkeypatch.setattr(
        "tok.tok_orchestrator.fetch_model_pricing", lambda *_: (0.1, 0.2)
    )
    monkeypatch.setattr(
        "tok.tok_orchestrator.OpenAI", lambda **_kwargs: FakeClient()
    )

    orchestrator = TokOrchestrator(model="openai/gpt-4.1-mini")

    def fake_prepare_turn(**kwargs):
        return (
            [{"role": "user", "content": "audit"}],
            SimpleNamespace(
                behavior_signals={"repeat_search": 1},
                input_saved_tokens=60,
                type_breakdown={"tool_result": 120},
            ),
        )

    def fake_finalize(*, text, model, behavior_signals=None):
        return SimpleNamespace(
            mode="tool-compatible",
            output_saved_tokens=9,
            behavior_signals={"tool_compatible_response": 1},
            content_blocks=[{"type": "text", "text": "ok"}],
            updated_memory="",
        )

    class FakeSession:
        @staticmethod
        def consume_behavior_signals():
            return {"answer_anchor_present": 1}

    class FakeExecutor:
        @staticmethod
        def execute_normalized_tool(_event):
            return {"status": "SUCCESS", "message": "unused"}

        @staticmethod
        def get_pending_deltas():
            return []

        @staticmethod
        def clear_pending_deltas():
            return None

    orchestrator.adapter.session = FakeSession()
    monkeypatch.setattr(orchestrator, "_load_grammar", lambda: "")
    monkeypatch.setattr(
        orchestrator, "_get_pulse_prompt", lambda: "tok system"
    )
    monkeypatch.setattr(
        orchestrator.adapter, "prepare_turn", fake_prepare_turn
    )
    monkeypatch.setattr(orchestrator.adapter, "finalize", fake_finalize)
    monkeypatch.setattr(
        "tok.runtime_tools.get_default_executor", lambda: FakeExecutor()
    )

    orchestrator.chat("audit the codebase", verbose=False)

    summary = orchestrator.tracker.session_summary()

    assert summary is not None
    assert int(summary["actual_tokens"]) == 200
    assert int(summary["baseline_tokens"]) == 269
    assert int(summary["tokens_saved"]) == 69


def test_orchestrator_chat_records_savings_tracker_metrics(monkeypatch):
    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_kwargs):
                    return SimpleNamespace(
                        usage=SimpleNamespace(
                            prompt_tokens=220,
                            completion_tokens=40,
                        ),
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content="plain assistant reply"
                                )
                            )
                        ],
                    )

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(
        "tok.tok_orchestrator.fetch_model_pricing", lambda *_: (0.1, 0.2)
    )
    monkeypatch.setattr(
        "tok.tok_orchestrator.OpenAI", lambda **_kwargs: FakeClient()
    )

    orchestrator = TokOrchestrator(model="openai/gpt-4.1-mini")
    tracker_calls = {}

    def fake_prepare_turn(**kwargs):
        return (
            [{"role": "user", "content": "audit"}],
            SimpleNamespace(
                behavior_signals={"repeat_search": 1},
                input_saved_tokens=90,
                type_breakdown={"tool_result": 120},
            ),
        )

    def fake_finalize(*, text, model, behavior_signals=None):
        # In the real path, process_response() merges session signals internally.
        # Simulate that by including the session signal in the returned behavior_signals.
        return SimpleNamespace(
            mode="tool-compatible",
            output_saved_tokens=11,
            behavior_signals={
                "tool_compatible_response": 1,
                "answer_anchor_present": 1,
            },
            content_blocks=[{"type": "text", "text": "ok"}],
            updated_memory="",
        )

    class FakeSession:
        @staticmethod
        def consume_behavior_signals():
            return {}

    class FakeExecutor:
        @staticmethod
        def execute_normalized_tool(_event):
            return {"status": "SUCCESS", "message": "unused"}

        @staticmethod
        def get_pending_deltas():
            return []

        @staticmethod
        def clear_pending_deltas():
            return None

    orchestrator.adapter.session = FakeSession()
    orchestrator.tracker = SimpleNamespace(
        record_call=lambda **kwargs: tracker_calls.update(kwargs)
    )
    monkeypatch.setattr(orchestrator, "_load_grammar", lambda: "")
    monkeypatch.setattr(
        orchestrator, "_get_pulse_prompt", lambda: "tok system"
    )
    monkeypatch.setattr(
        orchestrator.adapter, "prepare_turn", fake_prepare_turn
    )
    monkeypatch.setattr(orchestrator.adapter, "finalize", fake_finalize)
    monkeypatch.setattr(
        "tok.runtime_tools.get_default_executor", lambda: FakeExecutor()
    )

    result = orchestrator.chat("audit the codebase", verbose=False)

    assert result == "plain assistant reply"
    assert tracker_calls["model"] == "openai/gpt-4.1-mini"
    assert tracker_calls["actual_input"] == 220
    assert tracker_calls["actual_output"] == 40
    assert tracker_calls["input_saved"] == 90
    assert tracker_calls["output_saved"] == 11
    assert tracker_calls["type_breakdown"] == {"tool_result": 120}
    assert tracker_calls["behavior_signals"]["tool_compatible_response"] == 1
    assert tracker_calls["behavior_signals"]["answer_anchor_present"] == 1


def test_orchestrator_handshake_routes_response_through_finalize(monkeypatch):
    class FakeChunk:
        def __init__(self, content):
            self.choices = [
                SimpleNamespace(delta=SimpleNamespace(content=content))
            ]

    class FakeStream:
        usage = None

        def __iter__(self):
            yield FakeChunk(">>> t:1|agt:handshake|state:init")

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_kwargs):
                    return FakeStream()

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(
        "tok.tok_orchestrator.fetch_model_pricing", lambda *_: (0.1, 0.2)
    )
    monkeypatch.setattr(
        "tok.tok_orchestrator.OpenAI", lambda **_kwargs: FakeClient()
    )

    orchestrator = TokOrchestrator(model="openai/gpt-4.1-mini")
    finalize_calls = {}

    def fake_finalize(*, text, model, behavior_signals=None):
        finalize_calls["text"] = text
        finalize_calls["model"] = model
        return SimpleNamespace(
            mode="tok-native",
            output_saved_tokens=0,
            behavior_signals={},
            content_blocks=[],
            updated_memory="",
        )

    monkeypatch.setattr(orchestrator.adapter, "finalize", fake_finalize)
    monkeypatch.setattr("builtins.print", lambda *_a, **_kw: None)

    orchestrator.handshake()

    assert (
        "text" in finalize_calls
    ), "adapter.finalize() was never called during handshake()"
    assert "t:1|agt:handshake" in finalize_calls["text"]
    assert finalize_calls["model"] == "openai/gpt-4.1-mini"
