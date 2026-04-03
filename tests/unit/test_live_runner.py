from types import SimpleNamespace

from tok.testing.live_runner import LiveAgent


def test_live_agent_passes_prepared_behavior_signals_to_finalize(monkeypatch):
    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_kwargs):
                    return SimpleNamespace(
                        usage=SimpleNamespace(
                            prompt_tokens=120,
                            completion_tokens=30,
                            total_tokens=150,
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
        "tok.testing.live_runner.OpenAI", lambda **_kwargs: FakeClient()
    )
    monkeypatch.setattr("tok.testing.live_runner.config.API_KEY", "test-key")

    agent = LiveAgent(model="openai/gpt-4.1-mini")
    finalize_calls = {}

    def fake_build_chat_messages(*, model, user_text, system_prompt=None):
        return (
            [{"role": "user", "content": user_text}],
            SimpleNamespace(behavior_signals={"repeat_search": 1}),
        )

    def fake_finalize(*, text, model, behavior_signals=None):
        finalize_calls["text"] = text
        finalize_calls["model"] = model
        finalize_calls["behavior_signals"] = behavior_signals
        return SimpleNamespace(content_blocks=[{"type": "text", "text": "ok"}])

    monkeypatch.setattr(
        agent.adapter, "build_chat_messages", fake_build_chat_messages
    )
    monkeypatch.setattr(agent.adapter, "finalize", fake_finalize)
    monkeypatch.setattr(agent.adapter, "visible_text", lambda processed: "ok")

    response, usage = agent("hello")

    assert response == "ok"
    assert usage.total_tokens == 150
    assert finalize_calls["text"] == "plain assistant reply"
    assert finalize_calls["model"] == "openai/gpt-4.1-mini"
    assert finalize_calls["behavior_signals"] == {"repeat_search": 1}
