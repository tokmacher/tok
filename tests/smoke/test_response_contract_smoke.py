from __future__ import annotations

from tok.runtime.pipeline.response_processing import response_contract_for_mode


def test_smoke_response_contract_classifies_tok_native_and_plain_text() -> None:
    tok_text = ">>> turns:1|goal:smoke\n@msg role:assistant\n  |> bridge response ok"
    plain_text = "This is a complete non-streaming response."

    native = response_contract_for_mode(tok_text, tool_compatible=False)
    plain = response_contract_for_mode(plain_text, tool_compatible=False)

    assert native.mode == "tok-native"
    assert native.behavior_signals.get("tok_native_response") == 1
    assert native.content_blocks == [{"type": "text", "text": "bridge response ok"}]

    assert plain.mode == "markdown"
    assert plain.behavior_signals.get("non_tok_response") == 1
    assert plain.behavior_signals.get("tok_native_response", 0) == 0
