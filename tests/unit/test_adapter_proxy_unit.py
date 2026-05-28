"""Unit tests for gateway._adapter_proxy helper functions and fallback paths."""

from __future__ import annotations

import json

from tok.gateway._adapter_proxy import (
    _adapter_diagnostic_signals,
    _conversation_id,
    _decode_provider_payload,
    _default_provider_echo,
    _extract_response_text,
    handle_adapter_proxy_request,
    unstable_adapters_enabled,
)
from tok.runtime.core import RuntimeSession

# ---------------------------------------------------------------------------
# unstable_adapters_enabled
# ---------------------------------------------------------------------------


def test_unstable_adapters_enabled_returns_true_when_set(monkeypatch) -> None:
    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", "1")
    assert unstable_adapters_enabled() is True


def test_unstable_adapters_enabled_returns_false_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("TOK_UNSTABLE_ADAPTERS", raising=False)
    assert unstable_adapters_enabled() is False


def test_unstable_adapters_enabled_returns_false_for_non_one_value(monkeypatch) -> None:
    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", "true")
    assert unstable_adapters_enabled() is False


def test_unstable_adapters_enabled_returns_true_with_surrounding_whitespace(monkeypatch) -> None:
    # The implementation strips before comparing, so " 1 " is treated as "1".
    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", " 1 ")
    assert unstable_adapters_enabled() is True


# ---------------------------------------------------------------------------
# _default_provider_echo
# ---------------------------------------------------------------------------


def test_default_provider_echo_returns_last_message_content() -> None:
    body = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    result = _default_provider_echo(body)
    assert result["choices"][0]["message"]["content"] == "a"


def test_default_provider_echo_returns_ok_for_empty_messages() -> None:
    result = _default_provider_echo({"messages": []})
    assert result["choices"][0]["message"]["content"] == "ok"


def test_default_provider_echo_returns_ok_when_messages_key_absent() -> None:
    result = _default_provider_echo({})
    assert result["choices"][0]["message"]["content"] == "ok"


def test_default_provider_echo_returns_ok_for_non_dict_last_message() -> None:
    result = _default_provider_echo({"messages": ["plain string"]})
    assert result["choices"][0]["message"]["content"] == "ok"


# ---------------------------------------------------------------------------
# _decode_provider_payload
# ---------------------------------------------------------------------------


def test_decode_provider_payload_passes_dict_through() -> None:
    payload: dict[str, object] = {"choices": []}
    assert _decode_provider_payload(payload) is payload


def test_decode_provider_payload_decodes_json_bytes() -> None:
    raw = json.dumps({"output": "hello"}).encode()
    assert _decode_provider_payload(raw) == {"output": "hello"}


def test_decode_provider_payload_wraps_invalid_json_bytes() -> None:
    result = _decode_provider_payload(b"not-json")
    assert "output" in result


def test_decode_provider_payload_wraps_non_dict_json() -> None:
    result = _decode_provider_payload(json.dumps([1, 2]).encode())
    assert "output" in result


# ---------------------------------------------------------------------------
# _extract_response_text
# ---------------------------------------------------------------------------


def test_extract_response_text_from_openai_choices() -> None:
    payload = {"choices": [{"message": {"content": "hello"}}]}
    assert _extract_response_text(payload) == "hello"


def test_extract_response_text_from_output_string() -> None:
    assert _extract_response_text({"output": "direct"}) == "direct"


def test_extract_response_text_from_output_list() -> None:
    payload = {"output": [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]}
    assert _extract_response_text(payload) == "a\nb"


def test_extract_response_text_returns_empty_for_empty_payload() -> None:
    assert _extract_response_text({}) == ""


def test_extract_response_text_returns_empty_for_empty_choices() -> None:
    assert _extract_response_text({"choices": []}) == ""


# ---------------------------------------------------------------------------
# _adapter_diagnostic_signals
# ---------------------------------------------------------------------------


def test_adapter_diagnostic_signals_returns_empty_for_none() -> None:
    assert _adapter_diagnostic_signals(None) == {}


def test_adapter_diagnostic_signals_parses_valid_todo_json() -> None:
    todo = json.dumps({"diagnostic_signals": {"adapter_probe_parsed": 1, "adapter_probe_fallback": 0}})
    assert _adapter_diagnostic_signals(todo) == {"adapter_probe_parsed": 1, "adapter_probe_fallback": 0}


def test_adapter_diagnostic_signals_returns_empty_for_invalid_json() -> None:
    assert _adapter_diagnostic_signals("{bad") == {}


def test_adapter_diagnostic_signals_returns_empty_when_no_signals_key() -> None:
    assert _adapter_diagnostic_signals(json.dumps({"other": 1})) == {}


def test_adapter_diagnostic_signals_returns_empty_for_non_dict_signals() -> None:
    assert _adapter_diagnostic_signals(json.dumps({"diagnostic_signals": [1, 2]})) == {}


# ---------------------------------------------------------------------------
# _conversation_id
# ---------------------------------------------------------------------------


def test_conversation_id_extracts_from_json() -> None:
    raw = json.dumps({"conversation_id": "abc-123"}).encode()
    assert _conversation_id(raw) == "abc-123"


def test_conversation_id_returns_empty_string_when_missing() -> None:
    assert _conversation_id(json.dumps({"model": "x"}).encode()) == ""


def test_conversation_id_returns_empty_string_for_invalid_json() -> None:
    assert _conversation_id(b"not-json") == ""


# ---------------------------------------------------------------------------
# handle_adapter_proxy_request — runtime prepare failure path
# ---------------------------------------------------------------------------


def test_handle_adapter_proxy_request_fails_open_on_runtime_prepare_error(monkeypatch) -> None:
    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", "1")

    from tok.runtime.core import UniversalTokRuntime

    def _fail(*args, **kwargs):
        raise RuntimeError("simulated runtime failure")

    monkeypatch.setattr(UniversalTokRuntime, "prepare_request", _fail)

    raw = json.dumps({"model": "test-model", "input": [{"role": "user", "content": "hi"}]}).encode()
    result = handle_adapter_proxy_request(raw, session=RuntimeSession())

    assert result.status_code == 200
    assert result.fallback is True
    assert result.diagnostics["fallback_reason"] == "runtime_prepare_failed"
    assert result.diagnostics["execution_path"] == "raw_passthrough"
    assert result.body == raw
