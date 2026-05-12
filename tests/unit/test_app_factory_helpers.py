from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tok.gateway import BridgeSession
from tok.gateway._app_factory import (
    _build_response_signals,
    _handle_nonstreaming_failopen,
    _handle_retried_without_tok,
    _rebuild_and_record_response,
)
from tok.runtime.smoothness.models import SmoothnessEvent, SmoothnessEventType, TokMode, TurnSmoothnessReport


def _make_session(**overrides: Any) -> BridgeSession:
    defaults: dict[str, Any] = {"fail_open": True}
    defaults.update(overrides)
    return BridgeSession(**defaults)


def test_handle_retried_without_tok_resets_state() -> None:
    session = _make_session()
    behavior_signals: dict[str, int] = {}
    request_state: dict[str, bool] = {"fallback_recorded": False}

    compressed, saved_toks, tool_breakdown, prompt_metrics = _handle_retried_without_tok(
        behavior_signals,
        session,
        request_state,
    )

    assert compressed is False
    assert saved_toks == 0
    assert tool_breakdown == {}
    assert prompt_metrics == {
        "baseline_prompt_tokens": 0,
        "prepared_prompt_tokens": 0,
        "saved_prompt_tokens": 0,
        "hot_hint_tokens_added": 0,
        "reacquisition_tokens_avoided_estimate": 0,
    }
    assert behavior_signals.get("tok_fail_open_retry") == 1
    assert behavior_signals.get("tok_fallback_activated") == 1
    assert request_state.get("fallback_recorded") is True


def test_build_response_signals_empty_text_passes_through() -> None:
    session = _make_session()
    behavior_signals: dict[str, int] = {}
    resp_json: dict[str, Any] = {"content": [], "model": "test"}

    session.runtime_session._bump_signals({"some_signal": 2})

    response_signals, total_output_saved = _build_response_signals(
        "",
        resp_json,
        session,
        behavior_signals,
        request_tool_compatible=False,
    )

    assert total_output_saved == 0
    assert response_signals.get("some_signal") == 2


def test_handle_nonstreaming_failopen_records_signals() -> None:
    session = _make_session()
    behavior_signals: dict[str, int] = {}
    request_state: dict[str, bool] = {"fallback_recorded": False}
    resp_json: dict[str, Any] = {
        "model": "claude-sonnet-4",
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    exc = RuntimeError("processing failure")

    _handle_nonstreaming_failopen(
        exc,
        session,
        behavior_signals,
        request_state,
        resp_json,
        saved_toks=10,
        compressed=True,
        tool_breakdown={"tool_a": 5},
        prompt_metrics={
            "baseline_prompt_tokens": 100,
            "prepared_prompt_tokens": 90,
            "saved_prompt_tokens": 10,
            "hot_hint_tokens_added": 0,
            "reacquisition_tokens_avoided_estimate": 0,
        },
    )

    assert behavior_signals.get("processing_error") == 1
    assert behavior_signals.get("tok_fallback_activated") == 1
    assert request_state.get("fallback_recorded") is True


def test_handle_nonstreaming_failopen_raises_when_closed() -> None:
    session = _make_session(fail_open=False)
    behavior_signals: dict[str, int] = {}
    request_state: dict[str, bool] = {"fallback_recorded": False}
    resp_json: dict[str, Any] = {}
    exc = RuntimeError("processing failure")

    with pytest.raises(RuntimeError, match="processing failure"):
        _handle_nonstreaming_failopen(
            exc,
            session,
            behavior_signals,
            request_state,
            resp_json,
            saved_toks=0,
            compressed=False,
            tool_breakdown={},
            prompt_metrics={
                "baseline_prompt_tokens": 0,
                "prepared_prompt_tokens": 0,
                "saved_prompt_tokens": 0,
                "hot_hint_tokens_added": 0,
                "reacquisition_tokens_avoided_estimate": 0,
            },
        )


def test_rebuild_and_record_response_calls_tracker() -> None:
    session = _make_session()

    mock_report = TurnSmoothnessReport(
        turn_id="t1",
        task_id="task1",
        score=95,
        labour_index=1,
        mode=TokMode.FULL_TOK,
        events=[
            SmoothnessEvent(
                event_type=SmoothnessEventType.DIRECT_ACTION_AFTER_FIRST_READ,
                turn_id="t1",
                task_id="task1",
                penalty=0,
            ),
        ],
    )
    session.smoothness_tracker.start_turn(task_id="task1")
    session.smoothness_tracker.finish_turn()
    session.smoothness_tracker.start_turn = MagicMock()
    session.smoothness_tracker.finish_turn = MagicMock(return_value=mock_report)

    session.tracker.record_call = MagicMock()

    resp_json: dict[str, Any] = {
        "model": "claude-sonnet-4",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }

    _rebuild_and_record_response(
        resp_json,
        session,
        saved_toks=10,
        compressed=True,
        tool_breakdown={"tool_a": 5},
        response_signals={"some_signal": 1},
        prompt_metrics={
            "baseline_prompt_tokens": 100,
            "prepared_prompt_tokens": 90,
            "saved_prompt_tokens": 10,
            "hot_hint_tokens_added": 0,
            "reacquisition_tokens_avoided_estimate": 0,
        },
        total_output_saved=20,
        request_policy="tool_compatible",
        request_tool_compatible=True,
    )

    session.smoothness_tracker.finish_turn.assert_called_once()
    session.tracker.record_call.assert_called_once()
    call_kwargs = session.tracker.record_call.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4"
    assert call_kwargs["actual_input"] == 100
    assert call_kwargs["actual_output"] == 50
    assert call_kwargs["input_saved"] == 10
    assert call_kwargs["output_saved"] == 20
    assert call_kwargs["behavior_signals"] == {"some_signal": 1}


def test_build_response_signals_normalizes_tool_use_on_textless_response() -> None:
    session = _make_session()
    behavior_signals: dict[str, int] = {}
    tool_use_block = {
        "type": "tool_use",
        "id": "",
        "name": "Read",
        "input": {"file_path": "src/foo.py"},
    }
    resp_json: dict[str, Any] = {
        "content": [tool_use_block.copy()],
        "model": "claude-sonnet-4",
    }

    response_signals, _ = _build_response_signals(
        "",
        resp_json,
        session,
        behavior_signals,
        request_tool_compatible=False,
    )

    assert resp_json["content"][0]["id"] != "", "tool_use id must be normalized even when full_response_text is empty"
    assert "tool_use_blank_id_synthesized" in behavior_signals, (
        "normalization signals must be accumulated into behavior_signals"
    )
