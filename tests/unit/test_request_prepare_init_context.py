from __future__ import annotations

from dataclasses import fields

from tok.runtime.core import RuntimeSession
from tok.runtime.memory.bridge_memory import MemoryEntry
from tok.runtime.pipeline._prepare_init_context import Step1Result, run_step_1
from tok.runtime.types import RuntimeRequest


def _make_request(**overrides) -> RuntimeRequest:
    defaults = dict(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "hello"}],
        adapter_kind="unknown",
        tool_compatible=False,
    )
    defaults.update(overrides)
    return RuntimeRequest(**defaults)


class TestStep1ResultDefaults:
    def test_step1_result_has_correct_defaults(self) -> None:
        r = Step1Result()
        assert r.body == {}
        assert r.original_body == {}
        assert r.thinking_snapshot is None
        assert r.last_user_msg == ""
        assert r.is_bridge_adapter is False
        assert r.initial_answer_facts_present is False
        assert r.initial_exact_search_evidence_present is False
        assert r.compressed is False
        assert r.pre_existing_session_signals == {}
        assert r.seen_mutation_pairs == set()

    def test_step1_result_all_fields_present(self) -> None:
        expected = {
            "body",
            "original_body",
            "thinking_snapshot",
            "last_user_msg",
            "is_bridge_adapter",
            "initial_answer_facts_present",
            "initial_exact_search_evidence_present",
            "compressed",
            "pre_existing_session_signals",
            "seen_mutation_pairs",
        }
        actual = {f.name for f in fields(Step1Result)}
        assert actual == expected


class TestStep1InitContext:
    def test_basic_request_sets_body_and_original(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        result = run_step_1(req, session)
        assert result.body["model"] == "claude-sonnet-4"
        assert result.body["messages"] == [{"role": "user", "content": "hello"}]
        assert result.original_body == result.body
        assert result.body is not result.original_body

    def test_system_prompt_copied_into_body(self) -> None:
        session = RuntimeSession()
        req = _make_request(system="You are a helpful assistant.")
        result = run_step_1(req, session)
        assert result.body["system"] == "You are a helpful assistant."

    def test_no_system_prompt_means_no_system_key(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        result = run_step_1(req, session)
        assert "system" not in result.body

    def test_last_user_msg_extracted(self) -> None:
        session = RuntimeSession()
        req = _make_request(
            messages=[
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "what is 2+2?"},
            ]
        )
        result = run_step_1(req, session)
        assert result.last_user_msg == "what is 2+2?"

    def test_last_user_msg_from_block_content(self) -> None:
        session = RuntimeSession()
        req = _make_request(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "block msg"}],
                }
            ]
        )
        result = run_step_1(req, session)
        assert result.last_user_msg == "block msg"

    def test_empty_messages_gives_empty_last_user_msg(self) -> None:
        session = RuntimeSession()
        req = _make_request(messages=[])
        result = run_step_1(req, session)
        assert result.last_user_msg == ""

    def test_bridge_adapter_detected(self) -> None:
        session = RuntimeSession()
        req = _make_request(adapter_kind="claude-bridge")
        result = run_step_1(req, session)
        assert result.is_bridge_adapter is True

    def test_orchestrator_is_bridge_adapter(self) -> None:
        session = RuntimeSession()
        req = _make_request(adapter_kind="orchestrator")
        result = run_step_1(req, session)
        assert result.is_bridge_adapter is True

    def test_unknown_adapter_is_not_bridge(self) -> None:
        session = RuntimeSession()
        req = _make_request(adapter_kind="openai")
        result = run_step_1(req, session)
        assert result.is_bridge_adapter is False

    def test_session_flags_set(self) -> None:
        session = RuntimeSession()
        req = _make_request(request_has_tools=True)
        run_step_1(req, session)
        assert session._request_has_tools is True
        assert session._answer_phase_expected_this_turn is False
        assert session._natural_response_acceptable_this_turn is False

    def test_no_tools_flag_set(self) -> None:
        session = RuntimeSession()
        req = _make_request(request_has_tools=False)
        run_step_1(req, session)
        assert session._request_has_tools is False

    def test_initial_answer_facts_present(self) -> None:
        session = RuntimeSession()
        session.bridge_memory.hot["facts"] = [MemoryEntry(value="answer_step_complete")]
        req = _make_request()
        result = run_step_1(req, session)
        assert result.initial_answer_facts_present is True

    def test_initial_answer_facts_absent(self) -> None:
        session = RuntimeSession()
        session.bridge_memory.hot["facts"] = [MemoryEntry(value="not_answer_related")]
        req = _make_request()
        result = run_step_1(req, session)
        assert result.initial_answer_facts_present is False

    def test_thinking_snapshot_none_for_no_thinking(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        result = run_step_1(req, session)
        assert result.thinking_snapshot is None

    def test_thinking_snapshot_captured(self) -> None:
        session = RuntimeSession()
        req = _make_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "deep thought"},
                        {"type": "text", "text": "answer"},
                    ],
                },
                {"role": "user", "content": "follow-up"},
            ]
        )
        result = run_step_1(req, session)
        assert result.thinking_snapshot is not None
        import json

        snap = json.loads(result.thinking_snapshot)
        assert "full_content" in snap
        assert "content_hash" in snap
        assert snap["block_types"] == ["thinking", "text"]

    def test_pre_existing_session_signals_captured(self) -> None:
        session = RuntimeSession()
        session.pending_behavior_signals["existing_signal"] = 5
        req = _make_request()
        result = run_step_1(req, session)
        assert result.pre_existing_session_signals == {"existing_signal": 5}

    def test_compressed_defaults_false(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        result = run_step_1(req, session)
        assert result.compressed is False

    def test_seen_mutation_pairs_defaults_empty(self) -> None:
        session = RuntimeSession()
        req = _make_request()
        result = run_step_1(req, session)
        assert result.seen_mutation_pairs == set()
