from __future__ import annotations

from dataclasses import fields

from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline._prepare_bridge_cut_search import Step7aResult
from tok.runtime.pipeline._prepare_compress_history import Step7Result
from tok.runtime.pipeline.context_dependency import ContextDependencyDecision
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


class TestStep7ResultDefaults:
    def test_step7_result_has_correct_defaults(self) -> None:
        r = Step7Result()
        assert r.body == {}
        assert r.recent == []
        assert r.tok_state == ""
        assert r.session_memory == ""
        assert r.compressed is False
        assert r.behavior_signals == {}
        assert r.type_breakdown == {}
        assert r.should_skip_history is False
        assert r.skip_reason == ""
        assert r.history_skip_reason == ""
        assert r.saved_tokens == 0
        assert r.injected_state_payload == ""
        assert r.keep_turns == 3
        assert r.bridge_keep_turns == 3

    def test_step7_result_all_fields_present(self) -> None:
        expected = {
            "body",
            "recent",
            "tok_state",
            "session_memory",
            "compressed",
            "behavior_signals",
            "type_breakdown",
            "should_skip_history",
            "skip_reason",
            "history_skip_reason",
            "saved_tokens",
            "injected_state_payload",
            "keep_turns",
            "bridge_keep_turns",
        }
        actual = {f.name for f in fields(Step7Result)}
        assert actual == expected


class TestStep7aResultDefaults:
    def test_step7a_result_has_correct_defaults(self) -> None:
        r = Step7aResult()
        assert r.recent == []
        assert r.tok_state == ""
        assert r.recent_breakdown == {}
        assert r.bridge_search_success is False
        assert r.behavior_signals == {}

    def test_step7a_result_all_fields_present(self) -> None:
        expected = {
            "recent",
            "tok_state",
            "recent_breakdown",
            "bridge_search_success",
            "behavior_signals",
        }
        actual = {f.name for f in fields(Step7aResult)}
        assert actual == expected


class TestStep7aBridgeCutSearch:
    def test_non_bridge_adapter_returns_early(self) -> None:
        from tok.runtime.pipeline._prepare_bridge_cut_search import run_step_7a_bridge_cut_search

        session = RuntimeSession()
        req = _make_request(adapter_kind="openai")
        result = run_step_7a_bridge_cut_search(
            session=session,
            request=req,
            recent=[{"role": "user", "content": "hello"}],
            original_messages=[{"role": "user", "content": "hello"}],
            system="You are helpful.",
            id_to_context={},
            keep_turns=2,
            bridge_keep_turns=4,
            bridge_profile={},
            history_baseline_prompt_tokens=100,
            seen_mutation_pairs=None,
            preserve_exact_search_evidence=False,
            exact_search_evidence_keys_in_request=set(),
            _first_exact_evidence_seen_for_compression=frozenset(),
            effective_tool_compatible=False,
        )
        assert result.bridge_search_success is False
        assert result.recent == [{"role": "user", "content": "hello"}]

    def test_non_tool_material_recent_returns_early(self) -> None:
        from tok.runtime.pipeline._prepare_bridge_cut_search import run_step_7a_bridge_cut_search

        session = RuntimeSession()
        req = _make_request(adapter_kind="claude-bridge")
        result = run_step_7a_bridge_cut_search(
            session=session,
            request=req,
            recent=[{"role": "user", "content": "hello"}],
            original_messages=[{"role": "user", "content": "hello"}],
            system="You are helpful.",
            id_to_context={},
            keep_turns=2,
            bridge_keep_turns=4,
            bridge_profile={},
            history_baseline_prompt_tokens=100,
            seen_mutation_pairs=None,
            preserve_exact_search_evidence=False,
            exact_search_evidence_keys_in_request=set(),
            _first_exact_evidence_seen_for_compression=frozenset(),
            effective_tool_compatible=False,
        )
        assert result.bridge_search_success is False

    def test_bridge_adapter_with_tool_material_proceeds(self) -> None:
        from tok.runtime.pipeline._prepare_bridge_cut_search import run_step_7a_bridge_cut_search

        session = RuntimeSession()
        req = _make_request(adapter_kind="claude-bridge")
        messages_with_tool = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool_1", "name": "read_file"}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "file content"}],
            },
        ]
        result = run_step_7a_bridge_cut_search(
            session=session,
            request=req,
            recent=messages_with_tool,
            original_messages=messages_with_tool,
            system="You are helpful.",
            id_to_context={},
            keep_turns=2,
            bridge_keep_turns=4,
            bridge_profile={"_bridge_cut_search": 1},
            history_baseline_prompt_tokens=100,
            seen_mutation_pairs=None,
            preserve_exact_search_evidence=False,
            exact_search_evidence_keys_in_request=set(),
            _first_exact_evidence_seen_for_compression=frozenset(),
            effective_tool_compatible=True,
        )
        assert result.bridge_search_success is False
        assert result.recent == messages_with_tool


class TestStep7CompressHistory:
    def test_skip_history_block_sets_should_skip(self) -> None:
        from tok.runtime.pipeline._prepare_compress_history import run_step_7

        session = RuntimeSession()
        req = _make_request(adapter_kind="claude-bridge")
        result = run_step_7(
            session=session,
            request=req,
            normalized_tool_events=[],
            body={"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]},
            id_to_context={},
            behavior_signals={},
            effective_tool_compatible=False,
            mode="balanced",
            policy={},
            should_skip_history=False,
            skip_reason="",
            history_skip_reason="",
            preserve_exact_search_evidence=False,
            plan_finalization_turn=False,
            context_dependency=ContextDependencyDecision(),
            broad_audit_batch=True,
            edit_reacquisition_signals={},
            stream_recovery_history_floor_active=False,
            session_memory="",
            history_baseline_prompt_tokens=100,
            seen_mutation_pairs=None,
            saved_tokens=0,
            compressed=False,
            current_pressure=0.0,
            request_policy="legacy_tool_compatible",
            exact_search_evidence_keys_in_request=set(),
            recent=[{"role": "user", "content": "hello"}],
            tok_state="",
            type_breakdown={},
            keep_turns=2,
            bridge_keep_turns=4,
            bridge_profile={},
            h_profile={},
            _first_exact_evidence_seen_for_compression=frozenset(),
        )
        assert result.should_skip_history is True
        assert result.skip_reason == "broad_audit"
        assert result.history_skip_reason == "broad_audit"
        assert result.behavior_signals.get("broad_audit_history_skipped") == 1

    def test_edit_reacquisition_signals_triggers_skip(self) -> None:
        from tok.runtime.pipeline._prepare_compress_history import run_step_7

        session = RuntimeSession()
        req = _make_request(adapter_kind="claude-bridge")
        result = run_step_7(
            session=session,
            request=req,
            normalized_tool_events=[],
            body={"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]},
            id_to_context={},
            behavior_signals={},
            effective_tool_compatible=False,
            mode="balanced",
            policy={},
            should_skip_history=False,
            skip_reason="",
            history_skip_reason="",
            preserve_exact_search_evidence=False,
            plan_finalization_turn=False,
            context_dependency=ContextDependencyDecision(),
            broad_audit_batch=False,
            edit_reacquisition_signals={"some_signal": 1},
            stream_recovery_history_floor_active=False,
            session_memory="",
            history_baseline_prompt_tokens=100,
            seen_mutation_pairs=None,
            saved_tokens=0,
            compressed=False,
            current_pressure=0.0,
            request_policy="legacy_tool_compatible",
            exact_search_evidence_keys_in_request=set(),
            recent=[{"role": "user", "content": "hello"}],
            tok_state="",
            type_breakdown={},
            keep_turns=2,
            bridge_keep_turns=4,
            bridge_profile={},
            h_profile={},
            _first_exact_evidence_seen_for_compression=frozenset(),
        )
        assert result.should_skip_history is True
        assert result.skip_reason == "evidence_exact_reacquisition"
        assert result.behavior_signals.get("evidence_history_compression_skipped") == 1

    def test_plan_finalization_triggers_skip(self) -> None:
        from tok.runtime.pipeline._prepare_compress_history import run_step_7

        session = RuntimeSession()
        req = _make_request(adapter_kind="claude-bridge")
        result = run_step_7(
            session=session,
            request=req,
            normalized_tool_events=[],
            body={"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]},
            id_to_context={},
            behavior_signals={},
            effective_tool_compatible=False,
            mode="balanced",
            policy={},
            should_skip_history=False,
            skip_reason="",
            history_skip_reason="",
            preserve_exact_search_evidence=False,
            plan_finalization_turn=True,
            context_dependency=ContextDependencyDecision(),
            broad_audit_batch=False,
            edit_reacquisition_signals={},
            stream_recovery_history_floor_active=False,
            session_memory="",
            history_baseline_prompt_tokens=100,
            seen_mutation_pairs=None,
            saved_tokens=0,
            compressed=False,
            current_pressure=0.0,
            request_policy="legacy_tool_compatible",
            exact_search_evidence_keys_in_request=set(),
            recent=[{"role": "user", "content": "hello"}],
            tok_state="",
            type_breakdown={},
            keep_turns=2,
            bridge_keep_turns=4,
            bridge_profile={},
            h_profile={},
            _first_exact_evidence_seen_for_compression=frozenset(),
        )
        assert result.should_skip_history is True
        assert result.skip_reason == "plan_finalization"
        assert result.behavior_signals.get("plan_finalization_history_skipped") == 1

    def test_stream_recovery_history_floor_triggers_skip(self) -> None:
        from tok.runtime.pipeline._prepare_compress_history import run_step_7

        session = RuntimeSession()
        req = _make_request(adapter_kind="claude-bridge")
        result = run_step_7(
            session=session,
            request=req,
            normalized_tool_events=[],
            body={"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]},
            id_to_context={},
            behavior_signals={},
            effective_tool_compatible=False,
            mode="balanced",
            policy={},
            should_skip_history=False,
            skip_reason="",
            history_skip_reason="",
            preserve_exact_search_evidence=False,
            plan_finalization_turn=False,
            context_dependency=ContextDependencyDecision(),
            broad_audit_batch=False,
            edit_reacquisition_signals={},
            stream_recovery_history_floor_active=True,
            session_memory="",
            history_baseline_prompt_tokens=100,
            seen_mutation_pairs=None,
            saved_tokens=0,
            compressed=False,
            current_pressure=0.0,
            request_policy="legacy_tool_compatible",
            exact_search_evidence_keys_in_request=set(),
            recent=[{"role": "user", "content": "hello"}],
            tok_state="",
            type_breakdown={},
            keep_turns=2,
            bridge_keep_turns=4,
            bridge_profile={},
            h_profile={},
            _first_exact_evidence_seen_for_compression=frozenset(),
        )
        assert result.should_skip_history is True
        assert result.skip_reason == "stream_recovery_history_floor"
        assert result.behavior_signals.get("stream_recovery_history_floor_applied") == 1

    def test_should_skip_history_passthrough(self) -> None:
        from tok.runtime.pipeline._prepare_compress_history import run_step_7

        session = RuntimeSession()
        session.bridge_memory.turn = 10
        req = _make_request(adapter_kind="claude-bridge")
        result = run_step_7(
            session=session,
            request=req,
            normalized_tool_events=[],
            body={"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]},
            id_to_context={},
            behavior_signals={},
            effective_tool_compatible=False,
            mode="balanced",
            policy={},
            should_skip_history=True,
            skip_reason="already_skipped",
            history_skip_reason="already_skipped",
            preserve_exact_search_evidence=False,
            plan_finalization_turn=False,
            context_dependency=ContextDependencyDecision(),
            broad_audit_batch=False,
            edit_reacquisition_signals={},
            stream_recovery_history_floor_active=False,
            session_memory="",
            history_baseline_prompt_tokens=100,
            seen_mutation_pairs=None,
            saved_tokens=0,
            compressed=False,
            current_pressure=0.0,
            request_policy="legacy_tool_compatible",
            exact_search_evidence_keys_in_request=set(),
            recent=[{"role": "user", "content": "hello"}],
            tok_state="",
            type_breakdown={},
            keep_turns=2,
            bridge_keep_turns=4,
            bridge_profile={},
            h_profile={},
            _first_exact_evidence_seen_for_compression=frozenset(),
        )
        assert result.should_skip_history is True
        assert result.skip_reason == "already_skipped"
        assert result.history_skip_reason == "already_skipped"

    def test_context_dependency_compresses_prefix_and_preserves_suffix(self) -> None:
        from tok.runtime.pipeline._prepare_compress_history import run_step_7

        prefix: list[dict[str, str]] = []
        for index in range(8):
            prefix.append({"role": "user", "content": f"older prefix {index} " + ("x " * 100)})
            prefix.append({"role": "assistant", "content": f"older reply {index} " + ("y " * 100)})
        suffix = [
            {
                "role": "assistant",
                "content": "Here is the implementation plan.\nPlan: guard\n- inspect\n- patch\n- test\n",
            },
            {"role": "user", "content": "proceed"},
        ]
        messages = prefix + suffix
        protected_start = len(prefix)
        session = RuntimeSession()
        session.bridge_memory.turn = 10
        req = _make_request(adapter_kind="claude-bridge", messages=messages)

        result = run_step_7(
            session=session,
            request=req,
            normalized_tool_events=[],
            body={"model": "claude-sonnet-4", "messages": list(messages)},
            id_to_context={},
            behavior_signals={},
            effective_tool_compatible=False,
            mode="balanced",
            policy={},
            should_skip_history=False,
            skip_reason="",
            history_skip_reason="",
            preserve_exact_search_evidence=False,
            plan_finalization_turn=False,
            context_dependency=ContextDependencyDecision(
                depends_on_context=True,
                kind="plan_handoff",
                protected_suffix_start=protected_start,
                reason="test",
            ),
            broad_audit_batch=False,
            edit_reacquisition_signals={},
            stream_recovery_history_floor_active=False,
            session_memory="",
            history_baseline_prompt_tokens=1000,
            seen_mutation_pairs=None,
            saved_tokens=0,
            compressed=False,
            current_pressure=0.0,
            request_policy="legacy_tool_compatible",
            exact_search_evidence_keys_in_request=set(),
            recent=list(messages),
            tok_state="",
            type_breakdown={},
            keep_turns=2,
            bridge_keep_turns=2,
            bridge_profile={},
            h_profile={},
            _first_exact_evidence_seen_for_compression=frozenset(),
        )

        assert len(result.body["messages"]) < len(messages)
        assert result.body["messages"][-2:] == suffix
        assert result.behavior_signals.get("context_dependency_slice_preserved") == 1
        assert result.behavior_signals.get("plan_finalization_history_skipped", 0) == 0, (
            "context_dependency slice-preserved branch must not mislabel itself as plan_finalization"
        )

    def test_context_dependency_invalid_boundary_falls_back_to_full_history(self) -> None:
        from tok.runtime.pipeline._prepare_compress_history import run_step_7

        messages = [
            {"role": "assistant", "content": "Plan: keep exact\n- inspect\n- patch\n- test\n"},
            {"role": "user", "content": "proceed"},
        ]
        session = RuntimeSession()
        session.bridge_memory.turn = 10
        req = _make_request(adapter_kind="claude-bridge", messages=messages)

        result = run_step_7(
            session=session,
            request=req,
            normalized_tool_events=[],
            body={"model": "claude-sonnet-4", "messages": list(messages)},
            id_to_context={},
            behavior_signals={},
            effective_tool_compatible=False,
            mode="balanced",
            policy={},
            should_skip_history=False,
            skip_reason="",
            history_skip_reason="",
            preserve_exact_search_evidence=False,
            plan_finalization_turn=False,
            context_dependency=ContextDependencyDecision(
                depends_on_context=True,
                kind="plan_handoff",
                protected_suffix_start=None,
                reason="test",
            ),
            broad_audit_batch=False,
            edit_reacquisition_signals={},
            stream_recovery_history_floor_active=False,
            session_memory="",
            history_baseline_prompt_tokens=100,
            seen_mutation_pairs=None,
            saved_tokens=0,
            compressed=False,
            current_pressure=0.0,
            request_policy="legacy_tool_compatible",
            exact_search_evidence_keys_in_request=set(),
            recent=list(messages),
            tok_state="",
            type_breakdown={},
            keep_turns=2,
            bridge_keep_turns=2,
            bridge_profile={},
            h_profile={},
            _first_exact_evidence_seen_for_compression=frozenset(),
        )

        assert result.body["messages"] == messages
        assert result.should_skip_history is True
        assert result.skip_reason == "context_dependency"
        assert result.behavior_signals.get("context_dependency_fallback_full_history") == 1
        assert result.behavior_signals.get("context_dependency_history_skipped") == 1
        assert result.behavior_signals.get("plan_finalization_history_skipped", 0) == 0, (
            "context_dependency fallback branch must not mislabel itself as plan_finalization"
        )

    def test_keep_turns_and_bridge_keep_turns_preserved(self) -> None:
        from tok.runtime.pipeline._prepare_compress_history import run_step_7

        session = RuntimeSession()
        req = _make_request(adapter_kind="claude-bridge")
        result = run_step_7(
            session=session,
            request=req,
            normalized_tool_events=[],
            body={"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]},
            id_to_context={},
            behavior_signals={},
            effective_tool_compatible=False,
            mode="balanced",
            policy={},
            should_skip_history=False,
            skip_reason="",
            history_skip_reason="",
            preserve_exact_search_evidence=False,
            plan_finalization_turn=False,
            context_dependency=ContextDependencyDecision(),
            broad_audit_batch=False,
            edit_reacquisition_signals={},
            stream_recovery_history_floor_active=False,
            session_memory="",
            history_baseline_prompt_tokens=100,
            seen_mutation_pairs=None,
            saved_tokens=0,
            compressed=False,
            current_pressure=0.0,
            request_policy="legacy_tool_compatible",
            exact_search_evidence_keys_in_request=set(),
            recent=[{"role": "user", "content": "hello"}],
            tok_state="",
            type_breakdown={},
            keep_turns=3,
            bridge_keep_turns=6,
            bridge_profile={},
            h_profile={},
            _first_exact_evidence_seen_for_compression=frozenset(),
        )
        assert result.keep_turns == 3
        assert result.bridge_keep_turns == 6
