"""Signal exclusivity: each bypass path must emit only its own named signals.

This regression guard ensures that bypass causes are not mislabeled.  If a
future refactor accidentally reuses another path's signal name, these tests
will catch it immediately.
"""

from __future__ import annotations

from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline._prepare_compress_history import run_step_7
from tok.runtime.pipeline._prepare_compress_tool_results import run_step_6
from tok.runtime.pipeline.context_dependency import ContextDependencyDecision
from tok.runtime.types import RuntimeRequest


def _make_request(**overrides) -> RuntimeRequest:
    defaults = dict(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "hello"}],
        adapter_kind="claude-bridge",
        tool_compatible=True,
    )
    defaults.update(overrides)
    return RuntimeRequest(**defaults)


class _MockPolicy:
    def __init__(self) -> None:
        self.tool_levels = {"balanced": "balanced", "aggressive": "balanced", "full": "balanced"}


_HISTORY_SKIP_SIGNALS = frozenset(
    {
        "broad_audit_history_skipped",
        "evidence_history_compression_skipped",
        "plan_finalization_history_skipped",
        "stream_recovery_history_floor_applied",
        "short_session_history_skipped",
        "context_dependency_history_skipped",
        "context_dependency_fallback_full_history",
        "context_dependency_slice_preserved",
    }
)

_TOOL_RESULT_SKIP_SIGNALS = frozenset(
    {
        "broad_audit_tool_result_compression_skipped",
        "evidence_tool_result_compression_skipped",
        "plan_finalization_tool_result_compression_skipped",
        "stream_recovery_history_floor_tool_result_compression_skipped",
        "context_dependency_tool_result_compression_skipped",
        "context_dependency_fallback_full_history",
        "context_dependency_slice_preserved",
        "compress_tool_results_bypassed",
    }
)


def _drive_history(**override_flags) -> dict[str, int]:
    session = RuntimeSession()
    session.bridge_memory.turn = 10
    defaults = dict(
        session=session,
        request=_make_request(),
        normalized_tool_events=[],
        body={"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]},
        id_to_context={},
        behavior_signals={},
        effective_tool_compatible=True,
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
        keep_turns=2,
        bridge_keep_turns=4,
        bridge_profile={},
        h_profile={},
        _first_exact_evidence_seen_for_compression=frozenset(),
    )
    defaults.update(override_flags)
    result = run_step_7(**defaults)  # type: ignore[arg-type]
    return result.behavior_signals


def _drive_tool_results(**override_flags) -> dict[str, int]:
    session = RuntimeSession()
    defaults = dict(
        session=session,
        request=_make_request(),
        body={"messages": [{"role": "user", "content": "hello"}]},
        translated_messages=[{"role": "user", "content": "hello"}],
        id_to_context={},
        behavior_signals={},
        effective_tool_compatible=True,
        preserve_exact_search_evidence=False,
        broad_audit_batch=False,
        edit_reacquisition_signals={},
        stream_recovery_history_floor_active=False,
        plan_finalization_turn=False,
        context_dependency=ContextDependencyDecision(),
        mode="balanced",
        policy=_MockPolicy(),
        exact_search_evidence_keys_in_request=set(),
        current_pressure=0,
        saved_tokens=0,
        compressed=False,
        result_cache=None,
    )
    defaults.update(override_flags)
    result = run_step_6(**defaults)  # type: ignore[arg-type]
    return result.behavior_signals


def _only_expected_skip_signals(
    actual: dict[str, int],
    expected: set[str],
    universe: frozenset[str],
) -> None:
    wrong = set(k for k in actual if k in universe) - expected
    assert not wrong, f"unexpected skip signals: {sorted(wrong)}"


class TestHistorySignalExclusivity:
    def test_broad_audit(self) -> None:
        sigs = _drive_history(broad_audit_batch=True)
        _only_expected_skip_signals(sigs, {"broad_audit_history_skipped"}, _HISTORY_SKIP_SIGNALS)

    def test_edit_reacquisition(self) -> None:
        sigs = _drive_history(edit_reacquisition_signals={"x": 1})
        _only_expected_skip_signals(sigs, {"evidence_history_compression_skipped"}, _HISTORY_SKIP_SIGNALS)

    def test_plan_finalization(self) -> None:
        sigs = _drive_history(plan_finalization_turn=True)
        _only_expected_skip_signals(sigs, {"plan_finalization_history_skipped"}, _HISTORY_SKIP_SIGNALS)

    def test_stream_recovery_history_floor(self) -> None:
        sigs = _drive_history(stream_recovery_history_floor_active=True)
        _only_expected_skip_signals(sigs, {"stream_recovery_history_floor_applied"}, _HISTORY_SKIP_SIGNALS)

    def test_context_dependency_full_fallback(self) -> None:
        messages = [
            {"role": "assistant", "content": "Plan: keep\n- inspect\n- patch\n- test\n"},
            {"role": "user", "content": "proceed"},
        ]
        sigs = _drive_history(
            request=_make_request(messages=messages),
            body={"model": "claude-sonnet-4", "messages": list(messages)},
            recent=list(messages),
            context_dependency=ContextDependencyDecision(
                depends_on_context=True,
                kind="plan_handoff",
                protected_suffix_start=None,
                reason="test",
            ),
        )
        _only_expected_skip_signals(
            sigs,
            {
                "context_dependency_history_skipped",
                "context_dependency_fallback_full_history",
            },
            _HISTORY_SKIP_SIGNALS,
        )

    def test_context_dependency_slice_preserved(self) -> None:
        prefix: list[dict[str, str]] = []
        for i in range(8):
            prefix.append({"role": "user", "content": f"older {i} " + "x " * 100})
            prefix.append({"role": "assistant", "content": f"reply {i} " + "y " * 100})
        suffix = [
            {"role": "assistant", "content": "Plan: guard\n- inspect\n- patch\n- test\n"},
            {"role": "user", "content": "proceed"},
        ]
        messages = prefix + suffix
        sigs = _drive_history(
            request=_make_request(messages=messages),
            body={"model": "claude-sonnet-4", "messages": list(messages)},
            recent=list(messages),
            history_baseline_prompt_tokens=1000,
            context_dependency=ContextDependencyDecision(
                depends_on_context=True,
                kind="plan_handoff",
                protected_suffix_start=len(prefix),
                reason="test",
            ),
        )
        _only_expected_skip_signals(sigs, {"context_dependency_slice_preserved"}, _HISTORY_SKIP_SIGNALS)


class TestToolResultSignalExclusivity:
    def test_broad_audit(self) -> None:
        sigs = _drive_tool_results(broad_audit_batch=True)
        _only_expected_skip_signals(
            sigs,
            {"broad_audit_tool_result_compression_skipped", "compress_tool_results_bypassed"},
            _TOOL_RESULT_SKIP_SIGNALS,
        )

    def test_edit_reacquisition(self) -> None:
        sigs = _drive_tool_results(edit_reacquisition_signals={"x": 1})
        _only_expected_skip_signals(
            sigs,
            {"evidence_tool_result_compression_skipped", "compress_tool_results_bypassed"},
            _TOOL_RESULT_SKIP_SIGNALS,
        )

    def test_plan_finalization(self) -> None:
        sigs = _drive_tool_results(plan_finalization_turn=True)
        _only_expected_skip_signals(
            sigs,
            {"plan_finalization_tool_result_compression_skipped", "compress_tool_results_bypassed"},
            _TOOL_RESULT_SKIP_SIGNALS,
        )

    def test_stream_recovery_history_floor(self) -> None:
        sigs = _drive_tool_results(stream_recovery_history_floor_active=True)
        _only_expected_skip_signals(
            sigs,
            {
                "stream_recovery_history_floor_tool_result_compression_skipped",
                "compress_tool_results_bypassed",
            },
            _TOOL_RESULT_SKIP_SIGNALS,
        )

    def test_context_dependency_full_fallback(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "Plan: keep\n- inspect\n- patch\n- test\n",
                    }
                ],
            },
            {"role": "user", "content": "proceed"},
        ]
        sigs = _drive_tool_results(
            request=_make_request(messages=messages, tool_compatible=True),
            body={"messages": list(messages)},
            translated_messages=list(messages),
            context_dependency=ContextDependencyDecision(
                depends_on_context=True,
                kind="plan_handoff",
                protected_suffix_start=None,
                reason="test",
            ),
        )
        _only_expected_skip_signals(
            sigs,
            {
                "context_dependency_fallback_full_history",
                "compress_tool_results_bypassed",
            },
            _TOOL_RESULT_SKIP_SIGNALS,
        )
