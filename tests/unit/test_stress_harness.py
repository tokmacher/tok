from pathlib import Path
from types import SimpleNamespace

from tok.testing.stress import (
    DEFAULT_REQUIRED_CLASSES,
    ReadOnlyToolExecutor,
    StressHarness,
    StressHarnessConfig,
    StressObservation,
    StressTask,
    StressTurnRecord,
    _followthrough_evidence_sufficient,
    _late_tool_contract_grace_kind,
    _preprocess_runtime_contract_signals,
    _runtime_retry_context_signals,
    _runtime_turn_context_signals,
    _sanitize_tool_use_block,
    _strip_answer_labels,
    classify_breakpoints,
    extract_breakpoint_paths,
    render_language_refactor_plan,
    render_stress_report,
    required_class_coverage,
    should_stop_run,
    summarize_implicated_files,
)
from tok.universal_runtime import RuntimeSession


class _FakeCompletions:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )


class _FakeClient:
    def __init__(self, responses) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def _observation(**overrides):
    base = StressObservation(
        task_id="task",
        turn_index=1,
        prompt="prompt",
        phase="task",
        visible_response="File=src/tok/cli.py\nVerification=_gate_release_summary",
        active_tools=["grep_search"],
        input_behavior_signals={},
        output_behavior_signals={},
        state_payload_chars=220,
        resend_mode="full",
        transcript_slice=[{"role": "user", "content": "prompt"}],
    )
    data = base.__dict__ | overrides
    return StressObservation(**data)


def test_classify_breakpoints_detects_each_class() -> None:
    seen = set()
    drift = classify_breakpoints(
        _observation(output_behavior_signals={"semantic_drift_detected": 1}),
        seen,
    )
    reacq = classify_breakpoints(
        _observation(
            active_tools=["grep_search"],
            input_behavior_signals={"repeat_search": 1},
            target_already_validated=True,
        ),
        seen,
    )
    retention = classify_breakpoints(
        _observation(
            phase="checkpoint",
            expected_fields={
                "file": "src/tok/cli.py",
                "verification": "_gate_release_summary",
            },
            observed_fields={"file": "src/tok/cli.py"},
        ),
        seen,
    )
    compaction = classify_breakpoints(
        _observation(
            phase="checkpoint",
            expected_fields={"file": "src/tok/cli.py"},
            observed_fields={"file": "src/tok/cli.py"},
            input_behavior_signals={"answer_anchor_present": 0},
            state_payload_chars=80,
            resend_mode="suppressed",
            payload_pressure_ready=True,
        ),
        seen,
    )
    tool_failure = classify_breakpoints(_observation(tool_contract_failure=True), seen)
    fallback = classify_breakpoints(_observation(baseline_only=True), seen)

    classes = [
        drift[0].breakpoint_class,
        reacq[0].breakpoint_class,
        retention[0].breakpoint_class,
        compaction[0].breakpoint_class,
        tool_failure[0].breakpoint_class,
        fallback[0].breakpoint_class,
    ]
    assert classes == [
        "protocol_drift",
        "reacquisition_loop",
        "retention_loss",
        "compaction_loss",
        "tool_contract_failure",
        "baseline_fallback",
    ]


def test_classify_breakpoints_deduplicates_classes() -> None:
    seen = set()
    first = classify_breakpoints(
        _observation(output_behavior_signals={"semantic_drift_detected": 1}),
        seen,
    )
    second = classify_breakpoints(
        _observation(output_behavior_signals={"semantic_drift_detected": 1}),
        seen,
    )

    assert len(first) == 1
    assert second == []


def test_should_stop_run_covers_all_exit_conditions() -> None:
    config = StressHarnessConfig(max_tasks=5, required_classes=("protocol_drift",))
    assert should_stop_run(
        breakpoint_count=1,
        baseline_only=False,
        tasks_completed=1,
        seen_classes={"protocol_drift"},
        config=config,
    )
    assert should_stop_run(
        breakpoint_count=0,
        baseline_only=True,
        tasks_completed=1,
        seen_classes=set(),
        config=config,
    )
    assert should_stop_run(
        breakpoint_count=0,
        baseline_only=False,
        tasks_completed=5,
        seen_classes=set(),
        config=config,
    )
    assert not should_stop_run(
        breakpoint_count=1,
        baseline_only=False,
        tasks_completed=2,
        seen_classes=set(),
        config=config,
    )


def test_required_class_coverage_supports_alternative_classes() -> None:
    coverage = required_class_coverage(
        {"tool_contract_failure", "baseline_fallback"},
        DEFAULT_REQUIRED_CLASSES,
    )

    assert "tool_contract_failure" in coverage["covered"]
    assert "compaction_loss|baseline_fallback" in coverage["covered"]
    assert "retention_loss" in coverage["missing"]


def test_runtime_turn_context_signals_marks_payload_pressure() -> None:
    assert _runtime_turn_context_signals(payload_pressure_ready=True) == {"payload_pressure_ready": 1}


def test_runtime_retry_context_signals_marks_late_staged_retry() -> None:
    assert _runtime_retry_context_signals({"late_retry_contract_stage_tool_only": 1}) == {
        "late_retry_contract_stage_tool_only": 1,
        "late_staged_retry_context": 1,
    }


def test_preprocess_runtime_contract_signals_promotes_late_theless_fresh_signal() -> None:
    task = StressTask(
        id="tool_contract_toolless_fresh_answer",
        phase_name="tool-contract",
        prompt="fresh evidence required",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
    )

    signals = _preprocess_runtime_contract_signals(
        task=task,
        raw_response="File=src/tok/gateway.py\nVerification=health",
        attempt_tool_count_before_turn=0,
        payload_pressure_ready=True,
        request_behavior_signals={
            "late_staged_retry_context": 1,
            "late_retry_contract_stage_tool_only": 1,
        },
    )

    assert signals == {
        "late_staged_retry_context": 1,
        "late_retry_contract_stage_tool_only": 1,
        "payload_pressure_ready": 1,
        "toolless_fresh_answer_event": 1,
        "late_freshness_signal_promoted": 1,
    }


def test_preprocess_runtime_contract_signals_promotes_late_mixed_signal_without_freshness() -> None:
    task = StressTask(
        id="fresh_anchor_runtime",
        phase_name="fresh-grounding",
        prompt="collect fresh anchor",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=False,
    )

    signals = _preprocess_runtime_contract_signals(
        task=task,
        raw_response="@tool view_file id:call_1 path:src/tok/gateway.py\n\n@msg role:assistant\n  |> File=src/tok/gateway.py\n  |> Verification=health\n",
        attempt_tool_count_before_turn=0,
        payload_pressure_ready=True,
        request_behavior_signals={},
    )

    assert signals == {
        "payload_pressure_ready": 1,
        "mixed_answer_tool_event": 1,
        "late_mixed_signal_promoted": 1,
    }


def test_preprocess_runtime_contract_signals_does_not_promote_mixed_without_late_context() -> None:
    task = StressTask(
        id="fresh_anchor_runtime",
        phase_name="fresh-grounding",
        prompt="collect fresh anchor",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=False,
    )

    signals = _preprocess_runtime_contract_signals(
        task=task,
        raw_response="@tool view_file id:call_1 path:src/tok/gateway.py\n\n@msg role:assistant\n  |> File=src/tok/gateway.py\n  |> Verification=health\n",
        attempt_tool_count_before_turn=0,
        payload_pressure_ready=False,
        request_behavior_signals={},
    )

    assert signals == {}


def test_followthrough_evidence_sufficient_threshold() -> None:
    # Pre-payload: threshold 500
    assert _followthrough_evidence_sufficient(evidence_chars=0, payload_pressure_ready=False, tool_results=[]) is False
    assert (
        _followthrough_evidence_sufficient(
            evidence_chars=499,
            payload_pressure_ready=False,
            tool_results=[{"content": "ok"}],
        )
        is False
    )
    assert (
        _followthrough_evidence_sufficient(
            evidence_chars=500,
            payload_pressure_ready=False,
            tool_results=[{"content": "ok"}],
        )
        is True
    )

    # Post-payload: threshold 800, min 2 tools
    assert (
        _followthrough_evidence_sufficient(
            evidence_chars=799,
            payload_pressure_ready=True,
            tool_results=[{"content": "ok1"}, {"content": "ok2"}],
        )
        is False
    )
    assert (
        _followthrough_evidence_sufficient(
            evidence_chars=800,
            payload_pressure_ready=True,
            tool_results=[{"content": "ok1"}],
        )
        is False
    )
    assert (
        _followthrough_evidence_sufficient(
            evidence_chars=800,
            payload_pressure_ready=True,
            tool_results=[{"content": "ok1"}, {"content": "ok2"}],
        )
        is True
    )

    # Require successful tools
    assert (
        _followthrough_evidence_sufficient(evidence_chars=1000, payload_pressure_ready=False, tool_results=[]) is False
    )
    assert (
        _followthrough_evidence_sufficient(
            evidence_chars=1000,
            payload_pressure_ready=False,
            tool_results=[{"is_error": True, "content": "err"}],
        )
        is False
    )
    assert (
        _followthrough_evidence_sufficient(
            evidence_chars=1000,
            payload_pressure_ready=True,
            tool_results=[{"is_error": True}],
        )
        is False
    )
    assert (
        _followthrough_evidence_sufficient(
            evidence_chars=1000,
            payload_pressure_ready=True,
            tool_results=[{"content": "ok1"}, {"content": "ok2"}],
        )
        is True
    )


def test_read_only_tool_executor_blocks_mutating_tools(tmp_path) -> None:
    executor = ReadOnlyToolExecutor(tmp_path)
    result, blocked = executor.execute({"id": "x1", "name": "write", "input": {"path": "a.txt", "text": "x"}})

    assert blocked is True
    assert result["is_error"] is True
    assert "disabled" in result["content"]


def test_read_only_tool_executor_accepts_text_fallback_inputs(
    tmp_path,
) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text("def demo():\n    return 1\n")
    executor = ReadOnlyToolExecutor(tmp_path)

    read_result, read_blocked = executor.execute({"id": "x2", "name": "view_file", "input": {"text": "sample.py"}})
    list_result, list_blocked = executor.execute({"id": "x3", "name": "list_dir", "input": {"text": "."}})

    assert read_blocked is False
    assert "sample.py" in read_result["content"]
    assert list_blocked is False
    assert "sample.py" in list_result["content"]


def test_renderers_include_evidence_and_targets() -> None:
    seen = set()
    breakpoint = classify_breakpoints(
        _observation(output_behavior_signals={"semantic_drift_detected": 1}),
        seen,
    )[0]
    result = SimpleNamespace(
        model="qwen/qwen3-coder-next",
        provider="openrouter",
        started_at="2026-03-21T00:00:00+00:00",
        completed_at="2026-03-21T00:10:00+00:00",
        required_classes=DEFAULT_REQUIRED_CLASSES,
        tasks_completed=2,
        total_tokens=240,
        baseline_only=False,
        validated_anchor_count=1,
        reuse_checks_run=0,
        checkpoint_checks_run=0,
        reuse_probe_attempts=1,
        reuse_probe_successes=0,
        retention_probe_attempts=0,
        retention_probe_successes=0,
        late_retention_probe_attempts=0,
        late_retention_probe_successes=0,
        tool_contract_probe_attempts=1,
        tool_contract_failure_events_seen=1,
        mixed_answer_tool_events_seen=1,
        unsupported_tool_events_seen=0,
        bad_tool_args_events_seen=0,
        toolless_fresh_answer_events_seen=1,
        reacquisition_events_seen=0,
        validated_target_reacquisition_events_seen=0,
        validated_target_exact_reacquisition_events_seen=0,
        validated_target_reconfirmation_events_seen=1,
        answer_anchor_reacquisition_events_seen=0,
        answer_ready_reacquisition_events_seen=0,
        repair_phase_reacquisition_events_seen=0,
        benign_reverification_events_seen=0,
        answer_ready_repair_requested_count=1,
        answer_ready_repair_active_count=1,
        answer_ready_repair_resolved_count=0,
        answer_ready_repair_failed_count=1,
        late_freshness_signal_promoted_count=2,
        late_freshness_signal_consumed_by_tok_count=1,
        late_mixed_signal_promoted_count=3,
        late_mixed_signal_consumed_by_tok_count=2,
        late_answer_assembly_repair_answer_only_requested_count=2,
        late_answer_assembly_repair_answer_only_resolved_count=1,
        late_answer_assembly_repair_answer_only_failed_count=1,
        late_answer_followthrough_requested_count=2,
        late_answer_followthrough_active_count=1,
        late_answer_followthrough_resolved_count=1,
        late_answer_followthrough_failed_count=1,
        late_answer_followthrough_after_tool_only_repair_count=1,
        late_answer_followthrough_blocked_insufficient_evidence_count=1,
        late_tool_contract_reconfirmation_grace_count=1,
        late_tool_contract_mixed_grace_count=2,
        late_tool_contract_toolless_grace_count=1,
        late_tool_contract_reconfirmation_retry_failure_count=1,
        late_tool_contract_mixed_retry_failure_count=2,
        late_tool_contract_toolless_retry_failure_count=1,
        fallback_pressure_incremented_count=2,
        fallback_pressure_suppressed_count=1,
        fallback_pressure_cause_exact_reacquisition_count=1,
        fallback_pressure_cause_mixed_turn_count=2,
        fallback_pressure_cause_toolless_fresh_count=1,
        fallback_pressure_cause_bad_args_count=0,
        fallback_pressure_cause_unsupported_tool_count=0,
        retry_prompt_shape_exact_target_reread_count=1,
        retry_prompt_shape_mixed_turn_count=2,
        retry_prompt_shape_toolless_fresh_count=1,
        retry_prompt_shape_unsupported_tool_count=0,
        retry_prompt_shape_bad_args_count=0,
        retry_prompt_shape_generic_retry_count=3,
        retry_prompt_no_exact_reread_count=4,
        retry_prompt_requires_supporting_tool_count=3,
        retry_prompt_supporting_tool_satisfied_count=2,
        retry_prompt_supporting_tool_missed_count=1,
        retry_prompt_supporting_tool_missed_mixed_count=1,
        retry_prompt_supporting_tool_missed_toolless_count=0,
        exact_target_reread_after_no_exact_retry_count=1,
        early_retry_contract_stage_tool_only_count=2,
        early_retry_contract_stage_answer_only_count=1,
        early_retry_bad_args_tool_only_count=1,
        early_retry_tool_only_satisfied_count=1,
        early_retry_tool_only_failed_mixed_count=1,
        early_retry_tool_only_failed_toolless_count=0,
        early_retry_answer_only_satisfied_count=1,
        early_retry_answer_only_failed_tool_count=0,
        late_retry_contract_stage_tool_only_count=2,
        late_retry_contract_stage_answer_only_count=1,
        late_retry_tool_only_satisfied_count=1,
        late_retry_tool_only_failed_mixed_count=1,
        late_retry_tool_only_failed_toolless_count=0,
        late_retry_answer_only_satisfied_count=1,
        late_retry_answer_only_failed_tool_count=1,
        late_retry_no_exact_target_count=2,
        exact_target_reread_after_late_retry_no_exact_target_count=0,
        failed_tasks_before_any_retry_contract_count=1,
        failed_tasks_after_generic_retry_only_count=0,
        failed_tasks_after_early_staged_retry_count=1,
        failed_tasks_after_late_staged_retry_count=0,
        failed_tasks_after_validated_target_retry_count=0,
        first_failed_phase="tool-contract",
        first_failed_task="tool_contract_mixed_answer_and_tool",
        first_irreversible_miss_kind="first_miss_mixed_answer_tool",
        dominant_failure_locus="agent",
        first_payload_pressure_turn=2,
        first_payload_pressure_task="payload_task",
        first_compaction_eligible_turn=None,
        first_compaction_eligible_task="",
        first_baseline_fallback_turn=3,
        first_baseline_fallback_task="tool_contract_mixed_answer_and_tool",
        baseline_fallback_turns_after_payload_pressure=1,
        baseline_fallback_turns_after_compaction_eligible=0,
        fallback_after_payload_pressure=True,
        fallback_after_compaction_eligible=False,
        retention_substitution_events_seen=0,
        anchors_before_baseline=1,
        seed_searches=1,
        seed_direct_reads=1,
        seed_answer_attempts=1,
        seed_evidence_sufficient=True,
        first_anchor_failure_mode="answer_assembly",
        tool_backed_turns=1,
        resend_modes_seen=["full"],
        payload_pressure_reached=False,
        compaction_eligible=False,
        run_diagnosis="weak_harness_pressure",
        weak_run_reasons=["never_left_full_resend"],
        turns=[
            SimpleNamespace(
                turn_index=3,
                task_id="tool_contract_mixed_answer_and_tool",
                resend_mode="full",
                resend_decision_reason="answer_anchor_present_kept_full",
                state_payload_chars=220,
                tool_result_volume_chars=500,
                tool_dense_session=True,
                answer_fact_projection_present=True,
                phase_name="tool-contract",
            )
        ],
        breakpoints=[breakpoint],
    )

    report = render_stress_report(result)  # type: ignore[arg-type]
    plan = render_language_refactor_plan(result)  # type: ignore[arg-type]

    assert "protocol_drift" in report
    assert "response contract pressure" in report
    assert "Language Refactor Plan" in plan
    assert "response contract pressure" in plan
    assert "Required coverage complete" in report
    assert "First-Anchor Failure Mode" in report
    assert "Memory Probe Coverage" in report
    assert "Frontier Diagnostics" in report
    assert "Resend Analysis" in report
    assert "Tool contract probes" in report
    assert "Reacquisition diagnostics" in report
    assert "Answer-ready repair totals" in report
    assert "Late freshness handoff" in report
    assert "Late mixed handoff" in report
    assert "Late mixed answer-only repair" in report
    assert "Late answer follow-through" in report
    assert "Late answer follow-through after tool-only repair" in report
    assert "Late answer follow-through blocked" in report
    assert "Late tool-contract grace" in report
    assert "Late tool-contract retry failures" in report
    assert "Fallback pressure totals" in report
    assert "Fallback pressure causes" in report
    assert "Retry prompt shapes" in report
    assert "Retry no-exact-reread" in report
    assert "Retry supporting-tool totals" in report
    assert "Retry supporting-tool misses" in report
    assert "Early staged retries" in report
    assert "Early bad-args tool-only" in report
    assert "Early tool-only outcomes" in report
    assert "Early answer-only outcomes" in report
    assert "Staged late retries" in report
    assert "Tool-only stage outcomes" in report
    assert "Answer-only stage outcomes" in report
    assert "Staged retries with no-exact-target" in report
    assert "Failure Locus" in report
    assert "Dominant failure locus" in report
    assert "First irreversible miss kind" in report
    assert "Fallback after payload pressure" in report
    assert "validated_target_exact" in report
    assert "validated_target_reconfirmation" in report


def test_implicated_file_summary_extracts_and_counts_paths() -> None:
    seen = set()
    first = classify_breakpoints(
        _observation(
            output_behavior_signals={"semantic_drift_detected": 1},
            visible_response="File=src/tok/cli.py\nVerification=_gate_release_summary",
            transcript_slice=[
                {
                    "role": "user",
                    "content": "check src/tok/cli.py and src/tok/universal_runtime.py",
                }
            ],
        ),
        seen,
    )[0]
    second = classify_breakpoints(
        _observation(
            baseline_only=True,
            visible_response="File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        ),
        seen,
    )[0]

    assert extract_breakpoint_paths(first) == [
        "src/tok/cli.py",
        "src/tok/universal_runtime.py",
    ]
    summary = summarize_implicated_files([first, second])
    assert summary[0] == {"path": "src/tok/universal_runtime.py", "count": 2}


def test_implicated_file_summary_ignores_harness_and_generated_paths() -> None:
    item = SimpleNamespace(
        prompt="inspect src/tok/stress_harness.py and src/tok/gateway.py",
        visible_response="File=src/tok/stress_harness.py\nVerification=helper",
        transcript_slice=[
            {
                "role": "user",
                "content": "see tmp/stress_language/20260321_174110/stress_report.md and src/tok/gateway.py",
            }
        ],
    )

    assert extract_breakpoint_paths(item) == ["src/tok/gateway.py"]


def test_extract_breakpoint_paths_normalizes_stitched_tok_path() -> None:
    item = SimpleNamespace(
        prompt="inspect src/tok/g.tok/gateway.py",
        visible_response="File=src/tok/g.tok/gateway.py\nVerification=health",
        transcript_slice=[],
    )

    assert extract_breakpoint_paths(item) == ["src/tok/gateway.py"]


def test_classify_breakpoints_does_not_treat_initial_oracle_miss_as_retention() -> None:
    seen = set()
    breakpoints = classify_breakpoints(
        _observation(
            phase="task",
            expected_fields={"file": "src/tok/universal_runtime.py"},
            observed_fields={"file": "src/tok/cli.py"},
            tool_contract_failure=True,
        ),
        seen,
    )

    assert [item.breakpoint_class for item in breakpoints] == ["tool_contract_failure"]


def test_classify_breakpoints_includes_tool_contract_failure_for_mixed_answer_tool() -> None:
    seen = set()
    breakpoints = classify_breakpoints(
        _observation(
            tool_contract_failure=True,
            output_behavior_signals={"mixed_answer_tool_event": 1},
            observed_fields={
                "file": "src/tok/gateway.py",
                "verification": "health",
            },
        ),
        seen,
    )

    assert [item.breakpoint_class for item in breakpoints] == ["tool_contract_failure"]


def test_read_only_tool_executor_filters_excluded_paths_from_search(
    tmp_path,
) -> None:
    src_dir = tmp_path / "src" / "tok"
    src_dir.mkdir(parents=True)
    (src_dir / "gateway.py").write_text("def bridge():\n    return 'ok'\n")
    (src_dir / "stress_harness.py").write_text("def bridge():\n    return 'no'\n")
    executor = ReadOnlyToolExecutor(tmp_path)

    result, blocked = executor.execute(
        {
            "id": "x4",
            "name": "grep_search",
            "input": {"query": "bridge", "search_path": "src/tok"},
        }
    )

    assert blocked is False
    assert "gateway.py" in result["content"]
    assert "stress_harness.py" not in result["content"]


def test_strip_answer_labels_removes_file_and_verification_suffixes() -> None:
    assert (
        _strip_answer_labels("src/tok/gateway.pyFile=src/tok/gateway.py\nVerification=bridge") == "src/tok/gateway.py"
    )


def test_sanitize_tool_use_block_removes_recursive_tool_noise() -> None:
    block = {
        "type": "tool_use",
        "name": "grep_search",
        "input": {
            "Tool use (grep_search)": '{"text":"bad"}',
            "text": "src/tok/gateway.py fallbackFile=src/tok/gateway.py\nVerification=bridge",
        },
    }

    sanitized = _sanitize_tool_use_block(block)

    assert "Tool use (grep_search)" not in sanitized["input"]
    assert sanitized["input"]["text"] == "src/tok/gateway.py fallback"


def test_stress_harness_run_records_distinct_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "1")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/cli.py"',
        "File=src/tok/cli.py\nVerification=_gate_release_summary",
        "File=src/tok/cli.py",
        '@Tool view_file id:"s2"\n  path:"src/tok/cli.py"',
        "This is a long plain prose reply that ignores the Tok contract entirely and keeps talking "
        "without any structured anchor or compact answer format so semantic drift should be detected quickly.",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        model="qwen/qwen3-coder-next",
        target_breakpoints=4,
        max_tasks=2,
        max_tool_rounds=4,
        fallback_threshold=1,
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    classes = [item.breakpoint_class for item in result.breakpoints]
    result_dict = result.to_dict()

    assert "tool_contract_failure" in classes
    assert "baseline_fallback" in classes
    assert result.baseline_only is True
    assert "first_payload_pressure_turn" in result_dict
    assert "first_baseline_fallback_turn" in result_dict
    assert "answer_ready_repair_requested_count" in result_dict
    assert "late_freshness_signal_promoted_count" in result_dict
    assert "late_freshness_signal_consumed_by_tok_count" in result_dict
    assert "late_mixed_signal_promoted_count" in result_dict
    assert "late_mixed_signal_consumed_by_tok_count" in result_dict
    assert "late_answer_assembly_repair_answer_only_requested_count" in result_dict
    assert "late_answer_assembly_repair_answer_only_resolved_count" in result_dict
    assert "late_answer_assembly_repair_answer_only_failed_count" in result_dict
    assert "late_answer_followthrough_requested_count" in result_dict
    assert "late_answer_followthrough_active_count" in result_dict
    assert "late_answer_followthrough_resolved_count" in result_dict
    assert "late_answer_followthrough_failed_count" in result_dict
    assert "late_answer_followthrough_after_tool_only_repair_count" in result_dict
    assert "late_answer_followthrough_blocked_insufficient_evidence_count" in result_dict
    assert "fallback_after_payload_pressure" in result_dict
    assert "validated_target_exact_reacquisition_events_seen" in result_dict
    assert "validated_target_reconfirmation_events_seen" in result_dict
    assert "late_tool_contract_reconfirmation_grace_count" in result_dict
    assert "late_tool_contract_mixed_grace_count" in result_dict
    assert "late_tool_contract_toolless_grace_count" in result_dict
    assert "late_tool_contract_reconfirmation_retry_failure_count" in result_dict
    assert "late_tool_contract_mixed_retry_failure_count" in result_dict
    assert "late_tool_contract_toolless_retry_failure_count" in result_dict
    assert "fallback_pressure_incremented_count" in result_dict
    assert "fallback_pressure_suppressed_count" in result_dict
    assert "fallback_pressure_cause_exact_reacquisition_count" in result_dict
    assert "fallback_pressure_cause_mixed_turn_count" in result_dict
    assert "fallback_pressure_cause_toolless_fresh_count" in result_dict
    assert "retry_prompt_shape_exact_target_reread_count" in result_dict
    assert "retry_prompt_shape_mixed_turn_count" in result_dict
    assert "retry_prompt_shape_toolless_fresh_count" in result_dict
    assert "retry_prompt_shape_unsupported_tool_count" in result_dict
    assert "retry_prompt_shape_bad_args_count" in result_dict
    assert "retry_prompt_shape_generic_retry_count" in result_dict
    assert "retry_prompt_no_exact_reread_count" in result_dict
    assert "retry_prompt_requires_supporting_tool_count" in result_dict
    assert "retry_prompt_supporting_tool_satisfied_count" in result_dict
    assert "retry_prompt_supporting_tool_missed_count" in result_dict
    assert "retry_prompt_supporting_tool_missed_mixed_count" in result_dict
    assert "retry_prompt_supporting_tool_missed_toolless_count" in result_dict
    assert "early_retry_contract_stage_tool_only_count" in result_dict
    assert "early_retry_contract_stage_answer_only_count" in result_dict
    assert "early_retry_bad_args_tool_only_count" in result_dict
    assert "early_retry_tool_only_satisfied_count" in result_dict
    assert "early_retry_tool_only_failed_mixed_count" in result_dict
    assert "early_retry_tool_only_failed_toolless_count" in result_dict
    assert "early_retry_answer_only_satisfied_count" in result_dict
    assert "early_retry_answer_only_failed_tool_count" in result_dict
    assert "late_retry_contract_stage_tool_only_count" in result_dict
    assert "late_retry_contract_stage_answer_only_count" in result_dict
    assert "late_retry_tool_only_satisfied_count" in result_dict
    assert "late_retry_tool_only_failed_mixed_count" in result_dict
    assert "late_retry_tool_only_failed_toolless_count" in result_dict
    assert "late_retry_answer_only_satisfied_count" in result_dict
    assert "late_retry_answer_only_failed_tool_count" in result_dict
    assert "late_retry_no_exact_target_count" in result_dict
    assert "exact_target_reread_after_late_retry_no_exact_target_count" in result_dict
    assert "exact_target_reread_after_no_exact_retry_count" in result_dict
    assert "failed_tasks_before_any_retry_contract_count" in result_dict
    assert "failed_tasks_after_generic_retry_only_count" in result_dict
    assert "failed_tasks_after_early_staged_retry_count" in result_dict
    assert "failed_tasks_after_late_staged_retry_count" in result_dict
    assert "failed_tasks_after_validated_target_retry_count" in result_dict
    assert "first_failed_phase" in result_dict
    assert "first_failed_task" in result_dict
    assert "first_irreversible_miss_kind" in result_dict
    assert "dominant_failure_locus" in result_dict


def test_strict_gate_retries_until_fresh_evidence_and_creates_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        '@Tool view_file id:"s1"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        "File=src/tok/gateway.py\nVerification=health",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=2,
        max_tool_rounds=3,
        max_retries_per_task=2,
        task_catalog=(
            StressTask(
                id="seed_one",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="seed_two",
                phase_name="anchor-seed",
                prompt="Find fallback.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="record_fallback_event",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()

    assert result.validated_anchor_count == 2
    assert result.checkpoint_checks_run == 1
    assert any("required_coverage_missing" in note for note in result.notes)
    assert any(turn.retry_index == 1 for turn in result.turns)
    assert result.turns[-1].phase_name == "checkpoint"


def test_reuse_task_flags_validated_target_reacquisition(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=2,
        max_tool_rounds=2,
        max_retries_per_task=1,
        task_catalog=(
            StressTask(
                id="fresh_fallback",
                phase_name="fresh-grounding",
                prompt="Find fallback.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="record_fallback_event",
                require_fresh_evidence=True,
                require_tool_count=1,
            ),
            StressTask(
                id="reuse_fallback",
                phase_name="reuse-vs-reacquire",
                prompt="Recover fallback from memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="record_fallback_event",
                forbid_reacquisition=True,
                min_validated_anchors=1,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    classes = [item.breakpoint_class for item in result.breakpoints]

    assert "reacquisition_loop" in classes
    assert result.reuse_checks_run == 1
    assert result.validated_target_exact_reacquisition_events_seen == 1
    assert result.validated_target_reconfirmation_events_seen == 0


def test_validated_target_reconfirmation_is_tracked_without_reacquisition_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool list_dir id:"s2"\n  path:"src/tok"',
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=2,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="fresh_health",
                phase_name="fresh-grounding",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reconfirm_health",
                phase_name="tool-contract",
                prompt="Reconfirm health with fresh evidence.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                min_validated_anchors=1,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    classes = [item.breakpoint_class for item in result.breakpoints]

    assert result.validated_target_exact_reacquisition_events_seen == 0
    assert result.validated_target_reconfirmation_events_seen == 1
    assert result.reacquisition_events_seen == 0
    assert "reacquisition_loop" not in classes
    assert any(turn.validated_target_reconfirmation_attempt for turn in result.turns)


def test_exact_target_reread_still_increments_fallback_pressure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "1")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/gateway.py"',
        "I reopened the file but I am not returning the structured answer.",
        '@Tool view_file id:"s3"\n  path:"src/tok/gateway.py"',
        "I still am not returning the structured answer.",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=2,
        max_tool_rounds=2,
        max_retries_per_task=1,
        fallback_threshold=1,
        min_payload_pressure_bytes=1,
        task_catalog=(
            StressTask(
                id="fresh_health",
                phase_name="fresh-grounding",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reopen_health",
                phase_name="tool-contract",
                prompt="Reconfirm health with fresh evidence.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                min_validated_anchors=1,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    classes = [item.breakpoint_class for item in result.breakpoints]
    contract_turn = next(
        turn
        for turn in result.turns
        if turn.task_id == "reopen_health" and turn.output_behavior_signals.get("fallback_pressure_incremented")
    )

    assert result.baseline_only is True
    assert "reacquisition_loop" in classes
    assert result.validated_target_exact_reacquisition_events_seen >= 1
    assert result.fallback_pressure_incremented_count == 1
    assert result.fallback_pressure_suppressed_count == 0
    assert result.fallback_pressure_cause_exact_reacquisition_count == 1
    assert contract_turn.output_behavior_signals.get("fallback_pressure_incremented") == 1
    assert contract_turn.output_behavior_signals.get("fallback_pressure_cause_exact_reacquisition") == 1
    assert "late_tool_contract_reconfirmation_grace" not in contract_turn.output_behavior_signals


def test_late_reconfirmation_failure_shape_can_be_graced_without_incrementing_fallback(
    tmp_path,
) -> None:
    task = StressTask(
        id="reconfirm_health",
        phase_name="tool-contract",
        prompt="Reconfirm health with fresh evidence.\nFile=<the primary file>\nVerification=<the function>",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
        min_validated_anchors=2,
    )
    harness = StressHarness(
        StressHarnessConfig(fallback_threshold=1),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    assert (
        _late_tool_contract_grace_kind(
            task=task,
            retry_index=0,
            payload_pressure_ready=True,
            input_signals={"validated_target_reconfirmation_attempt": 1},
            output_signals={},
            protocol_failure=True,
            tool_contract_failure=False,
        )
        == "reconfirmation"
    )

    incremented = harness._update_failure_counter(
        protocol_failure=True,
        tool_contract_failure=False,
        suppress_failure_increment=True,
    )
    retry_incremented = harness._update_failure_counter(
        protocol_failure=True,
        tool_contract_failure=False,
        suppress_failure_increment=False,
    )

    assert incremented is False
    assert retry_incremented is True
    assert harness._consecutive_failures == 1
    assert harness.session._baseline_only is True


def test_late_mixed_failure_shape_is_classified_without_reconfirmation() -> None:
    task = StressTask(
        id="mixed_reconfirm_health",
        phase_name="tool-contract",
        prompt="Reconfirm health with fresh evidence.\nFile=<the primary file>\nVerification=<the function>",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
        min_validated_anchors=2,
    )

    assert (
        _late_tool_contract_grace_kind(
            task=task,
            retry_index=0,
            payload_pressure_ready=True,
            input_signals={"validated_target_reconfirmation_attempt": 1},
            output_signals={"mixed_answer_tool_event": 1},
            protocol_failure=True,
            tool_contract_failure=True,
        )
        == "mixed"
    )


def test_late_toolless_failure_shape_is_classified() -> None:
    task = StressTask(
        id="toolless_reconfirm_health",
        phase_name="tool-contract",
        prompt="Reconfirm health with fresh evidence.\nFile=<the primary file>\nVerification=<the function>",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
        min_validated_anchors=2,
    )

    assert (
        _late_tool_contract_grace_kind(
            task=task,
            retry_index=0,
            payload_pressure_ready=True,
            input_signals={},
            output_signals={"toolless_fresh_answer_event": 1},
            protocol_failure=True,
            tool_contract_failure=True,
        )
        == "toolless"
    )


def test_late_grace_does_not_apply_to_unsupported_or_exact_reread() -> None:
    task = StressTask(
        id="unsupported_reconfirm_health",
        phase_name="tool-contract",
        prompt="Reconfirm health with fresh evidence.\nFile=<the primary file>\nVerification=<the function>",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
        min_validated_anchors=2,
    )

    assert (
        _late_tool_contract_grace_kind(
            task=task,
            retry_index=0,
            payload_pressure_ready=True,
            input_signals={"validated_target_reconfirmation_attempt": 1},
            output_signals={"unsupported_tool_event": 1},
            protocol_failure=True,
            tool_contract_failure=True,
        )
        is None
    )
    assert (
        _late_tool_contract_grace_kind(
            task=task,
            retry_index=0,
            payload_pressure_ready=True,
            input_signals={"validated_target_exact_reacquired": 1},
            output_signals={"mixed_answer_tool_event": 1},
            protocol_failure=True,
            tool_contract_failure=True,
        )
        is None
    )


def test_retry_prompt_uses_mixed_turn_template_for_late_validated_target(
    tmp_path,
) -> None:
    task = StressTask(
        id="mixed_reconfirm_health",
        phase_name="tool-contract",
        prompt="Reconfirm health.",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
    )
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    prompt, signals = harness._retry_prompt(
        task=task,
        expected_fields={
            "file": "src/tok/gateway.py",
            "verification": "health",
        },
        observed_fields={},
        attempt_tool_count=1,
        attempt_tool_names={"grep_search"},
        validated_reacquisition=False,
        target_already_validated=True,
        payload_pressure_ready=True,
        validated_target_exact_reacquired=False,
        validated_target_reconfirmation_attempt=False,
        mixed_answer_tool_event=True,
        toolless_fresh_answer_event=False,
        unsupported_tool_event=False,
        bad_tool_args_event=False,
        prior_turn_has_valid_supporting_tool_backing=False,
        current_turn_was_tool_only_retry=False,
        current_turn_satisfied_tool_only_stage=False,
        retry_index=1,
    )

    assert "your previous turn mixed tool use with a final answer" in prompt
    assert "Do not reopen the exact target again on this retry" in prompt
    assert "You must use exactly one supported read-only tool before answering" in prompt
    assert signals == {
        "retry_prompt_shape_mixed_turn": 1,
        "retry_prompt_no_exact_reread": 1,
        "retry_prompt_requires_supporting_tool": 1,
    }


def test_retry_prompt_uses_toolless_template_for_late_validated_target(
    tmp_path,
) -> None:
    task = StressTask(
        id="toolless_reconfirm_health",
        phase_name="tool-contract",
        prompt="Reconfirm health.",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
    )
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    prompt, signals = harness._retry_prompt(
        task=task,
        expected_fields={
            "file": "src/tok/gateway.py",
            "verification": "health",
        },
        observed_fields={},
        attempt_tool_count=0,
        attempt_tool_names=set(),
        validated_reacquisition=False,
        target_already_validated=True,
        payload_pressure_ready=True,
        validated_target_exact_reacquired=False,
        validated_target_reconfirmation_attempt=False,
        mixed_answer_tool_event=False,
        toolless_fresh_answer_event=True,
        unsupported_tool_event=False,
        bad_tool_args_event=False,
        prior_turn_has_valid_supporting_tool_backing=False,
        current_turn_was_tool_only_retry=False,
        current_turn_satisfied_tool_only_stage=False,
        retry_index=1,
    )

    assert "you answered without satisfying the fresh-evidence requirement" in prompt
    assert "Do not reopen the exact validated target on this retry" in prompt
    assert "You must use exactly one supported read-only tool before answering" in prompt
    assert signals == {
        "retry_prompt_shape_toolless_fresh": 1,
        "retry_prompt_no_exact_reread": 1,
        "retry_prompt_requires_supporting_tool": 1,
    }


def test_retry_prompt_uses_exact_target_reread_template(tmp_path) -> None:
    task = StressTask(
        id="reopen_health",
        phase_name="tool-contract",
        prompt="Reconfirm health.",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
    )
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    prompt, signals = harness._retry_prompt(
        task=task,
        expected_fields={
            "file": "src/tok/gateway.py",
            "verification": "health",
        },
        observed_fields={},
        attempt_tool_count=1,
        attempt_tool_names={"grep_search"},
        validated_reacquisition=True,
        target_already_validated=True,
        payload_pressure_ready=True,
        validated_target_exact_reacquired=True,
        validated_target_reconfirmation_attempt=False,
        mixed_answer_tool_event=True,
        toolless_fresh_answer_event=False,
        unsupported_tool_event=False,
        bad_tool_args_event=False,
        prior_turn_has_valid_supporting_tool_backing=False,
        current_turn_was_tool_only_retry=False,
        current_turn_satisfied_tool_only_stage=False,
        retry_index=2,
    )

    assert "you reopened an already validated exact target" in prompt
    assert "Do not read or search the exact validated target again on this retry" in prompt
    assert signals == {
        "retry_prompt_shape_exact_target_reread": 1,
        "retry_prompt_no_exact_reread": 1,
    }


def test_retry_prompt_preserves_late_validated_bad_args_guidance(
    tmp_path,
) -> None:
    task = StressTask(
        id="bad_args_health",
        phase_name="tool-contract",
        prompt="Find health.",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
    )
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    prompt, signals = harness._retry_prompt(
        task=task,
        expected_fields={
            "file": "src/tok/gateway.py",
            "verification": "health",
        },
        observed_fields={},
        attempt_tool_count=0,
        attempt_tool_names=set(),
        validated_reacquisition=False,
        target_already_validated=True,
        payload_pressure_ready=True,
        validated_target_exact_reacquired=False,
        validated_target_reconfirmation_attempt=False,
        mixed_answer_tool_event=False,
        toolless_fresh_answer_event=False,
        unsupported_tool_event=False,
        bad_tool_args_event=True,
        prior_turn_has_valid_supporting_tool_backing=False,
        current_turn_was_tool_only_retry=False,
        current_turn_satisfied_tool_only_stage=False,
        retry_index=1,
    )

    assert "If fresh evidence is required, use the read-only tools first." in prompt
    assert "Do not reopen the exact validated target" not in prompt
    assert "Do not read or search the exact validated target again" not in prompt
    assert signals == {"retry_prompt_shape_bad_args": 1}


def test_retry_prompt_mixed_turn_keeps_direct_answer_option_when_fresh_evidence_not_required(
    tmp_path,
) -> None:
    task = StressTask(
        id="mixed_reconfirm_health",
        phase_name="tool-contract",
        prompt="Reconfirm health.",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=False,
        require_tool_count=0,
    )
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    prompt, signals = harness._retry_prompt(
        task=task,
        expected_fields={
            "file": "src/tok/gateway.py",
            "verification": "health",
        },
        observed_fields={},
        attempt_tool_count=0,
        attempt_tool_names=set(),
        validated_reacquisition=False,
        target_already_validated=True,
        payload_pressure_ready=True,
        validated_target_exact_reacquired=False,
        validated_target_reconfirmation_attempt=False,
        mixed_answer_tool_event=True,
        toolless_fresh_answer_event=False,
        unsupported_tool_event=False,
        bad_tool_args_event=False,
        prior_turn_has_valid_supporting_tool_backing=False,
        current_turn_was_tool_only_retry=False,
        current_turn_satisfied_tool_only_stage=False,
        retry_index=1,
    )

    assert "Either use already validated evidence to answer directly" in prompt
    assert "You must use exactly one supported read-only tool before answering" not in prompt
    assert signals == {
        "retry_prompt_shape_mixed_turn": 1,
        "retry_prompt_no_exact_reread": 1,
    }


def test_retry_prompt_uses_early_answer_only_for_mixed_when_fresh_evidence_not_required(
    tmp_path,
) -> None:
    task = StressTask(
        id="mixed_contract_early",
        phase_name="tool-contract",
        prompt="Find health.",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=False,
        require_tool_count=0,
    )
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    prompt, signals = harness._retry_prompt(
        task=task,
        expected_fields={
            "file": "src/tok/gateway.py",
            "verification": "health",
        },
        observed_fields={},
        attempt_tool_count=0,
        attempt_tool_names=set(),
        validated_reacquisition=False,
        target_already_validated=False,
        payload_pressure_ready=False,
        validated_target_exact_reacquired=False,
        validated_target_reconfirmation_attempt=False,
        mixed_answer_tool_event=True,
        toolless_fresh_answer_event=False,
        unsupported_tool_event=False,
        bad_tool_args_event=False,
        prior_turn_has_valid_supporting_tool_backing=False,
        current_turn_was_tool_only_retry=False,
        current_turn_satisfied_tool_only_stage=False,
        retry_index=1,
    )

    assert "enough evidence is already available" in prompt
    assert "Do not call tools in this turn" in prompt
    assert "File=src/tok/gateway.py" in prompt
    assert signals == {
        "retry_prompt_shape_generic_retry": 1,
        "early_retry_contract_stage_answer_only": 1,
    }


def test_retry_prompt_preserves_generic_prompt_for_non_late_or_non_validated_retry(
    tmp_path,
) -> None:
    task = StressTask(
        id="generic_health",
        phase_name="reuse-probe",
        prompt="Find health.",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
    )
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    prompt, signals = harness._retry_prompt(
        task=task,
        expected_fields={
            "file": "src/tok/gateway.py",
            "verification": "health",
        },
        observed_fields={},
        attempt_tool_count=0,
        attempt_tool_names=set(),
        validated_reacquisition=False,
        target_already_validated=False,
        payload_pressure_ready=False,
        validated_target_exact_reacquired=False,
        validated_target_reconfirmation_attempt=False,
        mixed_answer_tool_event=True,
        toolless_fresh_answer_event=False,
        unsupported_tool_event=False,
        bad_tool_args_event=False,
        prior_turn_has_valid_supporting_tool_backing=False,
        current_turn_was_tool_only_retry=False,
        current_turn_satisfied_tool_only_stage=False,
        retry_index=1,
    )

    assert "Focus on the expected target class" in prompt
    assert "Do not reopen the exact target again on this retry" not in prompt
    assert signals == {"retry_prompt_shape_generic_retry": 1}


def test_retry_prompt_uses_early_tool_only_for_mixed_when_fresh_evidence_required(
    tmp_path,
) -> None:
    task = StressTask(
        id="mixed_contract",
        phase_name="tool-contract",
        prompt="Find health.",
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
    )
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    prompt, signals = harness._retry_prompt(
        task=task,
        expected_fields={
            "file": "src/tok/gateway.py",
            "verification": "health",
        },
        observed_fields={},
        attempt_tool_count=0,
        attempt_tool_names=set(),
        validated_reacquisition=False,
        target_already_validated=False,
        payload_pressure_ready=False,
        validated_target_exact_reacquired=False,
        validated_target_reconfirmation_attempt=False,
        mixed_answer_tool_event=True,
        toolless_fresh_answer_event=False,
        unsupported_tool_event=False,
        bad_tool_args_event=False,
        prior_turn_has_valid_supporting_tool_backing=False,
        current_turn_was_tool_only_retry=False,
        current_turn_satisfied_tool_only_stage=False,
        retry_index=1,
    )

    assert "Do not answer yet" in prompt
    assert "Use exactly one supported read-only tool in this turn and nothing else" in prompt
    assert "Do not include File= or Verification=" in prompt
    assert signals == {
        "retry_prompt_shape_generic_retry": 1,
        "early_retry_contract_stage_tool_only": 1,
    }


def test_retry_prompt_uses_tool_only_stage_for_late_generic_retry(
    tmp_path,
) -> None:
    task = StressTask(
        id="payload_pressure_collect",
        phase_name="payload-pressure",
        prompt="Collect payload behavior.",
        expected_file="src/tok/gateway.py",
        expected_verification="collect_behavior",
        require_fresh_evidence=True,
        require_tool_count=1,
    )
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    prompt, signals = harness._retry_prompt(
        task=task,
        expected_fields={
            "file": "src/tok/gateway.py",
            "verification": "collect_behavior",
        },
        observed_fields={},
        attempt_tool_count=0,
        attempt_tool_names=set(),
        validated_reacquisition=False,
        target_already_validated=False,
        payload_pressure_ready=True,
        validated_target_exact_reacquired=False,
        validated_target_reconfirmation_attempt=False,
        mixed_answer_tool_event=False,
        toolless_fresh_answer_event=False,
        unsupported_tool_event=False,
        bad_tool_args_event=False,
        prior_turn_has_valid_supporting_tool_backing=False,
        current_turn_was_tool_only_retry=False,
        current_turn_satisfied_tool_only_stage=False,
        retry_index=1,
    )

    assert "do not answer yet" in prompt.lower()
    assert "Use exactly one supported read-only tool in this turn and nothing else" in prompt
    assert "Do not include File= or Verification=" in prompt
    assert signals == {
        "retry_prompt_shape_generic_retry": 1,
        "late_retry_contract_stage_tool_only": 1,
    }


def test_retry_prompt_uses_answer_only_stage_when_supporting_tool_backing_exists(
    tmp_path,
) -> None:
    task = StressTask(
        id="payload_pressure_collect",
        phase_name="payload-pressure",
        prompt="Collect payload behavior.",
        expected_file="src/tok/gateway.py",
        expected_verification="collect_behavior",
        require_fresh_evidence=True,
        require_tool_count=1,
    )
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    prompt, signals = harness._retry_prompt(
        task=task,
        expected_fields={
            "file": "src/tok/gateway.py",
            "verification": "collect_behavior",
        },
        observed_fields={},
        attempt_tool_count=1,
        attempt_tool_names={"view_file"},
        validated_reacquisition=False,
        target_already_validated=True,
        payload_pressure_ready=True,
        validated_target_exact_reacquired=False,
        validated_target_reconfirmation_attempt=False,
        mixed_answer_tool_event=False,
        toolless_fresh_answer_event=False,
        unsupported_tool_event=False,
        bad_tool_args_event=False,
        prior_turn_has_valid_supporting_tool_backing=True,
        current_turn_was_tool_only_retry=False,
        current_turn_satisfied_tool_only_stage=False,
        retry_index=2,
    )

    assert "enough evidence is already available" in prompt
    assert "Do not call tools in this turn" in prompt
    assert "File=src/tok/gateway.py" in prompt
    assert "Verification=collect_behavior" in prompt
    assert "Do not reopen the exact validated target" in prompt
    assert signals == {
        "retry_prompt_shape_generic_retry": 1,
        "late_retry_contract_stage_answer_only": 1,
        "late_retry_no_exact_target": 1,
    }


def test_retry_prompt_transitions_tool_only_retry_to_answer_only(
    tmp_path,
) -> None:
    task = StressTask(
        id="payload_pressure_collect",
        phase_name="payload-pressure",
        prompt="Collect payload behavior.",
        expected_file="src/tok/gateway.py",
        expected_verification="collect_behavior",
        require_fresh_evidence=True,
        require_tool_count=1,
    )
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    prompt, signals = harness._retry_prompt(
        task=task,
        expected_fields={
            "file": "src/tok/gateway.py",
            "verification": "collect_behavior",
        },
        observed_fields={},
        attempt_tool_count=1,
        attempt_tool_names={"view_file"},
        validated_reacquisition=False,
        target_already_validated=False,
        payload_pressure_ready=True,
        validated_target_exact_reacquired=False,
        validated_target_reconfirmation_attempt=False,
        mixed_answer_tool_event=False,
        toolless_fresh_answer_event=False,
        unsupported_tool_event=False,
        bad_tool_args_event=False,
        prior_turn_has_valid_supporting_tool_backing=True,
        current_turn_was_tool_only_retry=True,
        current_turn_satisfied_tool_only_stage=True,
        retry_index=2,
    )

    assert "Do not call tools in this turn" in prompt
    assert signals == {
        "retry_prompt_shape_generic_retry": 1,
        "late_retry_contract_stage_answer_only": 1,
    }


def test_tool_only_retry_stage_satisfaction_requires_one_clean_read_only_tool(
    tmp_path,
) -> None:
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    satisfied_turn = SimpleNamespace(
        tool_uses=[{"name": "view_file", "input": {"path": "src/tok/gateway.py"}}],
        validated_target_exact_reacquired=False,
        output_behavior_signals={},
        visible_response="",
    )
    mixed_turn = SimpleNamespace(
        tool_uses=[{"name": "view_file", "input": {"path": "src/tok/gateway.py"}}],
        validated_target_exact_reacquired=False,
        output_behavior_signals={"mixed_answer_tool_event": 1},
        visible_response="File=src/tok/gateway.py\nVerification=health",
    )
    toolless_turn = SimpleNamespace(
        tool_uses=[],
        validated_target_exact_reacquired=False,
        output_behavior_signals={},
        visible_response="",
    )

    assert (
        harness._turn_satisfies_tool_only_retry_stage(satisfied_turn) is True  # type: ignore[arg-type]
    )
    assert harness._turn_satisfies_tool_only_retry_stage(mixed_turn) is False  # type: ignore[arg-type]
    assert (
        harness._turn_satisfies_tool_only_retry_stage(toolless_turn) is False  # type: ignore[arg-type]
    )


def test_answer_only_retry_stage_requires_two_line_answer_without_tools(
    tmp_path,
) -> None:
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    satisfied_turn = SimpleNamespace(
        tool_uses=[],
        output_behavior_signals={},
        visible_response="File=src/tok/gateway.py\nVerification=health",
    )
    tool_violation_turn = SimpleNamespace(
        tool_uses=[{"name": "view_file", "input": {"path": "src/tok/gateway.py"}}],
        output_behavior_signals={},
        visible_response="",
    )

    assert (
        harness._turn_satisfies_answer_only_retry_stage(satisfied_turn) is True  # type: ignore[arg-type]
    )
    assert (
        harness._turn_satisfies_answer_only_retry_stage(tool_violation_turn)  # type: ignore[arg-type]
        is False
    )


def test_first_irreversible_miss_kind_classifies_failure_shapes(
    tmp_path,
) -> None:
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    def turn(**kwargs):
        base = {
            "task_id": "test_task",
            "phase_name": "test_phase",
            "turn_index": 0,
            "phase": "test",
            "prompt": "test prompt",
            "raw_response": "test response",
            "visible_response": "",
            "tool_uses": [],
            "tool_results": [],
            "evidence_chars": 0,
            "retry_index": 0,
            "validated": False,
            "input_behavior_signals": {},
            "output_behavior_signals": {},
            "input_saved_tokens": 0,
            "output_saved_tokens": 0,
            "tool_contract_failure": False,
            "state_payload_chars": 0,
            "resend_mode": "none",
        }
        base.update(kwargs)
        return StressTurnRecord(**base)

    assert (
        harness._first_irreversible_miss_kind([turn(output_behavior_signals={"bad_tool_args_event": 1})])
        == "first_miss_bad_args"
    )
    assert (
        harness._first_irreversible_miss_kind([turn(output_behavior_signals={"unsupported_tool_event": 1})])
        == "first_miss_unsupported_tool"
    )
    assert (
        harness._first_irreversible_miss_kind([turn(output_behavior_signals={"mixed_answer_tool_event": 1})])
        == "first_miss_mixed_answer_tool"
    )
    assert (
        harness._first_irreversible_miss_kind([turn(output_behavior_signals={"toolless_fresh_answer_event": 1})])
        == "first_miss_toolless_fresh"
    )
    assert (
        harness._first_irreversible_miss_kind([turn(tool_uses=[{"name": "delta", "input": {}}])])
        == "first_miss_tool_only_insufficient"
    )
    assert (
        harness._first_irreversible_miss_kind(
            [
                turn(
                    tool_uses=[{"name": "view_file", "input": {"path": "x"}}],
                    output_behavior_signals={"unsupported_tool_event": 1},
                )
            ]
        )
        == "first_miss_unsupported_tool"
    )
    assert (
        harness._first_irreversible_miss_kind(
            [
                turn(
                    tool_uses=[
                        {
                            "name": "view_file",
                            "input": {"path": "src/tok/gateway.py"},
                        }
                    ],
                    visible_response="",
                ),
                turn(visible_response="File=src/tok/gateway.py"),
            ]
        )
        == "first_miss_answer_after_grounding"
    )
    assert (
        harness._first_irreversible_miss_kind([turn(visible_response="just prose reply")]) == "first_miss_prose_no_tool"
    )


def test_failed_task_summaries_and_locus_capture_retry_families(
    tmp_path,
) -> None:
    harness = StressHarness(
        StressHarnessConfig(),
        client=_FakeClient([]),
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    turns = [
        SimpleNamespace(
            task_id="task_one",
            phase_name="tool-contract",
            task_completed_validated=False,
            retry_index=0,
            tool_uses=[],
            validated_target_exact_reacquired=False,
            visible_response="just prose",
            output_behavior_signals={},
        ),
        SimpleNamespace(
            task_id="task_two",
            phase_name="payload-pressure",
            task_completed_validated=False,
            retry_index=1,
            tool_uses=[],
            validated_target_exact_reacquired=False,
            visible_response="",
            output_behavior_signals={"late_retry_contract_stage_tool_only": 1},
        ),
    ]

    failed = harness._failed_task_summaries(turns)  # type: ignore[arg-type]

    assert failed[0]["retry_family"] == "none"
    assert failed[0]["first_irreversible_miss_kind"] == "first_miss_prose_no_tool"
    assert failed[1]["retry_family"] == "late_staged"
    assert (
        harness._dominant_failure_locus(
            failed_task_summaries=failed,
            answer_anchor_reacquisition_events_seen=0,
            answer_ready_reacquisition_events_seen=0,
            repair_phase_reacquisition_events_seen=0,
            answer_ready_repair_failed_count=0,
            fallback_after_compaction_eligible=False,
        )
        == "agent"
    )


def test_repeat_tool_use_before_validation_does_not_count_as_reacquisition(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        '@Tool view_file id:"s2"\n  path:"src/tok/gateway.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=1,
        max_tool_rounds=3,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="fresh_fallback",
                phase_name="fresh-grounding",
                prompt="Find fallback.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="record_fallback_event",
                require_fresh_evidence=True,
                require_tool_count=1,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    classes = [item.breakpoint_class for item in result.breakpoints]

    assert "reacquisition_loop" not in classes


def test_anchor_seed_requires_direct_file_read_not_only_search(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        '@Tool grep_search id:"s1"\n  query:"health"\n  search_path:"src/tok"',
        "File=src/tok/gateway.py\nVerification=health",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=1,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()

    assert result.validated_anchor_count == 0


def test_anchor_seed_succeeds_after_narrow_search_then_direct_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        '@Tool grep_search id:"s1"\n  query:"health"\n  search_path:"src/tok/gateway.py"',
        '@Tool view_file id:"s2"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=1,
        max_tool_rounds=3,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()

    assert result.validated_anchor_count == 1
    assert result.seed_searches == 1
    assert result.seed_direct_reads == 1
    assert result.seed_evidence_sufficient is True
    assert any("Use the evidence you just retrieved." in turn.prompt for turn in result.turns[1:])


def test_repeated_seed_tool_use_after_evidence_counts_as_failure_pressure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        '@Tool grep_search id:"s2"\n  query:"health"\n  search_path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=1,
        max_tool_rounds=3,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    classes = [item.breakpoint_class for item in result.breakpoints]

    assert result.validated_anchor_count == 1
    assert "tool_contract_failure" in classes


def test_later_tasks_wait_until_reuse_and_checkpoint_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=3,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="anchor_seed_2",
                phase_name="anchor-seed",
                prompt="Find answer memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reuse_check_1",
                phase_name="reuse-vs-reacquire",
                prompt="Recover seed one.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                forbid_reacquisition=True,
                min_validated_anchors=2,
            ),
            StressTask(
                id="payload_late",
                phase_name="payload-pressure",
                prompt="Late payload task.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="collect_behavior_signals",
                require_fresh_evidence=True,
                require_tool_count=2,
                requires_memory_surfaces=True,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    phase_names = [turn.phase_name for turn in result.turns]

    assert result.reuse_checks_run == 1
    assert result.checkpoint_checks_run == 1
    assert "payload-pressure" not in phase_names


def test_reuse_probe_from_memory_increments_success_without_reacquisition(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=3,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="seed1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="seed2",
                phase_name="anchor-seed",
                prompt="Find memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reuse_probe_exact",
                phase_name="reuse-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    classes = [item.breakpoint_class for item in result.breakpoints]

    assert result.reuse_probe_attempts == 1
    assert result.reuse_probe_successes == 1
    assert result.reacquisition_events_seen == 0
    assert "reacquisition_loop" not in classes


def test_reuse_probe_tool_use_triggers_reacquisition_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s3"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=3,
        max_tool_rounds=2,
        max_retries_per_task=1,
        task_catalog=(
            StressTask(
                id="seed1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="seed2",
                phase_name="anchor-seed",
                prompt="Find memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reuse_probe_exact",
                phase_name="reuse-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    classes = [item.breakpoint_class for item in result.breakpoints]

    assert result.reuse_probe_attempts == 1
    assert result.reacquisition_events_seen >= 1
    assert "reacquisition_loop" in classes


def test_retention_probe_can_trigger_latest_anchor_substitution(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s3"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        '@Tool view_file id:"s4"\n  path:"src/tok/cli.py"',
        "File=src/tok/cli.py\nVerification=_gate_release_summary",
        "File=src/tok/cli.py\nVerification=_gate_release_summary",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=6,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="seed1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="seed2",
                phase_name="anchor-seed",
                prompt="Find memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reuse_probe_exact",
                phase_name="reuse-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="fresh3",
                phase_name="fresh-grounding",
                prompt="Find fallback.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="record_fallback_event",
                require_fresh_evidence=True,
                require_tool_count=1,
            ),
            StressTask(
                id="fresh4",
                phase_name="fresh-grounding",
                prompt="Find release summary.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/cli.py",
                expected_verification="_gate_release_summary",
                require_fresh_evidence=True,
                require_tool_count=1,
            ),
            StressTask(
                id="retention_probe_oldest",
                phase_name="retention-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=4,
                min_checkpoint_checks=1,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    classes = [item.breakpoint_class for item in result.breakpoints]

    assert result.retention_probe_attempts == 1
    assert result.retention_substitution_events_seen >= 1
    assert "retention_loss" in classes


def test_early_retention_probe_runs_before_payload_pressure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "5")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=_response_contract_for_mode",
        '@Tool view_file id:"s3"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool grep_search id:"s4"\n  query:"pressure"\n  search_path:"src/tok"',
        "File=src/tok/universal_runtime.py\nVerification=collect_behavior_signals",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=7,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="anchor_seed_2",
                phase_name="anchor-seed",
                prompt="Find memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reuse_probe_exact",
                phase_name="reuse-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="retention_probe_early",
                phase_name="retention-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="reuse_probe_near_neighbor",
                phase_name="reuse-probe",
                prompt="Differentiate helper.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="_response_contract_for_mode",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="fresh_anchor_runtime",
                phase_name="fresh-grounding",
                prompt="Find fallback.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="record_fallback_event",
                require_fresh_evidence=True,
                require_tool_count=1,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="collect_behavior_anchor",
                phase_name="payload-pressure",
                prompt="Trace pressure.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="collect_behavior_signals",
                require_fresh_evidence=True,
                require_tool_count=1,
                force_payload=True,
                requires_memory_surfaces=True,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    phase_order = [(turn.task_id, turn.phase_name) for turn in result.turns]
    retention_index = next(i for i, (task_id, _) in enumerate(phase_order) if task_id == "retention_probe_early")
    payload_index = next(i for i, (_, phase_name) in enumerate(phase_order) if phase_name == "payload-pressure")

    assert retention_index < payload_index


def test_tool_contract_probe_runs_before_fresh_anchor_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "5")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s3"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s4"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=_response_contract_for_mode",
        '@Tool view_file id:"s5"\n  path:"src/tok/cli.py"',
        "File=src/tok/cli.py\nVerification=_gate_release_summary",
        '@Tool view_file id:"s6"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=8,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="anchor_seed_2",
                phase_name="anchor-seed",
                prompt="Find memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reuse_probe_exact",
                phase_name="reuse-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="retention_probe_early",
                phase_name="retention-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="tool_contract_mixed_answer_and_tool",
                phase_name="tool-contract",
                prompt="Reconfirm health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="tool_contract_bad_args_shape",
                phase_name="tool-contract",
                prompt="Find helper.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="_response_contract_for_mode",
                require_fresh_evidence=True,
                require_tool_count=1,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="tool_contract_toolless_fresh_answer",
                phase_name="tool-contract",
                prompt="Find release summary.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/cli.py",
                expected_verification="_gate_release_summary",
                require_fresh_evidence=True,
                require_tool_count=1,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="fresh_anchor_runtime",
                phase_name="fresh-grounding",
                prompt="Find fallback.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="record_fallback_event",
                require_fresh_evidence=True,
                require_tool_count=1,
                min_reuse_checks=1,
                min_checkpoint_checks=1,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    phase_order = [(turn.task_id, turn.phase_name) for turn in result.turns]
    first_contract_index = next(i for i, (_, phase_name) in enumerate(phase_order) if phase_name == "tool-contract")
    fresh_index = next(i for i, (task_id, _) in enumerate(phase_order) if task_id == "fresh_anchor_runtime")

    assert first_contract_index < fresh_index
    assert result.tool_contract_probe_attempts >= 1


def test_early_retention_probe_can_hold_and_set_retention_surface_held(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "5")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=_response_contract_for_mode",
        '@Tool view_file id:"s3"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        "File=src/tok/gateway.py\nVerification=health",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=6,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="anchor_seed_2",
                phase_name="anchor-seed",
                prompt="Find memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reuse_probe_exact",
                phase_name="reuse-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="retention_probe_early",
                phase_name="retention-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="reuse_probe_near_neighbor",
                phase_name="reuse-probe",
                prompt="Differentiate helper.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="_response_contract_for_mode",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="fresh_anchor_runtime",
                phase_name="fresh-grounding",
                prompt="Find fallback.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="record_fallback_event",
                require_fresh_evidence=True,
                require_tool_count=1,
                min_checkpoint_checks=1,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    classes = [item.breakpoint_class for item in result.breakpoints]

    assert result.retention_probe_attempts >= 1
    assert result.retention_probe_successes >= 1
    assert result.retention_substitution_events_seen == 0
    assert "retention_loss" not in classes
    assert result.run_diagnosis == "retention_surface_held_early_only"


def test_late_retention_probe_runs_before_fallback_and_can_hold(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "5")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s3"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        "File=src/tok/gateway.py\nVerification=_response_contract_for_mode",
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s4"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        '@Tool view_file id:"s5"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=8,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="anchor_seed_2",
                phase_name="anchor-seed",
                prompt="Find memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reuse_probe_exact",
                phase_name="reuse-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="retention_probe_early",
                phase_name="retention-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="fresh_anchor_runtime",
                phase_name="fresh-grounding",
                prompt="Find fallback.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="record_fallback_event",
                require_fresh_evidence=True,
                require_tool_count=1,
                min_reuse_checks=1,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="reuse_probe_near_neighbor",
                phase_name="reuse-probe",
                prompt="Differentiate helper.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="_response_contract_for_mode",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="retention_probe_late",
                phase_name="retention-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=3,
                min_reuse_checks=1,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="fallback_anchor",
                phase_name="fresh-grounding",
                prompt="Find fallback.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="record_fallback_event",
                require_fresh_evidence=True,
                require_tool_count=1,
                requires_memory_surfaces=True,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    phase_order = [(turn.task_id, turn.phase_name) for turn in result.turns]
    late_retention_index = next(i for i, (task_id, _) in enumerate(phase_order) if task_id == "retention_probe_late")
    fallback_index = next(i for i, (task_id, _) in enumerate(phase_order) if task_id == "fallback_anchor")

    assert late_retention_index < fallback_index
    assert result.late_retention_probe_attempts >= 1
    assert result.late_retention_probe_successes >= 1
    assert result.run_diagnosis == "retention_surface_held"


def test_late_retention_probe_tool_use_does_not_count_as_success(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "5")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s3"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=record_fallback_event",
        "File=src/tok/gateway.py\nVerification=_response_contract_for_mode",
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s4"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=7,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="anchor_seed_2",
                phase_name="anchor-seed",
                prompt="Find memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reuse_probe_exact",
                phase_name="reuse-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="retention_probe_early",
                phase_name="retention-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="fresh_anchor_runtime",
                phase_name="fresh-grounding",
                prompt="Find fallback.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="record_fallback_event",
                require_fresh_evidence=True,
                require_tool_count=1,
                min_reuse_checks=1,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="reuse_probe_near_neighbor",
                phase_name="reuse-probe",
                prompt="Differentiate helper.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="_response_contract_for_mode",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="retention_probe_late",
                phase_name="retention-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=3,
                min_reuse_checks=1,
                min_checkpoint_checks=1,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()

    assert result.late_retention_probe_attempts >= 1
    assert result.late_retention_probe_successes == 0


def test_missing_memory_classes_without_probes_are_diagnosed_as_unexercised(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "5")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/gateway.py\nVerification=health",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=2,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="seed1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="seed2",
                phase_name="anchor-seed",
                prompt="Find memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()

    assert result.reuse_probe_attempts == 0
    assert result.retention_probe_attempts == 0
    assert result.run_diagnosis == "reuse_surface_unexercised"


def test_payload_eligibility_distinguishes_high_bytes_from_compaction_ready(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "5")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/gateway.py\nVerification=health",
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool grep_search id:"s3"\n  query:"pressure"\n  search_path:"src/tok"',
        '@Tool view_file id:"s4"\n  path:"src/tok/universal_runtime.py"\n@Tool view_file id:"s5"\n  path:"src/tok/gateway.py"',
        "File=src/tok/universal_runtime.py\nVerification=collect_behavior_signals",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=4,
        max_tool_rounds=3,
        max_retries_per_task=0,
        min_payload_pressure_bytes=100,
        task_catalog=(
            StressTask(
                id="seed1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="seed2",
                phase_name="anchor-seed",
                prompt="Find memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="reuse_probe_exact",
                phase_name="reuse-probe",
                prompt="Recover oldest.\nFile={file}\nVerification={verification}",
                dynamic_anchor="oldest",
                forbid_reacquisition=True,
                min_validated_anchors=2,
                min_checkpoint_checks=1,
            ),
            StressTask(
                id="collect_behavior_anchor",
                phase_name="payload-pressure",
                prompt="Trace pressure.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="collect_behavior_signals",
                require_fresh_evidence=True,
                require_tool_count=3,
                min_validated_anchors=2,
                force_payload=True,
                requires_memory_surfaces=True,
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()

    assert result.payload_pressure_reached is True
    assert result.compaction_eligible is True


def test_early_baseline_is_diagnosed_as_early_contract_collapse(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "1")
    responses = [
        "File=src/tok/gateway.py\nVerification=_persist_answer_anchor",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=1,
        max_tool_rounds=1,
        max_retries_per_task=0,
        fallback_threshold=1,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()

    assert result.baseline_only is True
    assert result.run_diagnosis == "early_contract_collapse:navigation"
    assert result.first_anchor_failure_mode == "navigation"
    assert result.anchors_before_baseline == 0


def test_early_baseline_can_identify_answer_assembly_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "1")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "I found the evidence in the file, but I am deliberately responding in plain prose without the required File or Verification fields so this should count as answer assembly failure.",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=1,
        max_tool_rounds=2,
        max_retries_per_task=0,
        fallback_threshold=1,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()

    assert result.baseline_only is True
    assert result.run_diagnosis == "early_contract_collapse:answer_assembly"
    assert result.first_anchor_failure_mode == "answer_assembly"
    assert result.seed_evidence_sufficient is True


def test_first_checkpoint_can_trigger_retention_loss(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_FALLBACK_THRESHOLD", "3")
    responses = [
        '@Tool view_file id:"s1"\n  path:"src/tok/gateway.py"',
        "File=src/tok/gateway.py\nVerification=health",
        '@Tool view_file id:"s2"\n  path:"src/tok/universal_runtime.py"',
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
        "File=src/tok/universal_runtime.py\nVerification=_process_answer_memory",
    ]
    client = _FakeClient(responses)
    config = StressHarnessConfig(
        max_tasks=2,
        max_tool_rounds=2,
        max_retries_per_task=0,
        task_catalog=(
            StressTask(
                id="anchor_seed_1",
                phase_name="anchor-seed",
                prompt="Find health.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/gateway.py",
                expected_verification="health",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
            StressTask(
                id="anchor_seed_2",
                phase_name="anchor-seed",
                prompt="Find answer memory.\nFile=<the primary file>\nVerification=<the function>",
                expected_file="src/tok/universal_runtime.py",
                expected_verification="_process_answer_memory",
                require_fresh_evidence=True,
                require_tool_count=1,
                required_tool_names=("view_file", "read"),
            ),
        ),
    )
    harness = StressHarness(
        config,
        client=client,
        session=RuntimeSession(memory_dir=tmp_path),
        workspace_root=Path(__file__).parent.parent.parent,
    )

    result = harness.run()
    classes = [item.breakpoint_class for item in result.breakpoints]

    assert "retention_loss" in classes
    assert result.checkpoint_checks_run == 1
