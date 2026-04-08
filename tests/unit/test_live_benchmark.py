import json
from types import SimpleNamespace
from typing import Any

from tok.testing.live_benchmark import (
    BenchmarkDefinition,
    BenchmarkResult,
    LiveBenchmarkRunner,
    ProviderUsageSnapshot,
    _chunk_messages,
    _turn_prompts,
    compare_results,
    load_benchmark_definition,
    normalize_fixture_messages,
    render_comparison_markdown,
    render_stability_markdown,
    select_preferred_mode,
    summarize_compare_runs,
)


class _FakeCompletions:
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int) -> None:
        self._response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def create(self, **kwargs):
        return self._response


class _FakeClient:
    def __init__(
        self,
        content: str,
        prompt_tokens: int = 100,
        completion_tokens: int = 20,
    ) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content, prompt_tokens, completion_tokens))


def _result(
    *,
    mode: str,
    total_tokens: int,
    prompt_tokens: int,
    success: bool = True,
    tok_overhead_tokens: int = 0,
    total_saved_tokens: int = 0,
    response_signals: dict[str, int] | None = None,
    reacquisition_cost_tokens: int = 0,
    invisible_pressure: int = 0,
) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark="coding-loop",
        mode=mode,
        model="m",
        provider="p",
        fixture_path="f",
        provider_usage=ProviderUsageSnapshot(
            prompt_tokens=prompt_tokens,
            completion_tokens=total_tokens - prompt_tokens,
            total_tokens=total_tokens,
            latency_ms=10.0,
        ),
        compression_metrics={
            "input_saved_tokens": max(0, total_saved_tokens),
            "output_saved_tokens": 0,
            "total_saved_tokens": total_saved_tokens,
            "input_behavior_signals": {},
            "type_breakdown": {},
        },
        prompt_metrics={
            "system_prompt_tokens": 10,
            "normalized_messages_tokens": 20,
            "prepared_messages_tokens": 20,
            "tok_system_additions_tokens": 0,
            "tok_overhead_tokens": tok_overhead_tokens,
            "estimated_prompt_delta_tokens": 0,
            "outbound_prompt_estimate_tokens": 30,
        },
        response_metrics={
            "response_behavior_signals": response_signals or {},
            "invisible_pressure": invisible_pressure,
            "reacquisition_cost_tokens": reacquisition_cost_tokens,
            "family_mode": "",
            "response_mode": mode,
        },
        diagnostics={
            "tool_compatible_requested": mode == "tok-tool-compatible",
            "request_messages_before": 3,
            "request_messages_after": 2,
            "session_turns": 3,
            "response_warning_signal_count": sum((response_signals or {}).values()),
        },
        task_success=success,
        matched_success_terms=["gateway.py", "passed"] if success else [],
        request_messages=2,
        turn_count=3,
        turns=[],
        visible_response="ok",
        raw_response="ok",
    )


def test_normalize_fixture_messages_converts_tools_and_appends_followup() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Read the file"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "a1",
                    "name": "view_file",
                    "input": {"path": "x.py"},
                }
            ],
        },
        {"role": "tool_result", "tool_use_id": "a1", "content": "print('ok')"},
    ]

    normalized = normalize_fixture_messages(messages, "What changed?")

    assert normalized[0] == {"role": "user", "content": "Read the file"}
    assert any("Tool use (view_file)" in str(b) for b in normalized[1]["content"])
    assert "Tool result (a1)" in normalized[2]["content"]
    assert normalized[-1]["content"] == "What changed?"


def test_chunk_messages_respects_replay_turn_boundaries() -> None:
    fixture: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Find where history compression is implemented.",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r1",
                    "name": "grep_search",
                    "input": {
                        "search_path": "src",
                        "query": "compress_history",
                    },
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "r1",
            "content": "src/tok/compression.py:305: def compress_history(",
        },
        {
            "role": "user",
            "content": "What's the main implementation entry point?",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r2",
                    "name": "view_file",
                    "input": {"path": "src/tok/compression.py"},
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "r2",
            "content": "def compress_history(messages, keep_turns=2, profile=None):",
        },
        {"role": "user", "content": "Is there any related memory structure?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r3",
                    "name": "grep_search",
                    "input": {
                        "search_path": "src",
                        "query": "BridgeMemoryState",
                    },
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "r3",
            "content": "src/tok/bridge_memory.py:14: class BridgeMemoryState:",
        },
        {"role": "user", "content": "Please summarize the answer."},
    ]

    normalized = normalize_fixture_messages(
        fixture,
        "Based on the conversation so far, respond in exactly two lines:\n"
        "File=<the primary file that answered the question>\n"
        "Verification=<the function, class, or finding that supports the answer>",
    )
    chunks = _chunk_messages(normalized[:-1], 5)

    assert len(chunks) == 5
    assert chunks[0][-1]["content"].startswith("Tool result (r1)")
    assert chunks[1][-1]["content"].startswith("Tool result (r2)")
    assert chunks[2] == []
    assert chunks[3][-1]["content"].startswith("Tool result (r3)")
    assert chunks[4] == [{"role": "user", "content": "Please summarize the answer."}]


def test_compare_results_emits_bootstrap_diagnosis() -> None:
    baseline = _result(mode="baseline", total_tokens=120, prompt_tokens=100)
    candidate = _result(
        mode="tok-native",
        total_tokens=160,
        prompt_tokens=140,
        tok_overhead_tokens=80,
        total_saved_tokens=40,
    )

    comparison = compare_results(baseline, candidate)

    assert comparison.total_token_delta == 40
    assert comparison.total_token_delta_pct == 33.3
    assert comparison.diagnosis == "lost_on_bootstrap_overhead"
    assert comparison.tok_improved is False


def test_compare_results_emits_response_drift_diagnosis() -> None:
    baseline = _result(mode="baseline", total_tokens=120, prompt_tokens=100)
    candidate = _result(
        mode="tok-native",
        total_tokens=130,
        prompt_tokens=110,
        response_signals={"non_tok_response": 1, "tok_drift_healed": 1},
        invisible_pressure=2,
    )

    comparison = compare_results(baseline, candidate)

    assert comparison.diagnosis == "lost_on_response_drift"


def test_select_preferred_mode_ignores_failing_candidate() -> None:
    baseline = _result(mode="baseline", total_tokens=120, prompt_tokens=100, success=True)
    failing = _result(mode="tok-minimal", total_tokens=50, prompt_tokens=40, success=False)
    winning = _result(
        mode="tok-tool-compatible",
        total_tokens=90,
        prompt_tokens=70,
        success=True,
    )

    preferred = select_preferred_mode(
        baseline,
        [
            compare_results(baseline, failing),
            compare_results(baseline, winning),
        ],
    )

    assert preferred == "tok-tool-compatible"


def test_render_comparison_markdown_includes_success_and_turns() -> None:
    baseline = _result(mode="baseline", total_tokens=120, prompt_tokens=100, success=True)
    candidate = _result(mode="tok-minimal", total_tokens=90, prompt_tokens=70, success=False)
    comparison = compare_results(baseline, candidate)

    markdown = render_comparison_markdown(baseline, [comparison])

    assert "Session turns" in markdown
    assert "| baseline | True |" in markdown
    assert "| tok-minimal | False |" in markdown
    assert "Candidate task success" in markdown
    assert "Directive tokens estimate" in markdown
    assert "State payload tokens estimate" in markdown


def test_load_benchmark_definition_supports_five_turn_variant() -> None:
    definition = load_benchmark_definition("coding-loop-5")

    assert definition.default_turns == 5
    assert definition.fixture_path.name == "claude_coding_loop.jsonl"


def test_load_benchmark_definition_supports_research_variant() -> None:
    definition = load_benchmark_definition("research-loop-5")

    assert definition.default_turns == 5
    assert definition.fixture_path.name == "research_loop.jsonl"
    # Accept both original (compression.py) and related (bridge_memory.py) findings
    assert definition.expected_file_terms == (
        "compression.py",
        "bridge_memory.py",
    )
    assert definition.expected_verification_terms == (
        "compress_history",
        "BridgeMemoryState",
    )
    prompts = _turn_prompts(definition, 5)
    assert "primary file that answered the original question" in prompts[0]
    assert "Related=<the related file or class mentioned during the investigation>" in prompts[3]


def test_summarize_compare_runs_reports_preferred_counts_and_medians() -> None:
    run_one = {
        "baseline": _result(mode="baseline", total_tokens=120, prompt_tokens=100),
        "tok-minimal": _result(
            mode="tok-minimal",
            total_tokens=90,
            prompt_tokens=70,
            success=False,
        ),
        "tok-native": _result(mode="tok-native", total_tokens=110, prompt_tokens=90),
        "tok-tool-compatible": _result(mode="tok-tool-compatible", total_tokens=80, prompt_tokens=60),
    }
    run_two = {
        "baseline": _result(mode="baseline", total_tokens=140, prompt_tokens=115),
        "tok-minimal": _result(
            mode="tok-minimal",
            total_tokens=95,
            prompt_tokens=75,
            success=False,
        ),
        "tok-native": _result(mode="tok-native", total_tokens=100, prompt_tokens=80),
        "tok-tool-compatible": _result(mode="tok-tool-compatible", total_tokens=85, prompt_tokens=65),
    }

    summary = summarize_compare_runs([run_one, run_two])
    markdown = render_stability_markdown("coding-loop-5", "m", summary)

    assert summary["runs"] == 2
    assert summary["preferred_mode_counts"]["tok-tool-compatible"] == 2
    assert summary["mode_summaries"]["tok-tool-compatible"]["median_total_tokens"] == 82
    assert "Preferred Mode Counts" in markdown
    assert "tok-tool-compatible" in markdown


def test_live_benchmark_runner_reports_prompt_and_response_metrics(
    tmp_path,
) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Fix failing tests in src/tok/gateway.py",
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Investigation complete."}],
                    },
                ]
            }
        )
        + "\n"
    )
    definition = BenchmarkDefinition(
        name="coding-loop",
        fixture_path=fixture,
        system_prompt="You are analyzing a coding loop.",
        followup_prompt="File=<the file that was changed>\nVerification=<the result>",
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
    )
    client = _FakeClient("File=gateway.py\nVerification=1 passed in 0.05s")
    runner = LiveBenchmarkRunner(model="gpt-4o-mini", client=client)

    baseline = runner.run(definition, mode="baseline", turns=3)
    minimal = runner.run(definition, mode="tok-minimal", turns=3)
    native = runner.run(definition, mode="tok-native", turns=3)
    tool_compatible = runner.run(definition, mode="tok-tool-compatible", turns=3)

    assert baseline.task_success is True
    assert baseline.compression_metrics["total_saved_tokens"] == 0
    assert baseline.prompt_metrics["tok_overhead_tokens"] == 0
    assert baseline.prompt_metrics["directive_tokens_estimate"] == 0
    assert baseline.prompt_metrics["state_payload_tokens_estimate"] == 0
    assert baseline.turn_count == 3
    assert len(baseline.turns) == 3
    assert "outbound_payload" in baseline.turns[0]

    assert minimal.task_success is True
    assert minimal.diagnostics["tool_compatible_requested"] is True
    assert len(minimal.turns) == 3
    assert minimal.turns[0]["outbound_payload"]["system"]
    assert "state_resend_suppressed_turns" in minimal.diagnostics
    assert "state_resend_delta_turns" in minimal.diagnostics
    assert "state_resend_full_turns" in minimal.diagnostics
    assert "directive_tokens_estimate" in minimal.prompt_metrics
    assert "state_payload_tokens_estimate" in minimal.prompt_metrics

    assert native.task_success is True
    assert native.compression_metrics["input_saved_tokens"] >= 0
    assert native.prompt_metrics["prepared_messages_tokens"] >= 0
    assert native.response_metrics["response_mode"] in {
        "tok-native",
        "tool-compatible",
        "tok",
        "empty",
    }

    assert tool_compatible.task_success is True
    assert tool_compatible.diagnostics["tool_compatible_requested"] is True
    assert tool_compatible.turns[0]["diagnostics"]["request_policy"] == ("natural_first")
    assert tool_compatible.response_metrics["response_mode"] == "tool-compatible"


def test_live_benchmark_runner_rejects_placeholder_success(tmp_path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps({"messages": [{"role": "user", "content": "Fix it"}]}) + "\n")
    definition = BenchmarkDefinition(
        name="coding-loop",
        fixture_path=fixture,
        system_prompt="You are analyzing a coding loop.",
        followup_prompt="File=<the file that was changed>\nVerification=<the result>",
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed",),
    )
    client = _FakeClient(
        "File=<the specific filename from the code change>\n"
        "Verification=<the command run or test outcome that confirmed the fix>"
    )
    runner = LiveBenchmarkRunner(model="gpt-4o-mini", client=client)

    result = runner.run(definition, mode="tok-minimal", turns=1)

    assert result.task_success is False
    assert "placeholder_file_field" in result.notes
    assert "placeholder_verification_field" in result.notes


def test_live_benchmark_runner_accepts_plain_pytest_verification(
    tmp_path,
) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps({"messages": [{"role": "user", "content": "Fix gateway"}]}) + "\n")
    definition = BenchmarkDefinition(
        name="coding-loop",
        fixture_path=fixture,
        system_prompt="You are analyzing a coding loop.",
        followup_prompt="File=<the file that was changed>\nVerification=<the result>",
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "1 passed", "pytest"),
    )
    client = _FakeClient("File=src/tok/gateway.py\nVerification=pytest src/tok/gateway.py")
    runner = LiveBenchmarkRunner(model="gpt-4o-mini", client=client)

    result = runner.run(definition, mode="baseline", turns=1)

    assert result.task_success is True
    assert result.notes == []


def test_live_benchmark_runner_rejects_bare_failed_verification(
    tmp_path,
) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps({"messages": [{"role": "user", "content": "Fix gateway"}]}) + "\n")
    definition = BenchmarkDefinition(
        name="coding-loop",
        fixture_path=fixture,
        system_prompt="You are analyzing a coding loop.",
        followup_prompt="File=<the file that was changed>\nVerification=<the result>",
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "1 passed", "pytest"),
    )
    client = _FakeClient("File=src/tok/gateway.py\nVerification=FAILED")
    runner = LiveBenchmarkRunner(model="gpt-4o-mini", client=client)

    result = runner.run(definition, mode="baseline", turns=1)

    assert result.task_success is False
    assert "unexpected_verification_field" in result.notes
