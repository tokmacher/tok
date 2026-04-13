import json
from types import SimpleNamespace
from typing import Any

from tok.gateway import BridgeSession
from tok.gateway._bridge_runtime_pipeline import BridgePreparedPayload
from tok.gateway._request_policy import default_request_policy
from tok.testing.live_benchmark import (
    BenchmarkDefinition,
    BenchmarkResult,
    LiveBenchmarkRunner,
    ProviderUsageSnapshot,
    _adapt_tool_results_for_openai,
    _chunk_messages,
    _turn_prompts,
    compare_results,
    load_benchmark_definition,
    normalize_fixture_messages,
    normalize_fixture_messages_for_bridge,
    render_comparison_markdown,
    render_stability_markdown,
    select_preferred_mode,
    summarize_compare_runs,
    summarize_compare_triage,
)


class _FakeCompletions:
    def __init__(self, content: Any, prompt_tokens: int, completion_tokens: int) -> None:
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
        content: Any,
        prompt_tokens: int = 100,
        completion_tokens: int = 20,
    ) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content, prompt_tokens, completion_tokens))


class _FakeOpenAIProtocolRetryCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise RuntimeError("Error code: 400 - No tool call found for function call output with call_id call_1")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="File=src/tok/gateway.py\nVerification=pytest"))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=25, total_tokens=145),
        )


class _AlwaysFailCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        del kwargs
        self.calls += 1
        raise RuntimeError("Error code: 500 - internal provider fault")


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
    notes: list[str] | None = None,
    diagnostics_extra: dict[str, Any] | None = None,
    cost_usd: float | None = None,
) -> BenchmarkResult:
    diagnostics = {
        "tool_compatible_requested": mode == "tok-universal",
        "request_messages_before": 3,
        "request_messages_after": 2,
        "session_turns": 3,
        "response_warning_signal_count": sum((response_signals or {}).values()),
    }
    if diagnostics_extra:
        diagnostics.update(diagnostics_extra)
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
            cost_usd=cost_usd,
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
        diagnostics=diagnostics,
        task_success=success,
        matched_success_terms=["gateway.py", "passed"] if success else [],
        request_messages=2,
        turn_count=3,
        turns=[],
        visible_response="ok",
        raw_response="ok",
        notes=notes or [],
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


def test_bridge_fixture_normalization_preserves_tool_result_pairing() -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "x.py"}},
            ],
        },
        {"role": "tool_result", "tool_use_id": "t1", "content": "print('ok')"},
    ]
    normalized = normalize_fixture_messages_for_bridge(messages, "final prompt")
    chunks = _chunk_messages(normalized[:-1], 2)
    merged = [message for chunk in chunks for message in chunk]

    assert normalized[1]["role"] == "user"
    assert normalized[1]["content"][0]["type"] == "tool_result"
    assert merged[0]["role"] == "assistant"
    assert merged[1]["content"][0]["type"] == "tool_result"


def test_bridge_fixture_normalization_downgrades_orphan_tool_results_to_text() -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "n7", "name": "view_file", "input": {"path": "src/tok/gateway.py"}},
            ],
        },
        {"role": "tool_result", "tool_use_id": "n7", "content": "class Gateway:"},
        {"role": "user", "content": "Summary please."},
        # Orphan/replayed result: should not be emitted as structured tool_result for the bridge path.
        {"role": "tool_result", "tool_use_id": "n7", "content": "class Gateway:"},
    ]

    normalized = normalize_fixture_messages_for_bridge(messages, "final prompt")

    assert normalized[1]["role"] == "user"
    assert isinstance(normalized[1]["content"], list)
    assert normalized[1]["content"][0]["type"] == "tool_result"

    assert normalized[3]["role"] == "user"
    assert isinstance(normalized[3]["content"], str)
    assert normalized[3]["content"].startswith("Tool result (n7):")


def test_adapt_tool_results_for_openai_converts_tool_use_and_tool_result_pairing() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking file first."},
                {"type": "tool_use", "id": "call_1", "name": "view_file", "input": {"path": "src/app.py"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "def run() -> None:\n    pass"},
            ],
        },
    ]

    adapted = _adapt_tool_results_for_openai(messages)

    assert adapted[0]["role"] == "assistant"
    assert adapted[0]["content"] == "Checking file first."
    assert adapted[0]["tool_calls"][0]["id"] == "call_1"
    assert adapted[0]["tool_calls"][0]["function"]["name"] == "view_file"
    assert adapted[1] == {"role": "tool", "tool_call_id": "call_1", "content": "def run() -> None:\n    pass"}


def test_adapt_tool_results_for_openai_keeps_mixed_user_content_when_tool_result_present() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_2", "name": "grep_search", "input": {"pattern": "foo"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_2", "content": "src/app.py:12: foo()"},
                {"type": "text", "text": "Thanks, continue."},
            ],
        },
    ]

    adapted = _adapt_tool_results_for_openai(messages)

    assert adapted[0]["role"] == "assistant"
    assert adapted[1] == {"role": "tool", "tool_call_id": "call_2", "content": "src/app.py:12: foo()"}
    assert adapted[2] == {"role": "user", "content": "Thanks, continue."}


def test_adapt_tool_results_for_openai_downgrades_orphan_tool_result_to_text() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_3", "name": "view_file", "input": {"path": "src/app.py"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_3", "content": "ok"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_3", "content": "replayed"}]},
    ]

    adapted = _adapt_tool_results_for_openai(messages)

    assert adapted[1] == {"role": "tool", "tool_call_id": "call_3", "content": "ok"}
    assert adapted[2]["role"] == "user"
    assert adapted[2]["content"] == "Tool result (call_3): replayed"


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


def test_compare_results_detects_message_normalization_asymmetry() -> None:
    baseline = _result(
        mode="baseline",
        total_tokens=120,
        prompt_tokens=100,
        diagnostics_extra={"message_normalization_path": "normalize_fixture_messages"},
    )
    candidate = _result(
        mode="tok-universal",
        total_tokens=110,
        prompt_tokens=90,
        diagnostics_extra={"message_normalization_path": "normalize_fixture_messages_for_bridge"},
    )

    comparison = compare_results(baseline, candidate)

    assert "message_normalization_path_mismatch" in comparison.fairness_diagnostics["asymmetry_flags"]


def test_compare_results_flags_token_savings_without_cost_savings() -> None:
    baseline = _result(mode="baseline", total_tokens=120, prompt_tokens=100, cost_usd=0.25)
    candidate = _result(mode="tok-universal", total_tokens=100, prompt_tokens=80, cost_usd=0.30)

    comparison = compare_results(baseline, candidate)

    assert comparison.total_token_delta < 0
    assert comparison.cost_delta_usd is not None and comparison.cost_delta_usd > 0
    assert comparison.token_savings_without_cost_savings is True


def test_select_preferred_mode_ignores_failing_candidate() -> None:
    baseline = _result(mode="baseline", total_tokens=120, prompt_tokens=100, success=True)
    failing = _result(mode="tok-universal", total_tokens=50, prompt_tokens=40, success=False)
    winning = _result(
        mode="tok-universal",
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

    assert preferred == "tok-universal"


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


def test_load_benchmark_definition_supports_research_current_variant() -> None:
    definition = load_benchmark_definition("research-loop-current")

    assert definition.default_turns == 3
    assert definition.fixture_path.name == "research_loop.jsonl"
    assert "compression/__init__.py" in definition.success_terms
    assert "runtime/memory/bridge_memory.py" in definition.expected_file_terms


def test_summarize_compare_runs_reports_preferred_counts_and_medians() -> None:
    run_one = {
        "baseline": _result(mode="baseline", total_tokens=120, prompt_tokens=100),
        "tok-universal": _result(mode="tok-universal", total_tokens=80, prompt_tokens=60),
    }
    run_two = {
        "baseline": _result(mode="baseline", total_tokens=140, prompt_tokens=115),
        "tok-universal": _result(mode="tok-universal", total_tokens=85, prompt_tokens=65),
    }

    summary = summarize_compare_runs([run_one, run_two])
    markdown = render_stability_markdown("coding-loop-5", "m", summary)

    assert summary["runs"] == 2
    assert summary["preferred_mode_counts"]["tok-universal"] == 2
    assert summary["mode_summaries"]["tok-universal"]["median_total_tokens"] == 82
    assert "Preferred Mode Counts" in markdown
    assert "tok-universal" in markdown


def test_summarize_compare_triage_reports_dual_success_and_failures() -> None:
    baseline = _result(
        mode="baseline",
        total_tokens=120,
        prompt_tokens=100,
        success=True,
        diagnostics_extra={"repo_grounded_task_success": True},
    )
    tok = _result(
        mode="tok-universal",
        total_tokens=90,
        prompt_tokens=70,
        success=False,
        response_signals={"non_tok_response": 1, "tok_drift_healed": 1},
        notes=["response_contract_friction_detected"],
        diagnostics_extra={"repo_grounded_task_success": False, "repo_grounded_failures": ["file_not_found"]},
    )

    summary = summarize_compare_triage([{"baseline": baseline, "tok-universal": tok}])

    assert summary["runs"] == 1
    assert summary["mode_summaries"]["baseline"]["legacy_success_rate"] == 1.0
    assert summary["mode_summaries"]["tok-universal"]["legacy_success_rate"] == 0.0
    assert summary["mode_summaries"]["tok-universal"]["repo_grounded_success_rate"] == 0.0
    reasons = summary["mode_summaries"]["tok-universal"]["top_failure_reasons"]
    assert any(item["reason"] == "repo_grounded:file_not_found" for item in reasons)
    assert summary["mode_summaries"]["tok-universal"]["response_contract_friction_runs"] == 1


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
    universal = runner.run(definition, mode="tok-universal", turns=3)

    assert baseline.task_success is True
    assert baseline.compression_metrics["total_saved_tokens"] == 0
    assert baseline.prompt_metrics["tok_overhead_tokens"] == 0
    assert baseline.prompt_metrics["directive_tokens_estimate"] == 0
    assert baseline.prompt_metrics["state_payload_tokens_estimate"] == 0
    assert baseline.turn_count == 3
    assert len(baseline.turns) == 3
    assert "outbound_payload" in baseline.turns[0]

    assert universal.task_success is True
    assert universal.mode == "tok-universal"
    assert universal.diagnostics["tool_compatible_requested"] is True
    assert len(universal.turns) == 3
    assert universal.turns[0]["outbound_payload"]["system"]
    assert "state_resend_suppressed_turns" in universal.diagnostics
    assert "state_resend_delta_turns" in universal.diagnostics
    assert "state_resend_full_turns" in universal.diagnostics
    assert "legacy_task_success" in universal.diagnostics
    assert "repo_grounded_task_success" in universal.diagnostics
    assert "repo_grounded_failures" in universal.diagnostics
    assert "schema_forensics" in universal.turns[0]["diagnostics"]
    assert "before" in universal.turns[0]["diagnostics"]["schema_forensics"]
    assert "after" in universal.turns[0]["diagnostics"]["schema_forensics"]
    assert "directive_tokens_estimate" in universal.prompt_metrics
    assert "state_payload_tokens_estimate" in universal.prompt_metrics
    assert universal.turns[0]["diagnostics"]["request_policy"] == default_request_policy()
    assert universal.turns[0]["diagnostics"]["execution_path"] == "claude-bridge"
    assert universal.turns[0]["diagnostics"]["bridge_preflight_applied"] == 1


def test_live_benchmark_tok_universal_uses_shared_bridge_pipeline(tmp_path, monkeypatch) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Fix failing tests in src/tok/gateway.py"},
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
    runner = LiveBenchmarkRunner(
        model="gpt-4o-mini",
        client=_FakeClient("File=gateway.py\nVerification=1 passed in 0.05s"),
    )

    captured_paths: list[str] = []

    def _fake_prepare_bridge_payload(*, session, body, headers, path, **kwargs):
        del session, headers, kwargs
        captured_paths.append(path)
        return (
            BridgePreparedPayload(
                body={
                    "model": body.get("model", ""),
                    "system": body.get("system", ""),
                    "messages": list(body.get("messages", [])),
                },
                behavior_signals={},
                request_policy=default_request_policy(),
                request_tool_compatible=True,
                compressed=False,
                saved_toks=0,
                tool_breakdown={},
                prompt_metrics={
                    "baseline_prompt_tokens": 0,
                    "prepared_prompt_tokens": 0,
                    "saved_prompt_tokens": 0,
                    "hot_hint_tokens_added": 0,
                    "reacquisition_tokens_avoided_estimate": 0,
                },
                retry_forbidden=False,
            ),
            None,
        )

    monkeypatch.setattr("tok.testing.live_benchmark.prepare_bridge_payload", _fake_prepare_bridge_payload)

    result = runner.run(definition, mode="tok-universal", turns=2)

    assert captured_paths == ["v1/messages", "v1/messages"]
    assert result.turns[0]["diagnostics"]["execution_path"] == "claude-bridge"
    assert result.turns[0]["diagnostics"]["bridge_preflight_applied"] == 1


def test_live_benchmark_tok_universal_uses_runtime_response_contract_hook(tmp_path, monkeypatch) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Confirm the final answer format"},
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
    runner = LiveBenchmarkRunner(
        model="gpt-4o-mini",
        client=_FakeClient("File=src/tok/gateway.py\nVerification=passed"),
    )

    def _fake_prepare_bridge_payload(*, session, body, headers, path, **kwargs):
        del session, headers, kwargs
        return (
            BridgePreparedPayload(
                body={
                    "model": body.get("model", ""),
                    "system": body.get("system", ""),
                    "messages": list(body.get("messages", [])),
                },
                behavior_signals={},
                request_policy=default_request_policy(),
                request_tool_compatible=False,
                compressed=False,
                saved_toks=0,
                tool_breakdown={},
                prompt_metrics={
                    "baseline_prompt_tokens": 0,
                    "prepared_prompt_tokens": 0,
                    "saved_prompt_tokens": 0,
                    "hot_hint_tokens_added": 0,
                    "reacquisition_tokens_avoided_estimate": 0,
                },
                retry_forbidden=False,
            ),
            None,
        )

    monkeypatch.setattr("tok.testing.live_benchmark.prepare_bridge_payload", _fake_prepare_bridge_payload)

    import tok.runtime._runtime_orchestration as runtime_orchestration

    captured_tool_compatible: list[bool] = []
    original_contract = runtime_orchestration.response_contract_for_mode

    def _capture_contract(
        text: str,
        tool_compatible: bool = False,
        _family: str = "",
        _model: str = "",
        session=None,
    ):
        captured_tool_compatible.append(tool_compatible)
        return original_contract(
            text,
            tool_compatible=tool_compatible,
            _family=_family,
            _model=_model,
            session=session,
        )

    monkeypatch.setattr("tok.runtime._runtime_orchestration.response_contract_for_mode", _capture_contract)

    result = runner.run(definition, mode="tok-universal", turns=1)

    assert result.turns[0]["diagnostics"]["execution_path"] == "claude-bridge"
    assert result.turns[0]["diagnostics"]["tool_compatible_requested"] is False
    assert captured_tool_compatible
    assert set(captured_tool_compatible) == {False}


def test_live_benchmark_tok_universal_flattens_provider_payload_for_non_anthropic_models(
    tmp_path,
) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Find the entry point"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "s1",
                                "name": "grep_search",
                                "input": {"search_path": "src", "query": "compress_history"},
                            }
                        ],
                    },
                    {
                        "role": "tool_result",
                        "tool_use_id": "s1",
                        "content": "src/tok/compression.py:305: def compress_history(",
                    },
                ]
            }
        )
        + "\n"
    )
    definition = BenchmarkDefinition(
        name="research-loop",
        fixture_path=fixture,
        system_prompt="You are analyzing a research loop.",
        followup_prompt="File=<the primary file that answered the question>\nVerification=<the finding>",
        success_terms=("compression.py", "compress_history"),
        min_success_terms=2,
        expected_file_terms=("compression.py",),
        expected_verification_terms=("compress_history",),
    )
    runner = LiveBenchmarkRunner(
        model="deepseek/deepseek-v3.2",
        client=_FakeClient("File=src/tok/compression.py\nVerification=compress_history"),
    )

    captured_messages: list[dict[str, Any]] = []

    def _capture_create(**kwargs):
        captured_messages[:] = list(kwargs["messages"])
        return runner.client.chat.completions._response  # type: ignore[attr-defined]

    runner.client.chat.completions.create = _capture_create  # type: ignore[assignment]

    result = runner.run(definition, mode="tok-universal", turns=1)

    assert result.turns[0]["diagnostics"]["execution_path"] == "claude-bridge"
    assert result.turns[0]["diagnostics"]["bridge_preflight_applied"] == 1
    assert result.turns[0]["diagnostics"]["schema_forensics"]["provider_after"]["tool_use_blocks"] == 0
    assert result.turns[0]["diagnostics"]["schema_forensics"]["provider_after"]["tool_result_blocks"] == 0
    assert any(
        "non_anthropic_tool_block_payload" in warning
        for warning in result.turns[0]["diagnostics"]["schema_forensics"]["compatibility_warnings"]
    )
    assert captured_messages
    assert all(isinstance(message.get("content"), str) for message in captured_messages)
    assert any("Tool result" in message.get("content", "") for message in captured_messages)


def test_live_benchmark_tok_universal_plain_structured_answer_avoids_contract_friction(
    tmp_path,
) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps({"messages": [{"role": "user", "content": "Inspect gateway"}]}) + "\n")
    definition = BenchmarkDefinition(
        name="coding-loop",
        fixture_path=fixture,
        system_prompt="You are analyzing a coding loop.",
        followup_prompt="File=<the file that was changed>\nVerification=<the result>",
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "pytest"),
    )
    runner = LiveBenchmarkRunner(
        model="deepseek/deepseek-v3.2",
        client=_FakeClient("File=src/tok/gateway.py\nVerification=pytest tests/unit/test_gateway.py"),
    )

    result = runner.run(definition, mode="tok-universal", turns=1)

    assert result.task_success is True
    assert result.diagnostics["response_warning_signal_count"] == 0
    assert "response_contract_friction_detected" not in result.notes


def test_live_benchmark_tok_universal_handles_list_content_without_strip_failures(tmp_path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps({"messages": [{"role": "user", "content": "Inspect gateway"}]}) + "\n")
    definition = BenchmarkDefinition(
        name="coding-loop",
        fixture_path=fixture,
        system_prompt="You are analyzing a coding loop.",
        followup_prompt="File=<the file that was changed>\nVerification=<the result>",
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "pytest"),
    )
    runner = LiveBenchmarkRunner(
        model="deepseek/deepseek-v3.2",
        client=_FakeClient(
            [
                {"type": "text", "text": "File=src/tok/gateway.py\nVerification=pytest passed"},
            ]
        ),
    )

    result = runner.run(definition, mode="tok-universal", turns=1)

    assert isinstance(result.raw_response, str)
    assert "File=src/tok/gateway.py" in result.raw_response


def test_live_benchmark_retries_once_on_tool_protocol_pairing_400(tmp_path) -> None:
    completions = _FakeOpenAIProtocolRetryCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    runner = LiveBenchmarkRunner(model="openai/gpt-4.1", client=client, provider="openrouter")
    bridge_session = BridgeSession(memory_dir=tmp_path / "bridge_mem")
    session = bridge_session.runtime_session
    conversation = [
        {"role": "user", "content": "Inspect gateway."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "class Gateway:\n    pass"}],
        },
        {
            "role": "user",
            "content": "File=<the file that was changed>\nVerification=<the result>",
        },
    ]
    step = runner.run_conversation_step(
        conversation=conversation,
        system_prompt="You are analyzing a coding loop.",
        mode="tok-universal",
        session=session,
        bridge_session=bridge_session,
        allowed_tools=("view_file",),
    )

    assert len(completions.calls) == 2
    first_call = completions.calls[0]
    second_call = completions.calls[1]
    assert "tools" in first_call
    assert "tools" not in second_call
    assert all(isinstance(message.get("content"), str) for message in second_call["messages"])
    assert step.diagnostics["tool_protocol_retry_count"] == 1
    assert step.diagnostics["tool_protocol_retry_success"] == 1
    assert step.diagnostics["tool_protocol_retry_reason"] == "missing_tool_call_for_call_id"
    assert step.diagnostics["tool_protocol_retry_mode"] == "safe_provider_text_history"


def test_live_benchmark_does_not_retry_on_unrelated_errors(tmp_path) -> None:
    completions = _AlwaysFailCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    runner = LiveBenchmarkRunner(model="openai/gpt-4.1", client=client, provider="openrouter")
    bridge_session = BridgeSession(memory_dir=tmp_path / "bridge_mem_error")
    session = bridge_session.runtime_session
    conversation = [{"role": "user", "content": "Inspect gateway"}]

    import pytest

    with pytest.raises(RuntimeError, match="internal provider fault"):
        runner.run_conversation_step(
            conversation=conversation,
            system_prompt="You are analyzing a coding loop.",
            mode="tok-universal",
            session=session,
            bridge_session=bridge_session,
            allowed_tools=("view_file",),
        )
    assert completions.calls == 1


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


def test_research_dual_scoring_can_pass_legacy_and_fail_repo_grounded(tmp_path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps({"messages": [{"role": "user", "content": "Investigate compression"}]}) + "\n")
    definition = BenchmarkDefinition(
        name="research-loop",
        fixture_path=fixture,
        system_prompt="You are analyzing a research loop.",
        followup_prompt="File=<file>\nVerification=<verification>",
        success_terms=("compression.py", "compress_history"),
        min_success_terms=2,
        expected_file_terms=("compression.py",),
        expected_verification_terms=("compress_history",),
    )
    client = _FakeClient("File=src/compression.py\nVerification=compress_history in compression.py")
    runner = LiveBenchmarkRunner(model="gpt-4o-mini", client=client)

    result = runner.run(definition, mode="baseline", turns=1)

    assert result.task_success is True
    assert result.diagnostics["repo_grounded_task_success"] is False
    assert "file_not_found" in result.diagnostics["repo_grounded_failures"]


def test_research_dual_scoring_can_fail_legacy_and_pass_repo_grounded(tmp_path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps({"messages": [{"role": "user", "content": "Investigate compression"}]}) + "\n")
    definition = BenchmarkDefinition(
        name="research-loop-current",
        fixture_path=fixture,
        system_prompt="You are analyzing a research loop.",
        followup_prompt="File=<file>\nVerification=<verification>",
        success_terms=("compression.py", "compress_history"),
        min_success_terms=2,
        expected_file_terms=("compression.py",),
        expected_verification_terms=("compress_history",),
    )
    client = _FakeClient("File=src/tok/compression/__init__.py\nVerification=compress_history")
    runner = LiveBenchmarkRunner(model="gpt-4o-mini", client=client)

    result = runner.run(definition, mode="baseline", turns=1)

    assert result.task_success is False
    assert result.diagnostics["repo_grounded_task_success"] is True
    assert result.diagnostics["repo_grounded_failures"] == []


def test_live_benchmark_non_structured_success_without_labeled_fields(tmp_path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps({"messages": [{"role": "user", "content": "Is there drift?"}]}) + "\n")
    definition = BenchmarkDefinition(
        name="non-structured-loop",
        fixture_path=fixture,
        system_prompt="Classify drift.",
        followup_prompt="Has the grammar drifted?",
        success_terms=("yes", "no"),
        min_success_terms=1,
        require_file_field=False,
        require_verification_field=False,
    )
    client = _FakeClient("No, the grammar has not drifted.")
    runner = LiveBenchmarkRunner(model="gpt-4o-mini", client=client)

    result = runner.run(definition, mode="baseline", turns=1)

    assert result.task_success is True
    assert "missing_file_field" not in result.notes
    assert "missing_verification_field" not in result.notes


def test_live_benchmark_structured_scoring_still_requires_labeled_fields(tmp_path) -> None:
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
        expected_verification_terms=("passed",),
    )
    client = _FakeClient("The fix passed.")
    runner = LiveBenchmarkRunner(model="gpt-4o-mini", client=client)

    result = runner.run(definition, mode="baseline", turns=1)

    assert result.task_success is False
    assert "missing_file_field" in result.notes
    assert "missing_verification_field" in result.notes


def test_live_benchmark_grammar_drift_accepts_plain_yes_no_response() -> None:
    definition = load_benchmark_definition("grammar_drift")
    client = _FakeClient("No, grammar has not drifted.")
    runner = LiveBenchmarkRunner(model="gpt-4o-mini", client=client)

    result = runner.run(definition, mode="baseline", turns=1)

    assert result.task_success is True
    assert "missing_file_field" not in result.notes
    assert "missing_verification_field" not in result.notes


def test_live_benchmark_non_structured_scoring_still_requires_success_terms(tmp_path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(json.dumps({"messages": [{"role": "user", "content": "Is there drift?"}]}) + "\n")
    definition = BenchmarkDefinition(
        name="non-structured-loop",
        fixture_path=fixture,
        system_prompt="Classify drift.",
        followup_prompt="Has the grammar drifted?",
        success_terms=("yes", "no"),
        min_success_terms=1,
        require_file_field=False,
        require_verification_field=False,
    )
    client = _FakeClient("Maybe.")
    runner = LiveBenchmarkRunner(model="gpt-4o-mini", client=client)

    result = runner.run(definition, mode="baseline", turns=1)

    assert result.task_success is False
    assert "response_missing_success_terms" in result.notes


def test_live_benchmark_grammar_drift_rejects_ambiguous_unknown_response() -> None:
    definition = load_benchmark_definition("grammar_drift")
    client = _FakeClient("Unknown from this excerpt.")
    runner = LiveBenchmarkRunner(model="gpt-4o-mini", client=client)

    result = runner.run(definition, mode="baseline", turns=1)

    assert result.task_success is False
    assert "response_missing_success_terms" in result.notes


def test_live_benchmark_jit_loop_tok_universal_uses_textual_success_terms() -> None:
    definition = load_benchmark_definition("jit-loop")
    client = _FakeClient("I found parse_error in src/tok/cli.py and pytest src/tok/cli.py passed.")
    runner = LiveBenchmarkRunner(model="deepseek/deepseek-v3.2", client=client)

    result = runner.run(definition, mode="tok-universal", turns=1)

    assert result.task_success is True
    assert "missing_file_field" not in result.notes
    assert "missing_verification_field" not in result.notes
