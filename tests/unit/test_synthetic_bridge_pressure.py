from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import pytest

from tok.compression._history_pipeline import compress_tool_results_impl
from tok.exceptions import TokSafetyError
from tok.gateway import BridgeSession
from tok.gateway._bridge_runtime_pipeline import prepare_bridge_payload
from tok.runtime.core import RuntimeSession, UniversalTokRuntime
from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context
from tok.runtime.pipeline._tool_repeat_detection import _make_cache_key
from tok.runtime.pipeline.request_validation import (
    canonicalize_anthropic_bridge_body,
    validate_anthropic_outgoing_bridge_body,
)
from tok.runtime.repeat_targets import evidence_identity_key, normalize_path_target
from tok.runtime.types import NormalizedToolEvent, RuntimeRequest

RISK_SIGNALS = {
    "answer_ready_exact_search_evidence_history_preserved",
    "broad_audit_history_skipped",
    "broad_audit_system_additions_skipped",
    "broad_audit_tool_result_compression_skipped",
    "plan_finalization_history_skipped",
    "plan_finalization_tool_result_compression_skipped",
    "tok_bridge_assistant_block_order_normalized",
    "tok_bridge_canonicalized",
    "tok_bridge_pairing_degraded_to_provider_safe",
    "tok_bridge_provider_sensitive_blocked_local",
    "tok_bridge_provider_sensitive_degraded_to_provider_safe",
    "tok_history_compression_skipped",
    "tok_skip_broad_audit",
}

UGLY_PATH_SCENARIOS = (
    "high_fanout_tool_burst",
    "repeated_evidence_loop",
    "final_answer_after_compression",
    "malformed_tool_history",
    "streaming_path_damage",
    "provider_sensitive_shape",
    "baseline_degradation",
    "long_session_retention",
)


@dataclass(frozen=True)
class PressureMetrics:
    baseline_prompt_tokens: int
    prepared_prompt_tokens: int
    tok_overhead_tokens: int
    retained_tool_result_bytes: int
    behavior_signals: dict[str, int]
    outgoing_failures: list[str]

    @property
    def has_visible_risk_signal(self) -> bool:
        return any(self.behavior_signals.get(signal, 0) for signal in RISK_SIGNALS)


def _file_text(path: str, *, lines: int = 80) -> str:
    return "\n".join(f"{path}:{index}: def marker_{index}(): return {index}" for index in range(lines))


def _tool_use(tool_id: str, name: str, **input_kw: Any) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": name,
        "input": input_kw,
    }


def _tool_result(tool_id: str, content: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": content,
    }


def _parallel_file_read_messages(*, count: int = 16, lines: int = 80) -> list[dict[str, Any]]:
    tool_uses = [
        _tool_use(f"toolu_read_{index}", "read_file", path=f"src/tok/pressure_{index}.py") for index in range(count)
    ]
    tool_results = [
        _tool_result(
            f"toolu_read_{index}",
            _file_text(f"src/tok/pressure_{index}.py", lines=lines),
        )
        for index in range(count)
    ]
    return [
        {"role": "user", "content": [{"type": "text", "text": "Audit these files for bridge pressure."}]},
        {"role": "assistant", "content": tool_uses},
        {"role": "user", "content": tool_results},
    ]


def _varied_parallel_file_read_messages(*, count: int = 18) -> list[dict[str, Any]]:
    tool_uses = [
        _tool_use(f"toolu_varied_{index}", "read_file", path=f"src/tok/varied_{index}.py") for index in range(count)
    ]
    tool_results = [
        _tool_result(
            f"toolu_varied_{index}",
            _file_text(f"src/tok/varied_{index}.py", lines=12 + ((index % 5) * 42)),
        )
        for index in range(count)
    ]
    return [
        {"role": "user", "content": [{"type": "text", "text": "Sweep these files for subtle bugs."}]},
        {"role": "assistant", "content": tool_uses},
        {"role": "user", "content": tool_results},
    ]


def _claude_code_bug_sweep_messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "Find a subtle bridge usage spike bug."}]},
        {
            "role": "assistant",
            "content": [
                _tool_use("toolu_sweep_read_1", "read_file", path="src/tok/gateway/_app_factory.py"),
                _tool_use("toolu_sweep_read_2", "read_file", path="src/tok/runtime/_request_preparation.py"),
                _tool_use("toolu_sweep_search_1", "grep_search", query="fail_open", path="src/tok"),
                _tool_use("toolu_sweep_search_2", "grep_search", query="stable_result", path="src/tok"),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result(
                    "toolu_sweep_read_1",
                    _file_text("src/tok/gateway/_app_factory.py", lines=160),
                ),
                _tool_result(
                    "toolu_sweep_read_2",
                    _file_text("src/tok/runtime/_request_preparation.py", lines=190),
                ),
                _tool_result(
                    "toolu_sweep_search_1",
                    "\n".join(f"src/tok/gateway/file_{i}.py:{i}: fail_open branch" for i in range(80)),
                ),
                _tool_result(
                    "toolu_sweep_search_2",
                    "\n".join(f"src/tok/runtime/file_{i}.py:{i}: @stable_result marker" for i in range(80)),
                ),
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "I found likely pressure paths."}]},
        {"role": "user", "content": [{"type": "text", "text": "Now inspect the retry and streaming cleanup paths."}]},
        {
            "role": "assistant",
            "content": [
                _tool_use("toolu_sweep_read_3", "read_file", path="src/tok/gateway/_bridge_streaming.py"),
                _tool_use("toolu_sweep_bash_1", "bash", command="uv run pytest tests/unit/test_gateway.py -q"),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result(
                    "toolu_sweep_read_3",
                    _file_text("src/tok/gateway/_bridge_streaming.py", lines=180),
                ),
                _tool_result(
                    "toolu_sweep_bash_1",
                    "\n".join(f"tests/unit/test_gateway.py::test_{i} PASSED" for i in range(120)),
                ),
            ],
        },
    ]


def _final_answer_after_evidence_messages() -> list[dict[str, Any]]:
    messages = _claude_code_bug_sweep_messages()
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Ready to summarize the finding."}],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Based on the conversation so far, respond in exactly two lines:\nFile=<file>\nVerification=<evidence>",
                }
            ],
        }
    )
    return messages


def _long_alternating_history_messages(*, turns: int = 32) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index in range(turns):
        tool_id = f"toolu_long_{index}"
        tool_name = ("read_file", "grep_search", "bash", "find")[index % 4]
        if tool_name == "read_file":
            tool = _tool_use(tool_id, tool_name, path=f"src/tok/long_{index}.py")
            result = _file_text(f"src/tok/long_{index}.py", lines=70 + (index % 3) * 30)
        elif tool_name == "grep_search":
            tool = _tool_use(tool_id, tool_name, query="compression", path="src/tok")
            result = "\n".join(f"src/tok/long_{index}_{row}.py:{row}: compression" for row in range(90))
        elif tool_name == "bash":
            tool = _tool_use(tool_id, tool_name, command="uv run pytest -q")
            result = "\n".join(f"tests/unit/test_long.py::test_{row} PASSED" for row in range(80))
        else:
            tool = _tool_use(tool_id, tool_name, path="src/tok", pattern="*.py")
            result = "\n".join(f"src/tok/long_{index}_{row}.py" for row in range(120))
        messages.extend(
            [
                {"role": "user", "content": [{"type": "text", "text": f"Continue bridge audit sweep {index}."}]},
                {"role": "assistant", "content": [tool]},
                {"role": "user", "content": [_tool_result(tool_id, result)]},
            ]
        )
    return messages


def _reordered_tool_result_messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "Read and search before deciding."}]},
        {
            "role": "assistant",
            "content": [
                _tool_use("toolu_pair_1", "read_file", path="src/tok/a.py"),
                _tool_use("toolu_pair_2", "grep_search", query="bridge", path="src/tok"),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result("toolu_pair_2", "src/tok/a.py:1: bridge"),
                {"type": "text", "text": "interleaved note that should be split away"},
                _tool_result("toolu_pair_1", _file_text("src/tok/a.py", lines=40)),
            ],
        },
    ]


def _thinking_mutation_messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "Use thinking then inspect the file."}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private chain", "signature": "sig_1"},
                _tool_use("toolu_thinking_1", "read_file", path="src/tok/thinking.py"),
            ],
        },
        {
            "role": "user",
            "content": [_tool_result("toolu_thinking_1", _file_text("src/tok/thinking.py", lines=90))],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "mutated private chain", "signature": "sig_1"},
                {"type": "text", "text": "Continuing."},
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "Finish the audit."}]},
    ]


def _runtime_request(messages: list[dict[str, Any]], *, system: str | None = None) -> RuntimeRequest:
    return RuntimeRequest(
        model="claude-sonnet-4",
        messages=messages,
        system=system,
        adapter_kind="claude-bridge",
        tool_compatible=True,
        request_policy="natural_first",
        request_has_tools=True,
    )


def _tool_result_bytes(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                total += len(str(block.get("content", "")).encode())
    return total


def _prepare_runtime_pressure(
    messages: list[dict[str, Any]],
    tmp_path,
    *,
    session: RuntimeSession | None = None,
    system: str | None = None,
) -> tuple[PressureMetrics, RuntimeSession]:
    active_session = session or RuntimeSession(memory_dir=tmp_path / ".tok")
    runtime = UniversalTokRuntime()
    prepared = runtime.prepare_request(_runtime_request(messages, system=system), active_session)
    metrics = PressureMetrics(
        baseline_prompt_tokens=prepared.baseline_prompt_tokens,
        prepared_prompt_tokens=prepared.prepared_prompt_tokens,
        tok_overhead_tokens=max(0, prepared.prepared_prompt_tokens - prepared.baseline_prompt_tokens),
        retained_tool_result_bytes=_tool_result_bytes(prepared.body["messages"]),
        behavior_signals=dict(prepared.behavior_signals),
        outgoing_failures=validate_anthropic_outgoing_bridge_body(
            {
                "model": "claude-sonnet-4",
                "max_tokens": 8192,
                "messages": prepared.body["messages"],
                "stream": False,
            }
        ),
    )
    return metrics, active_session


def _prepare_bridge_pressure(messages: list[dict[str, Any]], tmp_path) -> PressureMetrics:
    session = BridgeSession(memory_dir=tmp_path / ".tok", fail_open=True)
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": messages,
        "stream": False,
    }
    payload, response = prepare_bridge_payload(
        session=session,
        body=body,
        headers={"x-api-key": "test"},
        path="v1/messages",
        tok_tool_header="1",
    )
    assert response is None
    return PressureMetrics(
        baseline_prompt_tokens=int(payload.prompt_metrics.get("baseline_prompt_tokens", 0)),
        prepared_prompt_tokens=int(payload.prompt_metrics.get("prepared_prompt_tokens", 0)),
        tok_overhead_tokens=max(
            0,
            int(payload.prompt_metrics.get("prepared_prompt_tokens", 0))
            - int(payload.prompt_metrics.get("baseline_prompt_tokens", 0)),
        ),
        retained_tool_result_bytes=_tool_result_bytes(payload.body["messages"]),
        behavior_signals=dict(payload.behavior_signals),
        outgoing_failures=validate_anthropic_outgoing_bridge_body(payload.body),
    )


def _assert_bounded_or_signaled(metrics: PressureMetrics, *, multiplier: float = 1.10, slack: int = 128) -> None:
    assert metrics.baseline_prompt_tokens > 0
    bounded = metrics.prepared_prompt_tokens <= int(metrics.baseline_prompt_tokens * multiplier) + slack
    assert bounded or metrics.has_visible_risk_signal, metrics


def _provider_sensitive_large_tool_batch_messages() -> list[dict[str, Any]]:
    tool_uses = [_tool_use(f"toolu_batch_{index + 1}", "read_file", path=f"file_{index + 1}.py") for index in range(18)]
    assistant_content = [
        *tool_uses[:9],
        {"type": "text", "text": "Collecting evidence."},
        *tool_uses[9:],
    ]
    tool_results = [_tool_result(f"toolu_batch_{index + 1}", f"result {index + 1}") for index in range(18)]
    return [
        {"role": "user", "content": [{"type": "text", "text": "Inspect concurrency path."}]},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": tool_results},
    ]


def _red_team_high_fanout_messages(*, batches: int = 4, tools_per_batch: int = 12) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Red-team Tok: push high fan-out tool batches, repeated evidence, and interleaved notes.",
                }
            ],
        }
    ]
    for batch in range(batches):
        tool_uses: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for index in range(tools_per_batch):
            tool_id = f"toolu_red_{batch}_{index}"
            path = f"src/tok/red_team/{batch}_{index}.py"
            tool_uses.append(_tool_use(tool_id, "read_file", path=path))
            repeated_payload = "\n".join(
                [
                    f"{path}:{line}: repeated pressure marker {line % 7}"
                    for line in range(90 + ((batch + index) % 4) * 35)
                ]
            )
            tool_results.append(_tool_result(tool_id, repeated_payload))
        messages.extend(
            [
                {"role": "assistant", "content": tool_uses},
                {"role": "user", "content": tool_results},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"Batch {batch} evidence retained; continue pressure."}],
                },
                {"role": "user", "content": [{"type": "text", "text": f"Escalate batch {batch + 1}."}]},
            ]
        )
    return messages


def _malformed_tool_result_flood_messages(*, count: int = 20) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "Here is an invalid flood from a broken client."}]},
        {
            "role": "user",
            "content": [
                _tool_result(f"unknown_toolu_{index}", f"orphaned tool result payload {index}")
                for index in range(count)
            ],
        },
    ]


def _near_neighbor_retention_messages(*, turns: int = 14) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "Seed the oldest anchor with exact evidence."}]},
        {"role": "assistant", "content": [_tool_use("toolu_anchor_oldest", "read_file", path="src/tok/gateway.py")]},
        {
            "role": "user",
            "content": [_tool_result("toolu_anchor_oldest", "src/tok/gateway.py:238: async def health()")],
        },
    ]
    for index in range(turns):
        path = f"src/tok/near_neighbor_{index}.py"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"Retaining anchor while checking {path}."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Differentiate this from the gateway health anchor {index}."}
                    ],
                },
                {"role": "assistant", "content": [_tool_use(f"toolu_neighbor_{index}", "read_file", path=path)]},
                {"role": "user", "content": [_tool_result(f"toolu_neighbor_{index}", _file_text(path, lines=35))]},
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Return the oldest anchor only, not a near neighbor. File=src/tok/gateway.py Verification=health",
                }
            ],
        }
    )
    return messages


def test_large_parallel_reads_do_not_expand_prepared_prompt_unboundedly(tmp_path) -> None:
    messages = _parallel_file_read_messages(count=16, lines=90)

    metrics, _session = _prepare_runtime_pressure(messages, tmp_path)

    _assert_bounded_or_signaled(metrics, multiplier=1.05, slack=64)
    assert metrics.retained_tool_result_bytes == _tool_result_bytes(messages)
    assert metrics.behavior_signals.get("broad_audit_tool_result_compression_skipped") == 1
    assert metrics.behavior_signals.get("broad_audit_history_skipped") == 1


def test_ugly_path_unit_matrix_declares_all_supported_scenarios() -> None:
    assert UGLY_PATH_SCENARIOS == (
        "high_fanout_tool_burst",
        "repeated_evidence_loop",
        "final_answer_after_compression",
        "malformed_tool_history",
        "streaming_path_damage",
        "provider_sensitive_shape",
        "baseline_degradation",
        "long_session_retention",
    )


@pytest.mark.parametrize(
    ("scenario_id", "messages", "multiplier", "slack"),
    [
        ("high_fanout_tool_burst", _red_team_high_fanout_messages(batches=3, tools_per_batch=10), 1.18, 384),
        ("repeated_evidence_loop", _long_alternating_history_messages(turns=10), 1.14, 256),
        ("final_answer_after_compression", _final_answer_after_evidence_messages(), 1.15, 180),
        ("provider_sensitive_shape", _provider_sensitive_large_tool_batch_messages(), 1.18, 384),
        ("long_session_retention", _near_neighbor_retention_messages(turns=12), 1.16, 320),
    ],
)
def test_ugly_path_unit_matrix_transport_behavior_and_economics(
    tmp_path,
    scenario_id: str,
    messages: list[dict[str, Any]],
    multiplier: float,
    slack: int,
) -> None:
    del scenario_id
    metrics = _prepare_bridge_pressure(messages, tmp_path)

    assert metrics.outgoing_failures == []
    _assert_bounded_or_signaled(metrics, multiplier=multiplier, slack=slack)
    assert metrics.retained_tool_result_bytes <= _tool_result_bytes(messages) or metrics.has_visible_risk_signal


def test_ugly_path_unit_matrix_malformed_tool_history_blocks_before_send(tmp_path) -> None:
    session = BridgeSession(memory_dir=tmp_path / ".tok", fail_open=False)
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": _malformed_tool_result_flood_messages(count=18),
        "stream": False,
    }

    payload, response = prepare_bridge_payload(
        session=session,
        body=body,
        headers={"x-api-key": "test"},
        path="v1/messages",
        tok_tool_header="1",
    )

    assert response is not None
    assert payload.retry_forbidden is True
    assert payload.behavior_signals.get("tok_bridge_invalid_tool_history_blocked") == 1
    assert payload.behavior_signals.get("tok_fallback_activated") == 1


def test_ugly_path_unit_matrix_baseline_degradation_is_visible(tmp_path) -> None:
    session = BridgeSession(memory_dir=tmp_path / ".tok", fail_open=True)
    session.runtime_session._baseline_only = True

    payload, response = prepare_bridge_payload(
        session=session,
        body={
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": _parallel_file_read_messages(count=3, lines=20),
            "stream": False,
        },
        headers={"x-api-key": "test"},
        path="v1/messages",
        tok_tool_header="1",
    )

    assert response is None
    assert validate_anthropic_outgoing_bridge_body(payload.body) == []
    assert payload.behavior_signals.get("tok_fallback_activated") == 1
    assert payload.prompt_metrics["saved_prompt_tokens"] == 0


def test_varied_parallel_reads_are_bounded_or_visibly_degraded(tmp_path) -> None:
    metrics, _session = _prepare_runtime_pressure(_varied_parallel_file_read_messages(count=18), tmp_path)

    _assert_bounded_or_signaled(metrics, multiplier=1.08, slack=96)
    assert metrics.outgoing_failures == []
    assert metrics.behavior_signals.get("broad_audit_tool_result_compression_skipped") == 1


def test_red_team_high_fanout_history_stays_provider_valid_and_spike_visible(tmp_path) -> None:
    messages = _red_team_high_fanout_messages(batches=4, tools_per_batch=12)

    metrics = _prepare_bridge_pressure(messages, tmp_path)

    assert metrics.outgoing_failures == []
    assert metrics.baseline_prompt_tokens > 12000
    assert (
        metrics.prepared_prompt_tokens <= int(metrics.baseline_prompt_tokens * 1.18) + 384
        or metrics.has_visible_risk_signal
    )
    assert metrics.retained_tool_result_bytes < _tool_result_bytes(messages) or metrics.has_visible_risk_signal


def test_red_team_repeated_runtime_preparation_does_not_accumulate_hidden_overhead(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    running_messages: list[dict[str, Any]] = []
    observed: list[PressureMetrics] = []
    for batch in range(10):
        tool_id = f"toolu_runtime_spike_{batch}"
        path = f"src/tok/runtime_spike_{batch}.py"
        running_messages.extend(
            [
                {"role": "user", "content": [{"type": "text", "text": f"Pressure turn {batch}: inspect {path}."}]},
                {"role": "assistant", "content": [_tool_use(tool_id, "read_file", path=path)]},
                {"role": "user", "content": [_tool_result(tool_id, _file_text(path, lines=110 + batch * 8))]},
            ]
        )
        metrics, session = _prepare_runtime_pressure(running_messages, tmp_path, session=session)
        observed.append(metrics)

    assert all(metrics.outgoing_failures == [] for metrics in observed)
    peak_overhead = max(metrics.tok_overhead_tokens for metrics in observed)
    final = observed[-1]
    assert peak_overhead <= 768 or final.has_visible_risk_signal
    assert (
        final.prepared_prompt_tokens <= int(final.baseline_prompt_tokens * 1.15) + 256 or final.has_visible_risk_signal
    )


def test_red_team_orphaned_tool_result_flood_blocks_locally_before_upstream(tmp_path) -> None:
    session = BridgeSession(memory_dir=tmp_path / ".tok", fail_open=False)
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": _malformed_tool_result_flood_messages(count=24),
        "stream": False,
    }

    payload, response = prepare_bridge_payload(
        session=session,
        body=body,
        headers={"x-api-key": "test"},
        path="v1/messages",
        tok_tool_header="1",
    )

    assert response is not None
    assert payload.retry_forbidden is True
    assert payload.behavior_signals.get("tok_bridge_preflight_failed_local") == 1
    assert payload.behavior_signals.get("tok_bridge_invalid_tool_history_blocked") == 1
    assert payload.behavior_signals.get("tok_fallback_activated") == 1


def test_broad_audit_turn_preserves_exact_evidence_before_compression(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    runtime = UniversalTokRuntime()
    messages = _parallel_file_read_messages(count=10, lines=24)
    original_payload = json.dumps(messages, sort_keys=True)

    prepared = runtime.prepare_request(_runtime_request(messages), session)

    assert json.dumps(prepared.body["messages"], sort_keys=True) == original_payload
    assert _tool_result_bytes(prepared.body["messages"]) == _tool_result_bytes(messages)
    assert prepared.saved_prompt_tokens == 0
    assert prepared.behavior_signals.get("broad_audit_tool_result_compression_skipped") == 1


def test_repeated_large_reads_across_turns_do_not_grow_without_signal(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    path = "src/tok/repeated_pressure.py"
    first_messages = [
        {"role": "user", "content": [{"type": "text", "text": "Read the file."}]},
        {"role": "assistant", "content": [_tool_use("toolu_repeat_1", "read_file", path=path)]},
        {"role": "user", "content": [_tool_result("toolu_repeat_1", _file_text(path, lines=220))]},
    ]
    first_metrics, session = _prepare_runtime_pressure(first_messages, tmp_path, session=session)
    repeat_messages = [
        *first_messages,
        {"role": "assistant", "content": [{"type": "text", "text": "Need to verify the same file again."}]},
        {"role": "user", "content": [{"type": "text", "text": "Read it again before editing."}]},
        {"role": "assistant", "content": [_tool_use("toolu_repeat_2", "read_file", path=path)]},
        {"role": "user", "content": [_tool_result("toolu_repeat_2", _file_text(path, lines=220))]},
    ]

    repeat_metrics, _session = _prepare_runtime_pressure(repeat_messages, tmp_path, session=session)

    _assert_bounded_or_signaled(repeat_metrics, multiplier=1.10, slack=160)
    assert repeat_metrics.outgoing_failures == []
    assert (
        repeat_metrics.prepared_prompt_tokens <= first_metrics.prepared_prompt_tokens + 800
        or repeat_metrics.has_visible_risk_signal
    )


def test_final_answer_turn_after_compressed_evidence_preserves_answer_context(tmp_path) -> None:
    metrics, _session = _prepare_runtime_pressure(_final_answer_after_evidence_messages(), tmp_path)

    _assert_bounded_or_signaled(metrics, multiplier=1.15, slack=180)
    assert metrics.outgoing_failures == []
    assert (
        metrics.behavior_signals.get("plan_finalization_turn", 0)
        or metrics.behavior_signals.get("answer_ready_reacquisition_attempt", 0)
        or metrics.behavior_signals.get("answer_ready_exact_search_evidence_history_preserved", 0)
        or metrics.prepared_prompt_tokens <= int(metrics.baseline_prompt_tokens * 1.15) + 180
    )


def test_large_recent_tool_results_after_history_compression_are_bounded_or_signaled(tmp_path) -> None:
    old_turns: list[dict[str, Any]] = []
    for index in range(8):
        old_turns.extend(
            [
                {"role": "user", "content": [{"type": "text", "text": f"Investigate older area {index}."}]},
                {
                    "role": "assistant",
                    "content": [_tool_use(f"toolu_old_{index}", "read_file", path=f"src/tok/old_{index}.py")],
                },
                {
                    "role": "user",
                    "content": [_tool_result(f"toolu_old_{index}", _file_text(f"src/tok/old_{index}.py", lines=80))],
                },
            ]
        )
    recent = [
        {"role": "user", "content": [{"type": "text", "text": "Now inspect the giant recent output."}]},
        {"role": "assistant", "content": [_tool_use("toolu_recent_big", "bash", command="uv run pytest -q")]},
        {
            "role": "user",
            "content": [
                _tool_result(
                    "toolu_recent_big",
                    "\n".join(f"tests/unit/test_pressure.py::test_case_{i} PASSED" for i in range(900)),
                )
            ],
        },
    ]

    metrics, _session = _prepare_runtime_pressure([*old_turns, *recent], tmp_path)

    _assert_bounded_or_signaled(metrics, multiplier=1.12, slack=256)
    assert metrics.outgoing_failures == []
    assert (
        metrics.retained_tool_result_bytes < _tool_result_bytes([*old_turns, *recent])
        or metrics.behavior_signals.get("tok_history_compression_skipped", 0)
        or metrics.has_visible_risk_signal
    )


def test_mixed_bug_finding_sweep_is_bounded_or_visibly_guarded(tmp_path) -> None:
    metrics, _session = _prepare_runtime_pressure(_claude_code_bug_sweep_messages(), tmp_path)

    _assert_bounded_or_signaled(metrics, multiplier=1.12, slack=256)
    assert metrics.outgoing_failures == []
    assert metrics.retained_tool_result_bytes <= _tool_result_bytes(_claude_code_bug_sweep_messages())


def test_log_shaped_long_history_with_many_tool_pairs_stays_valid_and_signaled(tmp_path) -> None:
    messages = _long_alternating_history_messages(turns=34)

    metrics = _prepare_bridge_pressure(messages, tmp_path)

    _assert_bounded_or_signaled(metrics, multiplier=1.12, slack=320)
    assert metrics.outgoing_failures == []
    assert metrics.retained_tool_result_bytes < _tool_result_bytes(messages) or metrics.has_visible_risk_signal


def test_reordered_tool_results_are_repaired_before_provider_send(tmp_path) -> None:
    messages = _reordered_tool_result_messages()
    original_failures = validate_anthropic_outgoing_bridge_body(
        {
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": messages,
            "stream": False,
        }
    )

    metrics = _prepare_bridge_pressure(messages, tmp_path)

    assert "tool_result_not_immediately_after_assistant_tool_use" in original_failures
    assert metrics.outgoing_failures == []
    assert metrics.behavior_signals.get("tok_bridge_tool_result_order_repaired") == 1
    assert metrics.behavior_signals.get("tok_bridge_tool_result_pairing_repaired") == 1


def test_signed_thinking_blocks_survive_pressure_canonicalization_unchanged(tmp_path) -> None:
    del tmp_path
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": _thinking_mutation_messages(),
        "stream": False,
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert validate_anthropic_outgoing_bridge_body(canonical) == []
    assert changed is False
    assert signals == {}
    assert canonical["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": "private chain",
        "signature": "sig_1",
    }
    assert canonical["messages"][3]["content"][0] == {
        "type": "thinking",
        "thinking": "mutated private chain",
        "signature": "sig_1",
    }


def test_repeated_file_read_can_compress_only_after_exact_observation() -> None:
    file_content = _file_text("/tmp/pressure.py", lines=120)
    messages = [
        {"role": "assistant", "content": [_tool_use("t1", "read_file", file_path="/tmp/pressure.py")]},
        {"role": "user", "content": [_tool_result("t1", file_content)]},
        {"role": "assistant", "content": [_tool_use("t2", "read_file", file_path="/tmp/pressure.py")]},
        {"role": "user", "content": [_tool_result("t2", file_content)]},
    ]
    tool_use_id_to_context = build_tool_use_id_to_context(messages)
    first_exact_evidence_seen: set[str] = set()
    result_cache: dict[str, tuple[str, str, float]] = {}
    digest = hashlib.sha256(file_content.encode()).hexdigest()[:8]
    cache_key = _make_cache_key("read_file", tool_use_id_to_context["t1"])
    result_cache[cache_key] = (digest, file_content, time.time())

    compressed, _breakdown = compress_tool_results_impl(
        messages,
        result_cache=result_cache,
        tool_use_id_to_context=tool_use_id_to_context,
        compression_level="balanced",
        first_exact_evidence_seen=first_exact_evidence_seen,
    )

    first_result = compressed[1]["content"][0]["content"]
    second_result = compressed[3]["content"][0]["content"]
    evidence_key = evidence_identity_key(
        "read_file",
        path="/tmp/pressure.py",
        args={"file_path": "/tmp/pressure.py"},
    )

    assert first_result == file_content
    assert evidence_key in first_exact_evidence_seen
    assert second_result == file_content or any(
        marker in second_result for marker in ("@stable_result", "tok_compressed", "|unchanged|", "[tok optimized]")
    )


def test_summary_or_skeleton_then_edit_intent_is_flagged(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    runtime = UniversalTokRuntime()
    path = "src/tok/pressure.py"
    session._skeleton_delivered_paths.add(normalize_path_target(path))

    event = NormalizedToolEvent(
        id="toolu_edit_1",
        name="edit_file",
        args={"path": path, "old_string": "before", "new_string": "after"},
        path=path,
    )

    try:
        runtime.execute_tool_event(event, session=session)
    except TokSafetyError as exc:
        message = str(exc)
    else:  # pragma: no cover - the assertion below documents the expected failure mode.
        raise AssertionError("edit-like intent after skeleton evidence was not blocked")

    assert "Cannot edit" in message
    assert "summary instead of full content" in message
    assert "Re-read" in message


def test_provider_sensitive_tool_pairing_never_silent_fallbacks(tmp_path) -> None:
    session = BridgeSession(memory_dir=tmp_path / ".tok", fail_open=True)
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": _provider_sensitive_large_tool_batch_messages(),
        "stream": False,
    }
    original_failures = validate_anthropic_outgoing_bridge_body(body)

    payload, response = prepare_bridge_payload(
        session=session,
        body=body,
        headers={"x-api-key": "test"},
        path="v1/messages",
        tok_tool_header="1",
    )

    assert response is None
    assert "provider_sensitive_large_tool_use_text_interleaving" in original_failures
    assert validate_anthropic_outgoing_bridge_body(payload.body) == []
    assert payload.behavior_signals.get("tok_bridge_assistant_block_order_normalized") == 1
    assert payload.behavior_signals.get("tok_bridge_canonicalized") == 1


def test_provider_sensitive_bridge_pressure_reports_clean_outgoing_and_signals(tmp_path) -> None:
    messages = _provider_sensitive_large_tool_batch_messages()
    original_failures = validate_anthropic_outgoing_bridge_body(
        {
            "model": "claude-sonnet-4",
            "max_tokens": 8192,
            "messages": messages,
            "stream": False,
        }
    )

    metrics = _prepare_bridge_pressure(messages, tmp_path)

    assert "provider_sensitive_large_tool_use_text_interleaving" in original_failures
    assert metrics.outgoing_failures == []
    assert metrics.has_visible_risk_signal


def test_answer_ready_failure_requests_visible_repair_after_tool_response(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    runtime = UniversalTokRuntime()
    messages = _final_answer_after_evidence_messages()
    prepared = runtime.prepare_request(_runtime_request(messages), session)

    processed = runtime.process_response(
        "I need to read src/tok/missed.py before answer.",
        model="claude-sonnet-4",
        session=session,
        behavior_signals=prepared.behavior_signals,
        tool_compatible=True,
    )

    assert (
        processed.behavior_signals.get("structured_answer_deferral_rejected", 0)
        or processed.behavior_signals.get("structured_answer_repair_failed", 0)
        or processed.behavior_signals.get("answer_ready_repair_requested", 0)
        or processed.behavior_signals.get("answer_ready_tool_violation", 0)
        or processed.behavior_signals.get("answer_ready_failed_to_answer", 0)
        or prepared.behavior_signals.get("plan_finalization_turn", 0)
    )
    assert processed.behavior_signals.get("structured_answer_visible_preserved", 0) == 0


def test_answer_ready_repair_active_second_miss_fails_closed(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    runtime = UniversalTokRuntime()
    session._answer_ready_repair_active = True
    messages = _final_answer_after_evidence_messages()
    prepared = runtime.prepare_request(_runtime_request(messages), session)
    session._answer_ready_repair_active = True

    processed = runtime.process_response(
        "I still need to inspect src/tok/again.py before answer.",
        model="claude-sonnet-4",
        session=session,
        behavior_signals=prepared.behavior_signals | {"answer_ready_turn": 1},
        tool_compatible=True,
    )

    assert processed.behavior_signals.get("answer_ready_repair_failed", 0) == 1
    assert session._answer_ready_repair_pending is False
