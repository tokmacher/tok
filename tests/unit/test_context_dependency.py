from __future__ import annotations

from tok.compression import compress_tool_results, text_of
from tok.runtime.pipeline.context_dependency import (
    classify_context_dependency,
    suffix_preserves_tool_pairs,
)


def test_plan_handoff_detects_recent_plan_and_protects_tool_pair() -> None:
    messages = [
        {"role": "user", "content": "make a plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "plan_read",
                    "name": "Read",
                    "input": {"file_path": "/tmp/.claude/plans/example.md"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "plan_read",
                    "content": "Plan: safer compression\n- inspect\n- patch\n- test\n",
                }
            ],
        },
        {"role": "user", "content": "go ahead"},
    ]

    decision = classify_context_dependency(messages)

    assert decision.depends_on_context is True
    assert decision.kind == "plan_handoff"
    assert decision.protected_suffix_start == 1
    assert suffix_preserves_tool_pairs(messages, decision.protected_suffix_start)


def test_self_contained_plan_prompt_does_not_trigger_context_dependency() -> None:
    messages = [
        {"role": "assistant", "content": "Plan: inspect, patch, test.\n- one\n- two\n"},
        {"role": "user", "content": "implement this plan: change src/app.py and run tests"},
    ]

    decision = classify_context_dependency(messages)

    assert decision.depends_on_context is False


def test_unrelated_short_continue_without_plan_evidence_does_not_trigger() -> None:
    messages = [
        {"role": "assistant", "content": "I found one bug in the request parser."},
        {"role": "user", "content": "continue"},
    ]

    decision = classify_context_dependency(messages)

    assert decision.depends_on_context is False


def test_plan_handoff_without_safe_pair_boundary_requests_fallback() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "missing_plan_read",
                    "content": "Plan: safer compression\n- inspect\n- patch\n- test\n",
                }
            ],
        },
        {"role": "user", "content": "proceed"},
    ]

    decision = classify_context_dependency(messages)

    assert decision.depends_on_context is True
    assert decision.kind == "plan_handoff"
    assert decision.protected_suffix_start is None


def test_protected_suffix_blocks_tool_result_compression_but_prefix_can_compress() -> None:
    raw = "\n".join(f"test_{i} PASSED" for i in range(200))
    cache = {}
    first = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "cmd_1", "name": "bash", "input": {"command": "pytest"}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "cmd_1", "content": raw}]},
    ]
    compress_tool_results(
        first,
        result_cache=cache,
        tool_use_id_to_context={"cmd_1": {"name": "bash", "args": {"command": "pytest"}, "command": "pytest"}},
    )

    plan_text = "Here is the implementation plan.\nPlan: safer compression\n- inspect\n- patch\n- test\n"
    second = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "cmd_2", "name": "bash", "input": {"command": "pytest"}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "cmd_2", "content": raw}]},
        {"role": "assistant", "content": plan_text},
        {"role": "user", "content": "proceed"},
    ]

    out, breakdown = compress_tool_results(
        second,
        result_cache=cache,
        tool_use_id_to_context={"cmd_2": {"name": "bash", "args": {"command": "pytest"}, "command": "pytest"}},
        protected_suffix_start=2,
    )

    assert breakdown.get("command_cached", 0) > 0
    assert "unchanged|cached" in text_of(out[1]["content"])
    assert out[2]["content"] == plan_text
    assert out[3]["content"] == "proceed"
