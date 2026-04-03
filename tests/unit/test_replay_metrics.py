import json

from tok.utils.replay_metrics import analyze_replay_fixture
from tok.universal_runtime import calculate_invisible_pressure


def test_analyze_replay_fixture_counts_response_contract_drift(tmp_path):
    fixture = tmp_path / "grammar_drift.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": ">>> t:1|usr:hello\n@msg role:assistant\n  | Missing the inverted marker here",
                            }
                        ],
                    },
                ]
            }
        )
        + "\n"
    )
    fixture.with_suffix(".jsonl.meta.json").write_text(
        json.dumps({"model": "claude-sonnet-4"})
    )

    metrics = analyze_replay_fixture(fixture)

    assert metrics.behavior_totals["malformed_tok_response"] == 1
    assert metrics.behavior_totals["fail_open_compat_response"] == 1
    assert calculate_invisible_pressure(metrics.behavior_totals) == 2


def test_analyze_replay_fixture_supports_message_per_line_format(tmp_path):
    fixture = tmp_path / "subtle_drift.jsonl"
    fixture.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "role": "user",
                        "content": "Tell me about the current project architecture.",
                    }
                ),
                json.dumps(
                    {
                        "role": "assistant",
                        "content": "I have carefully reviewed the codebase and here is a summarized overview.",
                    }
                ),
            ]
        )
        + "\n"
    )
    fixture.with_suffix(".jsonl.meta.json").write_text(
        json.dumps({"model": "claude-sonnet-4"})
    )

    metrics = analyze_replay_fixture(fixture)

    assert metrics.lines == 2
    assert metrics.behavior_totals["semantic_drift_detected"] == 1
    assert metrics.behavior_totals["non_tok_response"] == 1
    assert metrics.behavior_totals["fail_open_compat_response"] == 1
    assert calculate_invisible_pressure(metrics.behavior_totals) == 3


def test_analyze_replay_fixture_respects_tool_compatible_response_mode(
    tmp_path,
):
    fixture = tmp_path / "adapter_fixture.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Still GPT-4."}],
                    },
                ]
            }
        )
        + "\n"
    )
    fixture.with_suffix(".jsonl.meta.json").write_text(
        json.dumps({"model": "gpt-4", "response_mode": "tool-compatible"})
    )

    metrics = analyze_replay_fixture(fixture)

    assert metrics.behavior_totals["tool_compatible_response"] == 1
    assert "non_tok_response" not in metrics.behavior_totals
    assert "fail_open_compat_response" not in metrics.behavior_totals


def test_analyze_replay_fixture_counts_reacquisition_pressure_from_user_tool_results(
    tmp_path,
):
    fixture = tmp_path / "release_reacquisition.jsonl"
    search_result = "\n".join(
        [
            "src/tok/cli.py:1301:def _gate_release_summary(results, tracker)",
            "src/tok/cli.py:1686:release_summary = _gate_release_summary(results, tracker)",
            'src/tok/cli.py:1703:"release_summary": release_summary,',
        ]
        * 5
    )
    file_result = "\n".join(
        [
            "def _gate_release_summary(results, tracker):",
            '    savings_values = [float(r.get("savings_pct", 0.0)) for r in results]',
            '    pressure_values = [float(r.get("pressure", 0.0)) for r in results]',
            '    export.write_text(json.dumps({"release_summary": release_summary}, indent=2))',
        ]
        * 8
    )
    fixture.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Investigate why release_summary is missing from the gate export.",
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "s1",
                                "name": "grep_search",
                                "input": {
                                    "search_path": "src/tok",
                                    "query": "release_summary",
                                },
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "s1",
                                "content": search_result,
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "r1",
                                "name": "view_file",
                                "input": {"path": "src/tok/cli.py"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "r1",
                                "content": file_result,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": "Retry the inspection and confirm the export path.",
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "s2",
                                "name": "grep_search",
                                "input": {
                                    "search_path": "src/tok",
                                    "query": "release_summary",
                                },
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "s2",
                                "content": search_result,
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "r2",
                                "name": "view_file",
                                "input": {"path": "src/tok/cli.py"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "r2",
                                "content": file_result,
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": ">>> t:3|usr:audit|agt:tok|state:release\n@msg role:assistant\n  |> Confirmed. The gate export writes a structured release_summary.",
                            }
                        ],
                    },
                ]
            }
        )
        + "\n"
    )
    fixture.with_suffix(".jsonl.meta.json").write_text(
        json.dumps({"model": "claude-sonnet-4"})
    )

    metrics = analyze_replay_fixture(fixture)

    assert (
        metrics.behavior_totals.get("repeat_file_read", 0)
        + metrics.behavior_totals.get("cached_file_read", 0)
    ) == 1
    assert (
        metrics.behavior_totals.get("repeat_search", 0)
        + metrics.behavior_totals.get("cached_search", 0)
    ) == 1
    assert metrics.input_saved_tokens > 0
    assert metrics.savings_pct > 0
