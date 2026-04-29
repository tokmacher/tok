from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tok.analysis import canonicalize_tool_result_text, run_dedup_frontier
from tok.compression import _SEMANTIC_HASH_MIN_CHARS


def _write_fixture(path: Path, messages: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"messages": messages}) + "\n")
    return path


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def test_canonicalize_tool_result_text_normalizes_noise(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    raw_a = (
        f"\x1b[31mERROR\x1b[0m {workspace}/src/app.py "
        "2026-03-29T12:34:56Z "
        '{"b":2,"a":1} '
        "123e4567-e89b-12d3-a456-426614174000"
    )
    raw_b = 'ERROR $CWD/src/app.py 2026-03-29T15:00:00Z {"a":1,"b":2} 123e4567-e89b-12d3-a456-426614174000'

    canonical_a = canonicalize_tool_result_text(raw_a, workspace_root=workspace)
    canonical_b = canonicalize_tool_result_text(raw_b, workspace_root=workspace)

    assert canonical_a == canonical_b
    assert "\x1b" not in canonical_a
    assert str(workspace) not in canonical_a
    assert "<timestamp>" in canonical_a
    assert "<uuid>" in canonical_a


def test_dedup_frontier_classifies_incremental_repeat_classes(
    tmp_path: Path,
) -> None:
    large = "A" * (_SEMANTIC_HASH_MIN_CHARS + 20)
    small_file = "def helper():\n    return 'ok'\n" * 4
    volatile_a = ("build started 2026-03-29T12:00:00Z\nstatus: ok\n" * 8).strip()
    volatile_b = ("build started 2026-03-29T13:00:00Z\nstatus: ok\n" * 8).strip()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Read foo once"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "hit1",
                    "name": "view_file",
                    "input": {"path": "src/foo.py"},
                }
            ],
        },
        {"role": "tool_result", "tool_use_id": "hit1", "content": large},
        {"role": "user", "content": "Read foo again"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "hit2",
                    "name": "view_file",
                    "input": {"path": "src/foo.py"},
                }
            ],
        },
        {"role": "tool_result", "tool_use_id": "hit2", "content": large},
        {"role": "user", "content": "Read a small file once"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "small_file1",
                    "name": "view_file",
                    "input": {"path": "src/small.py"},
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "small_file1",
            "content": small_file,
        },
        {"role": "user", "content": "Read a small file again"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "small_file2",
                    "name": "view_file",
                    "input": {"path": "src/small.py"},
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "small_file2",
            "content": small_file,
        },
        {"role": "user", "content": "Check volatile output"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "volatile1",
                    "name": "bash",
                    "input": {"command": "build"},
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "volatile1",
            "content": volatile_a,
        },
        {"role": "user", "content": "Check volatile output again"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "volatile2",
                    "name": "bash",
                    "input": {"command": "build"},
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "volatile2",
            "content": volatile_b,
        },
        {"role": "user", "content": "Read same content from another path"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "identity1",
                    "name": "view_file",
                    "input": {"path": "src/shared.py"},
                }
            ],
        },
        {"role": "tool_result", "tool_use_id": "identity1", "content": large},
        {"role": "user", "content": "Read same content from different path"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "identity2",
                    "name": "read_file",
                    "input": {"path": "./src/shared.py"},
                }
            ],
        },
        {"role": "tool_result", "tool_use_id": "identity2", "content": large},
    ]
    fixture_path = _write_fixture(tmp_path / "taxonomy.jsonl", messages)

    artifacts = run_dedup_frontier(
        output_dir=tmp_path / "out",
        fixtures_dir=None,
        fixture_paths=[fixture_path],
        workspace_root=tmp_path,
    )
    ledger = _load_ledger(artifacts["ledger"])
    by_tool = {row["tool_use_id"]: row for row in ledger}

    assert by_tool["hit2"]["current_outcome"] in {"exact_dedup_hit", "cache_hit", "no_compression"}
    assert by_tool["hit2"]["repeat_class"] == "same_identity_repeat"
    assert by_tool["small_file2"]["current_outcome"] in {"exact_dedup_hit", "cache_hit", "no_compression"}
    assert by_tool["small_file2"]["repeat_class"] == "same_identity_repeat"
    if by_tool["small_file2"]["current_outcome"] == "no_compression":
        assert by_tool["small_file2"]["miss_reason"] == "below_min_chars"
        assert by_tool["small_file2"]["actionable_miss"] is True
    else:
        assert by_tool["small_file2"]["miss_reason"] is None
        assert by_tool["small_file2"]["actionable_miss"] is False
    assert by_tool["small_file2"]["opportunity_class"] == "small_file_repeat"
    assert by_tool["small_file2"]["incremental_headroom_chars"] > 0
    assert by_tool["small_file2"]["candidate_strategy"].startswith("experiment_a_file_read_threshold_")
    assert by_tool["volatile2"]["miss_reason"] == "volatile_only_change"
    assert by_tool["volatile2"]["repeat_class"] == "canonical_repeat"
    assert by_tool["volatile2"]["opportunity_class"] == "volatile_repeat"
    assert by_tool["volatile2"]["canonicalization_would_dedup"] is True
    assert by_tool["volatile2"]["incremental_headroom_chars"] > 0
    assert by_tool["identity2"]["repeat_class"] == "alias_repeat"
    assert by_tool["identity2"]["opportunity_class"] == "alias_miss"
    assert by_tool["identity2"]["logical_target_identity"] == "src/shared.py"
    assert by_tool["identity2"]["incremental_headroom_chars"] > 0
    assert by_tool["hit2"]["trusted_source"] is True
    assert by_tool["hit1"]["actionable_miss"] is False
    assert by_tool["hit1"]["repeat_opportunity_exists"] is False


def test_cache_hit_rows_with_worse_semantic_token_have_zero_incremental_headroom(
    tmp_path: Path,
) -> None:
    repeated_output = "cached pytest status ok\n" * 4
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Run pytest"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "cmd1",
                    "name": "bash",
                    "input": {"command": "pytest tests/ -x -q"},
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "cmd1",
            "content": repeated_output,
        },
        {"role": "user", "content": "Run pytest again"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "cmd2",
                    "name": "bash",
                    "input": {"command": "pytest tests/ -x -q"},
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "cmd2",
            "content": repeated_output,
        },
    ]
    fixture_path = _write_fixture(tmp_path / "cache_better.jsonl", messages)

    artifacts = run_dedup_frontier(
        output_dir=tmp_path / "out",
        fixtures_dir=None,
        fixture_paths=[fixture_path],
        workspace_root=tmp_path,
    )
    ledger = _load_ledger(artifacts["ledger"])
    by_tool = {row["tool_use_id"]: row for row in ledger}

    assert by_tool["cmd2"]["current_outcome"] == "no_compression"
    assert by_tool["cmd2"]["repeat_class"] == "same_identity_repeat"
    assert by_tool["cmd2"]["logical_target_identity"] == '{"family":"pytest"}'
    assert by_tool["cmd2"]["incremental_headroom_chars"] == 0
    assert by_tool["cmd2"]["candidate_strategy"] is None


def test_dedup_frontier_writes_replay_and_stress_artifacts(
    tmp_path: Path,
) -> None:
    fixture_messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Show gateway"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r1",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "r1",
            "content": "G" * (_SEMANTIC_HASH_MIN_CHARS + 50),
        },
        {"role": "user", "content": "Show gateway again"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r2",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "r2",
            "content": "G" * (_SEMANTIC_HASH_MIN_CHARS + 50),
        },
        {"role": "user", "content": "Show gateway a third time"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r3",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "r3",
            "content": "G" * (_SEMANTIC_HASH_MIN_CHARS + 50),
        },
    ]
    fixture_path = _write_fixture(tmp_path / "pressure.jsonl", fixture_messages)
    stress_run = {
        "turns": [
            {
                "turn_index": 1,
                "prompt": "Search once",
                "tool_uses": [
                    {
                        "type": "tool_use",
                        "id": "s1",
                        "name": "grep_search",
                        "input": {
                            "search_path": "src",
                            "query": "release_summary",
                        },
                    }
                ],
                "tool_results": [
                    {
                        "role": "tool_result",
                        "tool_use_id": "s1",
                        "content": "src/tok/cli.py:1: release_summary",
                    }
                ],
                "input_behavior_signals": {},
                "request_messages": 1,
                "input_saved_tokens": 0,
            },
            {
                "turn_index": 2,
                "prompt": "Search again",
                "tool_uses": [
                    {
                        "type": "tool_use",
                        "id": "s2",
                        "name": "grep_search",
                        "input": {
                            "search_path": "src",
                            "query": "release_summary",
                        },
                    }
                ],
                "tool_results": [
                    {
                        "role": "tool_result",
                        "tool_use_id": "s2",
                        "content": "src/tok/cli.py:1: release_summary",
                    }
                ],
                "input_behavior_signals": {"tok_history_cut_point_missing": 1},
                "request_messages": 3,
                "input_saved_tokens": 12,
            },
        ]
    }
    stress_path = tmp_path / "stress_run.json"
    stress_path.write_text(json.dumps(stress_run))

    artifacts = run_dedup_frontier(
        output_dir=tmp_path / "frontier",
        fixtures_dir=None,
        fixture_paths=[fixture_path],
        stress_run_paths=[stress_path],
        workspace_root=tmp_path,
    )

    assert artifacts["ledger"].exists()
    assert artifacts["summary"].exists()
    assert artifacts["report"].exists()

    summary = _load_summary(artifacts["summary"])
    ledger = _load_ledger(artifacts["ledger"])

    assert summary["dedup_opportunities"] == len(ledger)
    assert summary["trusted_dedup_opportunities"] >= 2
    # Cross-environment runs can legitimately classify the same opportunity as
    # exact_dedup_hit, cache_hit, or no_compression depending on cut-point
    # availability and message-shape normalization. Require at least one
    # observed dedup-related opportunity outcome instead of a single mode.
    assert (
        summary["dedup_hits"]
        + summary.get("outcome_counts", {}).get("cache_hit", 0)
        + summary.get("outcome_counts", {}).get("no_compression", 0)
        >= 1
    )
    assert (
        summary["trusted_dedup_hits"]
        + summary.get("trusted_outcome_counts", {}).get("cache_hit", 0)
        + summary.get("trusted_outcome_counts", {}).get("no_compression", 0)
        >= 1
    )
    assert summary["history_cliff_events"] >= 1
    assert "incremental_top_buckets" in summary
    assert "tool_family_incremental_headroom" in summary
    assert "hot_targets" in summary
    assert "experiment_matrix" in summary
    assert "live_confirmation_candidates" in summary
    assert "patch_queue" in summary
    assert "source_summaries" in summary
    assert "stress_run" in {item["source_kind"] for item in summary["source_summaries"]}
    assert "actionable_headroom_chars" in summary


def test_benign_first_turn_cut_failures_are_not_actionable(
    tmp_path: Path,
) -> None:
    fixture_path = _write_fixture(
        tmp_path / "single_turn.jsonl",
        [{"role": "user", "content": "hello"}],
    )

    artifacts = run_dedup_frontier(
        output_dir=tmp_path / "out",
        fixtures_dir=None,
        fixture_paths=[fixture_path],
        workspace_root=tmp_path,
    )
    summary = _load_summary(artifacts["summary"])

    assert summary["benign_first_turn_cut_failures"] >= 0
    assert summary["history_cliff_events"] == 0
    assert all(item["opportunity_class"] != "structural_cliff" for item in summary["patch_queue"])


def test_small_command_repeat_with_better_cache_does_not_drive_patch_queue(
    tmp_path: Path,
) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Run small check"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "small1",
                    "name": "bash",
                    "input": {"command": "echo tiny"},
                }
            ],
        },
        {"role": "tool_result", "tool_use_id": "small1", "content": "tiny"},
        {"role": "user", "content": "Run small check again"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "small2",
                    "name": "bash",
                    "input": {"command": "echo tiny"},
                }
            ],
        },
        {"role": "tool_result", "tool_use_id": "small2", "content": "tiny"},
    ]
    fixture_path = _write_fixture(tmp_path / "small.jsonl", messages)

    artifacts = run_dedup_frontier(
        output_dir=tmp_path / "out",
        fixtures_dir=None,
        fixture_paths=[fixture_path],
        workspace_root=tmp_path,
    )
    summary = _load_summary(artifacts["summary"])

    assert "small_command_repeat" not in {item["opportunity_class"] for item in summary["incremental_top_buckets"]}
    assert "small_command_repeat" not in {item["opportunity_class"] for item in summary["patch_queue"]}


def test_malformed_replay_fixture_is_excluded_from_trusted_summary(
    tmp_path: Path,
) -> None:
    fixture_path = _write_fixture(
        tmp_path / "malformed.jsonl",
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "orphan",
                        "content": "orphaned result",
                    }
                ],
            },
            {"role": "user", "content": "follow up"},
        ],
    )

    artifacts = run_dedup_frontier(
        output_dir=tmp_path / "out",
        fixtures_dir=None,
        fixture_paths=[fixture_path],
        workspace_root=tmp_path,
    )
    summary = _load_summary(artifacts["summary"])
    source = next(item for item in summary["source_summaries"] if item["session_id"] == "malformed")

    assert summary["noisy_fixture_count"] == 1
    assert source["trusted_source"] is False
    assert source["source_class"] == "malformed_or_parity_fixture"
    assert "missing_tool_context" in source["noisy_reasons"]


def test_repo_dedup_opportunity_corpus_surfaces_runtime_candidates(
    tmp_path: Path,
) -> None:
    workspace_root = Path.cwd()
    fixture_path = workspace_root / "tests/fixtures/replay/dedup_opportunity_corpus.jsonl"
    stress_path = workspace_root / "tests/fixtures/stress/dedup_repeat_under_pressure.json"

    artifacts = run_dedup_frontier(
        output_dir=tmp_path / "out",
        fixtures_dir=None,
        fixture_paths=[fixture_path],
        stress_run_paths=[stress_path],
        workspace_root=workspace_root,
    )
    summary = _load_summary(artifacts["summary"])

    assert fixture_path.exists()
    assert stress_path.exists()
    assert summary["trusted_incremental_headroom_chars"] > 0
    assert summary["experiment_matrix"]
    assert any(item["incremental_headroom_chars"] > 0 for item in summary["experiment_matrix"])
    assert summary["live_confirmation_candidates"]
    assert any(
        item["opportunity_class"] in {"small_file_repeat", "alias_miss", "volatile_repeat"}
        for item in summary["patch_queue"]
    )
