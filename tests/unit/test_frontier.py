from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NoReturn

from tok.testing.frontier import (
    CompressionFrontierReport,
    CompressionProfile,
    FrontierCheckpoint,
    FrontierCheckpointReport,
    FrontierOpenRouterSummary,
    FrontierOpenRouterTurn,
    FrontierProfileSummary,
    _aggregate_benchmark_runs,
    _frontier_release_profile,
    apply_frontier_env,
    check_frontier_report,
    classify_frontier_verdict,
    render_frontier_markdown,
    run_benchmark_frontier_for_checkpoint,
)


def test_apply_frontier_env_restores_original_values() -> None:
    os.environ["TOK_KEEP_TURNS"] = "9"
    with apply_frontier_env({"TOK_KEEP_TURNS": "1", "TOK_MODE": "tool-compatible"}):
        assert os.environ["TOK_KEEP_TURNS"] == "1"
        assert os.environ["TOK_MODE"] == "tool-compatible"
    assert os.environ["TOK_KEEP_TURNS"] == "9"
    assert "TOK_MODE" not in os.environ


def test_classify_frontier_verdict_prefers_degraded_conditions() -> None:
    verdict = classify_frontier_verdict(
        success_rate=1.0,
        recovery_attempts=0,
        recovery_holdovers=1,
        fail_open_count=0,
        malformed_count=0,
        non_tok_count=0,
        warning_signal_count=0,
        local_failure_count=0,
        max_pressure=0,
    )
    assert verdict == "degraded"


def test_aggregate_benchmark_runs_marks_watch_for_recovery_noise() -> None:
    summary = _aggregate_benchmark_runs(
        benchmark="coding-loop-5",
        turns=5,
        repeats=1,
        mode="tok-tool-compatible",
        runs=[
            {
                "candidate": {
                    "task_success": True,
                    "notes": [],
                    "compression_metrics": {
                        "total_saved_tokens": 60,
                        "input_behavior_signals": {},
                    },
                    "provider_usage": {
                        "total_tokens": 140,
                        "latency_ms": 120.0,
                    },
                    "response_metrics": {
                        "invisible_pressure": 1,
                        "response_behavior_signals": {
                            "stream_recovery_started": 1,
                            "stream_recovery_success_text": 1,
                        },
                    },
                    "diagnostics": {"response_warning_signal_count": 0},
                },
                "comparison": {"diagnosis": "won_on_prompt_reduction"},
            }
        ],
    )
    assert summary.verdict == "watch"
    assert summary.avg_savings_pct == 30.0
    assert summary.recovery_attempt_count == 1


def test_frontier_release_profile_uses_benchmark_stability_only() -> None:
    conservative = CompressionProfile("conservative", "tok-tool-compatible", "")
    aggressive = CompressionProfile("aggressive", "tok-native", "")
    benchmark_profiles = [
        FrontierProfileSummary(
            profile=conservative,
            benchmark_summaries=[],
            verdict="stable",
            stop_after=False,
        ),
        FrontierProfileSummary(
            profile=aggressive,
            benchmark_summaries=[],
            verdict="stable",
            stop_after=False,
        ),
    ]
    openrouter_profiles = [
        FrontierOpenRouterSummary(
            profile="conservative",
            model="m",
            prompt="p",
            turns_requested=5,
            turns_completed=5,
            success_rate=1.0,
            avg_savings_pct=20.0,
            p95_savings_pct=22.0,
            recovery_attempt_count=0,
            recovery_success_count=0,
            recovery_fallback_count=0,
            recovery_holdover_count=0,
            fail_open_count=0,
            malformed_count=0,
            non_tok_count=0,
            warning_signal_count=0,
            local_failure_count=0,
            verdict="stable",
            turns=[
                FrontierOpenRouterTurn(
                    turn=0,
                    input_saved_tokens=1,
                    output_saved_tokens=2,
                    total_saved_tokens=3,
                    text_preview="ok",
                    behavior_signals={},
                )
            ],
        )
    ]
    release_profile, experimental = _frontier_release_profile(benchmark_profiles, openrouter_profiles)
    assert release_profile == "aggressive"
    assert "conservative" not in experimental


def test_render_frontier_markdown_includes_release_lane() -> None:
    checkpoint = FrontierCheckpointReport(
        checkpoint=FrontierCheckpoint(label="current-head", ref="CURRENT"),
        benchmark_profiles=[],
        openrouter_profiles=[],
        default_release_profile="balanced",
        experimental_profiles=["aggressive"],
        notes=["OpenRouter probes are advisory only; benchmark stability selects the release lane."],
    )
    report = CompressionFrontierReport(
        model="m",
        checkpoints=[checkpoint],
        profiles=[],
        benchmarks=["coding-loop-5"],
        repeats=1,
        openrouter_prompt="prompt",
        openrouter_turn_sets=[5],
    )
    rendered = render_frontier_markdown(report)
    assert "Compression Frontier Report" in rendered
    assert "Release lane: `balanced`" in rendered
    assert "OpenRouter probes are advisory only" in rendered


def test_check_frontier_report_requires_non_baseline_release_lane(
    tmp_path,
) -> None:
    path = tmp_path / "frontier.json"
    path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "checkpoint": {
                            "label": "current-head",
                            "ref": "CURRENT",
                        },
                        "benchmark_profiles": [
                            {
                                "profile": {"name": "balanced"},
                                "verdict": "stable",
                            }
                        ],
                        "openrouter_profiles": [],
                        "default_release_profile": "balanced",
                        "experimental_profiles": ["aggressive"],
                    }
                ]
            }
        )
    )
    row = check_frontier_report(path)
    assert row["passed"] is True
    assert row["release_profile"] == "balanced"


def test_select_frontier_checkpoints_marks_current_head_as_current(
    tmp_path,
) -> None:
    from tok.testing.frontier import select_frontier_checkpoints

    checkpoints = select_frontier_checkpoints(Path())
    assert checkpoints[0].ref == "CURRENT"


def test_incompatible_historical_checkpoint_degrades_instead_of_raising(
    monkeypatch,
) -> None:
    def _boom(**_kwargs) -> NoReturn:
        msg = "older checkpoint API mismatch"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        "tok.testing.frontier._run_checkpoint_worker",
        _boom,
    )
    profile = CompressionProfile("conservative", "tok-tool-compatible", "")
    summaries = run_benchmark_frontier_for_checkpoint(
        repo_root=Path(),
        checkpoint=FrontierCheckpoint(
            label="pre-runtime-shaping",
            ref="deadbeef",
        ),
        profiles=[profile],
        benchmarks=["coding-loop-5"],
        model="m",
        repeats=1,
        temperature=0.0,
        max_tokens=10,
        timeout=1.0,
        api_key="k",
        api_base="https://example.com",
    )
    assert summaries[0].verdict == "degraded"
    assert "checkpoint_incompatible:" in summaries[0].notes[0]
