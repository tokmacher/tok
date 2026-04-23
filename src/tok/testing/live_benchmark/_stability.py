from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_stability_markdown(
    benchmark: str,
    model: str,
    summary: dict[str, Any],
) -> str:
    lines = [
        f"# Live Benchmark Stability: {benchmark}",
        "",
        f"- Model: `{model}`",
        f"- Repeats: `{summary.get('runs', 0)}`",
        "",
        "## Preferred Mode Counts",
        "",
    ]
    preferred_counts = summary.get("preferred_mode_counts", {})
    if preferred_counts:
        for mode, count in sorted(preferred_counts.items()):
            lines.append(f"- `{mode}`: `{count}`")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Median Metrics",
            "",
            "| Mode | Success Rate | Median Total | Min | Max | Median Prompt | Median Completion | Median Latency (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, metrics in summary.get("mode_summaries", {}).items():
        lines.append(
            f"| {mode} | {metrics['success_rate']:.3f} | {metrics['median_total_tokens']} | "
            f"{metrics['min_total_tokens']} | {metrics['max_total_tokens']} | "
            f"{metrics['median_prompt_tokens']} | {metrics['median_completion_tokens']} | "
            f"{metrics['median_latency_ms']} |"
        )

    lines.append("")
    return "\n".join(lines)


def check_stability_artifacts(
    stability_dir: Path,
    required_benchmarks: list[str],
) -> dict[str, dict[str, Any]]:
    """Validate checked-in stability artifacts for the release gate."""
    results: dict[str, dict[str, Any]] = {}

    for benchmark in required_benchmarks:
        artifact = stability_dir / f"{benchmark}_stability.json"
        row: dict[str, Any] = {
            "path": str(artifact),
            "passed": False,
        }

        if not artifact.exists():
            row["reason"] = "file not found"
            results[benchmark] = row
            continue

        try:
            payload = json.loads(artifact.read_text())
        except Exception as exc:
            row["reason"] = "invalid_json"
            row["error"] = str(exc)
            results[benchmark] = row
            continue

        mode_summaries = payload.get("mode_summaries", {})
        preferred_mode_counts = payload.get("preferred_mode_counts", {})
        tok_summary = mode_summaries.get("tok-universal", {}) or mode_summaries.get("tok-tool-compatible", {})
        runs = int(payload.get("runs", 0))
        preferred_count = int(
            preferred_mode_counts.get("tok-universal", 0) or preferred_mode_counts.get("tok-tool-compatible", 0)
        )
        success_rate = float(tok_summary.get("success_rate", 0.0))

        benchmark_name = str(payload.get("benchmark", benchmark))
        passed = benchmark_name == benchmark and runs > 0 and success_rate == 1.0 and preferred_count == runs

        row.update(
            {
                "benchmark": benchmark_name,
                "runs": runs,
                "success_rate": success_rate,
                "preferred_mode": preferred_count,
                "passed": passed,
            }
        )
        if not passed:
            row["reason"] = "criteria_failed"

        results[benchmark] = row

    return results
