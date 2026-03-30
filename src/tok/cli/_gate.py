from __future__ import annotations

"""Helpers for replay gate loading, classification, and release summaries."""

import json
from pathlib import Path
from typing import Any, cast

from ..stats import SavingsTracker


def load_gate_config(config_path: Path | None) -> dict[str, Any] | None:
    if config_path is None:
        default = Path.cwd() / "gate-config.json"
        if default.exists():
            config_path = default
    if config_path is None or not config_path.exists():
        return None
    try:
        return cast(dict[str, Any], json.loads(config_path.read_text()))
    except Exception:
        return None


def load_fixture_files(fixtures_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(fixtures_dir.rglob("*.jsonl")):
        if path.name.endswith(".meta.json"):
            continue
        files.append(path)
    return files


def read_fixture(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta: dict[str, Any] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSONL fixture: {path} ({exc})") from exc
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:  # pragma: no cover - guardrail
            raise ValueError(
                f"Invalid meta JSON: {meta_path} ({exc})"
            ) from exc
    return records, meta


def infer_fixture_format(records: list[dict[str, Any]]) -> str:
    if not records:
        return "empty"
    if all(isinstance(record.get("messages"), list) for record in records):
        return "session-records"
    if all("role" in record and "content" in record for record in records):
        return "message-per-line"
    return "mixed"


def is_common_path_fixture(fixture_name: str, meta: dict[str, Any]) -> bool:
    value = meta.get("common_path")
    if isinstance(value, bool):
        return value
    return fixture_name in {
        "claude_coding_loop",
        "gpt_coding_loop",
        "long_coding_session",
        "search_intensive_workflow",
        "file_heavy_operations",
        "context_pinned_file",
        "pressure_session",
        "runtime_conformance",
        "metric_long_debug",
        "burst_retries",
    }


def fixture_usage_weight(fixture_name: str, meta: dict[str, Any]) -> float:
    value = meta.get("usage_weight")
    if isinstance(value, int | float):
        return float(value)
    default_weights = {
        "claude_coding_loop": 1.5,
        "gpt_coding_loop": 1.1,
        "long_coding_session": 1.6,
        "search_intensive_workflow": 1.2,
        "file_heavy_operations": 1.8,
        "context_pinned_file": 1.7,
        "pressure_session": 1.3,
        "runtime_conformance": 1.0,
        "metric_long_debug": 1.4,
        "burst_retries": 1.0,
    }
    return float(default_weights.get(fixture_name, 1.0))


def evaluate_expected_failure(
    meta: dict[str, Any],
    *,
    actual_failures: list[str],
    behavior_signals: dict[str, int],
) -> tuple[bool, list[str]]:
    expected_failures = [str(x) for x in meta.get("expected_failures", [])]
    expected_signals = [str(x) for x in meta.get("expected_signals", [])]

    missing_failures = [
        name for name in expected_failures if name not in actual_failures
    ]
    missing_signals = [
        name for name in expected_signals if behavior_signals.get(name, 0) <= 0
    ]
    unexpected_failures = [
        name
        for name in actual_failures
        if expected_failures and name not in expected_failures
    ]

    failures: list[str] = []
    failures.extend(
        f"missing_expected_failure:{name}" for name in missing_failures
    )
    failures.extend(
        f"missing_expected_signal:{name}" for name in missing_signals
    )
    failures.extend(
        f"unexpected_failure:{name}" for name in unexpected_failures
    )
    return (not failures), failures


def load_session_trend(tracker: SavingsTracker) -> dict[str, Any]:
    trend = tracker.trend_summary(recent_sessions=5)
    invisible_pressure = trend.get("avg_invisible_pressure", 0)
    status = "clean"
    if invisible_pressure > 10:
        status = "noisy"
    elif invisible_pressure > 5:
        status = "watch"
    return {
        "trend": trend,
        "status": status,
    }


def gate_release_summary(
    results: list[dict[str, Any]],
    tracker: SavingsTracker,
    trend_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_rows = [r for r in results if "error" not in r]
    savings_values = [
        float(r.get("savings_pct", 0.0))
        for r in metric_rows
        if r.get("savings_pct") is not None
    ]
    pressure_values = [
        float(r.get("pressure", 0.0))
        for r in metric_rows
        if r.get("pressure") is not None
    ]
    fallback_rows = [
        r
        for r in metric_rows
        if int(
            r.get("behavior_signals", {}).get("fail_open_compat_response", 0)
        )
        > 0
    ]
    reacquisition_rows = [
        r
        for r in metric_rows
        if int(r.get("behavior_signals", {}).get("repeat_search", 0)) > 0
        or int(r.get("behavior_signals", {}).get("repeat_file_read", 0)) > 0
    ]
    return {
        "fixtures": len(metric_rows),
        "avg_savings_pct": (
            round(sum(savings_values) / len(savings_values), 1)
            if savings_values
            else 0.0
        ),
        "avg_invisible_pressure": (
            round(sum(pressure_values) / len(pressure_values), 1)
            if pressure_values
            else 0.0
        ),
        "fallback_fixture_rate": (
            round(len(fallback_rows) / len(metric_rows) * 100, 1)
            if metric_rows
            else 0.0
        ),
        "reacquisition_fixture_rate": (
            round(len(reacquisition_rows) / len(metric_rows) * 100, 1)
            if metric_rows
            else 0.0
        ),
        "billing_delta_usd": 0.0,
        "billing_delta_pct": 0.0,
    }
