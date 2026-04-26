"""Helpers for replay gate loading, classification, and release summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from tok.stats import SavingsTracker


def load_gate_config(config_path: Path | None) -> dict[str, Any] | None:
    """
    Load gate configuration from file.

    Args:
        config_path: Path to the gate configuration file, or None to use default.

    Returns:
        Loaded configuration as dictionary, or None if not found.

    """
    if config_path is None:
        default = Path.cwd() / "gate-config.json"
        if default.exists():
            config_path = default
    if config_path is None or not config_path.exists():
        return None
    try:
        return cast("dict[str, Any]", json.loads(config_path.read_text()))
    except Exception:
        return None


def select_fixture_set(fixture_set: str) -> list[str]:
    """Select a named fixture set for gate-check (feature/full/redteam)."""
    if fixture_set == "feature":
        return [
            "claude_coding_loop",
            "gpt_coding_loop",
            "long_coding_session",
            "search_intensive_workflow",
            "high_pressure_scenario",
            "runtime_conformance",
            "cache_stable_research_turns",
            "refined_search_recovery",
            # Green stress fixtures
            "metric_long_debug",
            "burst_retries",
            "verbose_payload",
            "straddling_boundary",
            "context_pinned_file",
            "alternating_adapters",
            "branching_tests",
            "compression_hypothesis_churn",
            "heavy_tool_event",
            "tool_density_micro",
            "episodes_multi_phase",
            "release_reacquisition",
            "cache_sensitivity",
        ]
    if fixture_set == "full":
        return [
            "claude_coding_loop",
            "gpt_coding_loop",
            "long_coding_session",
            "search_intensive_workflow",
            "high_pressure_scenario",
            "multi_model_session",
            "file_heavy_operations",
            "test_cli_fixture",
            "test_search_fixture",
            "test_coding_fixture",
            "comprehensive_test",
            "gemini_coding_loop",
            "pressure_session",
            "runtime_conformance",
            "cache_stable_research_turns",
            "refined_search_recovery",
            # Green stress fixtures
            "metric_long_debug",
            "burst_retries",
            "verbose_payload",
            "straddling_boundary",
            "context_pinned_file",
            "alternating_adapters",
            "branching_tests",
            "compression_hypothesis_churn",
            "heavy_tool_event",
            "tool_density_micro",
            "episodes_multi_phase",
            "release_reacquisition",
            "cache_sensitivity",
        ]
    if fixture_set == "redteam":
        return [
            "grammar_drift",
            "markdown_fallback",
            "subtle_drift",
            "healing_drift",
            "repeat_search_pressure",
        ]
    msg = f"Unknown fixture set: {fixture_set}"
    raise ValueError(msg)


def load_fixture_files(fixtures_dir: Path) -> list[Path]:
    """
    Load fixture files from directory.

    Args:
        fixtures_dir: Directory containing fixture files.

    Returns:
        List of fixture file paths.

    """
    files: list[Path] = []
    for path in sorted(fixtures_dir.rglob("*.jsonl")):
        if path.name.endswith(".meta.json"):
            continue
        files.append(path)
    return files


def read_fixture(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Read fixture file and its metadata.

    Args:
        path: Path to the fixture file.

    Returns:
        Tuple of (records, metadata) from the fixture.

    """
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
        msg = f"Invalid JSONL fixture: {path} ({exc})"
        raise ValueError(msg) from exc
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:  # pragma: no cover - guardrail
            msg = f"Invalid meta JSON: {meta_path} ({exc})"
            raise ValueError(msg) from exc
    return records, meta


def infer_fixture_format(records: list[dict[str, Any]]) -> str:
    """
    Infer the format of fixture records.

    Args:
        records: List of fixture records to analyze.

    Returns:
        String describing the fixture format.

    """
    if not records:
        return "empty"
    if all(isinstance(record.get("messages"), list) for record in records):
        return "session-records"
    if all("role" in record and "content" in record for record in records):
        return "message-per-line"
    return "mixed"


def is_common_path_fixture(fixture_name: str, meta: dict[str, Any]) -> bool:
    """
    Check if fixture is a common path fixture.

    Args:
        fixture_name: Name of the fixture.
        meta: Fixture metadata.

    Returns:
        True if this is a common path fixture.

    """
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
    """
    Get usage weight for fixture.

    Args:
        fixture_name: Name of the fixture.
        meta: Fixture metadata.

    Returns:
        Usage weight for the fixture.

    """
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
    """
    Evaluate if expected failures match actual failures.

    Args:
        meta: Fixture metadata.
        actual_failures: List of actual failure modes.
        behavior_signals: Dictionary of behavior signals.

    Returns:
        Tuple of (success, mismatched_failures).

    """
    expected_failures = [str(x) for x in meta.get("expected_failures", [])]
    expected_signals = [str(x) for x in meta.get("expected_signals", [])]

    missing_failures = [name for name in expected_failures if name not in actual_failures]
    missing_signals = [name for name in expected_signals if behavior_signals.get(name, 0) <= 0]
    unexpected_failures = [name for name in actual_failures if name not in expected_failures]

    failures: list[str] = []
    failures.extend(f"missing_expected_failure:{name}" for name in missing_failures)
    failures.extend(f"missing_expected_signal:{name}" for name in missing_signals)
    failures.extend(f"unexpected_failure:{name}" for name in unexpected_failures)
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
    _tracker: SavingsTracker,
    _trend_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_rows = [r for r in results if "error" not in r]
    savings_values = [float(r.get("savings_pct", 0.0)) for r in metric_rows if r.get("savings_pct") is not None]
    pressure_values = [float(r.get("pressure", 0.0)) for r in metric_rows if r.get("pressure") is not None]
    fallback_rows = [
        r for r in metric_rows if int(r.get("behavior_signals", {}).get("fail_open_compat_response", 0)) > 0
    ]
    reacquisition_rows = [
        r
        for r in metric_rows
        if int(r.get("behavior_signals", {}).get("repeat_search", 0)) > 0
        or int(r.get("behavior_signals", {}).get("repeat_file_read", 0)) > 0
    ]
    return {
        "fixtures": len(metric_rows),
        "avg_savings_pct": (round(sum(savings_values) / len(savings_values), 1) if savings_values else 0.0),
        "avg_invisible_pressure": (round(sum(pressure_values) / len(pressure_values), 1) if pressure_values else 0.0),
        "fallback_fixture_rate": (round(len(fallback_rows) / len(metric_rows) * 100, 1) if metric_rows else 0.0),
        "reacquisition_fixture_rate": (
            round(len(reacquisition_rows) / len(metric_rows) * 100, 1) if metric_rows else 0.0
        ),
        "billing_delta_usd": 0.0,
        "billing_delta_pct": 0.0,
    }
