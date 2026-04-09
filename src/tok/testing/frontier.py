"""Compression frontier harness for checkpoint and profile comparisons."""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Generator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from openai import OpenAI

from .live_benchmark import (
    BenchmarkDefinition,
    LiveBenchmarkRunner,
    compare_results,
    load_benchmark_definition,
)

DEFAULT_BASELINE_REF = "5aebb5d"
DEFAULT_OPENROUTER_TURNS = (5, 12)
DEFAULT_FRONTIER_BENCHMARKS = (
    "coding-loop-5",
    "research-loop-5",
    "research-loop-8",
)


@dataclass(frozen=True)
class CompressionProfile:
    name: str
    mode: str
    description: str
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "description": self.description,
            "env": dict(self.env),
        }


@dataclass(frozen=True)
class FrontierCheckpoint:
    label: str
    ref: str

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "ref": self.ref}


@dataclass(frozen=True)
class FrontierBenchmarkSummary:
    benchmark: str
    turns: int
    repeats: int
    mode: str
    success_rate: float
    avg_savings_pct: float
    p95_savings_pct: float
    avg_latency_ms: float
    avg_pressure: float
    max_pressure: int
    recovery_attempt_count: int
    recovery_success_count: int
    recovery_fallback_count: int
    recovery_holdover_count: int
    fail_open_count: int
    malformed_count: int
    non_tok_count: int
    warning_signal_count: int
    local_failure_count: int
    verdict: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrontierProfileSummary:
    profile: CompressionProfile
    benchmark_summaries: list[FrontierBenchmarkSummary]
    verdict: str
    stop_after: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "benchmark_summaries": [benchmark.to_dict() for benchmark in self.benchmark_summaries],
            "verdict": self.verdict,
            "stop_after": self.stop_after,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class FrontierOpenRouterTurn:
    turn: int
    input_saved_tokens: int
    output_saved_tokens: int
    total_saved_tokens: int
    text_preview: str
    behavior_signals: dict[str, int]
    local_failure: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrontierOpenRouterSummary:
    profile: str
    model: str
    prompt: str
    turns_requested: int
    turns_completed: int
    success_rate: float
    avg_savings_pct: float
    p95_savings_pct: float
    recovery_attempt_count: int
    recovery_success_count: int
    recovery_fallback_count: int
    recovery_holdover_count: int
    fail_open_count: int
    malformed_count: int
    non_tok_count: int
    warning_signal_count: int
    local_failure_count: int
    verdict: str
    notes: list[str] = field(default_factory=list)
    turns: list[FrontierOpenRouterTurn] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "model": self.model,
            "prompt": self.prompt,
            "turns_requested": self.turns_requested,
            "turns_completed": self.turns_completed,
            "success_rate": self.success_rate,
            "avg_savings_pct": self.avg_savings_pct,
            "p95_savings_pct": self.p95_savings_pct,
            "recovery_attempt_count": self.recovery_attempt_count,
            "recovery_success_count": self.recovery_success_count,
            "recovery_fallback_count": self.recovery_fallback_count,
            "recovery_holdover_count": self.recovery_holdover_count,
            "fail_open_count": self.fail_open_count,
            "malformed_count": self.malformed_count,
            "non_tok_count": self.non_tok_count,
            "warning_signal_count": self.warning_signal_count,
            "local_failure_count": self.local_failure_count,
            "verdict": self.verdict,
            "notes": list(self.notes),
            "turns": [turn.to_dict() for turn in self.turns],
        }


@dataclass(frozen=True)
class FrontierCheckpointReport:
    checkpoint: FrontierCheckpoint
    benchmark_profiles: list[FrontierProfileSummary]
    openrouter_profiles: list[FrontierOpenRouterSummary]
    default_release_profile: str
    experimental_profiles: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint.to_dict(),
            "benchmark_profiles": [profile.to_dict() for profile in self.benchmark_profiles],
            "openrouter_profiles": [profile.to_dict() for profile in self.openrouter_profiles],
            "default_release_profile": self.default_release_profile,
            "experimental_profiles": list(self.experimental_profiles),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CompressionFrontierReport:
    model: str
    checkpoints: list[FrontierCheckpointReport]
    profiles: list[CompressionProfile]
    benchmarks: list[str]
    repeats: int
    openrouter_prompt: str
    openrouter_turn_sets: Sequence[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "profiles": [profile.to_dict() for profile in self.profiles],
            "benchmarks": list(self.benchmarks),
            "repeats": self.repeats,
            "openrouter_prompt": self.openrouter_prompt,
            "openrouter_turn_sets": list(self.openrouter_turn_sets),
        }


DEFAULT_FRONTIER_PROFILES = (
    CompressionProfile(
        name="conservative",
        mode="tok-tool-compatible",
        description="Highest-calm lane: natural-first tool-compatible request policy with extra recent turns and short recovery windows.",
        env={
            "TOK_MODE": "tool-compatible",
            "TOK_REQUEST_POLICY": "natural_first",
            "TOK_KEEP_TURNS": "4",
            "TOK_REQUEST_POLICY_STICKY_TURNS": "1",
            "TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS": "1",
            "TOK_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT": "1",
            "TOK_NEURO_REACTOR": "0",
        },
    ),
    CompressionProfile(
        name="balanced",
        mode="tok-tool-compatible",
        description="Bridge-default leaning lane with moderate history shaping.",
        env={
            "TOK_MODE": "tool-compatible",
            "TOK_REQUEST_POLICY": "natural_first",
            "TOK_KEEP_TURNS": "2",
            "TOK_REQUEST_POLICY_STICKY_TURNS": "2",
            "TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS": "2",
            "TOK_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT": "2",
            "TOK_NEURO_REACTOR": "0",
        },
    ),
    CompressionProfile(
        name="assertive",
        mode="tok-native",
        description="Lower retained history with native Tok shaping enabled.",
        env={
            "TOK_MODE": "tool-compatible",
            "TOK_REQUEST_POLICY": "natural_first",
            "TOK_KEEP_TURNS": "2",
            "TOK_REQUEST_POLICY_STICKY_TURNS": "3",
            "TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS": "2",
            "TOK_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT": "2",
            "TOK_NEURO_REACTOR": "0",
        },
    ),
    CompressionProfile(
        name="aggressive",
        mode="tok-native",
        description="Reduced recent-window retention and longer aggressive shaping windows.",
        env={
            "TOK_MODE": "tool-compatible",
            "TOK_REQUEST_POLICY": "natural_first",
            "TOK_KEEP_TURNS": "1",
            "TOK_REQUEST_POLICY_STICKY_TURNS": "4",
            "TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS": "3",
            "TOK_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT": "2",
            "TOK_NEURO_REACTOR": "0",
        },
    ),
    CompressionProfile(
        name="extreme",
        mode="tok-neuro",
        description="Experimental high-compression lane that turns on neuro-reactor behavior.",
        env={
            "TOK_MODE": "tool-compatible",
            "TOK_REQUEST_POLICY": "natural_first",
            "TOK_KEEP_TURNS": "1",
            "TOK_REQUEST_POLICY_STICKY_TURNS": "5",
            "TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS": "4",
            "TOK_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT": "3",
            "TOK_NEURO_REACTOR": "1",
        },
    ),
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 1)
    ordered = sorted(values)
    rank = max(
        0,
        min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1))),
    )
    return round(float(ordered[rank]), 1)


def _savings_pct(total_saved_tokens: int, provider_total_tokens: int) -> float:
    baseline_total = total_saved_tokens + provider_total_tokens
    if baseline_total <= 0:
        return 0.0
    return round((total_saved_tokens / baseline_total) * 100.0, 1)


@contextlib.contextmanager
def apply_frontier_env(
    overrides: dict[str, str],
) -> Generator[None, None, None]:
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def select_frontier_checkpoints(
    repo_root: Path,
    *,
    baseline_ref: str = DEFAULT_BASELINE_REF,
) -> list[FrontierCheckpoint]:
    head_ref = _git_stdout(repo_root, ["git", "rev-parse", "HEAD"]).strip()
    recent = _git_stdout(
        repo_root,
        ["git", "log", "--since=4 days ago", "--format=%H"],
    ).splitlines()

    chosen_third = ""
    for ref in recent:
        if ref and ref not in {head_ref, baseline_ref}:
            chosen_third = ref
    if not chosen_third:
        chosen_third = recent[-1] if recent else head_ref

    return [
        FrontierCheckpoint(label="current-head", ref="CURRENT"),
        FrontierCheckpoint(label="pre-natural-first", ref=baseline_ref),
        FrontierCheckpoint(label="pre-runtime-shaping", ref=chosen_third),
    ]


def _git_stdout(repo_root: Path, cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def _worker_script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "frontier_checkpoint_runner.py"


def _run_checkpoint_worker(
    *,
    repo_root: Path,
    checkpoint: FrontierCheckpoint,
    model: str,
    profile: CompressionProfile,
    benchmark: str,
    turns: int,
    repeats: int,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_options: dict[str, Any] | None,
    pricing: dict[str, float] | None,
    api_key: str | None,
    api_base: str | None,
) -> list[dict[str, Any]]:
    worker = _worker_script_path()
    with tempfile.TemporaryDirectory(prefix="tok_frontier_checkpoint_") as tmpdir:
        export_dir = Path(tmpdir) / "repo"
        export_dir.mkdir(parents=True, exist_ok=True)
        archive_cmd = f"git archive {checkpoint.ref} | tar -x -C {export_dir}"  # nosec B602
        subprocess.run(
            archive_cmd,
            cwd=repo_root,
            shell=True,  # nosec B602
            check=True,
            text=True,
        )
        payload = {
            "repo_root": str(export_dir),
            "benchmark": benchmark,
            "turns": turns,
            "repeats": repeats,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "provider_options": provider_options,
            "pricing": pricing,
            "api_key": api_key,
            "api_base": api_base,
            "profile": profile.to_dict(),
        }
        proc = subprocess.run(
            [sys.executable, str(worker)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            cwd=repo_root,
            check=False,
        )
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or "worker failed"
            msg = f"checkpoint worker failed for {checkpoint.ref} {benchmark}/{profile.name}: {detail}"
            raise RuntimeError(msg)
        data = json.loads(proc.stdout)
        return list(data.get("runs", []))


def _count_local_failures(notes: list[str]) -> int:
    return sum(1 for note in notes if note.startswith("error:"))


def classify_frontier_verdict(
    *,
    success_rate: float,
    recovery_attempts: int,
    recovery_holdovers: int,
    fail_open_count: int,
    malformed_count: int,
    non_tok_count: int,
    warning_signal_count: int,
    local_failure_count: int,
    max_pressure: int,
) -> str:
    if (
        success_rate < 1.0
        or recovery_holdovers > 0
        or fail_open_count > 0
        or malformed_count > 0
        or non_tok_count > 0
        or local_failure_count > 0
    ):
        return "degraded"
    if recovery_attempts > 0 or warning_signal_count > 0 or max_pressure > 2:
        return "watch"
    return "stable"


def _aggregate_benchmark_runs(
    *,
    benchmark: str,
    turns: int,
    repeats: int,
    mode: str,
    runs: list[dict[str, Any]],
) -> FrontierBenchmarkSummary:
    if not runs:
        return FrontierBenchmarkSummary(
            benchmark=benchmark,
            turns=turns,
            repeats=repeats,
            mode=mode,
            success_rate=0.0,
            avg_savings_pct=0.0,
            p95_savings_pct=0.0,
            avg_latency_ms=0.0,
            avg_pressure=0.0,
            max_pressure=0,
            recovery_attempt_count=0,
            recovery_success_count=0,
            recovery_fallback_count=0,
            recovery_holdover_count=0,
            fail_open_count=0,
            malformed_count=0,
            non_tok_count=0,
            warning_signal_count=0,
            local_failure_count=0,
            verdict="degraded",
            notes=["no_runs"],
        )

    savings_values: list[float] = []
    latency_values: list[float] = []
    pressure_values: list[int] = []
    success_count = 0
    recovery_attempt_count = 0
    recovery_success_count = 0
    recovery_fallback_count = 0
    recovery_holdover_count = 0
    fail_open_count = 0
    malformed_count = 0
    non_tok_count = 0
    warning_signal_count = 0
    local_failure_count = 0
    notes: list[str] = []

    for run in runs:
        result = run["candidate"]
        comparison = run["comparison"]
        total_saved = int(result["compression_metrics"].get("total_saved_tokens", 0))
        provider_total = int(result["provider_usage"].get("total_tokens", 0))
        savings_values.append(_savings_pct(total_saved, provider_total))
        latency_values.append(float(result["provider_usage"].get("latency_ms", 0.0)))
        pressure = int(result["response_metrics"].get("invisible_pressure", 0))
        pressure_values.append(pressure)
        if bool(result.get("task_success")):
            success_count += 1
        response_signals = result["response_metrics"].get("response_behavior_signals", {})
        diagnostics = result.get("diagnostics", {})
        recovery_attempt_count += int(
            response_signals.get("stream_recovery_started", 0) or response_signals.get("stream_recovery_retry", 0)
        )
        recovery_success_count += int(response_signals.get("stream_recovery_success_text", 0)) + int(
            response_signals.get("stream_recovery_success_tool_use", 0)
        )
        recovery_fallback_count += int(response_signals.get("stream_recovery_fallback", 0))
        recovery_holdover_count += int(
            result["compression_metrics"]["input_behavior_signals"].get("request_policy_held_by_recovery", 0)
        )
        fail_open_count += int(response_signals.get("fail_open_compat_response", 0))
        malformed_count += int(response_signals.get("malformed_tok_response", 0))
        non_tok_count += int(response_signals.get("non_tok_response", 0))
        warning_signal_count += int(diagnostics.get("response_warning_signal_count", 0))
        local_failure_count += _count_local_failures(list(result.get("notes", [])))
        if comparison.get("diagnosis"):
            notes.append(str(comparison["diagnosis"]))

    success_rate = round(success_count / len(runs), 3)
    avg_pressure = round(sum(pressure_values) / len(pressure_values), 2)
    avg_latency = round(sum(latency_values) / len(latency_values), 2)
    verdict = classify_frontier_verdict(
        success_rate=success_rate,
        recovery_attempts=recovery_attempt_count,
        recovery_holdovers=recovery_holdover_count,
        fail_open_count=fail_open_count,
        malformed_count=malformed_count,
        non_tok_count=non_tok_count,
        warning_signal_count=warning_signal_count,
        local_failure_count=local_failure_count,
        max_pressure=max(pressure_values) if pressure_values else 0,
    )

    return FrontierBenchmarkSummary(
        benchmark=benchmark,
        turns=turns,
        repeats=repeats,
        mode=mode,
        success_rate=success_rate,
        avg_savings_pct=round(sum(savings_values) / len(savings_values), 1),
        p95_savings_pct=_percentile(savings_values, 95),
        avg_latency_ms=avg_latency,
        avg_pressure=avg_pressure,
        max_pressure=max(pressure_values) if pressure_values else 0,
        recovery_attempt_count=recovery_attempt_count,
        recovery_success_count=recovery_success_count,
        recovery_fallback_count=recovery_fallback_count,
        recovery_holdover_count=recovery_holdover_count,
        fail_open_count=fail_open_count,
        malformed_count=malformed_count,
        non_tok_count=non_tok_count,
        warning_signal_count=warning_signal_count,
        local_failure_count=local_failure_count,
        verdict=verdict,
        notes=sorted(set(notes)),
    )


def _incompatible_benchmark_summary(
    *,
    benchmark: str,
    turns: int,
    repeats: int,
    mode: str,
    detail: str,
) -> FrontierBenchmarkSummary:
    return FrontierBenchmarkSummary(
        benchmark=benchmark,
        turns=turns,
        repeats=repeats,
        mode=mode,
        success_rate=0.0,
        avg_savings_pct=0.0,
        p95_savings_pct=0.0,
        avg_latency_ms=0.0,
        avg_pressure=0.0,
        max_pressure=0,
        recovery_attempt_count=0,
        recovery_success_count=0,
        recovery_fallback_count=0,
        recovery_holdover_count=0,
        fail_open_count=0,
        malformed_count=0,
        non_tok_count=0,
        warning_signal_count=0,
        local_failure_count=1,
        verdict="degraded",
        notes=[f"checkpoint_incompatible:{detail}"],
    )


def _profile_summary(
    profile: CompressionProfile,
    benchmark_summaries: list[FrontierBenchmarkSummary],
) -> FrontierProfileSummary:
    verdict = "stable"
    notes: list[str] = []
    for summary in benchmark_summaries:
        if summary.verdict == "degraded":
            verdict = "degraded"
        elif summary.verdict == "watch" and verdict == "stable":
            verdict = "watch"
        if summary.notes:
            notes.extend(summary.notes)
    stop_after = verdict == "degraded"
    return FrontierProfileSummary(
        profile=profile,
        benchmark_summaries=benchmark_summaries,
        verdict=verdict,
        stop_after=stop_after,
        notes=sorted(set(notes)),
    )


def _frontier_release_profile(
    benchmark_profiles: list[FrontierProfileSummary],
    _openrouter_profiles: list[FrontierOpenRouterSummary],
) -> tuple[str, list[str]]:
    stable_profiles = {profile.profile.name for profile in benchmark_profiles if profile.verdict == "stable"}
    if stable_profiles:
        ordered = [profile.name for profile in DEFAULT_FRONTIER_PROFILES if profile.name in stable_profiles]
        default_release = ordered[-1]
        experimental = [profile.name for profile in DEFAULT_FRONTIER_PROFILES if profile.name not in ordered]
        return default_release, experimental
    return "baseline", [profile.name for profile in DEFAULT_FRONTIER_PROFILES]


def run_benchmark_frontier_for_checkpoint(
    *,
    repo_root: Path,
    checkpoint: FrontierCheckpoint,
    profiles: list[CompressionProfile],
    benchmarks: list[str],
    model: str,
    repeats: int,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_options: dict[str, Any] | None = None,
    pricing: dict[str, float] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> list[FrontierProfileSummary]:
    benchmark_profiles: list[FrontierProfileSummary] = []
    benchmark_defs = [load_benchmark_definition(name) for name in benchmarks]

    for profile in profiles:
        benchmark_summaries: list[FrontierBenchmarkSummary] = []
        checkpoint_incompatible = False
        for definition in benchmark_defs:
            if checkpoint.ref == "CURRENT":
                runs = _run_current_checkpoint_benchmark(
                    definition=definition,
                    repeats=repeats,
                    model=model,
                    profile=profile,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    provider_options=provider_options,
                    pricing=pricing,
                    api_key=api_key,
                    api_base=api_base,
                )
            else:
                try:
                    runs = _run_checkpoint_worker(
                        repo_root=repo_root,
                        checkpoint=checkpoint,
                        model=model,
                        profile=profile,
                        benchmark=definition.name,
                        turns=definition.default_turns,
                        repeats=repeats,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        provider_options=provider_options,
                        pricing=pricing,
                        api_key=api_key,
                        api_base=api_base,
                    )
                except RuntimeError as exc:
                    checkpoint_incompatible = True
                    benchmark_summaries.append(
                        _incompatible_benchmark_summary(
                            benchmark=definition.name,
                            turns=definition.default_turns,
                            repeats=repeats,
                            mode=profile.mode,
                            detail=str(exc),
                        )
                    )
                    break
            if checkpoint_incompatible:
                break
            benchmark_summaries.append(
                _aggregate_benchmark_runs(
                    benchmark=definition.name,
                    turns=definition.default_turns,
                    repeats=repeats,
                    mode=profile.mode,
                    runs=runs,
                )
            )
        profile_summary = _profile_summary(profile, benchmark_summaries)
        benchmark_profiles.append(profile_summary)
        if profile_summary.stop_after or checkpoint_incompatible:
            break
    return benchmark_profiles


def _run_current_checkpoint_benchmark(
    *,
    definition: BenchmarkDefinition,
    repeats: int,
    model: str,
    profile: CompressionProfile,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_options: dict[str, Any] | None,
    pricing: dict[str, float] | None,
    api_key: str | None,
    api_base: str | None,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for _ in range(max(1, repeats)):
        runner = LiveBenchmarkRunner(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            pricing=pricing,
            provider_options=provider_options,
            api_key=api_key,
            api_base=api_base,
        )
        baseline = runner.run(definition, mode="baseline", turns=definition.default_turns)
        with apply_frontier_env(profile.env):
            candidate = runner.run(
                definition,
                mode=profile.mode,
                turns=definition.default_turns,
            )
        comparison = compare_results(baseline, candidate)
        runs.append(
            {
                "baseline": baseline.to_dict(),
                "candidate": candidate.to_dict(),
                "comparison": comparison.to_dict(),
            }
        )
    return runs


def _load_openai_client(base_url: str, api_key: str) -> object:
    import openai

    return openai.OpenAI(base_url=base_url, api_key=api_key)


def run_openrouter_probe(
    *,
    profile: CompressionProfile,
    model: str,
    prompt: str,
    turns: int,
    delay_seconds: float,
    base_url: str,
    api_key: str,
) -> FrontierOpenRouterSummary:
    import time

    import tok

    client = _load_openai_client(base_url=base_url, api_key=api_key)
    session = tok.RuntimeSession()
    turn_rows: list[FrontierOpenRouterTurn] = []
    total_failures = 0
    savings_values: list[float] = []
    response_signals: dict[str, int] = {}
    input_signals: dict[str, int] = {}

    with apply_frontier_env(profile.env):
        for index in range(turns):
            local_failure = ""
            turn_behavior: dict[str, int] = {}
            try:
                prepared = tok.wrap(
                    [{"role": "user", "content": f"{prompt} (turn {index})"}],
                    model=model,
                    session=session,
                    tool_compatible=profile.mode != "tok-native",
                )
                body_messages = (
                    [{"role": "system", "content": prepared.body["system"]}] if prepared.body.get("system") else []
                ) + prepared.body["messages"]
                response = cast("OpenAI", client).chat.completions.create(
                    model=model,
                    messages=body_messages,
                    temperature=0.2,
                    max_tokens=200,
                )
                text = response.choices[0].message.content or ""
                processed = tok.process(
                    text,
                    model=model,
                    session=session,
                    tool_compatible=profile.mode != "tok-native",
                )
                turn_behavior = dict(prepared.behavior_signals)
                for key, value in processed.behavior_signals.items():
                    turn_behavior[key] = turn_behavior.get(key, 0) + int(value)
                total_saved = int(prepared.input_saved_tokens) + int(processed.output_saved_tokens)
                provider_total = int(getattr(response.usage, "total_tokens", 0))
                savings_values.append(_savings_pct(total_saved, provider_total))
                turn_rows.append(
                    FrontierOpenRouterTurn(
                        turn=index,
                        input_saved_tokens=int(prepared.input_saved_tokens),
                        output_saved_tokens=int(processed.output_saved_tokens),
                        total_saved_tokens=total_saved,
                        text_preview=text.strip()[:120],
                        behavior_signals=turn_behavior,
                    )
                )
            except Exception as exc:
                total_failures += 1
                local_failure = f"error:{exc}"
                turn_rows.append(
                    FrontierOpenRouterTurn(
                        turn=index,
                        input_saved_tokens=0,
                        output_saved_tokens=0,
                        total_saved_tokens=0,
                        text_preview="",
                        behavior_signals={},
                        local_failure=local_failure,
                    )
                )
            for key, value in turn_behavior.items():
                if key.startswith(("stream_", "non_tok", "malformed", "fail_open")):
                    response_signals[key] = response_signals.get(key, 0) + int(value)
                else:
                    input_signals[key] = input_signals.get(key, 0) + int(value)
            if delay_seconds > 0 and index + 1 < turns:
                time.sleep(delay_seconds)

    success_rate = round((turns - total_failures) / max(1, turns), 3)
    recovery_attempt_count = int(response_signals.get("stream_recovery_started", 0)) + int(
        response_signals.get("stream_recovery_retry", 0)
    )
    recovery_success_count = int(response_signals.get("stream_recovery_success_text", 0)) + int(
        response_signals.get("stream_recovery_success_tool_use", 0)
    )
    recovery_fallback_count = int(response_signals.get("stream_recovery_fallback", 0))
    recovery_holdover_count = int(input_signals.get("request_policy_held_by_recovery", 0))
    fail_open_count = int(response_signals.get("fail_open_compat_response", 0))
    malformed_count = int(response_signals.get("malformed_tok_response", 0))
    non_tok_count = int(response_signals.get("non_tok_response", 0))
    warning_signal_count = fail_open_count + malformed_count + non_tok_count
    verdict = classify_frontier_verdict(
        success_rate=success_rate,
        recovery_attempts=recovery_attempt_count,
        recovery_holdovers=recovery_holdover_count,
        fail_open_count=fail_open_count,
        malformed_count=malformed_count,
        non_tok_count=non_tok_count,
        warning_signal_count=warning_signal_count,
        local_failure_count=total_failures,
        max_pressure=0,
    )
    return FrontierOpenRouterSummary(
        profile=profile.name,
        model=model,
        prompt=prompt,
        turns_requested=turns,
        turns_completed=turns - total_failures,
        success_rate=success_rate,
        avg_savings_pct=round(sum(savings_values) / len(savings_values), 1) if savings_values else 0.0,
        p95_savings_pct=_percentile(savings_values, 95),
        recovery_attempt_count=recovery_attempt_count,
        recovery_success_count=recovery_success_count,
        recovery_fallback_count=recovery_fallback_count,
        recovery_holdover_count=recovery_holdover_count,
        fail_open_count=fail_open_count,
        malformed_count=malformed_count,
        non_tok_count=non_tok_count,
        warning_signal_count=warning_signal_count,
        local_failure_count=total_failures,
        verdict=verdict,
        notes=[turn.local_failure for turn in turn_rows if turn.local_failure],
        turns=turn_rows,
    )


def render_frontier_markdown(report: CompressionFrontierReport) -> str:
    lines = [
        "# Compression Frontier Report",
        "",
        f"- Model: `{report.model}`",
        f"- Benchmarks: `{', '.join(report.benchmarks)}`",
        f"- Repeats: `{report.repeats}`",
        f"- OpenRouter prompt: `{report.openrouter_prompt}`",
        "",
    ]
    for checkpoint in report.checkpoints:
        lines.extend(
            [
                f"## {checkpoint.checkpoint.label}",
                "",
                f"- Ref: `{checkpoint.checkpoint.ref}`",
                f"- Release lane: `{checkpoint.default_release_profile}`",
                f"- Experimental profiles: `{', '.join(checkpoint.experimental_profiles) or 'none'}`",
                "",
                "### Benchmarks",
                "",
                "| Profile | Verdict | Avg savings | P95 savings | Avg pressure | Recovery attempts | Fail-open | Local failures |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for profile in checkpoint.benchmark_profiles:
            avg_savings = 0.0
            avg_pressure = 0.0
            recovery_attempts = 0
            fail_open = 0
            local_failures = 0
            if profile.benchmark_summaries:
                avg_savings = round(
                    statistics.mean(summary.avg_savings_pct for summary in profile.benchmark_summaries),
                    1,
                )
                avg_pressure = round(
                    statistics.mean(summary.avg_pressure for summary in profile.benchmark_summaries),
                    2,
                )
                recovery_attempts = sum(summary.recovery_attempt_count for summary in profile.benchmark_summaries)
                fail_open = sum(summary.fail_open_count for summary in profile.benchmark_summaries)
                local_failures = sum(summary.local_failure_count for summary in profile.benchmark_summaries)
            p95 = max(
                (summary.p95_savings_pct for summary in profile.benchmark_summaries),
                default=0.0,
            )
            lines.append(
                f"| {profile.profile.name} | {profile.verdict} | {avg_savings:.1f}% | {p95:.1f}% | {avg_pressure:.2f} | {recovery_attempts} | {fail_open} | {local_failures} |"
            )
        lines.extend(["", "### OpenRouter", ""])
        lines.append(
            "| Profile | Verdict | Success rate | Avg savings | P95 savings | Recovery attempts | Holdovers | Local failures |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for probe in checkpoint.openrouter_profiles:
            lines.append(
                f"| {probe.profile} | {probe.verdict} | {probe.success_rate:.3f} | {probe.avg_savings_pct:.1f}% | {probe.p95_savings_pct:.1f}% | {probe.recovery_attempt_count} | {probe.recovery_holdover_count} | {probe.local_failure_count} |"
            )
        if checkpoint.notes:
            lines.extend(["", "### Notes", ""])
            for note in checkpoint.notes:
                lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def load_frontier_report(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text()))


def check_frontier_report(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "passed": False,
    }
    if not path.exists():
        row["reason"] = "file not found"
        return row

    try:
        payload = load_frontier_report(path)
    except Exception as exc:
        row["reason"] = "invalid_json"
        row["error"] = str(exc)
        return row

    checkpoints = payload.get("checkpoints", [])
    current = next(
        (checkpoint for checkpoint in checkpoints if checkpoint.get("checkpoint", {}).get("label") == "current-head"),
        None,
    )
    if current is None:
        row["reason"] = "missing_current_head"
        return row

    release_profile = str(current.get("default_release_profile", "baseline"))
    benchmark_profiles = current.get("benchmark_profiles", [])
    stable_benchmarks = [
        profile.get("profile", {}).get("name", "")
        for profile in benchmark_profiles
        if profile.get("verdict") == "stable"
    ]
    openrouter_profiles = current.get("openrouter_profiles", [])
    release_probe = next(
        (probe for probe in openrouter_profiles if str(probe.get("profile", "")) == release_profile),
        None,
    )
    release_probe_verdict = str(release_probe.get("verdict", "")) if isinstance(release_probe, dict) else ""

    passed = bool(stable_benchmarks) and release_profile != "baseline"
    if openrouter_profiles:
        passed = passed and release_probe_verdict in {"stable", "watch"}

    row.update(
        {
            "checkpoint_ref": current.get("checkpoint", {}).get("ref", ""),
            "release_profile": release_profile,
            "stable_profiles": stable_benchmarks,
            "experimental_profiles": list(current.get("experimental_profiles", [])),
            "openrouter_probe_present": bool(openrouter_profiles),
            "release_probe_verdict": release_probe_verdict,
            "passed": passed,
        }
    )
    if not passed:
        if release_profile == "baseline":
            row["reason"] = "baseline_release_lane"
        elif not stable_benchmarks:
            row["reason"] = "no_stable_profile"
        elif openrouter_profiles and release_probe_verdict not in {
            "stable",
            "watch",
        }:
            row["reason"] = "frontier_probe_failed"
        else:
            row["reason"] = "criteria_failed"
    return row


def run_checkpoint_frontier(
    *,
    repo_root: Path,
    checkpoint: FrontierCheckpoint,
    profiles: list[CompressionProfile],
    benchmarks: list[str],
    model: str,
    repeats: int,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_options: dict[str, Any] | None = None,
    pricing: dict[str, float] | None = None,
    openrouter_prompt: str,
    openrouter_turn_sets: Sequence[int],
    openrouter_delay_seconds: float,
    openrouter_api_key: str | None = None,
    openrouter_api_base: str = "https://openrouter.ai/api/v1",
) -> FrontierCheckpointReport:
    benchmark_profiles = run_benchmark_frontier_for_checkpoint(
        repo_root=repo_root,
        checkpoint=checkpoint,
        profiles=profiles,
        benchmarks=benchmarks,
        model=model,
        repeats=repeats,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        provider_options=provider_options,
        pricing=pricing,
        api_key=openrouter_api_key,
        api_base=openrouter_api_base,
    )

    openrouter_profiles: list[FrontierOpenRouterSummary] = []
    if openrouter_api_key:
        for profile in profiles[: len(benchmark_profiles)]:
            profile_turns = max(openrouter_turn_sets) if openrouter_turn_sets else 0
            if profile_turns <= 0:
                continue
            if checkpoint.label != "current-head":
                # The checkpoint worker is benchmark-only for now; OpenRouter probes
                # stay on the current checkout to keep cheap iteration simple.
                break
            openrouter_profiles.append(
                run_openrouter_probe(
                    profile=profile,
                    model=model,
                    prompt=openrouter_prompt,
                    turns=profile_turns,
                    delay_seconds=openrouter_delay_seconds,
                    base_url=openrouter_api_base,
                    api_key=openrouter_api_key,
                )
            )

    default_release_profile, experimental_profiles = _frontier_release_profile(benchmark_profiles, openrouter_profiles)
    notes: list[str] = []
    if openrouter_profiles:
        notes.append("OpenRouter probes are advisory only; benchmark stability selects the release lane.")
    if benchmark_profiles and benchmark_profiles[0].verdict == "degraded" and checkpoint.label == "current-head":
        notes.append("Current head degrades on the first rung; simplify before pushing compression further.")

    return FrontierCheckpointReport(
        checkpoint=checkpoint,
        benchmark_profiles=benchmark_profiles,
        openrouter_profiles=openrouter_profiles,
        default_release_profile=default_release_profile,
        experimental_profiles=experimental_profiles,
        notes=notes,
    )


def run_frontier_report(
    *,
    repo_root: Path,
    checkpoints: list[FrontierCheckpoint],
    profiles: list[CompressionProfile],
    benchmarks: list[str],
    model: str,
    repeats: int,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_options: dict[str, Any] | None = None,
    pricing: dict[str, float] | None = None,
    openrouter_prompt: str = "Give me a one-line repo summary.",
    openrouter_turn_sets: Sequence[int] | None = None,
    openrouter_delay_seconds: float = 0.2,
    openrouter_api_key: str | None = None,
    openrouter_api_base: str = "https://openrouter.ai/api/v1",
) -> CompressionFrontierReport:
    if not openrouter_api_key:
        msg = "compression frontier requires OPENROUTER_API_KEY so the live benchmark and probe runs can execute"
        raise RuntimeError(msg)
    effective_turns = list(openrouter_turn_sets) if openrouter_turn_sets is not None else list(DEFAULT_OPENROUTER_TURNS)
    checkpoint_reports = [
        run_checkpoint_frontier(
            repo_root=repo_root,
            checkpoint=checkpoint,
            profiles=profiles,
            benchmarks=benchmarks,
            model=model,
            repeats=repeats,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            provider_options=provider_options,
            pricing=pricing,
            openrouter_prompt=openrouter_prompt,
            openrouter_turn_sets=effective_turns,
            openrouter_delay_seconds=openrouter_delay_seconds,
            openrouter_api_key=openrouter_api_key,
            openrouter_api_base=openrouter_api_base,
        )
        for checkpoint in checkpoints
    ]
    return CompressionFrontierReport(
        model=model,
        checkpoints=checkpoint_reports,
        profiles=profiles,
        benchmarks=benchmarks,
        repeats=repeats,
        openrouter_prompt=openrouter_prompt,
        openrouter_turn_sets=effective_turns,
    )
