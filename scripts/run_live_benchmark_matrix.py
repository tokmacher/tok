#!/usr/bin/env python3

"""Run the existing live-benchmark loop across a model and benchmark matrix."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UV_RUN_PREFIX: tuple[str, ...] = ("uv", "run")
DEFAULT_MODELS: tuple[str, ...] = ("deepseek/deepseek-v3.2",)
TARGETED_BENCHMARKS: tuple[str, ...] = (
    "coding-loop",
    "research-loop-15",
    "research-loop-25",
    "jit-loop",
    "grammar_drift",
)
FULL_BENCHMARKS: tuple[str, ...] = (
    "coding-loop",
    "coding-loop-5",
    "coding-loop-8",
    "coding-loop-15",
    "coding-loop-25",
    "research-loop",
    "research-loop-current",
    "research-loop-5",
    "research-loop-8",
    "research-loop-15",
    "research-loop-25",
    "neuro-loop",
    "jit-loop",
    "grammar_drift",
)


@dataclass(frozen=True)
class MatrixStep:
    name: str
    command: tuple[str, ...]


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return ROOT / "tmp" / f"live_benchmark_matrix_{stamp}"


def _safe_model_name(model: str) -> str:
    safe: list[str] = []
    for char in model:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe)


def _selected_benchmarks(*, profile: str, explicit: tuple[str, ...]) -> tuple[str, ...]:
    if explicit:
        return explicit
    if profile == "full":
        return FULL_BENCHMARKS
    return TARGETED_BENCHMARKS


def _triage_path(output_dir: Path, benchmark: str) -> Path:
    return output_dir / f"{benchmark}_triage.json"


def _catalog_report_path(output_dir: Path) -> Path:
    return output_dir / "catalog" / "report.json"


def _build_step(
    *,
    benchmark: str,
    model: str,
    output_dir: Path,
    repeats: int | None,
) -> MatrixStep:
    command: list[str] = [
        *UV_RUN_PREFIX,
        "tok",
        "dev",
        "live-benchmark",
        "--program",
        "replay",
        "--benchmark",
        benchmark,
        "--mode",
        "compare",
        "--model",
        model,
        "--output",
        str(output_dir),
    ]
    if repeats is not None:
        command.extend(["--repeats", str(repeats)])
    return MatrixStep(
        name=f"{model} :: {benchmark}",
        command=tuple(command),
    )


def _build_catalog_step(
    *,
    model: str,
    output_dir: Path,
    catalog_root: Path,
    repeats: int | None,
) -> MatrixStep:
    command: list[str] = [
        *UV_RUN_PREFIX,
        "tok",
        "dev",
        "live-benchmark",
        "--program",
        "catalog",
        "--mode",
        "compare",
        "--model",
        model,
        "--catalog-root",
        str(catalog_root),
        "--output",
        str(output_dir / "catalog"),
        "--public-release-only",
        "--family",
        "execution_patch",
        "--family",
        "repo_grounding",
    ]
    if repeats is not None:
        command.extend(["--repeats", str(repeats)])
    return MatrixStep(
        name=f"{model} :: catalog",
        command=tuple(command),
    )


def _run_step(step: MatrixStep, *, dry_run: bool) -> int:
    print(f"[matrix] {step.name}")
    print("  $ " + " ".join(step.command))
    if dry_run:
        return 0
    completed = subprocess.run(step.command, cwd=ROOT, check=False)
    return int(completed.returncode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("replay", "catalog", "both", "legacy"),
        default="both",
        help="Run only the replay probes, only the catalog benchmarks, or both",
    )
    parser.add_argument(
        "--profile",
        choices=("targeted", "full"),
        default="targeted",
        help="Benchmark set to run when --benchmark is not specified",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Replay benchmark name to run; repeat to select a custom set",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model identifier to run; repeat to run multiple models",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help="Parent directory for per-model benchmark outputs",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Optional repeat count passed through to tok dev live-benchmark",
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=ROOT / "benchmarks",
        help="Benchmark catalog root for the new production suite",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run benchmarks even when replay triage or catalog report artifacts already exist",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to later benchmarks if one run fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    models = tuple(args.model) or DEFAULT_MODELS
    benchmarks = _selected_benchmarks(profile=args.profile, explicit=tuple(args.benchmark))
    catalog_root = args.catalog_root.resolve()
    had_failure = False

    suite = "replay" if args.suite == "legacy" else args.suite

    for model in models:
        model_output = output_root / _safe_model_name(model)
        model_output.mkdir(parents=True, exist_ok=True)
        if suite in {"replay", "both"}:
            for benchmark in benchmarks:
                triage_path = _triage_path(model_output, benchmark)
                if triage_path.exists() and not args.force:
                    print(f"[matrix] skipping {model} :: {benchmark} (found {triage_path})")
                    continue
                exit_code = _run_step(
                    _build_step(
                        benchmark=benchmark,
                        model=model,
                        output_dir=model_output,
                        repeats=args.repeats,
                    ),
                    dry_run=bool(args.dry_run),
                )
                if exit_code != 0:
                    had_failure = True
                    if not args.keep_going:
                        return exit_code
        if suite in {"catalog", "both"}:
            report_path = _catalog_report_path(model_output)
            if report_path.exists() and not args.force:
                print(f"[matrix] skipping {model} :: catalog (found {report_path})")
            else:
                exit_code = _run_step(
                    _build_catalog_step(
                        model=model,
                        output_dir=model_output,
                        catalog_root=catalog_root,
                        repeats=args.repeats,
                    ),
                    dry_run=bool(args.dry_run),
                )
                if exit_code != 0:
                    had_failure = True
                    if not args.keep_going:
                        return exit_code

    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
