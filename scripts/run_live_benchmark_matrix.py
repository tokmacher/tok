#!/usr/bin/env python3

"""Run the existing live-benchmark loop across a model and benchmark matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tok.testing.benchmark_executor import FamilyEvaluator
from tok.testing.benchmark_suite import load_benchmark_catalog

ROOT = Path(__file__).resolve().parent.parent
UV_RUN_PREFIX: tuple[str, ...] = ("uv", "run")
DEFAULT_MODELS: tuple[str, ...] = ("deepseek/deepseek-v3.2",)
ENV_PRIVATE_EVALUATOR_ROOTS = ("TOK_PRIVATE_EVALUATOR_ROOT", "PRIVATE_EVALUATOR_ROOT")
DISCOVERY_DIR_NAMES = (
    "private-evaluator-overlay",
    "private_evaluator_overlay",
    "benchmark-private-evaluator-overlay",
)
DISCOVERY_GLOBS = ("*private*evaluator*", "*evaluator*overlay*")
DEFAULT_PRIVATE_EVALUATOR_FILENAMES = ("evaluator.json",)
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


def _search_roots() -> tuple[Path, ...]:
    home = Path.home()
    roots = (
        ROOT,
        ROOT.parent,
        home,
        home / "Desktop",
        home / "code",
        home / "src",
    )
    return tuple(root.resolve() for root in roots if root.exists())


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return tuple(unique)


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


def _public_execution_hidden_refs(catalog_root: Path) -> tuple[str, ...]:
    catalog = load_benchmark_catalog(catalog_root)
    refs: list[str] = []
    for task in catalog.tasks:
        hidden_ref = task.hidden_evaluator_ref()
        if task.public_release and task.family == "execution_patch" and hidden_ref:
            refs.append(hidden_ref)
    return tuple(refs)


def _public_execution_tasks(catalog_root: Path) -> tuple[object, ...]:
    catalog = load_benchmark_catalog(catalog_root)
    return tuple(
        task
        for task in catalog.tasks
        if task.public_release and task.family == "execution_patch" and task.hidden_evaluator_ref()
    )


def _candidate_overlay_dirs(explicit_root: Path | None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root)
    for env_name in ENV_PRIVATE_EVALUATOR_ROOTS:
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value))
    for root in _search_roots():
        for name in DISCOVERY_DIR_NAMES:
            candidates.append(root / name)
        for pattern in DISCOVERY_GLOBS:
            candidates.extend(sorted(path for path in root.glob(pattern) if path.is_dir()))
    return _unique_paths(candidates)


def _hidden_spec_has_runner(spec: dict[str, object]) -> bool:
    if str(spec.get("command") or "").strip():
        return True
    selectors = spec.get("selectors") or spec.get("hidden_tests") or ()
    return bool(tuple(str(item) for item in selectors))


def _overlay_is_usable(overlay_root: Path, *, catalog_root: Path) -> bool:
    tasks = _public_execution_tasks(catalog_root)
    if not tasks:
        return False
    evaluator = FamilyEvaluator(private_evaluator_root=overlay_root)
    try:
        evaluator.validate_private_overlay(tasks, require_private_overlay=True)
        for task in tasks:
            payload = evaluator._load_hidden_evaluator_spec(task)
            if not _hidden_spec_has_runner(payload):
                return False
    except (RuntimeError, json.JSONDecodeError):
        return False
    return True


def _overlay_search_paths(*, explicit_root: Path | None = None) -> tuple[Path, ...]:
    return _candidate_overlay_dirs(explicit_root)


def _overlay_search_message(*, explicit_root: Path | None = None) -> str:
    searched = _overlay_search_paths(explicit_root=explicit_root)
    searched_text = "\n".join(f"- {path}" for path in searched) if searched else "- <none>"
    return (
        "No usable private evaluator overlay was found for the public execution_patch tasks.\n"
        "Set TOK_PRIVATE_EVALUATOR_ROOT, pass --private-evaluator-root, or use "
        "--catalog-profile grounding-only for a debug-only subset.\n"
        "Searched:\n"
        f"{searched_text}"
    )


def _overlay_has_hidden_specs(overlay_root: Path, *, catalog_root: Path) -> bool:
    hidden_refs = _public_execution_hidden_refs(catalog_root)
    if not hidden_refs:
        return False
    if _overlay_is_usable(overlay_root, catalog_root=catalog_root):
        return True
    for hidden_ref in hidden_refs:
        candidates = [
            overlay_root / f"{hidden_ref}.json",
            overlay_root / hidden_ref,
            *(overlay_root / hidden_ref / name for name in DEFAULT_PRIVATE_EVALUATOR_FILENAMES),
        ]
        if not any(candidate.is_file() for candidate in candidates):
            return False
    return True


def discover_private_evaluator_root(*, catalog_root: Path, explicit_root: Path | None = None) -> Path | None:
    for candidate in _candidate_overlay_dirs(explicit_root):
        if candidate.is_dir() and _overlay_has_hidden_specs(candidate, catalog_root=catalog_root):
            return candidate
    return None


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
        "legacy",
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
    catalog_profile: str,
    private_evaluator_root: Path | None,
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
    ]
    if repeats is not None:
        command.extend(["--repeats", str(repeats)])
    if catalog_profile == "public":
        command.append("--public-release-only")
        if private_evaluator_root is not None:
            command.extend(["--private-evaluator-root", str(private_evaluator_root)])
    elif catalog_profile == "grounding-only":
        command.extend(["--family", "repo_grounding", "--public-release-only"])
    else:
        raise ValueError(f"unsupported catalog profile: {catalog_profile}")
    return MatrixStep(
        name=f"{model} :: catalog ({catalog_profile})",
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
        choices=("legacy", "catalog", "both"),
        default="both",
        help="Run only the older legacy probes, only the new catalog benchmarks, or both",
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
        help="Legacy benchmark name to run; repeat to select a custom set",
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
        "--catalog-profile",
        choices=("auto", "public", "grounding-only", "none"),
        default="public",
        help="How to include the new production catalog benchmarks",
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=ROOT / "benchmarks",
        help="Benchmark catalog root for the new production suite",
    )
    parser.add_argument(
        "--private-evaluator-root",
        type=Path,
        default=None,
        help="Optional explicit private evaluator overlay root for public execution_patch tasks",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run benchmarks even when legacy triage or catalog report artifacts already exist",
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
    private_evaluator_root: Path | None = None
    if args.catalog_profile in {"auto", "public"}:
        private_evaluator_root = discover_private_evaluator_root(
            catalog_root=catalog_root,
            explicit_root=args.private_evaluator_root,
        )
    if args.catalog_profile == "auto":
        effective_catalog_profile = "public" if private_evaluator_root is not None else "grounding-only"
    elif args.catalog_profile == "public":
        if private_evaluator_root is None:
            raise SystemExit(_overlay_search_message(explicit_root=args.private_evaluator_root))
        effective_catalog_profile = "public"
    else:
        effective_catalog_profile = args.catalog_profile
    if effective_catalog_profile == "public":
        print(f"[matrix] using full public catalog benchmarks with overlay: {private_evaluator_root}")
    elif effective_catalog_profile == "grounding-only":
        print("[matrix] no usable private evaluator overlay found; running public repo-grounding catalog tasks only")
    had_failure = False

    for model in models:
        model_output = output_root / _safe_model_name(model)
        model_output.mkdir(parents=True, exist_ok=True)
        if args.suite in {"legacy", "both"}:
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
        if args.suite in {"catalog", "both"} and effective_catalog_profile != "none":
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
                        catalog_profile=effective_catalog_profile,
                        private_evaluator_root=private_evaluator_root,
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
