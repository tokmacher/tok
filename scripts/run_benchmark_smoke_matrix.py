#!/usr/bin/env python3

"""Run the benchmark smoke/public sweep for one or more models."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tok.testing.benchmark_executor import FamilyEvaluator
from tok.testing.benchmark_suite import BenchmarkTaskManifest, load_benchmark_catalog

ROOT = Path(__file__).resolve().parent.parent
ENV_PRIVATE_EVALUATOR_ROOTS = ("TOK_PRIVATE_EVALUATOR_ROOT", "PRIVATE_EVALUATOR_ROOT")
DEFAULT_MODELS = ("deepseek/deepseek-v3.2",)
DISCOVERY_DIR_NAMES = (
    "private-evaluator-overlay",
    "private_evaluator_overlay",
    "benchmark-private-evaluator-overlay",
)
DISCOVERY_GLOBS = ("*private*evaluator*", "*evaluator*overlay*")
UV_RUN_PREFIX: tuple[str, ...] = ("uv", "run")


@dataclass(frozen=True)
class MatrixStep:
    name: str
    command: tuple[str, ...]


def _date_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def _default_output_root() -> Path:
    return ROOT / "tmp" / f"benchmark_smoke_multimodel_{_date_stamp()}"


def _safe_model_name(model: str) -> str:
    safe = []
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


def _public_execution_tasks(catalog_root: Path) -> tuple[BenchmarkTaskManifest, ...]:
    catalog = load_benchmark_catalog(catalog_root)
    return tuple(
        task
        for task in catalog.tasks
        if task.public_release and task.family == "execution_patch" and task.hidden_evaluator_ref()
    )


def _hidden_spec_has_runner(spec: dict[str, object]) -> bool:
    if str(spec.get("command") or "").strip():
        return True
    selectors = spec.get("selectors") or spec.get("hidden_tests") or ()
    return bool(tuple(str(item) for item in selectors))


def _overlay_is_usable(overlay_root: Path, *, catalog_root: Path) -> bool:
    tasks = _public_execution_tasks(catalog_root)
    evaluator = FamilyEvaluator(private_evaluator_root=overlay_root)
    try:
        evaluator.validate_private_overlay(tasks, require_private_overlay=True)
        for task in tasks:
            payload = evaluator._load_hidden_evaluator_spec(task)
            if not _hidden_spec_has_runner(payload):
                return False
    except RuntimeError:
        return False
    return True


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


def discover_private_evaluator_root(*, catalog_root: Path, explicit_root: Path | None = None) -> Path:
    searched: list[Path] = []
    for candidate in _candidate_overlay_dirs(explicit_root):
        searched.append(candidate)
        if not candidate.is_dir():
            continue
        if _overlay_is_usable(candidate, catalog_root=catalog_root):
            return candidate
    searched_text = "\n".join(f"- {path}" for path in searched) if searched else "- <none>"
    msg = (
        "Could not find a usable private evaluator overlay.\n"
        "Pass --private-evaluator-root explicitly or set TOK_PRIVATE_EVALUATOR_ROOT.\n"
        "Searched:\n"
        f"{searched_text}"
    )
    raise RuntimeError(msg)


def _run_step(step: MatrixStep, *, dry_run: bool) -> int:
    print(f"[matrix] {step.name}")
    print("  $ " + " ".join(step.command))
    if dry_run:
        return 0
    completed = subprocess.run(step.command, cwd=ROOT, check=False)
    return int(completed.returncode)


def _asset_verify_step(catalog_root: Path) -> MatrixStep:
    return MatrixStep(
        "Verify benchmark assets",
        (*UV_RUN_PREFIX, "python", "scripts/prepare_benchmark_assets.py", "--root", str(catalog_root), "verify"),
    )


def _smoke_step(
    *,
    model: str,
    mode: str,
    output_root: Path,
    private_evaluator_root: Path,
) -> MatrixStep:
    return MatrixStep(
        f"Run {mode} for {model}",
        (
            *UV_RUN_PREFIX,
            "python",
            "scripts/run_release_smoke.py",
            "--benchmark-mode",
            mode,
            "--benchmark-output",
            str(output_root),
            "--model",
            model,
            "--private-evaluator-root",
            str(private_evaluator_root),
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "public_full"),
        default="smoke",
        help="Benchmark mode passed to scripts/run_release_smoke.py",
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=ROOT / "benchmarks",
        help="Benchmark catalog root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help="Parent directory for per-model benchmark outputs",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model identifier to run; repeat to run multiple models",
    )
    parser.add_argument(
        "--private-evaluator-root",
        type=Path,
        default=None,
        help="Optional explicit private evaluator overlay root",
    )
    parser.add_argument(
        "--skip-asset-verify",
        action="store_true",
        help="Skip the benchmark asset verification preflight",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to later models if one model fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalog_root = args.catalog_root.resolve()
    output_root = args.output_root.resolve()
    models = tuple(args.model) or DEFAULT_MODELS
    private_evaluator_root = discover_private_evaluator_root(
        catalog_root=catalog_root,
        explicit_root=args.private_evaluator_root,
    )
    print(f"[matrix] Using private evaluator overlay: {private_evaluator_root}")

    steps: list[MatrixStep] = []
    if not args.skip_asset_verify:
        steps.append(_asset_verify_step(catalog_root))
    for model in models:
        steps.append(
            _smoke_step(
                model=model,
                mode=args.mode,
                output_root=output_root / _safe_model_name(model),
                private_evaluator_root=private_evaluator_root,
            )
        )

    had_failure = False
    for step in steps:
        exit_code = _run_step(step, dry_run=bool(args.dry_run))
        if exit_code != 0:
            had_failure = True
            if not args.keep_going:
                return exit_code
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
