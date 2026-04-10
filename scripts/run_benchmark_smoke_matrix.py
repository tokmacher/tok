#!/usr/bin/env python3

"""Run the benchmark smoke/public sweep for one or more models."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = ("deepseek/deepseek-v3.2",)
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

    steps: list[MatrixStep] = []
    if not args.skip_asset_verify:
        steps.append(_asset_verify_step(catalog_root))
    for model in models:
        steps.append(
            _smoke_step(
                model=model,
                mode=args.mode,
                output_root=output_root / _safe_model_name(model),
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
