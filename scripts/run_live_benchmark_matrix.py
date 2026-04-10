#!/usr/bin/env python3

"""Run the existing live-benchmark loop across a model and benchmark matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _replay_fingerprint_path(output_dir: Path, benchmark: str) -> Path:
    return output_dir / f"{benchmark}_fingerprint.json"


def _catalog_fingerprint_path(output_dir: Path) -> Path:
    return output_dir / "catalog" / "fingerprint.json"


def _safe_git_commit(cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return (completed.stdout or "").strip() or "unknown"


def _load_fingerprint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _fingerprints_match(expected: dict[str, Any], observed: dict[str, Any] | None) -> bool:
    if observed is None:
        return False
    return observed == expected


def _normalized_provider_options(raw_value: str | None) -> dict[str, Any]:
    if not raw_value:
        return {}
    payload = json.loads(raw_value)
    if not isinstance(payload, dict):
        msg = "--provider-options must decode to a JSON object"
        raise ValueError(msg)
    return payload


def _build_fingerprint(
    *,
    kind: str,
    model: str,
    repeats: int | None,
    benchmark: str | None,
    catalog_root: Path,
    pricing_prompt: float | None,
    pricing_completion: float | None,
    provider_options: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": "1",
        "kind": kind,
        "git_commit": _safe_git_commit(ROOT),
        "model": model,
        "benchmark": benchmark or "",
        "repeats": repeats,
        "catalog_root": str(catalog_root),
        "pricing_profile": {
            "prompt": pricing_prompt,
            "completion": pricing_completion,
        },
        "provider_options": provider_options,
    }


def _preflight_catalog_root(*, suite: str, catalog_root: Path) -> tuple[bool, str]:
    if suite not in {"catalog", "both"}:
        return True, ""
    lanes_dir = catalog_root / "lanes"
    if not lanes_dir.exists():
        return False, f"benchmark lanes directory not found: {lanes_dir}"
    return True, ""


def _build_step(
    *,
    benchmark: str,
    model: str,
    output_dir: Path,
    repeats: int | None,
    pricing_prompt: float | None,
    pricing_completion: float | None,
    provider_options: str | None,
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
    if pricing_prompt is not None:
        command.extend(["--pricing-prompt", str(pricing_prompt)])
    if pricing_completion is not None:
        command.extend(["--pricing-completion", str(pricing_completion)])
    if provider_options:
        command.extend(["--provider-options", provider_options])
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
    pricing_prompt: float | None,
    pricing_completion: float | None,
    provider_options: str | None,
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
    if pricing_prompt is not None:
        command.extend(["--pricing-prompt", str(pricing_prompt)])
    if pricing_completion is not None:
        command.extend(["--pricing-completion", str(pricing_completion)])
    if provider_options:
        command.extend(["--provider-options", provider_options])
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
        default=5,
        help="Repeat count passed through to tok dev live-benchmark (release-grade default: 5)",
    )
    parser.add_argument(
        "--pricing-prompt",
        type=float,
        default=None,
        help="Prompt token price per 1M tokens (USD)",
    )
    parser.add_argument(
        "--pricing-completion",
        type=float,
        default=None,
        help="Completion token price per 1M tokens (USD)",
    )
    parser.add_argument(
        "--provider-options",
        default=None,
        help="JSON provider options passed through to tok dev live-benchmark",
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
    suite = "replay" if args.suite == "legacy" else args.suite
    catalog_root = args.catalog_root.resolve()
    preflight_ok, preflight_reason = _preflight_catalog_root(suite=suite, catalog_root=catalog_root)
    if not preflight_ok:
        print(f"[matrix] benchmark preflight failed: {preflight_reason}")
        return 2

    try:
        provider_options_dict = _normalized_provider_options(args.provider_options)
    except ValueError as exc:
        print(f"[matrix] {exc}")
        return 2

    output_root = args.output_root.resolve()
    models = tuple(args.model) or DEFAULT_MODELS
    benchmarks = _selected_benchmarks(profile=args.profile, explicit=tuple(args.benchmark))
    had_failure = False

    for model in models:
        model_output = output_root / _safe_model_name(model)
        model_output.mkdir(parents=True, exist_ok=True)
        if suite in {"replay", "both"}:
            for benchmark in benchmarks:
                triage_path = _triage_path(model_output, benchmark)
                replay_fingerprint = _build_fingerprint(
                    kind="replay",
                    model=model,
                    repeats=args.repeats,
                    benchmark=benchmark,
                    catalog_root=catalog_root,
                    pricing_prompt=args.pricing_prompt,
                    pricing_completion=args.pricing_completion,
                    provider_options=provider_options_dict,
                )
                fingerprint_path = _replay_fingerprint_path(model_output, benchmark)
                if (
                    triage_path.exists()
                    and not args.force
                    and _fingerprints_match(replay_fingerprint, _load_fingerprint(fingerprint_path))
                ):
                    print(f"[matrix] skipping {model} :: {benchmark} (fresh artifact + fingerprint)")
                    continue
                exit_code = _run_step(
                    _build_step(
                        benchmark=benchmark,
                        model=model,
                        output_dir=model_output,
                        repeats=args.repeats,
                        pricing_prompt=args.pricing_prompt,
                        pricing_completion=args.pricing_completion,
                        provider_options=args.provider_options,
                    ),
                    dry_run=bool(args.dry_run),
                )
                if exit_code != 0:
                    had_failure = True
                    if not args.keep_going:
                        return exit_code
                if not args.dry_run:
                    if not triage_path.exists():
                        print(f"[matrix] missing required artifact: {triage_path}")
                        had_failure = True
                        if not args.keep_going:
                            return 3
                    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
                    fingerprint_path.write_text(json.dumps(replay_fingerprint, indent=2))
        if suite in {"catalog", "both"}:
            report_path = _catalog_report_path(model_output)
            catalog_fingerprint = _build_fingerprint(
                kind="catalog",
                model=model,
                repeats=args.repeats,
                benchmark=None,
                catalog_root=catalog_root,
                pricing_prompt=args.pricing_prompt,
                pricing_completion=args.pricing_completion,
                provider_options=provider_options_dict,
            )
            catalog_fingerprint_path = _catalog_fingerprint_path(model_output)
            if (
                report_path.exists()
                and not args.force
                and _fingerprints_match(catalog_fingerprint, _load_fingerprint(catalog_fingerprint_path))
            ):
                print(f"[matrix] skipping {model} :: catalog (fresh artifact + fingerprint)")
            else:
                exit_code = _run_step(
                    _build_catalog_step(
                        model=model,
                        output_dir=model_output,
                        catalog_root=catalog_root,
                        repeats=args.repeats,
                        pricing_prompt=args.pricing_prompt,
                        pricing_completion=args.pricing_completion,
                        provider_options=args.provider_options,
                    ),
                    dry_run=bool(args.dry_run),
                )
                if exit_code != 0:
                    had_failure = True
                    if not args.keep_going:
                        return exit_code
                if not args.dry_run:
                    if not report_path.exists():
                        print(f"[matrix] missing required artifact: {report_path}")
                        had_failure = True
                        if not args.keep_going:
                            return 3
                    catalog_fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
                    catalog_fingerprint_path.write_text(json.dumps(catalog_fingerprint, indent=2))

    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
