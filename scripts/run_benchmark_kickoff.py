#!/usr/bin/env python3

"""Run the benchmark kickoff sequence for local smoke and supplemental diagnostics."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UV_RUN_PREFIX: tuple[str, ...] = ("uv", "run")
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
MOVING_PRIVATE_REFS = {"HEAD", "head", "main", "master", "trunk"}
SUPPLEMENTAL_TASKS: tuple[str, ...] = (
    "exec.tok.bridge-canonicalization",
    "exec.tok.first-exact-search",
    "session.context-reacquisition.answer",
    "session.response-contract.patch",
    "session.fail-open.answer",
    "session.answer-anchor.patch",
)


@dataclass(frozen=True)
class KickoffStep:
    name: str
    command: tuple[str, ...]


def _default_date_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def _validate_private_evaluator_ref(ref: str) -> str:
    value = ref.strip()
    if not value:
        raise ValueError("private evaluator ref must be set to a pinned tag or commit")
    if value in MOVING_PRIVATE_REFS or value.startswith("refs/heads/"):
        raise ValueError("private evaluator ref must be a pinned tag or commit, not a moving branch ref")
    return value


def _require_existing_dir(path: Path | None, *, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is required")
    resolved = path.resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"{label} must be an existing directory: {resolved}")
    return resolved


def _run_step(step: KickoffStep) -> int:
    print(f"[kickoff] {step.name}")
    print("  $ " + " ".join(step.command))
    completed = subprocess.run(step.command, cwd=ROOT, check=False)
    return int(completed.returncode)


def _preflight_step() -> KickoffStep:
    return KickoffStep(
        "Verify benchmark assets",
        (*UV_RUN_PREFIX, "python", "scripts/prepare_benchmark_assets.py", "--root", "benchmarks", "verify"),
    )


def _local_smoke_step(*, output_root: Path, model: str, private_evaluator_root: Path) -> KickoffStep:
    return KickoffStep(
        "Run local smoke shakedown",
        (
            *UV_RUN_PREFIX,
            "python",
            "scripts/run_release_smoke.py",
            "--benchmark-mode",
            "smoke",
            "--benchmark-output",
            str(output_root),
            "--model",
            model,
            "--private-evaluator-root",
            str(private_evaluator_root),
        ),
    )


def _supplemental_step(*, output_root: Path, model: str, private_evaluator_root: Path) -> KickoffStep:
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
        "--output",
        str(output_root),
        "--include-advisory",
        "--private-evaluator-root",
        str(private_evaluator_root),
    ]
    for task_id in SUPPLEMENTAL_TASKS:
        command.extend(["--task", task_id])
    return KickoffStep("Run supplemental diagnostics", tuple(command))


def _render_workflow_dispatch_markdown(
    *,
    date_stamp: str,
    model: str,
    private_evaluator_ref: str,
    smoke_output_root: Path,
    supplemental_output_root: Path,
) -> str:
    return "\n".join(
        [
            f"# Benchmark Kickoff Handoff ({date_stamp})",
            "",
            "## Local Outputs",
            "",
            f"- Smoke output: `{smoke_output_root}`",
            f"- Supplemental output: `{supplemental_output_root}`",
            "",
            "## Manual Workflow Dispatch",
            "",
            "Workflow: `.github/workflows/benchmark-smoke.yml`",
            "",
            "Smoke:",
            "- `mode=smoke`",
            f"- `model={model}`",
            f"- `private-evaluator-ref={private_evaluator_ref}`",
            "",
            "Public full:",
            "- `mode=public_full`",
            f"- `model={model}`",
            f"- `private-evaluator-ref={private_evaluator_ref}`",
            "",
            "Example `gh` commands:",
            "```bash",
            "gh workflow run benchmark-smoke.yml \\",
            "  -f mode=smoke \\",
            f"  -f model={model} \\",
            f"  -f private-evaluator-ref={private_evaluator_ref}",
            "",
            "gh workflow run benchmark-smoke.yml \\",
            "  -f mode=public_full \\",
            f"  -f model={model} \\",
            f"  -f private-evaluator-ref={private_evaluator_ref}",
            "```",
            "",
            "## Review Rules",
            "",
            "- Release gating comes from `catalog/report.json` and `catalog/report.md`.",
            "- Only the `Public Production Lane` section is release-relevant.",
            "- `consistency_gate_passed` and `public_claim_allowed` must both be `true`.",
            "- Ignore supplemental/internal sections for headline or public claims.",
            "",
        ]
    )


def _write_handoff_file(
    *,
    output_root: Path,
    date_stamp: str,
    model: str,
    private_evaluator_ref: str,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_output_root = output_root / f"{date_stamp}-local-smoke"
    supplemental_output_root = output_root / f"{date_stamp}-local-supplemental"
    handoff_path = output_root / f"{date_stamp}-workflow-dispatch.md"
    handoff_path.write_text(
        _render_workflow_dispatch_markdown(
            date_stamp=date_stamp,
            model=model,
            private_evaluator_ref=private_evaluator_ref,
            smoke_output_root=smoke_output_root,
            supplemental_output_root=supplemental_output_root,
        )
    )
    print(f"[kickoff] wrote hosted handoff instructions: {handoff_path}")
    return handoff_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("kickoff", "supplemental", "print-hosted"),
        default="kickoff",
        help="Kickoff local smoke, run supplemental diagnostics, or only render hosted workflow inputs",
    )
    parser.add_argument(
        "--date-stamp",
        default=_default_date_stamp(),
        help="Date stamp used for dated output directories (YYYYMMDD)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "tmp" / "benchmark-smoke",
        help="Parent directory for dated kickoff artifacts",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model identifier to use for kickoff benchmark commands",
    )
    parser.add_argument(
        "--private-evaluator-root",
        type=Path,
        default=None,
        help="Local private evaluator overlay directory for reportable execution tasks",
    )
    parser.add_argument(
        "--private-evaluator-ref",
        default=None,
        help="Pinned tag or commit for the private evaluator overlay repo used by the hosted workflow",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    smoke_output_root = output_root / f"{args.date_stamp}-local-smoke"
    supplemental_output_root = output_root / f"{args.date_stamp}-local-supplemental"

    try:
        if args.phase in {"kickoff", "supplemental"}:
            private_evaluator_root = _require_existing_dir(
                args.private_evaluator_root,
                label="--private-evaluator-root",
            )
        else:
            private_evaluator_root = None

        if args.phase in {"kickoff", "print-hosted"}:
            if args.private_evaluator_ref is None:
                raise ValueError("--private-evaluator-ref is required for hosted workflow handoff")
            private_evaluator_ref = _validate_private_evaluator_ref(args.private_evaluator_ref)
        else:
            private_evaluator_ref = None
    except ValueError as exc:
        parser.error(str(exc))

    if args.phase == "print-hosted":
        _write_handoff_file(
            output_root=output_root,
            date_stamp=args.date_stamp,
            model=args.model,
            private_evaluator_ref=private_evaluator_ref,
        )
        return 0

    steps = [_preflight_step()]
    if args.phase == "kickoff":
        steps.append(
            _local_smoke_step(
                output_root=smoke_output_root,
                model=args.model,
                private_evaluator_root=private_evaluator_root,
            )
        )
    elif args.phase == "supplemental":
        steps.append(
            _supplemental_step(
                output_root=supplemental_output_root,
                model=args.model,
                private_evaluator_root=private_evaluator_root,
            )
        )

    for step in steps:
        exit_code = _run_step(step)
        if exit_code != 0:
            return exit_code

    if args.phase == "kickoff":
        _write_handoff_file(
            output_root=output_root,
            date_stamp=args.date_stamp,
            model=args.model,
            private_evaluator_ref=private_evaluator_ref,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
