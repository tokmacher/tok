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


def _run_step(step: KickoffStep) -> int:
    print(f"[kickoff] {step.name}")
    print("  $ " + " ".join(step.command))
    completed = subprocess.run(step.command, cwd=ROOT, check=False)
    return int(completed.returncode)


def _preflight_step(*, catalog_root: Path) -> KickoffStep:
    return KickoffStep(
        "Verify benchmark assets",
        (*UV_RUN_PREFIX, "python", "scripts/prepare_benchmark_assets.py", "--root", str(catalog_root), "verify"),
    )


def _local_smoke_step(
    *,
    output_root: Path,
    model: str,
    catalog_root: Path,
    repeats: int,
    pricing_prompt: float | None,
    pricing_completion: float | None,
    provider_options: str | None,
) -> KickoffStep:
    command: list[str] = [
        *UV_RUN_PREFIX,
        "python",
        "scripts/run_release_smoke.py",
        "--benchmark-mode",
        "smoke",
        "--benchmark-output",
        str(output_root),
        "--model",
        model,
        "--catalog-root",
        str(catalog_root),
        "--repeats",
        str(max(1, repeats)),
    ]
    if pricing_prompt is not None:
        command.extend(["--pricing-prompt", str(pricing_prompt)])
    if pricing_completion is not None:
        command.extend(["--pricing-completion", str(pricing_completion)])
    if provider_options:
        command.extend(["--provider-options", provider_options])
    return KickoffStep(
        "Run local smoke shakedown",
        tuple(command),
    )


def _supplemental_step(
    *,
    output_root: Path,
    model: str,
    catalog_root: Path,
    repeats: int,
    pricing_prompt: float | None,
    pricing_completion: float | None,
    provider_options: str | None,
) -> KickoffStep:
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
        str(output_root),
        "--include-advisory",
        "--repeats",
        str(max(1, repeats)),
    ]
    if pricing_prompt is not None:
        command.extend(["--pricing-prompt", str(pricing_prompt)])
    if pricing_completion is not None:
        command.extend(["--pricing-completion", str(pricing_completion)])
    if provider_options:
        command.extend(["--provider-options", provider_options])
    for task_id in SUPPLEMENTAL_TASKS:
        command.extend(["--task", task_id])
    return KickoffStep("Run supplemental diagnostics", tuple(command))


def _preflight_catalog_root(catalog_root: Path) -> tuple[bool, str]:
    lanes_dir = catalog_root.resolve() / "lanes"
    if not lanes_dir.exists():
        return False, f"benchmark lanes directory not found: {lanes_dir}"
    return True, ""


def _render_workflow_dispatch_markdown(
    *,
    date_stamp: str,
    model: str,
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
            "",
            "Public full:",
            "- `mode=public_full`",
            f"- `model={model}`",
            "",
            "Example `gh` commands:",
            "```bash",
            "gh workflow run benchmark-smoke.yml \\",
            "  -f mode=smoke \\",
            f"  -f model={model}",
            "",
            "gh workflow run benchmark-smoke.yml \\",
            "  -f mode=public_full \\",
            f"  -f model={model}",
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
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_output_root = output_root / f"{date_stamp}-local-smoke"
    supplemental_output_root = output_root / f"{date_stamp}-local-supplemental"
    handoff_path = output_root / f"{date_stamp}-workflow-dispatch.md"
    handoff_path.write_text(
        _render_workflow_dispatch_markdown(
            date_stamp=date_stamp,
            model=model,
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
        "--catalog-root",
        type=Path,
        default=ROOT / "benchmarks",
        help="Benchmark catalog root used by kickoff preflight/smoke runs",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Repeat count for benchmark compare commands (release-grade default: 5)",
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
        help="JSON provider options passed through to benchmark commands",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    smoke_output_root = output_root / f"{args.date_stamp}-local-smoke"
    supplemental_output_root = output_root / f"{args.date_stamp}-local-supplemental"

    if args.phase == "print-hosted":
        _write_handoff_file(
            output_root=output_root,
            date_stamp=args.date_stamp,
            model=args.model,
        )
        return 0

    ok, reason = _preflight_catalog_root(args.catalog_root)
    if not ok:
        print(f"[kickoff] benchmark preflight failed: {reason}")
        return 2

    steps = [_preflight_step(catalog_root=args.catalog_root.resolve())]
    if args.phase == "kickoff":
        steps.append(
            _local_smoke_step(
                output_root=smoke_output_root,
                model=args.model,
                catalog_root=args.catalog_root.resolve(),
                repeats=max(1, int(args.repeats)),
                pricing_prompt=args.pricing_prompt,
                pricing_completion=args.pricing_completion,
                provider_options=args.provider_options,
            )
        )
    elif args.phase == "supplemental":
        steps.append(
            _supplemental_step(
                output_root=supplemental_output_root,
                model=args.model,
                catalog_root=args.catalog_root.resolve(),
                repeats=max(1, int(args.repeats)),
                pricing_prompt=args.pricing_prompt,
                pricing_completion=args.pricing_completion,
                provider_options=args.provider_options,
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
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
