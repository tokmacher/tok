#!/usr/bin/env python3

"""Agent smoke check: can an agent verify the repo shape quickly from a cold clone?"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UV_RUN: tuple[str, ...] = ("uv", "run")


@dataclass(frozen=True)
class SmokeStep:
    name: str
    command: tuple[str, ...]


SMOKE_STEPS: tuple[SmokeStep, ...] = (
    SmokeStep("CLI version", (*UV_RUN, "tok", "--version")),
    SmokeStep("CLI help", (*UV_RUN, "tok", "--help")),
    SmokeStep("Claude help", (*UV_RUN, "tok", "claude", "--help")),
    SmokeStep("Bridge status help", (*UV_RUN, "tok", "bridge", "status", "--help")),
    SmokeStep("Doctor help", (*UV_RUN, "tok", "doctor", "--help")),
    SmokeStep("Stats help", (*UV_RUN, "tok", "stats", "--help")),
    SmokeStep("Audit help", (*UV_RUN, "tok", "audit", "--help")),
    SmokeStep("Unit tests", (*UV_RUN, "pytest", "tests/unit", "-q")),
)


def build_steps() -> tuple[SmokeStep, ...]:
    return SMOKE_STEPS


def main(argv: list[str] | None = None) -> int:
    results: list[tuple[str, str]] = []
    failed = False

    for step in build_steps():
        completed = subprocess.run(step.command, cwd=ROOT, capture_output=True, text=True)
        status = "PASS" if completed.returncode == 0 else "FAIL"
        if completed.returncode != 0:
            failed = True
        results.append((step.name, status))
        print(f"  {step.name}: {status}")
        if status == "FAIL":
            if completed.stdout:
                print(f"    stdout: {completed.stdout[:500]}")
            if completed.stderr:
                print(f"    stderr: {completed.stderr[:500]}")
            break

    summary_status = "FAIL" if failed else "PASS"
    print(f"\nAgent smoke: {summary_status}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
