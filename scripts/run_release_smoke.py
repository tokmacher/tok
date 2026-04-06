#!/usr/bin/env python3

"""Run a bounded maintainer-facing release smoke sweep for Tok."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SmokeStep:
    name: str
    command: tuple[str, ...]


IMPORT_CHECK = (
    "import tok.stats",
    "import tok.savings_tracker",
    "import tok.gateway.stats",
    "import tok.universal_runtime",
    "import tok.runtime.core",
    "import tok.release_surface",
    "print('tok release smoke imports OK')",
)

SURFACE_GATE_CHECK = (
    "from typer.testing import CliRunner",
    "from tok.cli import app",
    "from tok.release_surface import validate_release_surface",
    "import tok",
    "help_output = CliRunner().invoke(app, ['--help']).output",
    "failures = validate_release_surface(exported_names=tok.__all__, cli_help_output=help_output, root_app=app)",
    "print('surface_gate_failures=' + repr(failures))",
    "raise SystemExit(1 if failures else 0)",
)


SMOKE_STEPS: tuple[SmokeStep, ...] = (
    SmokeStep("CLI help", ("uv", "run", "tok", "--help")),
    SmokeStep("Bridge help", ("uv", "run", "tok", "bridge", "--help")),
    SmokeStep("Doctor help", ("uv", "run", "tok", "doctor", "--help")),
    SmokeStep("Stats help", ("uv", "run", "tok", "stats", "--help")),
    SmokeStep(
        "Public imports",
        (
            "uv",
            "run",
            "python",
            "-c",
            "; ".join(IMPORT_CHECK),
        ),
    ),
    SmokeStep(
        "Release-surface gate",
        (
            "uv",
            "run",
            "python",
            "-c",
            "; ".join(SURFACE_GATE_CHECK),
        ),
    ),
    SmokeStep(
        "Focused bridge/runtime/compression smoke",
        (
            "uv",
            "run",
            "pytest",
            "tests/unit/test_compatibility_shims.py",
            "tests/unit/test_bridge_smoke.py",
            "tests/unit/test_packaging_smoke.py",
            "tests/unit/test_gateway.py::test_gateway_canonicalizes_tool_heavy_bridge_body_before_send",
            "tests/unit/test_universal_runtime.py::test_runtime_prepare_request_compression_in_tool_compatible_mode",
            "-q",
        ),
    ),
    SmokeStep(
        "Build smoke",
        (
            "uv",
            "run",
            "--with",
            "build",
            "--with",
            "hatchling",
            "python",
            "-m",
            "build",
        ),
    ),
)


def _run_step(step: SmokeStep) -> int:
    print(f"\n==> {step.name}", flush=True)
    print("$ " + " ".join(step.command), flush=True)
    completed = subprocess.run(step.command, cwd=ROOT, check=False)
    return completed.returncode


def main() -> int:
    print("Tok release smoke sweep", flush=True)
    print(f"repo: {ROOT}", flush=True)

    for step in SMOKE_STEPS:
        exit_code = _run_step(step)
        if exit_code != 0:
            print(f"\nRelease smoke failed during: {step.name}", flush=True)
            return exit_code

    print("\nRelease smoke passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
