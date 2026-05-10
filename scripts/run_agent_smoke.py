#!/usr/bin/env python3

"""Agent smoke check: can an agent verify the repo shape quickly from a cold clone?"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
REPORT_PATH = OUT_DIR / "agent_smoke_report.json"
UV_RUN: tuple[str, ...] = ("uv", "run")

CLAIM_LEVELS = ("source_only", "cli_smoke", "test_suite", "live_bridge", "live_bridge_with_measured_savings")


@dataclass(frozen=True)
class SmokeStep:
    name: str
    command: tuple[str, ...]


@dataclass
class SmokeResult:
    name: str
    status: str
    stdout: str = ""
    stderr: str = ""


SMOKE_STEPS: tuple[SmokeStep, ...] = (
    SmokeStep("CLI version", (*UV_RUN, "tok", "--version")),
    SmokeStep("CLI help", (*UV_RUN, "tok", "--help")),
    SmokeStep("Claude help", (*UV_RUN, "tok", "claude", "--help")),
    SmokeStep("Bridge status help", (*UV_RUN, "tok", "bridge", "status", "--help")),
    SmokeStep("Doctor help", (*UV_RUN, "tok", "doctor", "--help")),
    SmokeStep("Stats help", (*UV_RUN, "tok", "stats", "--help")),
    SmokeStep("Audit help", (*UV_RUN, "tok", "audit", "--help")),
    SmokeStep("Agent contract tests", (*UV_RUN, "pytest", "tests/unit/test_agent_docs_contract.py", "-q")),
)

LIVE_BRIDGE_STEPS: tuple[SmokeStep, ...] = (
    SmokeStep("Bridge status JSON", (*UV_RUN, "tok", "bridge", "status", "--json")),
    SmokeStep("Doctor JSON", (*UV_RUN, "tok", "doctor", "--json")),
    SmokeStep("Stats JSON", (*UV_RUN, "tok", "stats", "--json")),
)


def build_steps() -> tuple[SmokeStep, ...]:
    return SMOKE_STEPS


def _run_step(step: SmokeStep) -> SmokeResult:
    completed = subprocess.run(step.command, cwd=ROOT, capture_output=True, text=True)
    status = "PASS" if completed.returncode == 0 else "FAIL"
    return SmokeResult(
        name=step.name,
        status=status,
        stdout=(completed.stdout or "")[:500],
        stderr=(completed.stderr or "")[:500],
    )


def _determine_claim_level(
    results: list[SmokeResult],
    live_bridge_requested: bool,
    live_bridge_results: list[SmokeResult] | None,
) -> str:
    if live_bridge_requested and live_bridge_results:
        all_live_ok = all(r.status == "PASS" for r in live_bridge_results)
        if all_live_ok:
            for r in live_bridge_results:
                try:
                    data = json.loads(r.stdout)
                    ts = data.get("data", {}).get("tokens_saved", 0) or data.get("data", {}).get("session", {}).get(
                        "tokens_saved", 0
                    )
                    if ts > 0:
                        return "live_bridge_with_measured_savings"
                except (json.JSONDecodeError, KeyError):
                    pass
            return "live_bridge"
    all_base_ok = all(r.status == "PASS" for r in results)
    if not all_base_ok:
        return "source_only"
    has_test = any("test" in r.name.lower() for r in results)
    if has_test:
        return "test_suite"
    return "cli_smoke"


def _write_report(
    results: list[SmokeResult],
    *,
    live_bridge_requested: bool,
    live_bridge_results: list[SmokeResult] | None,
    claim_level: str,
    overall: str,
) -> dict:
    report = {
        "schema": "tok-agent-smoke-report/v0.1",
        "overall": overall,
        "checks": [{"name": r.name, "status": r.status} for r in results],
        "live_bridge_requested": live_bridge_requested,
        "live_bridge_result": "skipped",
        "claim_level": claim_level,
    }
    if live_bridge_requested and live_bridge_results is not None:
        live_ok = all(r.status == "PASS" for r in live_bridge_results)
        report["live_bridge_result"] = "pass" if live_ok else "fail"
        report["live_bridge_checks"] = [{"name": r.name, "status": r.status} for r in live_bridge_results]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    live_bridge_requested = "--live-bridge" in (argv or sys.argv[1:])

    results: list[SmokeResult] = []
    failed = False

    for step in build_steps():
        result = _run_step(step)
        results.append(result)
        print(f"  {result.name}: {result.status}")
        if result.status == "FAIL":
            failed = True
            if result.stdout:
                print(f"    stdout: {result.stdout}")
            if result.stderr:
                print(f"    stderr: {result.stderr}")
            break

    live_bridge_results: list[SmokeResult] | None = None
    if live_bridge_requested and not failed:
        live_bridge_results = []
        for step in LIVE_BRIDGE_STEPS:
            result = _run_step(step)
            live_bridge_results.append(result)
            print(f"  {result.name}: {result.status}")

    overall = "FAIL" if failed else "PASS"
    if live_bridge_requested and live_bridge_results is not None:
        if any(r.status == "FAIL" for r in live_bridge_results):
            overall = "FAIL"
    claim_level = _determine_claim_level(results, live_bridge_requested, live_bridge_results)

    _write_report(
        results,
        live_bridge_requested=live_bridge_requested,
        live_bridge_results=live_bridge_results,
        claim_level=claim_level,
        overall=overall,
    )

    print(f"\nAgent smoke: {overall} (claim_level={claim_level})")
    print(f"Report: {REPORT_PATH}")
    return 1 if overall == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
