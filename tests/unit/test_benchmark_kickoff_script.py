from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module() -> object:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "run_benchmark_kickoff.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_kickoff", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_kickoff_phase_runs_preflight_and_local_smoke_and_writes_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    commands: list[tuple[str, ...]] = []

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        del cwd, check
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    exit_code = module.main(
        [
            "--phase",
            "kickoff",
            "--date-stamp",
            "20260409",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert commands[0] == (
        "uv",
        "run",
        "python",
        "scripts/prepare_benchmark_assets.py",
        "--root",
        "benchmarks",
        "verify",
    )
    assert commands[1] == (
        "uv",
        "run",
        "python",
        "scripts/run_release_smoke.py",
        "--benchmark-mode",
        "smoke",
        "--benchmark-output",
        str((tmp_path / "out" / "20260409-local-smoke").resolve()),
        "--model",
        "anthropic/claude-sonnet-4.6",
    )
    handoff_path = tmp_path / "out" / "20260409-workflow-dispatch.md"
    assert handoff_path.exists()
    handoff = handoff_path.read_text()
    assert "benchmark-smoke.yml" in handoff
    assert "mode=smoke" in handoff
    assert "mode=public_full" in handoff


def test_supplemental_phase_runs_expected_catalog_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    commands: list[tuple[str, ...]] = []

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        del cwd, check
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    exit_code = module.main(
        [
            "--phase",
            "supplemental",
            "--date-stamp",
            "20260409",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert len(commands) == 2
    supplemental_command = commands[1]
    assert supplemental_command[:6] == (
        "uv",
        "run",
        "tok",
        "dev",
        "live-benchmark",
        "--program",
    )
    assert "--include-advisory" in supplemental_command
    assert str((tmp_path / "out" / "20260409-local-supplemental").resolve()) in supplemental_command
    for task_id in (
        "exec.tok.bridge-canonicalization",
        "exec.tok.first-exact-search",
        "session.context-reacquisition.answer",
        "session.response-contract.patch",
        "session.fail-open.answer",
        "session.answer-anchor.patch",
    ):
        assert task_id in supplemental_command
