from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from tok.testing.benchmark_suite import load_benchmark_catalog

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks"


def _load_module() -> object:
    module_path = REPO_ROOT / "scripts" / "run_live_benchmark_matrix.py"
    spec = importlib.util.spec_from_file_location("run_live_benchmark_matrix", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_public_execution_overlay(overlay_root: Path) -> None:
    overlay_root.mkdir(parents=True, exist_ok=True)
    catalog = load_benchmark_catalog(BENCHMARK_ROOT)
    for task in catalog.tasks:
        hidden_ref = task.hidden_evaluator_ref()
        if not task.public_release or task.family != "execution_patch" or not hidden_ref:
            continue
        (overlay_root / f"{hidden_ref}.json").write_text(
            json.dumps({"selectors": ["tests/test_placeholder.py::test_placeholder"]})
        )


def test_defaults_to_targeted_deepseek_matrix(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    commands: list[tuple[str, ...]] = []

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        del cwd, check
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(module, "_search_roots", lambda: (tmp_path,))

    exit_code = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--catalog-profile",
            "grounding-only",
        ]
    )

    assert exit_code == 0
    assert len(commands) == 6
    assert commands[0] == (
        "uv",
        "run",
        "tok",
        "dev",
        "live-benchmark",
        "--program",
        "legacy",
        "--benchmark",
        "coding-loop",
        "--mode",
        "compare",
        "--model",
        "deepseek/deepseek-v3.2",
        "--output",
        str((tmp_path / "out" / "deepseek_deepseek-v3.2").resolve()),
    )
    assert commands[4][8] == "grammar_drift"
    assert commands[5] == (
        "uv",
        "run",
        "tok",
        "dev",
        "live-benchmark",
        "--program",
        "catalog",
        "--mode",
        "compare",
        "--model",
        "deepseek/deepseek-v3.2",
        "--catalog-root",
        str(BENCHMARK_ROOT.resolve()),
        "--output",
        str((tmp_path / "out" / "deepseek_deepseek-v3.2" / "catalog").resolve()),
        "--family",
        "repo_grounding",
        "--public-release-only",
    )


def test_full_profile_and_multiple_models(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    commands: list[tuple[str, ...]] = []
    overlay_root = tmp_path / "private-evaluator-overlay"
    _write_public_execution_overlay(overlay_root)

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        del cwd, check
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(module, "_search_roots", lambda: (tmp_path,))

    exit_code = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--profile",
            "full",
            "--model",
            "deepseek/deepseek-v3.2",
            "--model",
            "openai/gpt-4.1",
            "--repeats",
            "2",
            "--benchmark",
            "coding-loop",
            "--benchmark",
            "jit-loop",
        ]
    )

    assert exit_code == 0
    assert len(commands) == 6
    assert "--repeats" in commands[0]
    assert "2" in commands[0]
    assert str((tmp_path / "out" / "openai_gpt-4.1").resolve()) in commands[4]
    assert commands[2][-3:] == (
        "--public-release-only",
        "--private-evaluator-root",
        str(overlay_root.resolve()),
    )
    assert commands[5][-3:] == (
        "--public-release-only",
        "--private-evaluator-root",
        str(overlay_root.resolve()),
    )


def test_skips_existing_triage_unless_forced(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    commands: list[tuple[str, ...]] = []
    model_output = tmp_path / "out" / "deepseek_deepseek-v3.2"
    model_output.mkdir(parents=True)
    (model_output / "coding-loop_triage.json").write_text("{}")

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        del cwd, check
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(module, "_search_roots", lambda: (tmp_path,))

    exit_code = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--catalog-profile",
            "grounding-only",
            "--benchmark",
            "coding-loop",
            "--benchmark",
            "jit-loop",
        ]
    )

    assert exit_code == 0
    assert len(commands) == 2
    assert commands[0][8] == "jit-loop"
    assert commands[1][6] == "catalog"


def test_catalog_suite_runs_only_new_benchmarks(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    commands: list[tuple[str, ...]] = []

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        del cwd, check
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(module, "_search_roots", lambda: (tmp_path,))

    exit_code = module.main(
        [
            "--suite",
            "catalog",
            "--output-root",
            str(tmp_path / "out"),
            "--catalog-profile",
            "grounding-only",
        ]
    )

    assert exit_code == 0
    assert len(commands) == 1
    assert commands[0][6] == "catalog"
    assert "--family" in commands[0]
    assert "repo_grounding" in commands[0]


def test_catalog_defaults_to_public_and_errors_without_overlay(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    monkeypatch.setattr(module, "_search_roots", lambda: (tmp_path,))

    try:
        module.main(
            [
                "--suite",
                "catalog",
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing private overlay to abort")

    assert "No usable private evaluator overlay was found" in message
    assert "--catalog-profile grounding-only" in message
