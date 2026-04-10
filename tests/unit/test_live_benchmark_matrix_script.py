from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

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


def test_defaults_to_targeted_deepseek_matrix(tmp_path: Path, monkeypatch) -> None:
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
            "--output-root",
            str(tmp_path / "out"),
            "--suite",
            "catalog",
        ]
    )

    assert exit_code == 0
    assert len(commands) == 1
    assert commands[0] == (
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
        "--public-release-only",
        "--family",
        "execution_patch",
        "--family",
        "repo_grounding",
    )


def test_full_profile_and_multiple_models(tmp_path: Path, monkeypatch) -> None:
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
    assert ("--family", "execution_patch", "--family", "repo_grounding") == commands[2][
        commands[2].index("--family") : commands[2].index("--family") + 4
    ]
    assert ("--family", "execution_patch", "--family", "repo_grounding") == commands[5][
        commands[5].index("--family") : commands[5].index("--family") + 4
    ]
