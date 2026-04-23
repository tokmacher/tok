from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks"


def _load_module() -> object:
    module_path = REPO_ROOT / "scripts" / "run_benchmark_smoke_matrix.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_smoke_matrix", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_defaults_to_deepseek(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    commands: list[tuple[str, ...]] = []
    catalog_root = tmp_path / "benchmarks"
    (catalog_root / "lanes").mkdir(parents=True, exist_ok=True)

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        del cwd, check
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    exit_code = module.main(
        [
            "--catalog-root",
            str(catalog_root),
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
        str(catalog_root.resolve()),
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
        str((tmp_path / "out" / "deepseek_deepseek-v3.2").resolve()),
        "--model",
        "deepseek/deepseek-v3.2",
        "--catalog-root",
        str(catalog_root.resolve()),
        "--repeats",
        "5",
    )


def test_main_loops_multiple_models(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    commands: list[tuple[str, ...]] = []
    catalog_root = tmp_path / "benchmarks"
    (catalog_root / "lanes").mkdir(parents=True, exist_ok=True)

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        del cwd, check
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    exit_code = module.main(
        [
            "--catalog-root",
            str(catalog_root),
            "--skip-asset-verify",
            "--mode",
            "public_full",
            "--output-root",
            str(tmp_path / "out"),
            "--model",
            "deepseek/deepseek-v3.2",
            "--model",
            "openai/gpt-4.1",
        ]
    )

    assert exit_code == 0
    assert len(commands) == 2
    assert commands[0][5] == "public_full"
    assert str((tmp_path / "out" / "deepseek_deepseek-v3.2").resolve()) in commands[0]
    assert str((tmp_path / "out" / "openai_gpt-4.1").resolve()) in commands[1]
