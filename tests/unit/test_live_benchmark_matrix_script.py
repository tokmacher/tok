from __future__ import annotations

import importlib.util
import json
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
    catalog_root = tmp_path / "benchmarks"
    (catalog_root / "lanes").mkdir(parents=True, exist_ok=True)

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        del cwd, check
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        if "--output" in command_tuple:
            output_index = command_tuple.index("--output") + 1
            output_root = Path(command_tuple[output_index])
            (output_root / "report.json").parent.mkdir(parents=True, exist_ok=True)
            (output_root / "report.json").write_text("{}")
        return subprocess.CompletedProcess(command_tuple, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    exit_code = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--suite",
            "catalog",
            "--catalog-root",
            str(catalog_root),
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
        str(catalog_root.resolve()),
        "--output",
        str((tmp_path / "out" / "deepseek_deepseek-v3.2" / "catalog").resolve()),
        "--public-release-only",
        "--family",
        "execution_patch",
        "--family",
        "repo_grounding",
        "--repeats",
        "5",
    )


def test_full_profile_and_multiple_models(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    commands: list[tuple[str, ...]] = []
    catalog_root = tmp_path / "benchmarks"
    (catalog_root / "lanes").mkdir(parents=True, exist_ok=True)

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        del cwd, check
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        if "--program" in command_tuple and command_tuple[command_tuple.index("--program") + 1] == "replay":
            output_path = Path(command_tuple[command_tuple.index("--output") + 1])
            benchmark = command_tuple[command_tuple.index("--benchmark") + 1]
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / f"{benchmark}_triage.json").write_text("{}")
        if "--program" in command_tuple and command_tuple[command_tuple.index("--program") + 1] == "catalog":
            output_path = Path(command_tuple[command_tuple.index("--output") + 1])
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "report.json").write_text("{}")
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
            "--catalog-root",
            str(catalog_root),
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


def test_preflight_fails_for_missing_catalog_root(tmp_path: Path) -> None:
    module = _load_module()
    exit_code = module.main(
        [
            "--suite",
            "catalog",
            "--catalog-root",
            str(tmp_path / "missing-benchmarks"),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 2


def test_fingerprint_mismatch_forces_rerun(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    commands: list[tuple[str, ...]] = []
    out_root = tmp_path / "out"
    model_dir = out_root / "deepseek_deepseek-v3.2"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "coding-loop_triage.json").write_text("{}")
    (model_dir / "coding-loop_fingerprint.json").write_text(json.dumps({"version": "0"}))

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        del cwd, check
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    exit_code = module.main(
        [
            "--suite",
            "replay",
            "--output-root",
            str(out_root),
            "--benchmark",
            "coding-loop",
        ]
    )

    assert exit_code == 0
    assert len(commands) == 1
