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
    module_path = REPO_ROOT / "scripts" / "run_benchmark_smoke_matrix.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_smoke_matrix", module_path)
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


def test_main_defaults_to_deepseek_and_autodiscovers_overlay(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
            "--catalog-root",
            str(BENCHMARK_ROOT),
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
        str(BENCHMARK_ROOT.resolve()),
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
        "--private-evaluator-root",
        str(overlay_root.resolve()),
    )


def test_main_loops_multiple_models(tmp_path: Path, monkeypatch) -> None:
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
            "--catalog-root",
            str(BENCHMARK_ROOT),
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


def test_discover_private_evaluator_root_errors_when_no_overlay_found(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_search_roots", lambda: (tmp_path,))

    try:
        module.discover_private_evaluator_root(catalog_root=BENCHMARK_ROOT)
    except RuntimeError as exc:
        assert "Could not find a usable private evaluator overlay" in str(exc)
    else:
        raise AssertionError("expected discovery to fail without an overlay")
