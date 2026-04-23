from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "check_repo_hygiene.py"
    spec = importlib.util.spec_from_file_location("check_repo_hygiene", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_violations_accepts_canonical_root_files() -> None:
    module = _load_module()

    violations = module.find_violations(
        [
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "roadmap.md",
            "src/tok/__init__.py",
            "docs/bridge.md",
        ]
    )

    assert violations == []


def test_find_violations_flags_backups_caches_and_temp_paths() -> None:
    module = _load_module()

    violations = module.find_violations(
        [
            "src/tok/gateway/__init__.py.bak",
            "src/tok/__pycache__/cli.cpython-312.pyc",
            "dist/tok.whl",
            "tmp/report.json",
        ]
    )

    assert "tracked backup file: src/tok/gateway/__init__.py.bak" in violations
    assert "tracked Python cache artifact: src/tok/__pycache__/cli.cpython-312.pyc" in violations
    assert "tracked build artifact under dist/: dist/tok.whl" in violations
    assert "tracked temporary artifact under tmp/: tmp/report.json" in violations


def test_find_violations_flags_runtime_and_ad_hoc_root_files() -> None:
    module = _load_module()

    violations = module.find_violations(
        [
            "memory.tok",
            "plan_structure.txt",
            "README.md",
        ]
    )

    assert "tracked runtime artifact in repo root: memory.tok" in violations
    assert "unexpected tracked top-level file: plan_structure.txt" in violations
