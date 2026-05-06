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
            ".editorconfig",
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


def test_find_violations_flags_root_release_report_artifacts() -> None:
    module = _load_module()

    violations = module.find_violations(
        [
            "subtle-bug-audit-report-2026-04-30.md",
            "stability-campaign-findings.md",
        ]
    )

    assert "tracked release report artifact in repo root: subtle-bug-audit-report-2026-04-30.md" in violations
    assert "tracked release report artifact in repo root: stability-campaign-findings.md" in violations


def test_public_docs_do_not_ship_competitor_or_positioning_pages() -> None:
    root = Path(__file__).resolve().parents[2]
    removed_public_docs = {
        "docs/claude-compaction-comparison.md",
        "docs/positioning-context-tools.md",
    }

    for relative_path in removed_public_docs:
        assert not (root / relative_path).exists(), f"{relative_path} should not be public repo documentation"

    public_docs = [
        root / "README.md",
        root / "docs" / "cli-reference.md",
        root / "docs" / "diagnostics.md",
    ]
    public_text = "\n".join(path.read_text() for path in public_docs)
    for relative_path in removed_public_docs:
        assert relative_path not in public_text

    for off_roadmap_term in ("TokenPak", "OpenClaw", "LLMLingua", "Context7", "Mem0", "Graphiti"):
        assert off_roadmap_term not in public_text
