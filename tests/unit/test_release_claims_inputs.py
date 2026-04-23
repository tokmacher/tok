"""Release-input guards for claims, pricing verification, and live-smoke evidence."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_claims_matrix_has_required_columns() -> None:
    content = _read("docs/claims_matrix.md")
    # Check for the 5-column header (with alignment padding)
    assert "Claim" in content and "Owner" in content and "Evidence Command" in content
    assert "Artifact" in content and "Status" in content
    assert "Verified" in content
    assert "Demoted" in content


def test_pricing_verification_points_to_canonical_source() -> None:
    content = _read("docs/pricing_verification.md")
    assert "src/tok/utils/pricing.py" in content
    assert "Last reviewed:" in content
    assert "2026-04-08" in content


def test_live_smoke_matrix_covers_manual_and_automated_paths() -> None:
    content = _read("docs/live_smoke_matrix.md")
    assert "## Automated Smokes" in content
    assert "## Manual Live Runs" in content
    assert "tests/smoke/test_live_claude_smoke_matrix.py" in content
