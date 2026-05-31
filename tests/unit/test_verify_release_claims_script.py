from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module() -> object:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "verify_release_claims.py"
    spec = importlib.util.spec_from_file_location("verify_release_claims", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verify_release_claims_passes_when_savings_band_matches(tmp_path) -> None:
    module = _load_module()
    gate_metrics = tmp_path / "gate_metrics.json"
    output = tmp_path / "claims.json"
    claims = tmp_path / "claims_matrix.md"
    gate_metrics.write_text('{"release_summary": {"avg_savings_pct": 50.0}}')
    claims.write_text(
        "| Claim | Owner | Evidence Command | Artifact | Status |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Supported release reference band is 45-55% on validated sessions | docs | cmd | artifact | Verified |\n"
    )

    payload = module.verify_release_claims(
        gate_metrics_path=gate_metrics,
        output_path=output,
        min_savings_pct=45.0,
        max_savings_pct=55.0,
        benchmark_report_path=None,
        claims_matrix_path=claims,
    )

    assert payload["passed"] is True
    assert output.exists()
    check_names = {check["name"] for check in payload["checks"]}
    assert "claims_matrix_savings_band" in check_names


def test_verify_release_claims_fails_when_savings_band_out_of_range(tmp_path) -> None:
    module = _load_module()
    gate_metrics = tmp_path / "gate_metrics.json"
    output = tmp_path / "claims.json"
    gate_metrics.write_text('{"release_summary": {"avg_savings_pct": 30.0}}')

    payload = module.verify_release_claims(
        gate_metrics_path=gate_metrics,
        output_path=output,
        min_savings_pct=45.0,
        max_savings_pct=55.0,
        benchmark_report_path=None,
    )

    assert payload["passed"] is False
    assert output.exists()


def test_verify_release_claims_fails_verified_doc_band_outside_range(tmp_path) -> None:
    module = _load_module()
    gate_metrics = tmp_path / "gate_metrics.json"
    output = tmp_path / "claims.json"
    claims = tmp_path / "claims_matrix.md"
    gate_metrics.write_text('{"release_summary": {"avg_savings_pct": 50.0}}')
    claims.write_text(
        "| Claim | Owner | Evidence Command | Artifact | Status |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Global 60-70% savings release claim | README.md | cmd | artifact | Verified |\n"
    )

    payload = module.verify_release_claims(
        gate_metrics_path=gate_metrics,
        output_path=output,
        min_savings_pct=45.0,
        max_savings_pct=55.0,
        benchmark_report_path=None,
        claims_matrix_path=claims,
    )

    assert payload["passed"] is False
    matrix_check = next(check for check in payload["checks"] if check["name"] == "claims_matrix_savings_band")
    assert matrix_check["passed"] is False
    assert matrix_check["details"]["out_of_band"]


# ---------------------------------------------------------------------------
# _verified_savings_bands_from_claims_matrix unit tests
# ---------------------------------------------------------------------------


def test_claims_matrix_parser_returns_empty_for_nonexistent_file(tmp_path) -> None:
    module = _load_module()
    result = module._verified_savings_bands_from_claims_matrix(tmp_path / "missing.md")
    assert result == []


def test_claims_matrix_parser_extracts_band_from_matching_row(tmp_path) -> None:
    module = _load_module()
    path = tmp_path / "matrix.md"
    path.write_text(
        "| Verified savings claims stay inside 45-55% band | owner | cmd | artifact | Verified |\n",
        encoding="utf-8",
    )
    bands = module._verified_savings_bands_from_claims_matrix(path)
    assert len(bands) == 1
    assert bands[0]["min"] == 45.0
    assert bands[0]["max"] == 55.0


def test_claims_matrix_parser_skips_row_without_savings_keyword(tmp_path) -> None:
    module = _load_module()
    path = tmp_path / "matrix.md"
    path.write_text(
        "| Supported reference band is 45-55% on sessions | docs | cmd | artifact | Verified |\n",
        encoding="utf-8",
    )
    bands = module._verified_savings_bands_from_claims_matrix(path)
    assert bands == []


def test_claims_matrix_parser_skips_row_without_verified_status(tmp_path) -> None:
    module = _load_module()
    path = tmp_path / "matrix.md"
    path.write_text(
        "| Global 60-70% savings claim | README | cmd | artifact | Demoted |\n",
        encoding="utf-8",
    )
    bands = module._verified_savings_bands_from_claims_matrix(path)
    assert bands == []


def test_claims_matrix_parser_skips_row_without_pipe_separator(tmp_path) -> None:
    module = _load_module()
    path = tmp_path / "matrix.md"
    path.write_text(
        "Verified savings band is 45-55% according to evidence\n",
        encoding="utf-8",
    )
    bands = module._verified_savings_bands_from_claims_matrix(path)
    assert bands == []


def test_claims_matrix_parser_returns_empty_for_empty_file(tmp_path) -> None:
    module = _load_module()
    path = tmp_path / "matrix.md"
    path.write_text("", encoding="utf-8")
    assert module._verified_savings_bands_from_claims_matrix(path) == []
