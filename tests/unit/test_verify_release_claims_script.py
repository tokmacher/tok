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
    gate_metrics.write_text('{"release_summary": {"avg_savings_pct": 50.0}}')

    payload = module.verify_release_claims(
        gate_metrics_path=gate_metrics,
        output_path=output,
        min_savings_pct=45.0,
        max_savings_pct=55.0,
        benchmark_report_path=None,
    )

    assert payload["passed"] is True
    assert output.exists()


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
