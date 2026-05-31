#!/usr/bin/env python3

"""Validate release-claim evidence and write a reviewable artifact."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tok.testing.benchmark_suite import check_benchmark_report


@dataclass(frozen=True)
class ClaimCheckResult:
    name: str
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
        }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return payload
    msg = f"expected JSON object in {path}"
    raise ValueError(msg)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _verified_savings_bands_from_claims_matrix(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    bands: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "|" not in line or "Verified" not in line or "savings" not in line.lower():
            continue
        for match in re.finditer(r"(?P<min>\d+(?:\.\d+)?)\s*-\s*(?P<max>\d+(?:\.\d+)?)\s*%", line):
            bands.append(
                {
                    "line": line.strip(),
                    "min": float(match.group("min")),
                    "max": float(match.group("max")),
                }
            )
    return bands


def verify_release_claims(
    *,
    gate_metrics_path: Path,
    output_path: Path,
    min_savings_pct: float,
    max_savings_pct: float,
    benchmark_report_path: Path | None = None,
    claims_matrix_path: Path | None = None,
) -> dict[str, Any]:
    gate_payload = _read_json(gate_metrics_path)
    release_summary = gate_payload.get("release_summary", {})
    if not isinstance(release_summary, dict):
        release_summary = {}

    avg_savings_pct = _as_float(release_summary.get("avg_savings_pct"), 0.0)
    checks: list[ClaimCheckResult] = [
        ClaimCheckResult(
            name="savings_band_claim",
            passed=min_savings_pct <= avg_savings_pct <= max_savings_pct,
            details={
                "avg_savings_pct": avg_savings_pct,
                "required_min": min_savings_pct,
                "required_max": max_savings_pct,
                "gate_metrics": str(gate_metrics_path),
            },
        )
    ]

    if claims_matrix_path is not None:
        verified_bands = _verified_savings_bands_from_claims_matrix(claims_matrix_path)
        out_of_band = [
            band for band in verified_bands if band["min"] < min_savings_pct or band["max"] > max_savings_pct
        ]
        checks.append(
            ClaimCheckResult(
                name="claims_matrix_savings_band",
                passed=not out_of_band,
                details={
                    "claims_matrix": str(claims_matrix_path),
                    "allowed_min": min_savings_pct,
                    "allowed_max": max_savings_pct,
                    "verified_bands": verified_bands,
                    "out_of_band": out_of_band,
                },
            )
        )

    if benchmark_report_path is not None:
        benchmark_check = check_benchmark_report(benchmark_report_path)
        checks.append(
            ClaimCheckResult(
                name="benchmark_headline_claim",
                passed=bool(benchmark_check.get("passed", False)),
                details=benchmark_check,
            )
        )

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "passed": all(check.passed for check in checks),
        "checks": [check.to_dict() for check in checks],
        "release_summary": release_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-metrics", type=Path, required=True, help="Gate-check metrics JSON path")
    parser.add_argument("--output", type=Path, required=True, help="Output artifact JSON path")
    parser.add_argument("--min-savings-pct", type=float, default=45.0, help="Minimum acceptable average savings pct")
    parser.add_argument("--max-savings-pct", type=float, default=55.0, help="Maximum acceptable average savings pct")
    parser.add_argument("--benchmark-report", type=Path, default=None, help="Optional benchmark report JSON path")
    parser.add_argument(
        "--claims-matrix",
        type=Path,
        default=Path("docs/claims_matrix.md"),
        help="Claims matrix Markdown path to check for verified savings bands",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = verify_release_claims(
        gate_metrics_path=args.gate_metrics,
        output_path=args.output,
        min_savings_pct=float(args.min_savings_pct),
        max_savings_pct=float(args.max_savings_pct),
        benchmark_report_path=args.benchmark_report,
        claims_matrix_path=args.claims_matrix,
    )
    print(f"[claims] wrote {args.output}")
    if not payload["passed"]:
        print("[claims] verification failed")
        return 1
    print("[claims] verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
