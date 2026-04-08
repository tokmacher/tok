#!/usr/bin/env python3
"""
PR comment generation script for Tok gate checks.

This script generates PR comments based on Tok gate check results,
providing detailed feedback on compression quality and system health.
"""

import argparse
import json
import os
import sys
from typing import Any


def load_results(results_file: str) -> dict[str, Any]:
    """Load gate check results from JSON file."""
    try:
        with open(results_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                return {"results": data}
            if isinstance(data, dict):
                return data
            msg = "Unsupported gate results shape"
            raise TypeError(msg)
    except Exception:
        sys.exit(1)


def generate_summary(
    results: list[dict[str, Any]],
    release_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate summary statistics from results."""
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    failed = total - passed

    # Calculate average metrics
    avg_savings = 0.0
    avg_pressure = 0.0
    if results:
        savings_values = [r.get("savings_pct", 0) for r in results if r.get("savings_pct") is not None]
        pressure_values = [r.get("pressure", 0) for r in results if r.get("pressure") is not None]

        if savings_values:
            avg_savings = sum(savings_values) / len(savings_values)
        if pressure_values:
            avg_pressure = sum(pressure_values) / len(pressure_values)

    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total * 100) if total > 0 else 0,
        "avg_savings_pct": round(avg_savings, 1),
        "avg_pressure": round(avg_pressure, 1),
    }
    if release_summary:
        summary["fallback_fixture_rate"] = release_summary.get("fallback_fixture_rate", 0.0)
        summary["billing_delta_usd"] = release_summary.get("billing_delta_usd", 0.0)
        summary["billing_delta_pct"] = release_summary.get("billing_delta_pct", 0.0)
    return summary


def _format_failed_fixtures(
    failed_results: list[dict[str, Any]],
) -> list[str]:
    if not failed_results:
        return []
    lines = ["### ❌ Failed Fixtures", ""]
    for result in failed_results[:10]:
        name = result.get("fixture", "Unknown")
        failures = result.get("failures", [])
        savings = result.get("savings_pct", 0)
        pressure = result.get("pressure", 0)
        lines.append(f"**{name}**")
        lines.append(f"- Savings: {savings}%, Pressure: {pressure}")
        if failures:
            lines.append(f"- Issues: {', '.join(failures)}")
        lines.append("")
    if len(failed_results) > 10:
        lines.append(f"... and {len(failed_results) - 10} more failed fixtures")
        lines.append("")
    return lines


def _build_recommendations(
    summary: dict[str, Any],
    failed_results: list[dict[str, Any]],
) -> list[str]:
    if summary["failed"] == 0:
        return [
            "✅ All gate checks passed! The compression system is performing well.",
            "",
        ]
    recommendations = []
    if summary["avg_savings_pct"] < 15:
        recommendations.append("💰 Consider improving compression strategies to increase token savings")
    if summary["avg_pressure"] > 5:
        recommendations.append("🔍 Investigate invisible pressure sources and optimize tool usage patterns")
    if any("trend_regressing" in r.get("failures", []) for r in failed_results):
        recommendations.append("📈 Analyze recent performance trends and address regression patterns")
    if any("min_savings_pct" in r.get("failures", []) for r in failed_results):
        recommendations.append("🎯 Review minimum savings thresholds and adjust compression targets")
    if recommendations:
        recommendations.append("")
        return recommendations
    return [
        "🔧 Review failed fixtures and address specific issues listed above",
        "",
    ]


def generate_comment(payload: dict[str, Any], fixture_set: str) -> str:
    """Generate PR comment from gate check results."""
    results = payload.get("results", [])
    release_summary = payload.get("release_summary")
    summary = generate_summary(results, release_summary)

    failed_results = [r for r in results if not r.get("passed", False)]

    comment_lines = [
        "## 🚪 Tok Gate Check Results",
        "",
        f"**Fixture Set:** `{fixture_set}`",
        f"**Status:** {'✅ PASSED' if summary['failed'] == 0 else '❌ FAILED'}",
        f"**Results:** {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.1f}%)",
        "",
        "### 📊 Metrics Summary",
        "",
        f"- **Average Savings:** {summary['avg_savings_pct']}%",
        f"- **Average Pressure:** {summary['avg_pressure']}",
        f"- **Fallback Fixture Rate:** {summary.get('fallback_fixture_rate', 0.0)}%",
        f"- **Billing Delta:** ${summary.get('billing_delta_usd', 0.0):.4f} ({summary.get('billing_delta_pct', 0.0)}%)",
        "",
    ]

    comment_lines.extend(_format_failed_fixtures(failed_results))

    comment_lines.extend(["### 💡 Recommendations", ""])
    comment_lines.extend(_build_recommendations(summary, failed_results))

    comment_lines.extend(
        [
            "---",
            "*Generated by Tok Gate Check System*",
            "",
            f"*Fixture set: {fixture_set} | Total fixtures: {summary['total']}*",
        ]
    )

    return "\n".join(comment_lines)


def post_to_github(_comment: str) -> int:
    _ = _comment
    """Post comment to GitHub (placeholder for future implementation)."""
    # For now, just print the comment (actual GitHub API integration would need
    # more setup)
    return 0


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate PR comments from Tok gate check results")
    parser.add_argument("--results", required=True, help="JSON file with gate check results")
    parser.add_argument(
        "--fixture-set",
        required=True,
        help="Fixture set used (feature or full)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print comment instead of posting to GitHub",
    )

    args = parser.parse_args()

    # Load results
    payload = load_results(args.results)
    results = payload.get("results", [])

    if not results:
        sys.exit(1)

    # Generate comment
    generate_comment(payload, args.fixture_set)

    # Output comment
    if args.dry_run:
        pass
    else:
        # Post to GitHub (placeholder)
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            sys.exit(1)

        # For now, just print the comment (actual GitHub API integration would need
        # more setup)

    return 0


if __name__ == "__main__":
    sys.exit(main())
