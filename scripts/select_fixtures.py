#!/usr/bin/env python3
"""
Fixture selection script for Tok gate checks.

This script selects fixtures based on a specified set (feature, full, or redteam) and
outputs the selected fixtures to a JSON file. This supports branch-based
fixture selection in CI/CD workflows.
"""

import argparse
import json
import sys
from pathlib import Path


def get_feature_fixtures() -> list[str]:
    """Get the feature fixture set (subset for faster PR testing)."""
    return [
        "claude_coding_loop",
        "gpt_coding_loop",
        "long_coding_session",
        "search_intensive_workflow",
        "high_pressure_scenario",
        "runtime_conformance",
        "cache_stable_research_turns",
        "refined_search_recovery",
        # Green stress fixtures
        "metric_long_debug",
        "burst_retries",
        "verbose_payload",
        "straddling_boundary",
        "context_pinned_file",
        "alternating_adapters",
        "branching_tests",
        "compression_hypothesis_churn",
        "heavy_tool_event",
        "tool_density_micro",
        "episodes_multi_phase",
        "release_reacquisition",
        "orchestrator_parity",
        "bridge_vs_orchestrator",
        "cache_sensitivity",
    ]


def get_full_fixtures() -> list[str]:
    """Get the full fixture set (comprehensive testing)."""
    return [
        "claude_coding_loop",
        "gpt_coding_loop",
        "long_coding_session",
        "search_intensive_workflow",
        "high_pressure_scenario",
        "multi_model_session",
        "file_heavy_operations",
        "test_cli_fixture",
        "test_search_fixture",
        "test_coding_fixture",
        "comprehensive_test",
        "gemini_coding_loop",
        "pressure_session",
        "runtime_conformance",
        "cache_stable_research_turns",
        "refined_search_recovery",
        # Green stress fixtures
        "metric_long_debug",
        "burst_retries",
        "verbose_payload",
        "straddling_boundary",
        "context_pinned_file",
        "alternating_adapters",
        "branching_tests",
        "compression_hypothesis_churn",
        "heavy_tool_event",
        "tool_density_micro",
        "episodes_multi_phase",
        "release_reacquisition",
        "orchestrator_parity",
        "bridge_vs_orchestrator",
        "cache_sensitivity",
    ]


def get_redteam_fixtures() -> list[str]:
    """Get the negative/expected-failure fixture set."""
    return [
        "grammar_drift",
        "markdown_fallback",
        "subtle_drift",
        "healing_drift",
        "repeat_search_pressure",
    ]


def select_fixtures(fixture_set: str) -> list[str]:
    """Select fixtures based on the specified set."""
    if fixture_set == "feature":
        return get_feature_fixtures()
    if fixture_set == "full":
        return get_full_fixtures()
    if fixture_set == "redteam":
        return get_redteam_fixtures()
    raise ValueError(f"Unknown fixture set: {fixture_set}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Select fixtures for Tok gate checks"
    )
    parser.add_argument(
        "--set",
        choices=["feature", "full", "redteam"],
        required=True,
        help="Fixture set to use",
    )
    parser.add_argument(
        "--output", default="fixtures.json", help="Output JSON file path"
    )

    args = parser.parse_args()

    try:
        fixtures = select_fixtures(args.set)

        # Create output data structure
        output_data = {
            "fixture_set": args.set,
            "fixtures": fixtures,
            "count": len(fixtures),
        }

        # Write to output file
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2))

        print(f"Selected {len(fixtures)} fixtures for '{args.set}' set")
        print(f"Output written to: {output_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
