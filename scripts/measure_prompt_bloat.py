"""Baseline measurement script for system prompt bloat in the Tok runtime.

Runs all scenario measurements and saves results to tmp/prompt_bloat_baseline.json.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tok.analysis.prompt_analyzer import (
    measure_base_prompts,
    measure_directive_sizes,
    measure_grammar_snippets,
    measure_pressure_impact,
    measure_dynamic_injections,
    measure_per_turn_actual,
    simulate_memory_growth,
    measure_memory_profiles,
    analyze_history_compression,
    measure_tool_compression_impact,
    analyze_directive_overlap,
    run_all_baselines,
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tmp", "prompt_bloat_baseline.json"
)


def _collect_results() -> dict:
    results: dict = {}
    steps = [
        (
            "\n[1/10] Base prompt components...",
            "base_prompts",
            measure_base_prompts,
        ),
        (
            "[2/10] Directive sizes...",
            "directive_sizes",
            measure_directive_sizes,
        ),
        (
            "[3/10] Grammar snippets...",
            "grammar_snippets",
            measure_grammar_snippets,
        ),
        (
            "[4/10] Pressure impact...",
            "pressure_impact",
            measure_pressure_impact,
        ),
        (
            "[5/10] Dynamic injections...",
            "dynamic_injections",
            measure_dynamic_injections,
        ),
        (
            "[6/10] Memory growth simulation...",
            "memory_growth",
            simulate_memory_growth,
        ),
        (
            "[7/10] Memory profiles...",
            "memory_profiles",
            measure_memory_profiles,
        ),
        (
            "[8/10] Memory compression...",
            "memory_compression",
            analyze_history_compression,
        ),
        (
            "[9/10] History compression...",
            "history_compression",
            analyze_history_compression,
        ),
        (
            "[10/10] Tool result compression...",
            "tool_compression",
            measure_tool_compression_impact,
        ),
    ]
    for msg, key, fn in steps:
        print(msg)
        results[key] = fn()

    print("[+] Per-turn actual injection (gateway path)...")
    results["per_turn_actual"] = measure_per_turn_actual()
    results["scenario_baselines"] = run_all_baselines()
    results["directive_overlap"] = analyze_directive_overlap()
    return results


def _row(label: str, data: dict) -> None:
    tokens = data.get("tokens", "?")
    chars = data.get("chars", "?")
    print(f"  {label:<43} {tokens:>7} {chars:>7}")


def _print_section(label: str, items: dict) -> None:
    print()
    for k, v in items.items():
        _row(k, v)


def _print_scenario_baselines(results: dict) -> None:
    print()
    print("Scenario baselines:")
    for k, v in results["scenario_baselines"].items():
        if isinstance(v, dict) and "tokens" in v:
            _row(k, v)
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, dict) and "tokens" in sv:
                    _row(f"  {k}/{sk}", sv)


def _print_per_turn_actual(results: dict) -> None:
    print()
    print("Per-turn actual (gateway path, grammar=None):")
    for k, v in results["per_turn_actual"].items():
        if k != "_component_costs" and isinstance(v, dict) and "tokens" in v:
            _row(k, v)


def _print_memory_growth(results: dict) -> None:
    print()
    print("Memory growth (wire_state tokens):")
    for k, v in results["memory_growth"].items():
        if "_full_tok" not in k:
            _row(k, v)


def _print_summary(results: dict) -> None:
    print("\n--- Quick Summary ---")
    print(f"{'Component':<45} {'Tokens':>7} {'Chars':>7}")
    print("-" * 62)
    _print_section("", results["base_prompts"])
    _print_section("", results["directive_sizes"])
    _print_section("", results["grammar_snippets"])
    _print_section("", results["pressure_impact"])
    _print_scenario_baselines(results)
    _print_per_turn_actual(results)
    _print_memory_growth(results)

    print()
    print("Scenario baselines:")
    for k, v in results["scenario_baselines"].items():
        if isinstance(v, dict) and "tokens" in v:
            _row(k, v)
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, dict) and "tokens" in sv:
                    _row(f"  {k}/{sk}", sv)

    print()
    print("Per-turn actual (gateway path, grammar=None):")
    for k, v in results["per_turn_actual"].items():
        if k != "_component_costs" and isinstance(v, dict) and "tokens" in v:
            _row(k, v)

    print()
    print("Memory growth (wire_state tokens):")
    for k, v in results["memory_growth"].items():
        if "_full_tok" not in k:
            _row(k, v)


def main() -> None:
    print("Tok System Prompt Bloat — Baseline Measurements")
    print("=" * 60)

    results = _collect_results()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline saved to: {OUTPUT_PATH}")
    _print_summary(results)


if __name__ == "__main__":
    main()
