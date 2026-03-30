"""Generate a comprehensive system prompt bloat attribution report.

Reads the baseline JSON (or re-runs measurements if not present),
computes component-wise attribution, trend analysis, and writes:
  - tmp/prompt_bloat_analysis_report.json
  - docs/prompt_bloat_findings.md
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BASELINE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tmp", "prompt_bloat_baseline.json"
)
REPORT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tmp", "prompt_bloat_analysis_report.json"
)
FINDINGS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "docs", "prompt_bloat_findings.md"
)


def _load_or_measure() -> dict:
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            return json.load(f)
    print("Baseline not found — running measurements...")
    import subprocess

    subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "measure_prompt_bloat.py"),
        ],
        check=True,
    )
    with open(BASELINE_PATH) as f:
        return json.load(f)


def _pct(part: float, total: float) -> str:
    if total == 0:
        return "0%"
    return f"{part / total * 100:.1f}%"


def compute_attribution(data: dict) -> dict:
    """Attribute token counts to each major component in a typical session."""
    bp = data["base_prompts"]
    ds = data["directive_sizes"]
    gs = data["grammar_snippets"]
    pi = data["pressure_impact"]
    mg = data["memory_growth"]
    sb = data["scenario_baselines"]

    # Typical session total (from injected system prompt)
    typical_total = sb["typical_session"]["tokens"]
    high_pressure_total = sb["high_pressure"]["tokens"]

    # Break down typical session:
    # - cold_start = minimal directive (MODE header + TOK_OUTPUT_DIRECTIVE_MINIMAL)
    cold_start_tokens = sb["cold_start"]["tokens"]
    # - additional state injection over cold
    state_overhead = sb["typical_session"]["tokens"] - cold_start_tokens

    # Incremental cost of pressure escalation
    pressure_delta_low_to_high = (
        pi["pressure_75"]["tokens"] - pi["pressure_0"]["tokens"]
    )
    law_tokens = ds["TOK_PROTOCOL_LAW"]["tokens"]
    reinforced_tokens = ds["TOK_OUTPUT_DIRECTIVE_REINFORCED"]["tokens"]
    minimal_tokens = ds["TOK_OUTPUT_DIRECTIVE_MINIMAL"]["tokens"]

    # Grammar contribution
    grammar_full_tokens = gs["full"]["tokens"]
    grammar_restricted_tokens = gs["restricted"]["tokens"]
    grammar_essentials_tokens = gs["essentials"]["tokens"]

    # Memory contribution at max growth
    turns_50 = mg["turns_50"]["tokens"]
    turns_1 = mg["turns_1"]["tokens"]
    memory_growth_delta = turns_50 - turns_1

    # Full TOK_SYSTEM_PROMPT cost if ever included
    system_prompt_full = bp["TOK_SYSTEM_PROMPT"]["tokens"]

    return {
        "typical_session_breakdown": {
            "total_tokens": typical_total,
            "minimal_directive_baseline": cold_start_tokens,
            "tok_state_overhead": state_overhead,
            "minimal_directive_pct": _pct(cold_start_tokens, typical_total),
            "tok_state_pct": _pct(state_overhead, typical_total),
        },
        "high_pressure_breakdown": {
            "total_tokens": high_pressure_total,
            "pressure_escalation_delta": pressure_delta_low_to_high,
            "pressure_escalation_pct_of_highpressure": _pct(
                pressure_delta_low_to_high, high_pressure_total
            ),
            "protocol_law_tokens": law_tokens,
            "reinforced_vs_minimal_extra": reinforced_tokens - minimal_tokens,
        },
        "grammar_bootstrap_cost": {
            "essentials": grammar_essentials_tokens,
            "restricted": grammar_restricted_tokens,
            "full_grammar": grammar_full_tokens,
            "full_vs_restricted_overhead": grammar_full_tokens
            - grammar_restricted_tokens,
            "full_vs_essentials_overhead": grammar_full_tokens
            - grammar_essentials_tokens,
        },
        "base_system_prompt": {
            "tok_system_prompt_tokens": system_prompt_full,
            "note": "Only included during explicit grammar=full delegations",
        },
        "memory_wire_state": {
            "at_1_turn": turns_1,
            "at_50_turns": turns_50,
            "growth_delta": memory_growth_delta,
            "plateau_note": "Wire state plateaus quickly due to HOT_LIMITS capping fields",
        },
        "directive_component_costs": {
            "minimal_directive": minimal_tokens,
            "full_directive": ds["TOK_OUTPUT_DIRECTIVE"]["tokens"],
            "reinforced_directive": reinforced_tokens,
            "protocol_law": law_tokens,
            "tool_compat_directive": ds["TOK_TOOL_COMPAT_DIRECTIVE"]["tokens"],
            "law_plus_reinforced": law_tokens + reinforced_tokens,
        },
    }


def compute_top_contributors(data: dict, attribution: dict) -> list[dict]:
    """Rank the top bloat contributors by token cost in realistic scenarios."""
    ds = data["directive_sizes"]
    gs = data["grammar_snippets"]
    bp = data["base_prompts"]

    contributors = [
        {
            "rank": 1,
            "component": "TOK_SYSTEM_PROMPT (grammar=full bootstrap)",
            "tokens": bp["TOK_SYSTEM_PROMPT"]["tokens"],
            "chars": bp["TOK_SYSTEM_PROMPT"]["chars"],
            "scenario": "Grammar delegation with level='full'",
            "frequency": "Each Delegate bootstrap with full grammar",
            "notes": (
                "891 tokens. Only used during @Delegate grammar='full' paths. "
                "Switching to 'restricted' (133t) or 'essentials' (87t) saves 758-804 tokens."
            ),
        },
        {
            "rank": 2,
            "component": "Pressure escalation: TOK_PROTOCOL_LAW + REINFORCED directive",
            "tokens": attribution["directive_component_costs"][
                "law_plus_reinforced"
            ],
            "chars": (
                ds["TOK_PROTOCOL_LAW"]["chars"]
                + ds["TOK_OUTPUT_DIRECTIVE_REINFORCED"]["chars"]
            ),
            "scenario": "pressure > 50 (protocol drift detected)",
            "frequency": "Triggered on any turn where invisible pressure exceeds 50",
            "notes": (
                "295 tokens combined vs 56 tokens at pressure=0 (minimal only). "
                "239 extra tokens per escalated turn. "
                "Law block (155t) fires at pressure>0; reinforced (140t) fires at >50."
            ),
        },
        {
            "rank": 3,
            "component": "TOK_OUTPUT_DIRECTIVE (full, not minimal)",
            "tokens": ds["TOK_OUTPUT_DIRECTIVE"]["tokens"],
            "chars": ds["TOK_OUTPUT_DIRECTIVE"]["chars"],
            "scenario": "Any session where full directive is injected instead of minimal",
            "frequency": "Appears in pressure 0-50 range if minimal is not selected",
            "notes": (
                "177 tokens vs 56 tokens for minimal variant. "
                "121 extra tokens. Current code defaults to minimal at 0<pressure<=50, "
                "so this is mainly a regression risk."
            ),
        },
        {
            "rank": 4,
            "component": "Grammar bootstrap: restricted vs essentials vs full",
            "tokens": gs["restricted"]["tokens"],
            "chars": gs["restricted"]["chars"],
            "scenario": "Delegate bootstrap with grammar level",
            "frequency": "Every delegated sub-agent call",
            "notes": (
                f"Essentials: {gs['essentials']['tokens']}t, "
                f"Restricted: {gs['restricted']['tokens']}t, "
                f"Full: {gs['full']['tokens']}t. "
                f"Full grammar costs 10x more than essentials."
            ),
        },
        {
            "rank": 5,
            "component": "Memory wire_state (tok_state injection)",
            "tokens": data["memory_growth"]["turns_50"]["tokens"],
            "chars": data["memory_growth"]["turns_50"]["chars"],
            "scenario": "Any session with active bridge memory state",
            "frequency": "Every turn after the first",
            "notes": (
                "Plateaus at ~75 tokens after 5 turns due to HOT_LIMITS caps. "
                "Growth is bounded — not a major accumulation risk. "
                "Pointer compression keeps long file paths token-efficient."
            ),
        },
    ]
    return contributors


def compute_compression_efficiency(data: dict) -> dict:
    """Summarize compression effectiveness across all compression paths."""
    hc = data["history_compression"]
    tc = data["tool_compression"]
    mc = data["memory_compression"]

    history_ratios = {k: v["compression_ratio"] for k, v in hc.items()}
    tool_ratios = {k: v["ratio"] for k, v in tc.items()}
    memory_ratios = {k: v["ratio"] for k, v in mc.items()}

    avg_history = sum(history_ratios.values()) / max(1, len(history_ratios))
    avg_tool = sum(tool_ratios.values()) / max(1, len(tool_ratios))
    avg_memory = sum(memory_ratios.values()) / max(1, len(memory_ratios))

    return {
        "history_compression": {
            "ratios_by_scenario": history_ratios,
            "avg_ratio": round(avg_history, 3),
            "note": "Ratio = (recent + tok_state tokens) / original tokens",
        },
        "tool_result_compression": {
            "ratios_by_type": tool_ratios,
            "avg_ratio": round(avg_tool, 3),
            "note": "Ratio = compressed tool result / original",
        },
        "memory_entry_compression": {
            "ratios_by_size": memory_ratios,
            "avg_ratio": round(avg_memory, 3),
            "note": "Ratio = compress_user_prompt output / input",
        },
    }


def compute_trend_analysis(data: dict) -> dict:
    """Show how wire_state size changes over turns."""
    mg = data["memory_growth"]
    trend = {}
    for n in [1, 5, 10, 20, 50]:
        key = f"turns_{n}"
        if key in mg:
            trend[f"turn_{n}"] = {
                "wire_state_tokens": mg[key]["tokens"],
                "wire_state_chars": mg[key]["chars"],
                "full_tok_tokens": mg.get(f"{key}_full_tok", {}).get(
                    "tokens", "n/a"
                ),
            }
    return trend


def generate_report(data: dict) -> dict:
    attribution = compute_attribution(data)
    top_contributors = compute_top_contributors(data, attribution)
    compression = compute_compression_efficiency(data)
    trend = compute_trend_analysis(data)

    # Optimization opportunity estimates
    optimizations = [
        {
            "contributor": "Pressure escalation overhead",
            "current_cost_tokens": attribution["directive_component_costs"][
                "law_plus_reinforced"
            ],
            "optimized_cost_tokens": attribution["directive_component_costs"][
                "minimal_directive"
            ],
            "savings_tokens": (
                attribution["directive_component_costs"]["law_plus_reinforced"]
                - attribution["directive_component_costs"]["minimal_directive"]
            ),
            "strategy": (
                "Merge TOK_PROTOCOL_LAW key rules into a compressed ~50-token enforcement block. "
                "Delay REINFORCED directive until pressure>75 instead of >50. "
                "Estimated savings: ~239 tokens per escalated turn."
            ),
        },
        {
            "contributor": "Grammar bootstrap level selection",
            "current_cost_tokens": data["grammar_snippets"]["full"]["tokens"],
            "optimized_cost_tokens": data["grammar_snippets"]["essentials"][
                "tokens"
            ],
            "savings_tokens": (
                data["grammar_snippets"]["full"]["tokens"]
                - data["grammar_snippets"]["essentials"]["tokens"]
            ),
            "strategy": (
                "Default delegation to 'essentials' (87t) or 'restricted' (133t) instead of 'full' (891t). "
                "Only use 'full' for first-turn or cold-start delegations. "
                "Estimated savings: 758-804 tokens per delegation turn."
            ),
        },
        {
            "contributor": "TOK_OUTPUT_DIRECTIVE full vs minimal",
            "current_cost_tokens": data["directive_sizes"][
                "TOK_OUTPUT_DIRECTIVE"
            ]["tokens"],
            "optimized_cost_tokens": data["directive_sizes"][
                "TOK_OUTPUT_DIRECTIVE_MINIMAL"
            ]["tokens"],
            "savings_tokens": (
                data["directive_sizes"]["TOK_OUTPUT_DIRECTIVE"]["tokens"]
                - data["directive_sizes"]["TOK_OUTPUT_DIRECTIVE_MINIMAL"][
                    "tokens"
                ]
            ),
            "strategy": (
                "Current code already uses TOK_OUTPUT_DIRECTIVE_MINIMAL at pressure<=50 (56t). "
                "Ensure no code path accidentally emits the full directive (177t). "
                "Consider further compressing the minimal directive itself — "
                "it still has 56 tokens of structural overhead."
            ),
        },
        {
            "contributor": "TOK_PROTOCOL_LAW firing threshold",
            "current_cost_tokens": data["directive_sizes"]["TOK_PROTOCOL_LAW"][
                "tokens"
            ],
            "optimized_cost_tokens": 0,
            "savings_tokens": data["directive_sizes"]["TOK_PROTOCOL_LAW"][
                "tokens"
            ],
            "strategy": (
                "TOK_PROTOCOL_LAW (155t) fires at pressure>0. "
                "Raise threshold to pressure>25 to skip it for minor drift. "
                "Or compress it to a ~50-token inline reminder instead of 155 tokens. "
                "Estimated savings: 155 tokens on turns 1-50 pressure range."
            ),
        },
    ]

    return {
        "summary": {
            "baseline_scenarios": {
                "cold_start_tokens": data["scenario_baselines"]["cold_start"][
                    "tokens"
                ],
                "typical_session_tokens": data["scenario_baselines"][
                    "typical_session"
                ]["tokens"],
                "high_pressure_tokens": data["scenario_baselines"][
                    "high_pressure"
                ]["tokens"],
                "memory_heavy_wire_state_tokens": data["scenario_baselines"][
                    "memory_heavy"
                ]["wire_state"]["tokens"],
                "memory_heavy_injected_tokens": data["scenario_baselines"][
                    "memory_heavy"
                ]["injected_system"]["tokens"],
            },
        },
        "attribution": attribution,
        "top_contributors": top_contributors,
        "compression_efficiency": compression,
        "trend_analysis": trend,
        "optimization_opportunities": optimizations,
        "directive_overlap": data.get("directive_overlap", {}),
        "memory_profiles": data["memory_profiles"],
    }


def write_markdown(report: dict) -> str:
    top = report["top_contributors"]
    attr = report["attribution"]
    opts = report["optimization_opportunities"]
    sb = report["summary"]["baseline_scenarios"]
    trend = report["trend_analysis"]
    ce = report["compression_efficiency"]

    lines = [
        "# Tok System Prompt Bloat — Findings",
        "",
        "Generated by `scripts/generate_bloat_report.py`.",
        "",
        "## Executive Summary",
        "",
        "System prompt size varies from **64 tokens** (cold start, no memory) to **335 tokens**",
        "(high-pressure with state). The three biggest contributors are:",
        "",
    ]
    for c in top[:3]:
        lines.append(f"1. **{c['component']}** — {c['tokens']} tokens")
    lines += ["", "---", ""]

    lines += [
        "## Baseline Scenarios",
        "",
        "| Scenario | Tokens | Chars |",
        "|----------|-------:|------:|",
        f"| Cold start (no memory, no directives) | {sb['cold_start_tokens']} | — |",
        f"| Typical session (state + minimal directive) | {sb['typical_session_tokens']} | — |",
        f"| High pressure (state + law + reinforced) | {sb['high_pressure_tokens']} | — |",
        f"| Memory-heavy wire state (50 turns) | {sb['memory_heavy_wire_state_tokens']} | — |",
        f"| Memory-heavy injected system | {sb['memory_heavy_injected_tokens']} | — |",
        "",
        "---",
        "",
        "## Component Token Costs",
        "",
        "### Base Directives",
        "",
        "| Component | Tokens | Chars |",
        "|-----------|-------:|------:|",
    ]
    dc = attr["directive_component_costs"]
    lines += [
        f"| TOK_OUTPUT_DIRECTIVE_MINIMAL | {dc['minimal_directive']} | — |",
        f"| TOK_OUTPUT_DIRECTIVE (full) | {dc['full_directive']} | — |",
        f"| TOK_OUTPUT_DIRECTIVE_REINFORCED | {dc['reinforced_directive']} | — |",
        f"| TOK_PROTOCOL_LAW | {dc['protocol_law']} | — |",
        f"| TOK_TOOL_COMPAT_DIRECTIVE | {dc['tool_compat_directive']} | — |",
        f"| **LAW + REINFORCED combined** | **{dc['law_plus_reinforced']}** | — |",
        "",
        "### Grammar Bootstrap Levels",
        "",
        "| Level | Tokens | Chars |",
        "|-------|-------:|------:|",
    ]
    gc = attr["grammar_bootstrap_cost"]
    lines += [
        f"| essentials | {gc['essentials']} | — |",
        f"| restricted | {gc['restricted']} | — |",
        f"| full (= TOK_SYSTEM_PROMPT) | {gc['full_grammar']} | — |",
        f"| Full vs. essentials overhead | **{gc['full_vs_essentials_overhead']}** | — |",
        "",
        "---",
        "",
        "## Top 3 Bloat Contributors",
        "",
    ]

    for c in top[:3]:
        lines += [
            f"### {c['rank']}. {c['component']}",
            "",
            f"- **Tokens:** {c['tokens']}",
            f"- **Scenario:** {c['scenario']}",
            f"- **Notes:** {c['notes']}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Wire State Growth Over Turns",
        "",
        "Wire state plateaus quickly — bounded by `HOT_LIMITS` field caps.",
        "",
        "| Turns | Wire State (tokens) | Full Tok Serialization (tokens) |",
        "|-------|-------------------:|--------------------------------:|",
    ]
    for k, v in trend.items():
        lines.append(
            f"| {k.replace('turn_', '')} | {v['wire_state_tokens']} | {v['full_tok_tokens']} |"
        )

    lines += [
        "",
        "**Key finding:** Wire state saturates at ~75 tokens after 5 turns.",
        "The `to_tok()` full serialization (with scores/timestamps) grows with history",
        "but is only used for persistent memory files, not injected per-turn.",
        "",
        "---",
        "",
        "## Compression Effectiveness",
        "",
        "### History Compression Ratios",
        "",
        "| Scenario | Compression Ratio |",
        "|----------|------------------:|",
    ]
    for k, v in ce["history_compression"]["ratios_by_scenario"].items():
        lines.append(f"| {k} | {v} |")

    lines += [
        f"| **Average** | **{ce['history_compression']['avg_ratio']}** |",
        "",
        "_(Ratio = (recent_kept + tok_state) / original. Lower = more compression.)_",
        "",
        "### Tool Result Compression Ratios",
        "",
        "| Content Type | Compression Ratio |",
        "|--------------|------------------:|",
    ]
    for k, v in ce["tool_result_compression"]["ratios_by_type"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        f"| **Average** | **{ce['tool_result_compression']['avg_ratio']}** |",
        "",
        "---",
        "",
        "## Optimization Recommendations",
        "",
    ]
    for i, opt in enumerate(opts, 1):
        lines += [
            f"### {i}. {opt['contributor']}",
            "",
            f"- **Current cost:** {opt['current_cost_tokens']} tokens",
            f"- **Potential optimized cost:** {opt['optimized_cost_tokens']} tokens",
            f"- **Estimated savings:** {opt['savings_tokens']} tokens",
            f"- **Strategy:** {opt['strategy']}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Investigation Answers",
        "",
        "### Q1: Token count of each base system prompt component",
        "",
        "| Component | Tokens |",
        "|-----------|-------:|",
        "| TOK_SYSTEM_PROMPT | 891 |",
        "| TOK_PROTOCOL_LAW | 155 |",
        "| TOK_OUTPUT_DIRECTIVE (full) | 177 |",
        "| TOK_OUTPUT_DIRECTIVE_MINIMAL | 56 |",
        "| TOK_OUTPUT_DIRECTIVE_REINFORCED | 140 |",
        "| Grammar: essentials | 87 |",
        "| Grammar: restricted | 133 |",
        "| Grammar: full | 891 |",
        "| Grammar: pulse | 49 |",
        "| Grammar: explore | 108 |",
        "",
        "### Q2: Bridge memory contribution over time",
        "",
        "- Wire state starts at ~49 tokens (turn 1) and plateaus at **~75 tokens** by turn 5.",
        "- Growth is bounded by `HOT_LIMITS` — a hard cap on field entries.",
        "- Compression ratio of wire state is excellent: HOT_LIMITS prevent runaway growth.",
        "- `to_tok()` full serialization (persistent file) grows but is not injected per-turn.",
        "",
        "### Q3: Directive escalation and bloat",
        "",
        "- Pressure > 0: adds **155 tokens** (TOK_PROTOCOL_LAW)",
        "- Pressure > 50: replaces minimal (56t) with reinforced (140t) = **+84 tokens**",
        "- Total escalation cost at pressure=75: **239 extra tokens** vs pressure=0",
        "",
        "### Q4: History compression effectiveness",
        "",
        "- Average compression ratio: **{:.2f}** across tested scenarios".format(
            ce["history_compression"]["avg_ratio"]
        ),
        "- Compresses old turns into a `>>> ...` state line (~75 tokens max)",
        "- Recent window kept verbatim (2 turns by default)",
        "- Tool result compression gives additional savings on large results",
        "",
        "### Q5: Top 3 biggest contributors",
        "",
    ]
    for c in top[:3]:
        lines.append(
            f"{c['rank']}. **{c['component']}** — {c['tokens']} tokens"
        )
        lines.append(f"   - {c['notes']}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    print("Tok System Prompt Bloat — Analysis Report")
    print("=" * 60)

    data = _load_or_measure()
    report = generate_report(data)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {REPORT_PATH}")

    os.makedirs(os.path.dirname(FINDINGS_PATH), exist_ok=True)
    md = write_markdown(report)
    with open(FINDINGS_PATH, "w") as f:
        f.write(md)
    print(f"Findings saved to: {FINDINGS_PATH}")

    # Print top contributors
    print("\n--- Top 3 Bloat Contributors ---")
    for c in report["top_contributors"][:3]:
        print(f"\n{c['rank']}. {c['component']}")
        print(f"   Tokens: {c['tokens']}")
        print(f"   Scenario: {c['scenario']}")
        print(f"   Notes: {c['notes'][:120]}...")

    print("\n--- Optimization Summary ---")
    for opt in report["optimization_opportunities"]:
        savings = opt["savings_tokens"]
        print(f"  [{savings:+d}t] {opt['contributor']}")


if __name__ == "__main__":
    main()
