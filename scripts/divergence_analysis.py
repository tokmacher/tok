"""Divergence analysis script for Tok benchmark regression investigation.

Identifies the turn where tok and baseline runs diverge and classifies
the divergence type (extra_read, redundant_search, loop, or novel).

Usage:
    python scripts/divergence_analysis.py --base tmp/tiny_allflags_20260411_070743 --output tmp/divergence_allflags.csv
    python scripts/divergence_analysis.py --base tmp/release_gate_openai41_post_pathfix_r5_20260413_102648 --output tmp/divergence_gpt41.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

FILE_LIKE_TOOLS = {"view", "view_file", "read", "read_file", "cat", "open_file", "get_file"}
SEARCH_LIKE_TOOLS = {"grep", "grep_search", "search", "find", "rg", "ripgrep"}
EDIT_LIKE_TOOLS = {"edit", "edit_file", "write", "write_file", "replace"}
COMMAND_LIKE_TOOLS = {"run", "run_command", "bash", "shell", "execute"}


def load_run(base_dir: str, task: str, repeat: int, mode: str) -> dict | None:
    path = Path(base_dir) / "tasks" / task / f"repeat_{repeat}" / mode / "run.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_tool_action(tool_record: dict) -> str:
    tool_name = tool_record.get("tool_name", tool_record.get("canonical_tool_name", ""))
    tool_input = tool_record.get("tool_input", {})
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            tool_input = {}
    if tool_name in FILE_LIKE_TOOLS:
        path_val = tool_input.get("path", tool_input.get("file_path", ""))
        return f"read:{path_val}"
    elif tool_name in SEARCH_LIKE_TOOLS:
        query = tool_input.get("query", tool_input.get("pattern", tool_input.get("search_term", "")))
        return f"search:{query}"
    elif tool_name in EDIT_LIKE_TOOLS:
        path_val = tool_input.get("path", tool_input.get("file_path", ""))
        return f"edit:{path_val}"
    elif tool_name in COMMAND_LIKE_TOOLS:
        cmd = tool_input.get("command", tool_input.get("cmd", ""))
        return f"cmd:{cmd[:60]}"
    else:
        return f"{tool_name}"


def classify_divergence(
    baseline_actions: list[str],
    tok_actions: list[str],
    all_baseline_actions: list[str],
    all_tok_actions: list[str],
) -> str:
    if not tok_actions and not baseline_actions:
        return "novel"
    if not tok_actions:
        return "novel"
    tok_action = tok_actions[0]
    if tok_action.startswith("read:"):
        file_path = tok_action[5:]
        if any(a.startswith(f"read:{file_path}") for a in all_baseline_actions):
            return "extra_read"
        if any(a.startswith(f"read:{file_path}") for a in all_tok_actions[: len(all_tok_actions) - 1]):
            return "loop"
        return "extra_read"
    elif tok_action.startswith("search:"):
        query = tok_action[8:]
        if any(a.startswith(f"search:{query}") for a in all_baseline_actions):
            return "redundant_search"
        if any(a.startswith(f"search:{query}") for a in all_tok_actions[: len(all_tok_actions) - 1]):
            return "loop"
        return "redundant_search"
    elif tok_action.startswith("edit:"):
        file_path = tok_action[5:]
        if any(a.startswith(f"edit:{file_path}") for a in all_tok_actions[: len(all_tok_actions) - 1]):
            return "loop"
        return "novel"
    else:
        if tok_action in [a for a in all_tok_actions[: len(all_tok_actions) - 1] if a == tok_action]:
            return "loop"
        return "novel"


def analyze_run_pair(baseline_run: dict, tok_run: dict) -> dict | None:
    baseline_records = baseline_run.get("tool_records", [])
    tok_records = tok_run.get("tool_records", [])

    baseline_steps = {tr.get("step_index", i + 1): tr for i, tr in enumerate(baseline_records)}
    tok_steps = {tr.get("step_index", i + 1): tr for i, tr in enumerate(tok_records)}

    max_baseline = max(baseline_steps.keys()) if baseline_steps else 0
    max_tok = max(tok_steps.keys()) if tok_steps else 0

    all_baseline_actions_so_far: list[str] = []
    divergence_turn = None
    baseline_action_at_div = ""
    tok_action_at_div = ""
    divergence_type = ""

    for step_idx in range(1, max(max_baseline, max_tok) + 1):
        baseline_tr = baseline_steps.get(step_idx)
        tok_tr = tok_steps.get(step_idx)

        if baseline_tr is None and tok_tr is not None:
            divergence_turn = step_idx
            tok_action = get_tool_action(tok_tr)
            if any(a == tok_action for a in all_baseline_actions_so_far):
                divergence_type = (
                    "extra_read"
                    if tok_action.startswith("read:")
                    else ("redundant_search" if tok_action.startswith("search:") else "loop")
                )
            else:
                divergence_type = "novel"
            baseline_action_at_div = "(none)"
            tok_action_at_div = tok_action
            break

        if baseline_tr is not None and tok_tr is None:
            divergence_turn = step_idx
            baseline_action_at_div = get_tool_action(baseline_tr)
            tok_action_at_div = "(none - baseline continued)"
            divergence_type = "novel"
            break

        if baseline_tr is not None and tok_tr is not None:
            baseline_action = get_tool_action(baseline_tr)
            tok_action = get_tool_action(tok_tr)
            if baseline_action != tok_action:
                divergence_turn = step_idx
                baseline_action_at_div = baseline_action
                tok_action_at_div = tok_action
                tok_actions_so_far = [get_tool_action(tr) for tr in tok_records[: step_idx - 1]]
                divergence_type = classify_divergence(
                    [baseline_action],
                    [tok_action],
                    all_baseline_actions_so_far,
                    tok_actions_so_far,
                )
                break

        if baseline_tr is not None:
            all_baseline_actions_so_far.append(get_tool_action(baseline_tr))

    if divergence_turn is None and len(tok_records) > len(baseline_records):
        divergence_turn = max_baseline + 1
        first_extra = tok_records[max_baseline] if max_baseline < len(tok_records) else tok_records[-1]
        tok_action_at_div = get_tool_action(first_extra)
        baseline_action_at_div = "(baseline ended)"
        divergence_type = "novel"

    baseline_usage = baseline_run.get("provider_usage", {})
    tok_usage = tok_run.get("provider_usage", {})

    return {
        "divergence_turn": divergence_turn or "",
        "divergence_type": divergence_type or "none",
        "baseline_action": baseline_action_at_div,
        "tok_action": tok_action_at_div,
        "baseline_total_tokens": baseline_usage.get("total_tokens", 0),
        "tok_total_tokens": tok_usage.get("total_tokens", 0),
        "baseline_tool_calls": baseline_run.get("tool_calls", len(baseline_records)),
        "tok_tool_calls": tok_run.get("tool_calls", len(tok_records)),
        "baseline_prompt_tokens": baseline_usage.get("prompt_tokens", 0),
        "tok_prompt_tokens": tok_usage.get("prompt_tokens", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze divergences between baseline and tok benchmark runs")
    parser.add_argument("--base", required=True, help="Base directory containing benchmark data")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--mode", default="tok-universal", help="Tok mode to compare (default: tok-universal)")
    args = parser.parse_args()

    base_path = Path(args.base)
    tasks_dir = base_path / "tasks"

    if not tasks_dir.exists():
        print(f"Error: tasks directory not found at {tasks_dir}")
        sys.exit(1)

    results = []
    divergence_counts = Counter()

    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name

        for repeat_dir in sorted(task_dir.iterdir()):
            if not repeat_dir.is_dir() or not repeat_dir.name.startswith("repeat_"):
                continue
            repeat = int(repeat_dir.name.split("_")[1])

            baseline_run = load_run(args.base, task_id, repeat, "baseline")
            tok_run = load_run(args.base, task_id, repeat, args.mode)

            if baseline_run is None or tok_run is None:
                continue

            analysis = analyze_run_pair(baseline_run, tok_run)
            if analysis is None:
                continue

            analysis["task"] = task_id
            analysis["repeat"] = repeat
            results.append(analysis)
            divergence_counts[analysis["divergence_type"]] += 1

    fieldnames = [
        "task",
        "repeat",
        "divergence_turn",
        "divergence_type",
        "baseline_action",
        "tok_action",
        "baseline_prompt_tokens",
        "tok_prompt_tokens",
        "baseline_total_tokens",
        "tok_total_tokens",
        "baseline_tool_calls",
        "tok_tool_calls",
    ]

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nDivergence analysis complete. {len(results)} paired runs analyzed.")
    print(f"Results written to {args.output}")
    print("\nDivergence type breakdown:")
    total = sum(divergence_counts.values())
    for dtype, count in divergence_counts.most_common():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {dtype}: {count} ({pct:.1f}%)")

    extra_turns = [r for r in results if r["tok_tool_calls"] > r["baseline_tool_calls"]]
    if extra_turns:
        avg_extra = sum(r["tok_tool_calls"] - r["baseline_tool_calls"] for r in extra_turns) / len(extra_turns)
        print(f"\nRuns with extra tok turns: {len(extra_turns)}/{len(results)}")
        print(f"Average extra turns: {avg_extra:.1f}")

    prompt_deltas = [
        r["tok_prompt_tokens"] - r["baseline_prompt_tokens"]
        for r in results
        if r["tok_prompt_tokens"] and r["baseline_prompt_tokens"]
    ]
    if prompt_deltas:
        avg_prompt_delta = sum(prompt_deltas) / len(prompt_deltas)
        print(f"\nAverage prompt token delta: {avg_prompt_delta:+.0f}")


if __name__ == "__main__":
    main()
