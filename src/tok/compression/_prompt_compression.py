"""Prompt compression helpers.

This module extracts the low-level heuristics used by `tok.compression.compress_user_prompt`
so that the package `__init__` can remain a thin facade.
"""

from __future__ import annotations

import re


def _extract_goal_from_line(line: str) -> str | None:
    lower = line.lower()
    if any(prefix in lower for prefix in ("task:", "goal:", "requirement:", "implement ", "add ")):
        return line[:60]
    if line.startswith(("- ", "* ", "1. ")) and any(
        keyword in lower for keyword in ("should", "must", "need to", "implement")
    ):
        return re.sub(r"^[-*1.\s]+", "", line)[:60]
    return None


def _extract_constraint_from_line(line: str) -> str | None:
    lower = line.lower()
    if any(keyword in lower for keyword in ("avoid", "don't", "do not", "never", "only")):
        return line[:60]
    return None


def _extract_files_from_line(line: str) -> set[str]:
    files: set[str] = set()
    for match in re.finditer(
        r"\b([\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|sh|txt|css|html|sql|rs|go|rb))\b",
        line,
    ):
        files.add(match.group(1))
    return files


def _filter_prompt_lines(lines: list[str]) -> list[str]:
    filtered: list[str] = []
    for line in lines:
        lower = line.lower()
        if (
            line.startswith(">>>")
            or "optimized task context" in lower
            or any(line.startswith(prefix) for prefix in ("goal:", "files:", "constraints:"))
        ):
            continue
        filtered.append(line)
    return filtered


def _extract_prompt_content(filtered_lines: list[str]) -> tuple[list[str], list[str], set[str]]:
    goals: list[str] = []
    constraints: list[str] = []
    files: set[str] = set()

    for line in filtered_lines:
        goal = _extract_goal_from_line(line)
        if goal:
            goals.append(goal)
        constraint = _extract_constraint_from_line(line)
        if constraint:
            constraints.append(constraint)
        files.update(_extract_files_from_line(line))

    return goals, constraints, files


def _build_prompt_result(
    goals: list[str],
    constraints: list[str],
    files: set[str],
    filtered_lines: list[str],
    original_prompt: str,
) -> str:
    parts: list[str] = []
    if goals:
        parts.append(f"goal:{','.join(goals[:2])}")
    if files:
        parts.append(f"files:{','.join(list(files)[:3])}")
    if constraints:
        parts.append(f"constraints:{','.join(constraints[:2])}")

    if not parts:
        for line in filtered_lines:
            if len(line) > 10:
                return f"goal:{line[:100].strip()}"
        return f"goal:{original_prompt[:100].strip()}"

    return "|".join(parts)


def compress_user_prompt(prompt: str) -> str:
    """Extract tasks, requirements, and constraints from a verbose prompt."""
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    filtered_lines = _filter_prompt_lines(lines)

    if not filtered_lines and "optimized task context" in prompt.lower():
        return re.sub(r"### Optimized Task Context\n", "", prompt, flags=re.IGNORECASE).strip()

    goals, constraints, files = _extract_prompt_content(filtered_lines)
    return _build_prompt_result(goals, constraints, files, filtered_lines, prompt)
