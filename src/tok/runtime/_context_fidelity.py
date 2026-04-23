"""Heuristics for preserving request-context fidelity during compression/optimization.

The runtime uses these helpers to decide when to bypass compression for recently
re-read files and when prompt-optimization is likely to drop user-required
anchors (paths/labels).
"""

from __future__ import annotations

import re
from typing import Any

_STRUCTURED_ANSWER_LABEL_RE = re.compile(r"(?<![\w-])(file|verification|related)(?![\w-])\s*[:=]", re.IGNORECASE)
_CONTEXT_FIDELITY_PATH_RE = re.compile(
    r"(?<!\w)([\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|yml|sh|txt|css|html|sql|rs|go|rb))(?!\w)"
)


def compute_fidelity_overrides(
    id_to_context: dict[str, dict],
    file_reads_by_turn: dict[str, int],
    last_elevated_path: str,
    current_turn: int,
) -> tuple[set[str], str]:
    """Return paths that should bypass compression due to recent re-read.

    Returns (overrides_set, elevated_path). elevated_path is the currently
    elevated path (for continued elevation) or empty string if not elevated.
    """
    repeat_paths: set[str] = set()
    elevated_path = ""

    paths_in_request = {ctx.get("path") for ctx in id_to_context.values() if ctx.get("path")}

    if last_elevated_path and last_elevated_path in paths_in_request:
        has_different_file = any(p != last_elevated_path for p in paths_in_request)
        if has_different_file:
            return repeat_paths, ""
        for ctx in id_to_context.values():
            path = ctx.get("path")
            if path == last_elevated_path:
                repeat_paths.add(path)
                elevated_path = path
        return repeat_paths, elevated_path

    for path in paths_in_request:
        last_turn = file_reads_by_turn.get(path)
        if last_turn and (current_turn - last_turn) <= 3:
            repeat_paths.add(path)
            if not elevated_path:
                elevated_path = path

    return repeat_paths, elevated_path


def extract_requested_answer_labels(text: str) -> tuple[str, ...]:
    if not text.strip():
        return ()
    labels: list[str] = []
    seen: set[str] = set()
    for match in _STRUCTURED_ANSWER_LABEL_RE.finditer(text):
        label = match.group(1).lower()
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return tuple(labels)


def _system_prompt_text(system_prompt: str | list[dict[str, Any]] | None) -> str:
    if system_prompt is None:
        return ""
    if isinstance(system_prompt, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in system_prompt
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(system_prompt)


def _collect_required_context_anchors(user_prompt: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    labels = extract_requested_answer_labels(user_prompt)
    paths: list[str] = []
    seen: set[str] = set()
    for match in _CONTEXT_FIDELITY_PATH_RE.finditer(user_prompt):
        path = match.group(1)
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
        if len(paths) >= 6:
            break
    return labels, tuple(paths)


def prompt_optimization_materially_degrades_context(
    original_system: str | list[dict[str, Any]] | None,
    optimized_system: str | list[dict[str, Any]] | None,
    user_prompt: str,
) -> tuple[bool, str]:
    """Return (degraded, reason) when optimization removes required context anchors."""
    original_text = _system_prompt_text(original_system)
    optimized_text = _system_prompt_text(optimized_system)
    if not original_text or not optimized_text:
        return False, ""
    labels, paths = _collect_required_context_anchors(user_prompt)
    has_explicit_requirements = bool(labels or paths or (user_prompt and len(user_prompt) > 160))
    if (
        has_explicit_requirements
        and len(original_text) >= 1200
        and len(optimized_text)
        < max(
            180,
            int(len(original_text) * 0.08),
        )
    ):
        return True, "overcompressed"

    original_lower = original_text.lower()
    optimized_lower = optimized_text.lower()
    for label in labels:
        if label in original_lower and label not in optimized_lower:
            return True, "missing_required_label"
    for path in paths:
        if path in original_text and path not in optimized_text:
            return True, "missing_required_path"

    if user_prompt and len(user_prompt) > 160:
        user_anchor = user_prompt[:120].strip()
        if (
            user_anchor
            and user_anchor in original_text
            and user_anchor not in optimized_text
            and len(optimized_text) < max(220, int(len(original_text) * 0.2))
        ):
            return True, "dropped_user_anchor"
    return False, ""
