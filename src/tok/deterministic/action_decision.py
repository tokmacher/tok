"""Pure action classification and exact-patch precondition checks.

Experimental — not part of the 0.2.x public API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tok.deterministic.models import (
    DeterministicActionDecision,
    ExactPatchPreconditions,
)

_VAGUE_KINDS = frozenset({"semantic", "vague", "natural_language"})
_VAGUE_KEYWORDS = (
    "fix this",
    "fix the",
    "improve this",
    "improve the",
    "refactor this",
    "refactor the",
    "clean up",
    "clean this",
    "make it",
    "make this",
    "update this",
    "update the",
    "change this",
    "change the",
)


def classify_action(action: dict[str, Any]) -> DeterministicActionDecision:
    """Classify whether an action can be deterministically executed.

    Rejects vague/semantic actions, multi-file requests, and actions missing
    an exact expected_before_hash. Returns should_execute only for actions
    with explicit path + search + replace + expected hash.
    """
    if _is_vague_semantic(action):
        return DeterministicActionDecision(
            should_execute=False,
            should_compress_only=False,
            should_reject=True,
            reason="action_is_vague_semantic",
            required_preconditions=("path", "search", "replace", "expected_before_hash"),
            missing_preconditions=("path", "search", "replace", "expected_before_hash"),
        )

    paths = action.get("paths", [])
    if isinstance(paths, list) and len(paths) > 1:
        return DeterministicActionDecision(
            should_execute=False,
            should_compress_only=False,
            should_reject=True,
            reason="multi_file_not_supported",
            required_preconditions=("single_path",),
            missing_preconditions=("single_path",),
        )

    missing: list[str] = []
    for key in ("path", "search", "expected_before_hash"):
        if not action.get(key):
            missing.append(key)
    if "replace" not in action or action.get("replace") is None:
        missing.append("replace")

    if missing:
        return DeterministicActionDecision(
            should_execute=False,
            should_compress_only=False,
            should_reject=True,
            reason="missing_required_preconditions",
            required_preconditions=("path", "search", "replace", "expected_before_hash"),
            missing_preconditions=tuple(missing),
        )

    return DeterministicActionDecision(
        should_execute=True,
        should_compress_only=False,
        should_reject=False,
        reason="all_preconditions_satisfied",
        required_preconditions=("path", "search", "replace", "expected_before_hash"),
        missing_preconditions=(),
    )


def _is_vague_semantic(action: dict[str, Any]) -> bool:
    kind = str(action.get("kind", "")).lower()
    if kind in _VAGUE_KINDS:
        return True
    instruction = str(action.get("instruction", "")).lower()
    return any(kw in instruction for kw in _VAGUE_KEYWORDS)


def check_preconditions(
    preconditions: ExactPatchPreconditions,
    workspace_root: Path,
    file_content: bytes,
) -> DeterministicActionDecision:
    """Check all safety conditions without touching the filesystem.

    Returns rejection decision if:
    - path traverses outside workspace_root
    - before_hash doesn't match SHA-256(file_content)
    - search string appears != 1 times in content
    """
    import hashlib

    # Path traversal check
    try:
        resolved = (workspace_root / preconditions.path).resolve()
        root_resolved = workspace_root.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            return DeterministicActionDecision(
                should_execute=False,
                should_compress_only=False,
                should_reject=True,
                reason="path_traversal_detected",
                required_preconditions=("safe_path",),
                missing_preconditions=("safe_path",),
            )
    except (ValueError, OSError):
        return DeterministicActionDecision(
            should_execute=False,
            should_compress_only=False,
            should_reject=True,
            reason="path_traversal_detected",
            required_preconditions=("safe_path",),
            missing_preconditions=("safe_path",),
        )

    # Hash check
    actual_hash = hashlib.sha256(file_content).hexdigest()
    if actual_hash != preconditions.expected_before_hash:
        return DeterministicActionDecision(
            should_execute=False,
            should_compress_only=False,
            should_reject=True,
            reason="hash_mismatch",
            required_preconditions=("before_hash_match",),
            missing_preconditions=("before_hash_match",),
        )

    # Occurrence check
    search_bytes = preconditions.search.encode("utf-8")
    count = _count_occurrences(search_bytes, file_content)
    if count == 0:
        return DeterministicActionDecision(
            should_execute=False,
            should_compress_only=False,
            should_reject=True,
            reason="search_not_found",
            required_preconditions=("single_occurrence",),
            missing_preconditions=("single_occurrence",),
        )
    if count > 1:
        return DeterministicActionDecision(
            should_execute=False,
            should_compress_only=False,
            should_reject=True,
            reason="ambiguous_search_multiple_occurrences",
            required_preconditions=("single_occurrence",),
            missing_preconditions=("single_occurrence",),
        )

    return DeterministicActionDecision(
        should_execute=True,
        should_compress_only=False,
        should_reject=False,
        reason="all_preconditions_satisfied",
        required_preconditions=("safe_path", "before_hash_match", "single_occurrence"),
        missing_preconditions=(),
    )


def _count_occurrences(search: bytes, content: bytes) -> int:
    if not search:
        return 0
    count = 0
    start = 0
    while True:
        idx = content.find(search, start)
        if idx == -1:
            break
        count += 1
        start = idx + len(search)
    return count
