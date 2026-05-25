"""Hash-checked exact patch application.

Experimental — not part of the 0.2.x public API.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tok.deterministic.action_decision import check_preconditions
from tok.deterministic.models import DeterministicActionReceipt, ExactPatchPreconditions


def apply_exact_patch(
    preconditions: ExactPatchPreconditions,
    workspace_root: Path,
) -> DeterministicActionReceipt:
    """Apply a hash-checked exact patch and return an audit receipt.

    Safety invariants enforced before any write:
    1. Resolved path is within workspace_root (no traversal).
    2. SHA-256 of current file bytes == preconditions.expected_before_hash.
    3. preconditions.search appears exactly once in file content.

    On any invariant failure: returns rejected receipt, file untouched.
    On success: writes new content, records before/after hashes and changed ranges.
    """
    action_id = str(uuid.uuid4())
    created_at = datetime.now(tz=timezone.utc).isoformat()

    # Resolve path safely
    try:
        resolved = _safe_resolve(preconditions.path, workspace_root)
    except ValueError as exc:
        return _rejected(
            action_id=action_id,
            path=preconditions.path,
            reason=f"traversal: {exc}",
            created_at=created_at,
        )

    # Read file
    try:
        file_content = resolved.read_bytes()
    except OSError as exc:
        return _rejected(
            action_id=action_id,
            path=preconditions.path,
            reason=f"read_error: {exc}",
            created_at=created_at,
        )

    # Check all preconditions
    decision = check_preconditions(preconditions, workspace_root, file_content)
    if decision.should_reject:
        return _rejected(
            action_id=action_id,
            path=preconditions.path,
            reason=decision.reason,
            created_at=created_at,
            before_hash=_sha256_hex(file_content),
        )

    before_hash = _sha256_hex(file_content)
    # Locate the single occurrence to compute byte range
    search_bytes = preconditions.search.encode("utf-8")
    replace_bytes = preconditions.replace.encode("utf-8")
    start_byte = file_content.find(search_bytes)
    end_byte = start_byte + len(search_bytes)

    new_content = file_content[:start_byte] + replace_bytes + file_content[end_byte:]
    after_hash = _sha256_hex(new_content)

    # Atomic write: write to temp file, then rename
    parent = resolved.parent
    try:
        fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=".tok_patch_")
        try:
            os.write(fd, new_content)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, resolved)
    except OSError as exc:
        # Clean up temp file if rename failed
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return _rejected(
            action_id=action_id,
            path=preconditions.path,
            reason=f"write_error: {exc}",
            created_at=created_at,
            before_hash=before_hash,
        )

    return DeterministicActionReceipt(
        action_id=action_id,
        action_kind="exact_patch",
        provider_surface=None,
        path=preconditions.path,
        paths=[preconditions.path],
        before_hash=before_hash,
        after_hash=after_hash,
        changed_ranges=[(start_byte, end_byte)],
        exactness_level="exact_patch",
        lossy=False,
        commands_run=[],
        verification_results={},
        rollback_ref=file_content.decode("latin-1"),
        exact_diff_ref=None,
        status="applied",
        rejection_reason=None,
        created_at=created_at,
    )


def _rejected(
    *,
    action_id: str,
    path: str,
    reason: str,
    created_at: str,
    before_hash: str | None = None,
) -> DeterministicActionReceipt:
    return DeterministicActionReceipt(
        action_id=action_id,
        action_kind="exact_patch",
        provider_surface=None,
        path=path,
        paths=[path],
        before_hash=before_hash,
        after_hash=None,
        changed_ranges=[],
        exactness_level="exact_patch",
        lossy=False,
        commands_run=[],
        verification_results={},
        rollback_ref=None,
        exact_diff_ref=None,
        status="rejected",
        rejection_reason=reason,
        created_at=created_at,
    )


def _sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _count_occurrences(search: str, content: str) -> int:
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


def _safe_resolve(path: str, workspace_root: Path) -> Path:
    """Resolve path; raise ValueError if outside workspace_root."""
    resolved = (workspace_root / path).resolve()
    root_resolved = workspace_root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(f"path '{path}' resolves outside workspace_root") from None
    return resolved
