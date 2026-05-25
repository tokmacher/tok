"""Frozen dataclasses for deterministic action receipts.

Experimental — not part of the 0.2.x public API. Import via explicit submodule path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExactPatchPreconditions:
    path: str
    expected_before_hash: str  # hex SHA-256 of file content before edit
    search: str  # exact literal string; no fuzzy matching
    replace: str  # exact replacement
    workspace_root: str  # all paths must resolve within this


@dataclass(frozen=True)
class DeterministicActionDecision:
    should_execute: bool
    should_compress_only: bool
    should_reject: bool
    reason: str
    required_preconditions: tuple[str, ...]
    missing_preconditions: tuple[str, ...]


@dataclass(frozen=True)
class DeterministicActionReceipt:
    action_id: str
    action_kind: str  # "exact_patch" | "exact_verification" | "lossy_summary_only"
    provider_surface: str | None
    path: str | None
    paths: list[str]
    before_hash: str | None
    after_hash: str | None
    changed_ranges: list[tuple[int, int]]  # (start_byte, end_byte)
    exactness_level: str  # mirrors action_kind
    lossy: bool
    commands_run: list[str]
    verification_results: dict[str, Any]
    rollback_ref: str | None  # inline before-content for exact recovery
    exact_diff_ref: str | None
    status: str  # "applied" | "rejected" | "verified" | "failed"
    rejection_reason: str | None
    created_at: str  # ISO-8601

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "provider_surface": self.provider_surface,
            "path": self.path,
            "paths": list(self.paths),
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "changed_ranges": [list(r) for r in self.changed_ranges],
            "exactness_level": self.exactness_level,
            "lossy": self.lossy,
            "commands_run": list(self.commands_run),
            "verification_results": dict(self.verification_results),
            "rollback_ref": self.rollback_ref,
            "exact_diff_ref": self.exact_diff_ref,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeterministicActionReceipt:
        return cls(
            action_id=str(d.get("action_id", "")),
            action_kind=str(d.get("action_kind", "")),
            provider_surface=d.get("provider_surface"),
            path=d.get("path"),
            paths=list(d.get("paths") or []),
            before_hash=d.get("before_hash"),
            after_hash=d.get("after_hash"),
            changed_ranges=[tuple(r) for r in (d.get("changed_ranges") or [])],
            exactness_level=str(d.get("exactness_level", "")),
            lossy=bool(d.get("lossy", False)),
            commands_run=list(d.get("commands_run") or []),
            verification_results=dict(d.get("verification_results") or {}),
            rollback_ref=d.get("rollback_ref"),
            exact_diff_ref=d.get("exact_diff_ref"),
            status=str(d.get("status", "")),
            rejection_reason=d.get("rejection_reason"),
            created_at=str(d.get("created_at", "")),
        )

    def compute_digest(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()
