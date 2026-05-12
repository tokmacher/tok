"""Session-local evidence exactness ledger for bridge compression safety.

Evidence form taxonomy (0.1.9):

- ``exact``: verbatim, first-hand observation content (what was actually read/searched).
- ``summary``: a lossy natural-language summary of an observation.
- ``skeleton``: a structural outline (e.g. headings, defs) that is not full content.
- ``reference``: a pointer or stable stub (e.g. hash-based stable result marker).

Hard rule: non-exact evidence must not authorize edit-like behavior without
re-observation (exact reacquisition).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

EvidenceForm = Literal["exact", "summary", "skeleton", "reference"]

logger = logging.getLogger("tok.evidence_safety")

EVIDENCE_DECISION_REASON_CODES = frozenset(
    {
        "exact_search_observation",
        "first_occurrence_guard",
        "first_session",
        "zero_heat",
        "detection_type_raw",
        "pytest_failed",
        "size_below_threshold",
        "no_compressor",
        "skip_file_skeleton",
        "zero_savings",
        "compressed",
    }
)


def record_evidence_decision(
    *,
    decision: str,
    reason: str,
    tool: str,
    kind: str | None = None,
    path: str | None = None,
    input_chars: int | None = None,
    output_chars: int | None = None,
    saved: int | None = None,
) -> None:
    if reason not in EVIDENCE_DECISION_REASON_CODES:
        logger.warning("evidence_decision: unknown reason=%s decision=%s tool=%s", reason, decision, tool)
    parts = [f"evidence_decision: decision={decision} reason={reason} tool={tool}"]
    if kind is not None:
        parts.append(f"kind={kind}")
    if path:
        parts.append(f"path={path}")
    if input_chars is not None:
        parts.append(f"input_chars={int(input_chars)}")
    if output_chars is not None:
        parts.append(f"output_chars={int(output_chars)}")
    if saved is not None:
        parts.append(f"saved={int(saved)}")
    logger.debug(" ".join(parts))


@dataclass
class EvidenceLedgerEntry:
    key: str
    latest_digest: str = ""
    first_exact_turn: int = 0
    latest_turn: int = 0
    latest_form: EvidenceForm = "exact"
    exact_reacquisition_required: bool = False
    exact_reacquisition_satisfied_turn: int = 0

    @property
    def has_exact(self) -> bool:
        return self.first_exact_turn > 0

    @property
    def latest_is_exact(self) -> bool:
        return self.latest_form == "exact"


@dataclass
class EvidenceSafetyState:
    """Grouped evidence-safety fields extracted from RuntimeSession.

    Tracks which evidence identities have been observed exactly,
    which are only summaries/skeletons, and which require reacquisition.
    """

    neighborhoods: dict[str, set[str]] = field(default_factory=dict)
    anchor_novelty_keys: dict[str, set[str]] = field(default_factory=dict)
    alias_map: dict[str, str] = field(default_factory=dict)
    first_exact_seen: set[str] = field(default_factory=set)
    ledger: dict[str, EvidenceLedgerEntry] = field(default_factory=dict)
    pending_exact_keys: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.neighborhoods.clear()
        self.anchor_novelty_keys.clear()
        self.alias_map.clear()
        self.first_exact_seen.clear()
        self.ledger.clear()
        self.pending_exact_keys.clear()


def evidence_safety_summary(ledger: dict[str, EvidenceLedgerEntry]) -> dict[str, int]:
    """Return compact audit counters for live trace metadata."""
    entries = list(ledger.values())
    return {
        "entries": len(entries),
        "exact_entries": sum(1 for entry in entries if entry.has_exact),
        "non_exact_latest": sum(1 for entry in entries if not entry.latest_is_exact),
        "reacquisition_required": sum(1 for entry in entries if entry.exact_reacquisition_required),
        "reacquisition_satisfied": sum(1 for entry in entries if entry.exact_reacquisition_satisfied_turn > 0),
    }
