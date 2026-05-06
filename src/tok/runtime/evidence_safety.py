"""Session-local evidence exactness ledger for bridge compression safety."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EvidenceForm = Literal["exact", "summary", "skeleton", "reference"]


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
