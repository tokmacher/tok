"""Session-local evidence exactness ledger for bridge compression safety."""

from __future__ import annotations

from dataclasses import dataclass
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
