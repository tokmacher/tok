from __future__ import annotations

from src.ledger import normalize_transactions


def reconcile(rows: list[dict[str, object]]) -> dict[str, float]:
    normalized = normalize_transactions(rows)
    totals: dict[str, float] = {}
    for row in normalized:
        acct = str(row["account"])
        amount = float(row["amount"])
        # BUG: duplicates should be ignored by id, but we double-count all rows.
        totals[acct] = totals.get(acct, 0.0) + amount
    return totals
