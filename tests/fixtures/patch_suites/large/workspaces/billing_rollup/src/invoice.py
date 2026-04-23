from __future__ import annotations

from src.rollup import account_totals


def render_invoice(events: list[dict[str, object]]) -> str:
    totals = account_totals(events)
    lines = ["account,total"]
    for account in sorted(totals):
        # BUG: should round to 2 decimal places.
        lines.append(f"{account},{totals[account]}")
    return "\n".join(lines)
