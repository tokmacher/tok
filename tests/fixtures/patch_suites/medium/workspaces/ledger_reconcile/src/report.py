from __future__ import annotations


def render_report(totals: dict[str, float]) -> str:
    lines = ["account,total"]
    for account in sorted(totals):
        # BUG: report should round to 2 decimals.
        lines.append(f"{account},{totals[account]}")
    return "\n".join(lines)
