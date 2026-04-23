from __future__ import annotations

from src.rates import unit_rate
from src.usage import normalize_usage


def account_totals(events: list[dict[str, object]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for event in normalize_usage(events):
        account = str(event["account"])
        service = str(event["service"])
        units = int(event["units"])
        totals[account] = totals.get(account, 0.0) + units * unit_rate(service)
    return totals
