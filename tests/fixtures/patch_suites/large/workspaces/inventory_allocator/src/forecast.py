from __future__ import annotations


def required_units(avg_daily_demand: float, lead_days: int, safety_days: int) -> int:
    # BUG: safety stock should be additive, not multiplicative.
    return int(avg_daily_demand * lead_days * safety_days)
