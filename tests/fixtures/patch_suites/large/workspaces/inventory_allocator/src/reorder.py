from __future__ import annotations


def reorder_point(avg_daily_demand: float, lead_days: int, buffer_units: int) -> int:
    return int(avg_daily_demand * lead_days) + buffer_units
