from __future__ import annotations

from src.allocator import allocate_stock
from src.forecast import required_units


def build_plan(
    avg_daily_demand: float,
    lead_days: int,
    safety_days: int,
    buffer_units: int,
    available: int,
) -> dict[str, int]:
    required = required_units(avg_daily_demand, lead_days, safety_days)
    allocated, backorder = allocate_stock(required, available)
    rp = int(avg_daily_demand * lead_days) + buffer_units
    return {
        "required": required,
        "allocated": allocated,
        "backorder": backorder,
        "reorder_point": rp,
    }
