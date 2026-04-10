from __future__ import annotations


def allocate_stock(required: int, available: int) -> tuple[int, int]:
    # BUG: should never allocate above available.
    allocated = required
    backorder = max(0, required - available)
    return allocated, backorder
