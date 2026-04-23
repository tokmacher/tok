from __future__ import annotations

from src.adjustments import apply_discount, apply_service_fee, apply_tax


def subtotal(items: list[tuple[int, float]]) -> float:
    total = 0.0
    for qty, price in items:
        # BUG: negative qty/price should raise ValueError.
        total += qty * price
    return total


def final_total(
    items: list[tuple[int, float]],
    discount_pct: float,
    tax_rate: float,
    service_fee_flat: float,
) -> float:
    base = subtotal(items)
    discounted = apply_discount(base, discount_pct)
    # BUG: tax should be applied before adding service fee.
    with_fee = apply_service_fee(discounted, service_fee_flat)
    taxed = apply_tax(with_fee, tax_rate)
    return round(taxed, 2)
