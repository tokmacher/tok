from __future__ import annotations


def apply_discount(amount: float, discount_pct: float) -> float:
    # BUG: discount should reduce amount, not increase it.
    return amount + (amount * discount_pct / 100.0)


def apply_tax(amount: float, tax_rate: float) -> float:
    # tax_rate is decimal, e.g. 0.2 for 20%
    # BUG: should multiply by (1 + tax_rate).
    return amount * (1 - tax_rate)


def apply_service_fee(amount: float, service_fee_flat: float) -> float:
    # BUG: service fee should never be negative.
    return amount + service_fee_flat
