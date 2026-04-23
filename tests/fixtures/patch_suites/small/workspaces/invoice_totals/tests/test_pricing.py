import pytest

from src.adjustments import apply_discount, apply_service_fee, apply_tax
from src.pricing import final_total, subtotal


def test_apply_discount_reduces_amount() -> None:
    assert apply_discount(100.0, 10.0) == 90.0


def test_apply_tax_increases_amount() -> None:
    assert apply_tax(100.0, 0.2) == 120.0


def test_service_fee_clamped_to_non_negative() -> None:
    assert apply_service_fee(25.0, -5.0) == 25.0


def test_subtotal_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        subtotal([(1, 10.0), (-1, 5.0)])


def test_final_total_with_discount_tax_and_fee() -> None:
    items = [(2, 10.0), (1, 5.0)]  # 25.0
    # discount 10% => 22.5, tax 20% => 27.0, + fee 2 => 29.0
    assert final_total(items, discount_pct=10.0, tax_rate=0.2, service_fee_flat=2.0) == 29.0
