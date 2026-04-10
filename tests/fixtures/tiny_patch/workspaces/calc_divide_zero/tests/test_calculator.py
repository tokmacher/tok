import pytest

from src.calculator import divide


def test_divide_by_zero_raises() -> None:
    with pytest.raises(ValueError):
        divide(4, 0)
