import pytest

from tok.utils.math_utils import add, divide, multiply, subtract


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_subtract():
    assert subtract(5, 2) == 3
    assert subtract(0, 0) == 0
    assert subtract(-1, 1) == -2


def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-1, 1) == -1
    assert multiply(0, 5) == 0


def test_divide():
    assert divide(6, 3) == 2
    assert divide(0, 5) == 0
    with pytest.raises(ValueError):
        divide(5, 0)
