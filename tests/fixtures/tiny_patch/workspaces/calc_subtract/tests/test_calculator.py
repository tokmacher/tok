from src.calculator import subtract


def test_subtract_two_numbers() -> None:
    assert subtract(9, 4) == 5
