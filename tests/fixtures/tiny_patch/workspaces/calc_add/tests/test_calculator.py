from src.calculator import add


def test_add_two_positive_numbers() -> None:
    assert add(2, 3) == 5
