from src.calculator import multiply


def test_multiply_two_numbers() -> None:
    assert multiply(6, 7) == 42
