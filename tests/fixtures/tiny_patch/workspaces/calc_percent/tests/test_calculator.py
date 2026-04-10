from src.calculator import percent_of


def test_percent_supports_fractional_result() -> None:
    assert percent_of(50, 15) == 7.5
