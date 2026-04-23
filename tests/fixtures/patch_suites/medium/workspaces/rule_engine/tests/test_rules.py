from src.evaluator import evaluate
from src.loader import load_rules


def test_load_rules_preserves_input_order_and_normalizes_keys() -> None:
    rules = load_rules([" Score >=10", "age>=18"])
    assert rules == [("score", 10), ("age", 18)]


def test_evaluate_passes_when_all_thresholds_met() -> None:
    rules = [("score", 10), ("age", 18)]
    assert evaluate({"score": 10, "age": 20}, rules) is True
    assert evaluate({"score": 9, "age": 20}, rules) is False


def test_evaluate_fails_when_required_field_missing() -> None:
    rules = [("score", 10), ("age", 18)]
    assert evaluate({"score": 11}, rules) is False
