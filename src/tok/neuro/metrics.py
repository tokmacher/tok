"""Metrics calculation for neural components."""

from collections.abc import Sequence
from typing import TypedDict


class RunMetrics(TypedDict):
    accuracy: float
    auc_score: float
    stored_memory_size: int
    retrieval_hit_rate: float
    verifier_retries: int
    lesson_reuse_counts: int
    compaction_ratio: float


def estimated_token_count(memory_list: Sequence[object]) -> int:
    """
    Very rough heuristic for token counting (approx 4 chars/token).
    Used to track relative compaction across distillation steps.
    """
    total_chars = 0
    for item in memory_list:
        question = getattr(item, "question", None)
        if isinstance(question, str):
            total_chars += len(question)

        answer = getattr(item, "answer", None)
        if isinstance(answer, str):
            total_chars += len(answer)

        lesson = getattr(item, "lesson", None)
        if isinstance(lesson, str):
            total_chars += len(lesson)

        definition = getattr(item, "definition", None)
        if isinstance(definition, str):
            total_chars += len(definition)

    return max(1, total_chars // 4)


def compute_auc(outcomes: Sequence[bool]) -> float:
    """
    Computes an AUC-like aggregate of sample efficiency.
    Calculates the average cumulative accuracy at each step over time.
    """
    if not outcomes:
        return 0.0

    cumulative_correct = 0
    area = 0.0
    for i, ok in enumerate(outcomes, start=1):
        if ok:
            cumulative_correct += 1
        area += cumulative_correct / i

    return area / len(outcomes)


class RollingRate:
    """Rolling window success rate tracker."""

    def __init__(self, window: int = 200) -> None:
        self.window = window
        self.values: list[int] = []

    def push(self, ok: bool) -> None:
        """Record a new success/failure outcome."""
        self.values.append(1 if ok else 0)
        if len(self.values) > self.window:
            self.values = self.values[-self.window :]

    def rate(self) -> float:
        """Calculate the current success rate."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
