"""Metrics calculation for neural components."""

from collections.abc import Sequence


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
