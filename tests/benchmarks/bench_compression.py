"""Benchmark: compression ratios at various conversation lengths."""

from __future__ import annotations

import time
from typing import Any

import pytest

from tok.compression import compress_history


def _make_conversation(turns: int) -> list[dict[str, Any]]:
    """Generate a synthetic conversation with N human turns."""
    msgs = []
    for i in range(turns):
        msgs.append(
            {
                "role": "user",
                "content": f"Question {i}: What about topic_{i}? variable_{i} = value_{i}",
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": f"Answer {i}: Here is a detailed response about topic_{i} with many words to simulate real output.",
            }
        )
    return msgs


@pytest.mark.benchmark
class TestCompressionBenchmarks:
    @pytest.mark.parametrize("turns", [10, 50, 200, 500])
    def test_compression_ratio(self, turns) -> None:
        msgs = _make_conversation(turns)
        orig_len = sum(len(m["content"]) for m in msgs)

        start = time.perf_counter()
        recent, state = compress_history(msgs, keep_turns=2)
        time.perf_counter() - start

        new_len = sum(len(m["content"]) for m in recent) + len(state)
        ratio = (1 - new_len / orig_len) * 100 if orig_len > 0 else 0

        assert ratio > 0 or turns <= 4  # Small conversations may not compress
