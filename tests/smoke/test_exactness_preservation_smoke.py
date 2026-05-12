from __future__ import annotations

from tok.compression._history_pipeline import compress_tool_results_impl
from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context


def test_exactness_preservation_smoke() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {"query": "x"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "match: exact content"}],
        },
    ]
    tool_use_id_to_context = build_tool_use_id_to_context(messages)
    first_exact_evidence_seen: set[str] = set()
    compressed, _breakdown = compress_tool_results_impl(
        messages,
        result_cache={},
        tool_use_id_to_context=tool_use_id_to_context,
        compression_level="balanced",
        first_exact_evidence_seen=first_exact_evidence_seen,
        preserve_exact_search_evidence=True,
    )
    assert compressed[1]["content"][0]["content"] == "match: exact content"
