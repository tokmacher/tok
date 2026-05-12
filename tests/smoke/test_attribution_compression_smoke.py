from __future__ import annotations

import copy

from tok.compression import compress_tool_results
from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context


def test_attribution_compression_smoke_records_compressed_repeat_read() -> None:
    repeated_content = "line\n" * 300
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "read-1", "name": "view_file", "input": {"path": "a.py"}},
                {"type": "tool_use", "id": "read-2", "name": "view_file", "input": {"path": "a.py"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "read-1", "content": repeated_content},
                {"type": "tool_result", "tool_use_id": "read-2", "content": repeated_content},
            ],
        },
    ]

    original = copy.deepcopy(messages)
    compressed, breakdown = compress_tool_results(
        messages,
        tool_use_id_to_context=build_tool_use_id_to_context(messages),
        semantic_hash_cache={},
        session_files_read=set(),
    )

    assert breakdown
    assert compressed != original
    compressed_text = str(compressed)
    assert any(marker in compressed_text for marker in ("@stable_result", "[tok optimized]", "|> [300 lines]"))
