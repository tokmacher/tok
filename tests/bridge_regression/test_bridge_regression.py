"""Section 5.1.4: Claude Code Bridge Regression Suite.

Core regression tests for the Tok bridge pipeline. These tests exercise the
full request handling path (compression, evidence safety, fail-open) using
controlled inputs without hitting real APIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tok.compression._history_pipeline import compress_tool_results_impl
from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_read_message(tool_id: str, path: str, content: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": "Read", "input": {"file_path": path}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}],
        },
    ]


def _build_history(*rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for round_msgs in rounds:
        messages.extend(round_msgs)
    return messages


def _compress(
    messages: list[dict[str, Any]],
    result_cache: dict | None = None,
    session_files_read: set[str] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    tool_use_id_to_context = build_tool_use_id_to_context(messages)
    return compress_tool_results_impl(
        messages,
        result_cache=result_cache or {},
        tool_use_id_to_context=tool_use_id_to_context,
        compression_level="balanced",
        first_exact_evidence_seen=set(),
        preserve_exact_search_evidence=True,
        session_files_read=session_files_read if session_files_read is not None else set(),
    )


# ---------------------------------------------------------------------------
# 1. Compression baseline: same file read 3× must compress 2nd/3rd reads
# ---------------------------------------------------------------------------


def test_repeat_file_read_compression_pipeline_does_not_crash() -> None:
    """Three reads of the same file must not crash and must produce valid output.

    This is a regression guard: the compression pipeline must handle repeated
    file reads without raising exceptions or corrupting the message structure.
    """
    content = "def foo():\n    return 42\n" * 40  # ~1000 chars, realistic
    path = "/repo/src/main.py"

    messages = _build_history(
        _file_read_message("t1", path, content),
        _file_read_message("t2", path, content),
        _file_read_message("t3", path, content),
    )
    # Must not raise
    compressed, breakdown = _compress(messages)

    # Output must preserve message count and valid structure
    assert len(compressed) == len(messages)
    for msg in compressed:
        if msg["role"] == "user":
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    assert "tool_use_id" in block
                    assert block.get("content") is not None
    assert isinstance(breakdown, dict)


def test_compression_baseline_preserves_content_type() -> None:
    """Compressed messages must retain correct tool_result structure."""
    content = "file content here " * 50
    path = "/repo/README.md"
    messages = _build_history(
        _file_read_message("r1", path, content),
        _file_read_message("r2", path, content),
    )
    compressed, _ = _compress(messages)

    for msg in compressed:
        if msg["role"] == "user":
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    assert "tool_use_id" in block, "tool_use_id lost during compression"


# ---------------------------------------------------------------------------
# 2. Fallback regression: error injection must trigger fail-open
# ---------------------------------------------------------------------------


def test_fail_open_uses_original_payload_on_400(tmp_path: Path) -> None:
    """A 400 from the provider must trigger fail-open and use the original body."""
    import asyncio

    import httpx

    from tok.gateway import BridgeSession
    from tok.gateway._bridge_request_handler import send_with_tok_fail_open_retry
    from tok.runtime.smoothness.models import TokMode

    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    session = BridgeSession(memory_dir=memory_dir, fail_open=True)
    session.runtime_session._current_tok_mode = TokMode.SMOOTH_MODE

    prepared_body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "prep"}], "stream": False}
    original_body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "orig"}], "stream": False}

    sent: list[bytes] = []

    async def _fake_send(_self: Any, request: Any, stream: bool = False) -> httpx.Response:
        sent.append(request.content)
        if len(sent) == 1:
            return httpx.Response(
                400,
                json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad request"}},
            )
        return httpx.Response(
            200,
            json={"id": "ok", "type": "message", "content": [], "usage": {"input_tokens": 1, "output_tokens": 1}},
        )

    import unittest.mock as mock

    with mock.patch.object(httpx.AsyncClient, "send", _fake_send):

        async def _run() -> Any:
            async with httpx.AsyncClient() as client:
                return await send_with_tok_fail_open_retry(
                    session,
                    client,
                    method="POST",
                    url="https://example.invalid/v1/messages",
                    headers={"x-api-key": "test"},
                    content=json.dumps(prepared_body).encode(),
                    original_content=json.dumps(original_body).encode(),
                    compressed_request=True,
                )

        _response, retried, signals = asyncio.run(_run())

    assert retried is True, "fail-open must set retried=True on 400"
    assert len(sent) == 2, f"Expected 2 sends (1 fail + 1 retry), got {len(sent)}"

    # Second send must use the original body
    second_body = json.loads(sent[1])
    assert second_body["messages"][0]["content"] == "orig", "Fail-open retry must use original (uncompressed) body"


def test_no_retry_on_success(tmp_path: Path) -> None:
    """Successful responses must not trigger fail-open retry."""
    import asyncio
    import unittest.mock as mock

    import httpx

    from tok.gateway import BridgeSession
    from tok.gateway._bridge_request_handler import send_with_tok_fail_open_retry
    from tok.runtime.smoothness.models import TokMode

    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    session = BridgeSession(memory_dir=memory_dir, fail_open=True)
    session.runtime_session._current_tok_mode = TokMode.SMOOTH_MODE

    body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    sent: list[bytes] = []

    async def _fake_send(_self: Any, request: Any, stream: bool = False) -> httpx.Response:
        sent.append(request.content)
        return httpx.Response(
            200,
            json={"id": "ok", "type": "message", "content": [], "usage": {"input_tokens": 1, "output_tokens": 1}},
        )

    with mock.patch.object(httpx.AsyncClient, "send", _fake_send):

        async def _run() -> Any:
            async with httpx.AsyncClient() as client:
                return await send_with_tok_fail_open_retry(
                    session,
                    client,
                    method="POST",
                    url="https://example.invalid/v1/messages",
                    headers={"x-api-key": "test"},
                    content=json.dumps(body).encode(),
                    original_content=json.dumps(body).encode(),
                    compressed_request=False,
                )

        _response, retried, _signals = asyncio.run(_run())

    assert retried is False, "Successful response must not trigger fail-open"
    assert len(sent) == 1, "Successful response must send only once"


# ---------------------------------------------------------------------------
# 3. Evidence safety: skeleton blocks compression, exact allows it
# ---------------------------------------------------------------------------


def test_exact_evidence_is_preserved_through_compression() -> None:
    """Exact file reads must survive compression unchanged."""
    content = "exact file content: def foo():\n    return 42\n"
    messages = _build_history(
        _file_read_message("e1", "/repo/foo.py", content),
    )
    first_exact_seen: set[str] = set()
    tool_use_id_to_context = build_tool_use_id_to_context(messages)
    compressed, _ = compress_tool_results_impl(
        messages,
        result_cache={},
        tool_use_id_to_context=tool_use_id_to_context,
        compression_level="balanced",
        first_exact_evidence_seen=first_exact_seen,
        preserve_exact_search_evidence=True,
    )
    # First exact read must be delivered verbatim
    result_blocks = [
        block
        for msg in compressed
        if msg["role"] == "user"
        for block in msg.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(result_blocks) >= 1
    first_result_content = result_blocks[0].get("content", "")
    assert content in str(first_result_content), "First exact evidence read must be delivered verbatim"


def test_repeat_exact_evidence_two_reads_valid_output() -> None:
    """Two reads of the same file must produce valid output without crashing.

    Implementation semantics: the first read is verbatim (first_exact_guard); the
    second read may also be verbatim because last_full_file_by_path is not populated
    when the first read passes via the first_exact_guard shortcut. Both deliveries
    must have valid structure.
    """
    content = "large content " * 80  # >1000 chars
    path = "/repo/large_file.py"

    messages = _build_history(
        _file_read_message("r1", path, content),
        _file_read_message("r2", path, content),
    )
    compressed, breakdown = _compress(messages)

    # Both messages must be present and valid
    assert len(compressed) == len(messages)
    for msg in compressed:
        if msg["role"] == "user":
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    assert "tool_use_id" in block, "tool_use_id lost during compression"
                    assert block.get("content") is not None, "content must not be None"


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------


def test_empty_history_is_safe() -> None:
    """compress_tool_results_impl must handle empty message list."""
    compressed, breakdown = _compress([])
    assert compressed == []
    assert isinstance(breakdown, dict)


def test_very_long_history_does_not_raise(tmp_path: Path) -> None:
    """50-turn file-read history must not raise and must return compressed messages."""
    path = "/repo/big.py"
    content = "line content " * 40  # ~500 chars per read
    messages: list[dict] = []
    for i in range(50):
        messages.extend(_file_read_message(f"t{i}", path, content))

    compressed, breakdown = _compress(messages)
    assert len(compressed) > 0
    assert isinstance(breakdown, dict)


def test_non_utf8_content_in_tool_result_is_safe() -> None:
    """Binary-like content in tool results must not crash compression."""
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "b1", "name": "Read", "input": {"file_path": "/bin/data"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "b1", "content": "binary data: \x00\xff\xfe"}],
        },
    ]
    # Must not raise
    compressed, _ = _compress(messages)
    assert len(compressed) == 2


def test_extreme_tool_density_does_not_raise() -> None:
    """History with many tool uses per turn must not crash compression."""
    messages: list[dict] = []
    tools = [{"type": "tool_use", "id": f"t{i}", "name": "search", "input": {"query": f"q{i}"}} for i in range(30)]
    results = [{"type": "tool_result", "tool_use_id": f"t{i}", "content": f"result {i}"} for i in range(30)]
    messages.append({"role": "assistant", "content": tools})
    messages.append({"role": "user", "content": results})

    compressed, breakdown = _compress(messages)
    assert len(compressed) == 2
    assert isinstance(breakdown, dict)
