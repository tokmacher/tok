"""Tests for repeated file-read detection and bypass-reacquire classification."""

from __future__ import annotations

import hashlib
from typing import Any

from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context
from tok.runtime.pipeline._tool_repeat_detection import (
    _make_cache_key,
)
from tok.runtime.pipeline.tool_processing import collect_behavior_signals


def _tool_use(tool_id: str, tool_name: str, **input_kw: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": input_kw,
            }
        ],
    }


def _tool_result(tool_id: str, content: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content,
            }
        ],
    }


def _cache_entry(
    tool_name: str, context: dict[str, Any], content: str
) -> tuple[str, tuple[str, str]]:
    key = _make_cache_key(tool_name, context)
    digest = hashlib.sha256(content.encode()).hexdigest()[:8]
    return key, (digest, content)


def test_bypass_reacquire_not_counted_as_penalized_repeat():
    file_content = "line1\nline2\nline3\n"
    messages = [
        _tool_use("t1", "Read", file_path="/tmp/foo.py"),
        _tool_result("t1", file_content),
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "@tok_bypass_next_read"},
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Read",
                    "input": {"file_path": "/tmp/foo.py"},
                },
            ],
        },
        _tool_result("t2", file_content),
    ]
    ctx = build_tool_use_id_to_context(messages)
    assert ctx["t2"]["args"].get("tok_bypass_cache") is True

    result_cache: dict[
        str, tuple[str, str, float] | tuple[str, str] | tuple[str]
    ] = {}
    cache_entry = _cache_entry("read", ctx["t1"], file_content)
    result_cache[cache_entry[0]] = cache_entry[1]

    signals = collect_behavior_signals(
        messages,
        tool_use_id_to_context=ctx,
        result_cache=result_cache,
    )
    assert signals.get("bypass_reacquire", 0) >= 1, (
        f"expected bypass_reacquire >= 1, got {signals}"
    )
    assert signals.get("repeat_file_read", 0) == 0, (
        f"expected no repeat_file_read, got {signals.get('repeat_file_read', 0)}"
    )


def test_cross_turn_reread_without_bypass_still_counted():
    content_v1 = "version one\n"
    content_v2 = "version two changed\n"
    messages = [
        _tool_use("t1", "Read", file_path="/tmp/bar.py"),
        _tool_result("t1", content_v1),
        _tool_use("t2", "Read", file_path="/tmp/bar.py"),
        _tool_result("t2", content_v2),
    ]
    ctx = build_tool_use_id_to_context(messages)
    result_cache: dict[
        str, tuple[str, str, float] | tuple[str, str] | tuple[str]
    ] = {}
    cache_entry = _cache_entry("read", ctx["t1"], content_v1)
    result_cache[cache_entry[0]] = cache_entry[1]

    signals = collect_behavior_signals(
        messages,
        tool_use_id_to_context=ctx,
        result_cache=result_cache,
    )
    assert signals.get("repeat_file_read", 0) >= 1, (
        f"expected repeat_file_read >= 1 for content-changed reread, got {signals}"
    )
    assert signals.get("bypass_reacquire", 0) == 0


def test_repeat_without_result_cache_still_counted():
    content = "same content\n"
    messages = [
        _tool_use("t1", "Read", file_path="/tmp/baz.py"),
        _tool_result("t1", content),
        _tool_use("t2", "Read", file_path="/tmp/baz.py"),
        _tool_result("t2", content),
    ]
    ctx = build_tool_use_id_to_context(messages)

    signals = collect_behavior_signals(
        messages,
        tool_use_id_to_context=ctx,
        result_cache=None,
    )
    assert signals.get("repeat_file_read", 0) >= 1, (
        f"expected repeat_file_read >= 1 when no result_cache, got {signals}"
    )


def test_cached_hit_not_counted_as_repeat():
    content = "cached content\n"
    messages = [
        _tool_use("t1", "Read", file_path="/tmp/qux.py"),
        _tool_result("t1", content),
        _tool_use("t2", "Read", file_path="/tmp/qux.py"),
        _tool_result("t2", content),
    ]
    ctx = build_tool_use_id_to_context(messages)
    result_cache: dict[
        str, tuple[str, str, float] | tuple[str, str] | tuple[str]
    ] = {}
    cache_entry = _cache_entry("read", ctx["t1"], content)
    result_cache[cache_entry[0]] = cache_entry[1]

    signals = collect_behavior_signals(
        messages,
        tool_use_id_to_context=ctx,
        result_cache=result_cache,
    )
    assert signals.get("repeat_file_read", 0) == 0, (
        f"expected no repeat_file_read for cache hit, got {signals}"
    )
    assert signals.get("cached_file_read", 0) >= 1, (
        f"expected cached_file_read >= 1, got {signals}"
    )
