"""Tests for the provider-neutral block-semantics seam.

These cover the single classification surface every request/response-shaping path
uses to ask "what kind of content block is this?" rather than hardcoding Anthropic
``type`` string comparisons at each call site. Centralizing the checks is what lets
a future provider adapter remap block shapes in one place.
"""

from __future__ import annotations

from tok.provider_block_semantics import (
    block_type,
    is_text_block,
    is_tool_result_block,
    is_tool_use_block,
)


def test_block_type_is_defensive() -> None:
    assert block_type({"type": "tool_use"}) == "tool_use"
    assert block_type({"type": 123}) is None
    assert block_type({}) is None
    assert block_type("not-a-dict") is None
    assert block_type(None) is None


def test_is_tool_use_block() -> None:
    assert is_tool_use_block({"type": "tool_use", "id": "t", "name": "Read", "input": {}})
    assert not is_tool_use_block({"type": "tool_result", "tool_use_id": "t", "content": "ok"})
    assert not is_tool_use_block({"type": "text", "text": "hi"})
    assert not is_tool_use_block("nope")
    assert not is_tool_use_block(None)


def test_is_tool_result_block() -> None:
    assert is_tool_result_block({"type": "tool_result", "tool_use_id": "t", "content": "ok"})
    assert not is_tool_result_block({"type": "tool_use", "id": "t"})
    assert not is_tool_result_block({"type": "text", "text": "hi"})
    assert not is_tool_result_block(42)


def test_is_text_block() -> None:
    assert is_text_block({"type": "text", "text": "hi"})
    assert not is_text_block({"type": "tool_use", "id": "t"})
    assert not is_text_block({"type": "thinking", "thinking": "x"})
    assert not is_text_block([])
