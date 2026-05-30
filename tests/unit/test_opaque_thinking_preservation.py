"""Regression tests for opaque provider block (thinking/redacted_thinking) preservation.

These cover the release-blocking bug where Tok mutated provider-owned thinking
blocks when forwarding request history upstream, causing:

    API Error: 400 messages.N.content.M: thinking or redacted_thinking blocks in
    the latest assistant message cannot be modified. These blocks must remain as
    they were in the original response.

Tok must treat ``thinking`` / ``redacted_thinking`` blocks as opaque immutable
pass-through data: never compress, summarize, reorder, merge, dedupe, strip, or
re-encode them. The fail-open retry/fallback path must reuse the original
unmodified body for those blocks.
"""

from __future__ import annotations

import copy
from typing import Any

from tok.gateway._bridge_request_handler import (
    _count_assistant_messages_with_opaque_blocks,
    _normalize_provider_safe_retry_payload,
)
from tok.provider_opaque_blocks import (
    OPAQUE_PROVIDER_BLOCK_TYPES,
    content_has_opaque_provider_blocks,
    is_opaque_provider_block,
)
from tok.runtime.pipeline.request_validation import canonicalize_anthropic_bridge_body

# --------------------------------------------------------------------------- #
# Central helper
# --------------------------------------------------------------------------- #


def test_is_opaque_provider_block_classifies_thinking_types() -> None:
    assert is_opaque_provider_block({"type": "thinking", "thinking": "x"})
    assert is_opaque_provider_block({"type": "redacted_thinking", "data": "blob"})
    assert OPAQUE_PROVIDER_BLOCK_TYPES == frozenset({"thinking", "redacted_thinking"})


def test_is_opaque_provider_block_rejects_ordinary_blocks() -> None:
    assert not is_opaque_provider_block({"type": "text", "text": "hi"})
    assert not is_opaque_provider_block({"type": "tool_use", "id": "t", "name": "n", "input": {}})
    assert not is_opaque_provider_block({"type": "tool_result", "tool_use_id": "t", "content": "ok"})
    assert not is_opaque_provider_block("not-a-dict")
    assert not is_opaque_provider_block(None)


def test_is_opaque_provider_block_treats_signed_block_as_opaque() -> None:
    # A non-empty signature is a provider-agnostic opacity signal.
    assert is_opaque_provider_block({"type": "thinking", "thinking": "x", "signature": "SIG"})
    assert is_opaque_provider_block({"type": "future_signed_kind", "signature": "SIG"})
    assert not is_opaque_provider_block({"type": "text", "text": "hi", "signature": ""})


def test_content_has_opaque_provider_blocks() -> None:
    assert content_has_opaque_provider_blocks(
        [{"type": "text", "text": "a"}, {"type": "redacted_thinking", "data": "b"}]
    )
    assert not content_has_opaque_provider_blocks([{"type": "text", "text": "a"}])
    assert not content_has_opaque_provider_blocks("str")


# --------------------------------------------------------------------------- #
# Canonicalization (first outgoing send) preserves opaque blocks
# --------------------------------------------------------------------------- #


def _assistant_then_tool_result(assistant_content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": "claude-opus-4-8",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
        ],
    }


def _canonicalize_preserves(assistant_content: list[dict[str, Any]]) -> None:
    body = _assistant_then_tool_result(assistant_content)
    before = copy.deepcopy(body["messages"][1]["content"])
    out, _changed, signals = canonicalize_anthropic_bridge_body(body)
    after = out["messages"][1]["content"]
    assert after == before, f"opaque content mutated: {before!r} -> {after!r}"
    assert not signals.get("thinking_block_mutated"), signals


def test_canonicalize_preserves_text_before_thinking() -> None:
    _canonicalize_preserves(
        [
            {"type": "text", "text": "let me reason"},
            {"type": "thinking", "thinking": "reasoning", "signature": "SIG1"},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/x"}},
        ]
    )


def test_canonicalize_preserves_thinking_between_text_blocks() -> None:
    _canonicalize_preserves(
        [
            {"type": "text", "text": "before"},
            {"type": "thinking", "thinking": "mid reasoning", "signature": "SIG2"},
            {"type": "text", "text": "after"},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/x"}},
        ]
    )


def test_canonicalize_preserves_redacted_thinking_between_text_blocks() -> None:
    _canonicalize_preserves(
        [
            {"type": "text", "text": "before"},
            {"type": "redacted_thinking", "data": "ENCRYPTED=="},
            {"type": "text", "text": "after"},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/x"}},
        ]
    )


def test_canonicalize_preserves_thinking_adjacent_to_tool_use() -> None:
    _canonicalize_preserves(
        [
            {"type": "thinking", "thinking": "plan", "signature": "SIG3"},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/x"}},
        ]
    )


def test_canonicalize_does_not_reorder_tool_use_before_thinking() -> None:
    # Even an unusual client order (tool_use before thinking) must be preserved:
    # Tok must not reorder opaque blocks to "fix" it.
    _canonicalize_preserves(
        [
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/x"}},
            {"type": "redacted_thinking", "data": "ENCRYPTED=="},
        ]
    )


def test_canonicalize_preserves_non_latest_assistant_redacted_thinking() -> None:
    # A redacted_thinking block in an earlier (non-latest) assistant turn must
    # also survive unchanged and keep its order.
    body = {
        "model": "claude-opus-4-8",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/x"}},
                    {"type": "redacted_thinking", "data": "EARLY=="},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ],
    }
    before = copy.deepcopy(body["messages"][1]["content"])
    out, _changed, _signals = canonicalize_anthropic_bridge_body(body)
    assert out["messages"][1]["content"] == before


# --------------------------------------------------------------------------- #
# Exact reported-error reproduction
# --------------------------------------------------------------------------- #


def test_regression_messages_1_content_1_redacted_thinking_not_modified() -> None:
    """Reproduce the exact reported 400: messages[1] assistant, content[1] redacted_thinking.

    This deep-equality check would have failed under the previous behavior
    (which stripped/reordered the opaque block) and passes after the fix.
    """
    body = {
        "model": "claude-opus-4-8",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "investigate"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll look into it."},
                    {"type": "redacted_thinking", "data": "OPAQUE_PROVIDER_BLOB=="},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/x"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
        ],
    }
    original_block = copy.deepcopy(body["messages"][1]["content"][1])
    assert original_block["type"] == "redacted_thinking"

    out, _changed, signals = canonicalize_anthropic_bridge_body(body)

    forwarded_block = out["messages"][1]["content"][1]
    assert forwarded_block == original_block
    assert out["messages"][1]["content"] == body["messages"][1]["content"]
    assert not signals.get("thinking_block_mutated"), signals


# --------------------------------------------------------------------------- #
# Fail-open retry / fallback path preserves opaque blocks
# --------------------------------------------------------------------------- #


def test_retry_payload_never_strips_latest_assistant_redacted_thinking() -> None:
    body = {
        "model": "claude-opus-4-8",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thinking..."},
                    {"type": "redacted_thinking", "data": "BLOB=="},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/x"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
        ],
    }
    before = copy.deepcopy(body)
    out, changed = _normalize_provider_safe_retry_payload(body)
    assert changed is False
    assert out is body
    assert out == before  # byte-for-byte equivalent, opaque blocks intact


def test_retry_payload_preserves_thinking_with_signature() -> None:
    block = {"type": "thinking", "thinking": "signed reasoning", "signature": "SIG_OPAQUE"}
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [copy.deepcopy(block), {"type": "tool_use", "id": "t", "name": "n", "input": {}}],
            },
        ]
    }
    out, changed = _normalize_provider_safe_retry_payload(body)
    assert changed is False
    assert out["messages"][1]["content"][0] == block


def test_count_assistant_messages_with_opaque_blocks() -> None:
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "no opaque here"}]},
            {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "b"}]},
        ]
    }
    assert _count_assistant_messages_with_opaque_blocks(body) == 2
    assert _count_assistant_messages_with_opaque_blocks({"messages": "bad"}) == 0
    assert _count_assistant_messages_with_opaque_blocks(None) == 0
