"""Edge-case tests for tok.adapters._utils after the walrus-operator refactor."""

from __future__ import annotations

from tok.adapters._utils import (  # noqa: PLC2701
    _render_text,
    _system_to_messages,
)

# ---------------------------------------------------------------------------
# _render_text edge cases
# ---------------------------------------------------------------------------


def test_render_text_integer_text_value_is_stringified() -> None:
    # block.get("text") returns 0 (int); str(0).strip() == "0" which is truthy
    blocks = [{"type": "text", "text": 0}]
    assert _render_text(blocks) == "0"


def test_render_text_non_text_type_with_text_key_is_skipped() -> None:
    blocks = [{"type": "tool_use", "text": "should not appear"}]
    assert _render_text(blocks) == ""


def test_render_text_missing_type_key_is_skipped() -> None:
    blocks = [{"text": "no type key"}]
    assert _render_text(blocks) == ""


def test_render_text_whitespace_only_text_is_skipped_by_walrus() -> None:
    # The walrus result is "" after strip, which is falsy — skipped
    blocks = [{"type": "text", "text": "   "}, {"type": "text", "text": "kept"}]
    assert _render_text(blocks) == "kept"


# ---------------------------------------------------------------------------
# _system_to_messages edge cases
# ---------------------------------------------------------------------------


def test_system_to_messages_non_text_block_is_filtered() -> None:
    # Image block has no "text" key — block.get("text") returns None (falsy) → skipped
    blocks = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
        {"text": "real"},
    ]
    result = _system_to_messages(blocks)
    assert result == [{"role": "system", "content": "real"}]


def test_system_to_messages_integer_text_filtered_out() -> None:
    # {"text": 0} — block.get("text") returns 0 (falsy) → skipped
    blocks = [{"text": 0}, {"text": "ok"}]
    result = _system_to_messages(blocks)
    assert result == [{"role": "system", "content": "ok"}]


def test_system_to_messages_empty_text_string_filtered_out() -> None:
    blocks = [{"text": ""}, {"text": "present"}]
    result = _system_to_messages(blocks)
    assert result == [{"role": "system", "content": "present"}]


def test_system_to_messages_non_dict_items_filtered() -> None:
    result = _system_to_messages(["plain string", {"text": "dict"}])  # type: ignore[arg-type]
    assert result == [{"role": "system", "content": "dict"}]


def test_system_to_messages_all_filtered_returns_empty() -> None:
    blocks = [{"type": "image"}, {"text": ""}]
    assert _system_to_messages(blocks) == []
