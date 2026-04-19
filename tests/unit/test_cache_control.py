"""Tests for gateway cache control helpers."""

from __future__ import annotations

from tok.gateway._cache_control import (
    _body_has_cache_control,
    _cache_control_counts_for_messages,
    _cache_control_counts_for_tools,
    _count_cache_control_entries,
)


class TestBodyHasCacheControl:
    def test_empty_dict(self) -> None:
        assert _body_has_cache_control({}) is False

    def test_empty_list(self) -> None:
        assert _body_has_cache_control([]) is False

    def test_dict_with_cache_control(self) -> None:
        assert _body_has_cache_control({"cache_control": {}}) is True

    def test_dict_without_cache_control(self) -> None:
        assert _body_has_cache_control({"key": "value"}) is False

    def test_nested_dict_with_cache_control(self) -> None:
        assert _body_has_cache_control({"outer": {"inner": {"cache_control": {}}}}) is True

    def test_nested_dict_without_cache_control(self) -> None:
        assert _body_has_cache_control({"outer": {"inner": "value"}}) is False

    def test_list_with_cache_control(self) -> None:
        assert _body_has_cache_control([{"cache_control": {}}]) is True

    def test_list_without_cache_control(self) -> None:
        assert _body_has_cache_control([{"key": "value"}]) is False

    def test_mixed_nested_structures(self) -> None:
        assert _body_has_cache_control({"outer": [{"cache_control": {}}, {"other": "value"}]}) is True

    def test_none(self) -> None:
        assert _body_has_cache_control(None) is False

    def test_string(self) -> None:
        assert _body_has_cache_control("cache_control") is False

    def test_int(self) -> None:
        assert _body_has_cache_control(123) is False


class TestCountCacheControlEntries:
    def test_empty_dict(self) -> None:
        assert _count_cache_control_entries({}) == 0

    def test_empty_list(self) -> None:
        assert _count_cache_control_entries([]) == 0

    def test_dict_with_cache_control(self) -> None:
        assert _count_cache_control_entries({"cache_control": {}}) == 1

    def test_dict_without_cache_control(self) -> None:
        assert _count_cache_control_entries({"key": "value"}) == 0

    def test_multiple_cache_controls_in_dict(self) -> None:
        result = _count_cache_control_entries({"a": {"cache_control": {}}, "b": {"cache_control": {}}})
        assert result == 2

    def test_nested_dict_counting(self) -> None:
        result = _count_cache_control_entries({"outer": {"inner": {"cache_control": {}}}})
        assert result == 1

    def test_list_counting(self) -> None:
        result = _count_cache_control_entries([{"cache_control": {}}, {"cache_control": {}}])
        assert result == 2

    def test_mixed_nested_counting(self) -> None:
        result = _count_cache_control_entries({"outer": [{"cache_control": {}}, {"cache_control": {}}]})
        assert result == 2

    def test_none(self) -> None:
        assert _count_cache_control_entries(None) == 0

    def test_string(self) -> None:
        assert _count_cache_control_entries("cache_control") == 0


class TestCacheControlCountsForMessages:
    def test_empty_messages(self) -> None:
        counts = _cache_control_counts_for_messages([])
        assert counts["message_text_blocks"] == 0
        assert counts["message_tool_result_blocks"] == 0
        assert counts["message_tool_use_blocks"] == 0

    def test_non_list_input(self) -> None:
        counts = _cache_control_counts_for_messages({})
        assert counts["message_text_blocks"] == 0
        assert counts["message_tool_result_blocks"] == 0
        assert counts["message_tool_use_blocks"] == 0

    def test_none_input(self) -> None:
        counts = _cache_control_counts_for_messages(None)
        assert counts["message_text_blocks"] == 0
        assert counts["message_tool_result_blocks"] == 0
        assert counts["message_tool_use_blocks"] == 0

    def test_message_with_text_block_cache_control(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello", "cache_control": {}},
                ],
            }
        ]
        counts = _cache_control_counts_for_messages(messages)
        assert counts["message_text_blocks"] == 1
        assert counts["message_tool_result_blocks"] == 0
        assert counts["message_tool_use_blocks"] == 0

    def test_message_with_tool_result_block_cache_control(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result", "cache_control": {}},
                ],
            }
        ]
        counts = _cache_control_counts_for_messages(messages)
        assert counts["message_text_blocks"] == 0
        assert counts["message_tool_result_blocks"] == 1
        assert counts["message_tool_use_blocks"] == 0

    def test_message_with_tool_use_block_cache_control(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "read", "input": {}, "cache_control": {}},
                ],
            }
        ]
        counts = _cache_control_counts_for_messages(messages)
        assert counts["message_text_blocks"] == 0
        assert counts["message_tool_result_blocks"] == 0
        assert counts["message_tool_use_blocks"] == 1

    def test_message_without_content(self) -> None:
        messages = [{"role": "user"}]
        counts = _cache_control_counts_for_messages(messages)
        assert counts["message_text_blocks"] == 0
        assert counts["message_tool_result_blocks"] == 0
        assert counts["message_tool_use_blocks"] == 0

    def test_message_with_non_list_content(self) -> None:
        messages = [{"role": "user", "content": "plain string"}]
        counts = _cache_control_counts_for_messages(messages)
        assert counts["message_text_blocks"] == 0
        assert counts["message_tool_result_blocks"] == 0
        assert counts["message_tool_use_blocks"] == 0

    def test_message_with_non_dict_block(self) -> None:
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        counts = _cache_control_counts_for_messages(messages)
        assert counts["message_text_blocks"] == 0
        assert counts["message_tool_result_blocks"] == 0
        assert counts["message_tool_use_blocks"] == 0

    def test_multiple_messages_multiple_blocks(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello", "cache_control": {}},
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result", "cache_control": {}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t2", "name": "read", "input": {}, "cache_control": {}},
                ],
            },
        ]
        counts = _cache_control_counts_for_messages(messages)
        assert counts["message_text_blocks"] == 1
        assert counts["message_tool_result_blocks"] == 1
        assert counts["message_tool_use_blocks"] == 1

    def test_block_without_type(self) -> None:
        messages = [{"role": "user", "content": [{"cache_control": {}}]}]
        counts = _cache_control_counts_for_messages(messages)
        assert counts["message_text_blocks"] == 0
        assert counts["message_tool_result_blocks"] == 0
        assert counts["message_tool_use_blocks"] == 0

    def test_block_with_empty_type(self) -> None:
        messages = [{"role": "user", "content": [{"type": "", "cache_control": {}}]}]
        counts = _cache_control_counts_for_messages(messages)
        assert counts["message_text_blocks"] == 0
        assert counts["message_tool_result_blocks"] == 0
        assert counts["message_tool_use_blocks"] == 0

    def test_block_with_whitespace_type(self) -> None:
        messages = [{"role": "user", "content": [{"type": "  text  ", "cache_control": {}}]}]
        counts = _cache_control_counts_for_messages(messages)
        assert counts["message_text_blocks"] == 1
        assert counts["message_tool_result_blocks"] == 0
        assert counts["message_tool_use_blocks"] == 0


class TestCacheControlCountsForTools:
    def test_empty_list(self) -> None:
        assert _cache_control_counts_for_tools([]) == 0

    def test_non_list_input(self) -> None:
        assert _cache_control_counts_for_tools({}) == 0
        assert _cache_control_counts_for_tools("tools") == 0
        assert _cache_control_counts_for_tools(123) == 0

    def test_tools_without_cache_control(self) -> None:
        tools = [{"name": "read", "description": "Read a file"}]
        assert _cache_control_counts_for_tools(tools) == 0

    def test_tool_with_cache_control(self) -> None:
        tools = [{"name": "read", "description": "Read a file", "cache_control": {}}]
        assert _cache_control_counts_for_tools(tools) == 1

    def test_multiple_tools_with_cache_control(self) -> None:
        tools = [
            {"name": "read", "cache_control": {}},
            {"name": "write", "cache_control": {}},
        ]
        assert _cache_control_counts_for_tools(tools) == 2

    def test_mixed_tools(self) -> None:
        tools = [
            {"name": "read", "cache_control": {}},
            {"name": "write"},
            {"name": "edit", "cache_control": {}},
        ]
        assert _cache_control_counts_for_tools(tools) == 2

    def test_tool_with_non_dict(self) -> None:
        tools = [{"name": "read", "cache_control": {}}, "not a dict", {"name": "write"}]
        assert _cache_control_counts_for_tools(tools) == 1

    def test_none_in_list(self) -> None:
        tools = [{"name": "read", "cache_control": {}}, None, {"name": "write"}]
        assert _cache_control_counts_for_tools(tools) == 1
