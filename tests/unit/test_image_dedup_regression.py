"""Regression tests: image dedup must not affect existing string tool_result compression."""

from __future__ import annotations


def _make_str_tool_result_message(tool_use_id: str, content: str) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
        ],
    }


def _make_image_tool_result_message(tool_use_id: str, data: str) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": data,
                        },
                    }
                ],
            }
        ],
    }


class TestStringToolResultUnaffected:
    def test_string_content_not_modified_by_image_dedup(self) -> None:
        """String-content tool_result blocks must be handled identically before and after."""
        from tok.compression._history_pipeline import compress_tool_results_impl

        text = "x" * 5000
        messages = [_make_str_tool_result_message("call_1", text)]
        _, breakdown = compress_tool_results_impl(messages)
        # image_dedup must NOT appear when there are no image blocks
        assert "image_dedup" not in breakdown

    def test_compress_tool_results_return_type_unchanged(self) -> None:
        from tok.compression._history_pipeline import compress_tool_results_impl

        messages = [_make_str_tool_result_message("call_2", "hello world")]
        result = compress_tool_results_impl(messages)
        msgs, bd = result
        assert isinstance(msgs, list)
        assert isinstance(bd, dict)

    def test_no_image_dedup_key_absent_not_zero(self) -> None:
        """The image_dedup key must be completely absent (not present as 0)."""
        from tok.compression._history_pipeline import compress_tool_results_impl

        messages = [_make_str_tool_result_message("call_3", "short")]
        _, breakdown = compress_tool_results_impl(messages)
        assert "image_dedup" not in breakdown

    def test_empty_messages_still_works(self) -> None:
        from tok.compression._history_pipeline import compress_tool_results_impl

        msgs, bd = compress_tool_results_impl([])
        assert msgs == []
        assert "image_dedup" not in bd


class TestImageDedupEndToEnd:
    def test_image_dedup_key_present_when_images_seen_twice(self) -> None:
        """image_dedup breakdown key appears when a repeated image is compressed."""
        from tok.compression._history_pipeline import compress_tool_results_impl

        data = "B" * 500
        msg1 = _make_image_tool_result_message("img_call_1", data)
        msg2 = _make_image_tool_result_message("img_call_2", data)  # same image, new tool id
        _, breakdown = compress_tool_results_impl([msg1, msg2])
        assert "image_dedup" in breakdown
        assert breakdown["image_dedup"] > 0

    def test_image_first_occurrence_content_preserved(self) -> None:
        """First occurrence of an image is never replaced."""
        from tok.compression._history_pipeline import compress_tool_results_impl

        data = "C" * 300
        msg = _make_image_tool_result_message("img_call_1", data)
        messages = [msg]
        result_msgs, breakdown = compress_tool_results_impl(messages)
        # First occurrence: no dedup, image block preserved
        assert "image_dedup" not in breakdown
        block = result_msgs[0]["content"][0]
        assert block["content"][0]["type"] == "image"

    def test_mixed_string_and_image_messages(self) -> None:
        """String results and image results in the same request don't interfere."""
        from tok.compression._history_pipeline import compress_tool_results_impl

        str_msg = _make_str_tool_result_message("str_call", "some long text " * 200)
        data = "D" * 400
        img_msg1 = _make_image_tool_result_message("img_1", data)
        img_msg2 = _make_image_tool_result_message("img_2", data)  # repeat

        _, breakdown = compress_tool_results_impl([str_msg, img_msg1, img_msg2])
        assert "image_dedup" in breakdown
        assert breakdown["image_dedup"] > 0
