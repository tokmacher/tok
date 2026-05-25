"""RED tests for image block deduplication in tool_result compression.

These tests are written against the planned public interface of
src/tok/compression/_image_dedup.py before that module exists.
"""

from __future__ import annotations


def _make_image_block(data: str, media_type: str = "image/png") -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


def _make_url_image_block(url: str) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "url",
            "url": url,
        },
    }


def _make_text_block(text: str) -> dict:
    return {"type": "text", "text": text}


class TestImageFingerprint:
    def test_fingerprint_is_deterministic(self) -> None:
        from tok.compression._image_dedup import _image_fingerprint

        block = _make_image_block("abc123def456")
        assert _image_fingerprint(block) == _image_fingerprint(block)

    def test_fingerprint_differs_for_different_images(self) -> None:
        from tok.compression._image_dedup import _image_fingerprint

        block_a = _make_image_block("aaaaaa")
        block_b = _make_image_block("bbbbbb")
        assert _image_fingerprint(block_a) != _image_fingerprint(block_b)

    def test_fingerprint_is_short_hex_string(self) -> None:
        from tok.compression._image_dedup import _image_fingerprint

        block = _make_image_block("somedata")
        fp = _image_fingerprint(block)
        assert isinstance(fp, str)
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)


class TestStripDuplicateImages:
    def test_first_occurrence_preserved(self) -> None:
        from tok.compression._image_dedup import strip_duplicate_images

        block = _make_image_block("uniquedata123")
        seen: set[str] = set()
        result, saved = strip_duplicate_images([block], seen)
        assert result[0]["type"] == "image"
        assert saved == 0

    def test_second_occurrence_replaced_with_stub(self) -> None:
        from tok.compression._image_dedup import _image_fingerprint, strip_duplicate_images

        block = _make_image_block("R" * 200)  # long enough to exceed stub length
        fp = _image_fingerprint(block)
        seen: set[str] = {fp}  # already seen
        result, saved = strip_duplicate_images([block], seen)
        assert result[0]["type"] == "text"
        assert fp in result[0]["text"]
        assert saved > 0

    def test_stub_contains_fingerprint(self) -> None:
        from tok.compression._image_dedup import _image_fingerprint, strip_duplicate_images

        block = _make_image_block("data_for_stub_test")
        fp = _image_fingerprint(block)
        seen: set[str] = {fp}
        result, _ = strip_duplicate_images([block], seen)
        assert fp in result[0]["text"]

    def test_url_images_never_deduped(self) -> None:
        from tok.compression._image_dedup import strip_duplicate_images

        block = _make_url_image_block("https://example.com/img.png")
        seen: set[str] = set()
        # Call twice — URL images should never be fingerprinted or stubbed
        result1, saved1 = strip_duplicate_images([block], seen)
        result2, saved2 = strip_duplicate_images([block], seen)
        assert result1[0]["type"] == "image"
        assert result2[0]["type"] == "image"
        assert saved1 == 0
        assert saved2 == 0

    def test_saved_chars_approximate(self) -> None:
        from tok.compression._image_dedup import _image_fingerprint, strip_duplicate_images

        data = "A" * 1000
        block = _make_image_block(data)
        fp = _image_fingerprint(block)
        seen: set[str] = {fp}
        _, saved = strip_duplicate_images([block], seen)
        # saved should be substantially positive (base64 data >> stub)
        assert saved > 100

    def test_mixed_content_partial_dedup(self) -> None:
        from tok.compression._image_dedup import _image_fingerprint, strip_duplicate_images

        img_block = _make_image_block("I" * 200)  # long enough to exceed stub length
        txt_block = _make_text_block("hello world")
        fp = _image_fingerprint(img_block)
        seen: set[str] = {fp}
        result, saved = strip_duplicate_images([img_block, txt_block], seen)
        # image replaced, text untouched
        assert result[0]["type"] == "text"
        assert fp in result[0]["text"]
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "hello world"
        assert saved > 0

    def test_empty_content_list_safe(self) -> None:
        from tok.compression._image_dedup import strip_duplicate_images

        result, saved = strip_duplicate_images([], set())
        assert result == []
        assert saved == 0

    def test_seen_set_updated_after_first_occurrence(self) -> None:
        from tok.compression._image_dedup import _image_fingerprint, strip_duplicate_images

        block = _make_image_block("trackme000")
        fp = _image_fingerprint(block)
        seen: set[str] = set()
        strip_duplicate_images([block], seen)
        assert fp in seen
