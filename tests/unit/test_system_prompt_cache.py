"""RED tests for system prompt static-section cache hint injection.

Written against the planned interface of src/tok/compression/_system_prompt_cache.py
before that module exists.
"""

from __future__ import annotations

_LONG_PARAGRAPH = "A" * 300
_ANOTHER_LONG_PARAGRAPH = "B" * 300
_SHORT_PARAGRAPH = "tiny"


class TestParagraphFingerprints:
    def test_paragraph_fingerprints_deterministic(self) -> None:
        from tok.compression._system_prompt_cache import paragraph_fingerprints

        text = f"{_LONG_PARAGRAPH}\n\n{_ANOTHER_LONG_PARAGRAPH}"
        assert paragraph_fingerprints(text) == paragraph_fingerprints(text)

    def test_paragraph_fingerprints_split_on_blank_lines(self) -> None:
        from tok.compression._system_prompt_cache import paragraph_fingerprints

        text = f"{_LONG_PARAGRAPH}\n\n{_ANOTHER_LONG_PARAGRAPH}"
        fps = paragraph_fingerprints(text)
        assert len(fps) == 2

    def test_paragraph_fingerprints_different_paragraphs(self) -> None:
        from tok.compression._system_prompt_cache import paragraph_fingerprints

        fps = paragraph_fingerprints(f"{_LONG_PARAGRAPH}\n\n{_ANOTHER_LONG_PARAGRAPH}")
        assert fps[0] != fps[1]

    def test_paragraph_fingerprints_empty_string(self) -> None:
        from tok.compression._system_prompt_cache import paragraph_fingerprints

        assert paragraph_fingerprints("") == []

    def test_paragraph_fingerprints_single_paragraph(self) -> None:
        from tok.compression._system_prompt_cache import paragraph_fingerprints

        fps = paragraph_fingerprints(_LONG_PARAGRAPH)
        assert len(fps) == 1
        assert isinstance(fps[0], str)


class TestFindStaticPrefixLength:
    def test_identical_returns_full_length(self) -> None:
        from tok.compression._system_prompt_cache import find_static_prefix_length

        fps = ["aaa", "bbb", "ccc"]
        assert find_static_prefix_length(fps, fps) == 3

    def test_diverges_at_third_paragraph(self) -> None:
        from tok.compression._system_prompt_cache import find_static_prefix_length

        current = ["aaa", "bbb", "NEW"]
        prior = ["aaa", "bbb", "ccc"]
        assert find_static_prefix_length(current, prior) == 2

    def test_completely_different_returns_zero(self) -> None:
        from tok.compression._system_prompt_cache import find_static_prefix_length

        assert find_static_prefix_length(["x", "y"], ["a", "b"]) == 0

    def test_empty_current_returns_zero(self) -> None:
        from tok.compression._system_prompt_cache import find_static_prefix_length

        assert find_static_prefix_length([], ["a", "b"]) == 0

    def test_empty_prior_returns_zero(self) -> None:
        from tok.compression._system_prompt_cache import find_static_prefix_length

        assert find_static_prefix_length(["a"], []) == 0


class TestApplySystemCacheHint:
    def test_no_prior_turn_system_unchanged(self) -> None:
        from tok.compression._system_prompt_cache import apply_system_cache_hint

        system = f"{_LONG_PARAGRAPH}\n\n{_ANOTHER_LONG_PARAGRAPH}"
        result, fps, static_chars = apply_system_cache_hint(system, None)
        assert result == system
        assert static_chars == 0
        assert isinstance(fps, list)
        assert len(fps) > 0

    def test_converts_string_to_blocks_on_second_turn(self) -> None:
        from tok.compression._system_prompt_cache import apply_system_cache_hint

        system = f"{_LONG_PARAGRAPH}\n\n{_ANOTHER_LONG_PARAGRAPH}"
        _, prior_fps, _ = apply_system_cache_hint(system, None)
        # Second turn with same system
        result, _, static_chars = apply_system_cache_hint(system, prior_fps)
        assert isinstance(result, list)
        assert static_chars > 0

    def test_returns_fingerprints_of_current_content(self) -> None:
        from tok.compression._system_prompt_cache import apply_system_cache_hint, paragraph_fingerprints

        system = f"{_LONG_PARAGRAPH}\n\n{_ANOTHER_LONG_PARAGRAPH}"
        _, fps, _ = apply_system_cache_hint(system, None)
        assert fps == paragraph_fingerprints(system)

    def test_below_min_chars_threshold_unchanged(self) -> None:
        from tok.compression._system_prompt_cache import apply_system_cache_hint

        short_system = f"{_SHORT_PARAGRAPH}\n\n{_SHORT_PARAGRAPH}"
        prior_fps = ["fp1", "fp2"]  # pretend prior exists
        result, _, static_chars = apply_system_cache_hint(short_system, prior_fps)
        # Below min threshold: no conversion, no cache hint
        assert result == short_system
        assert static_chars == 0

    def test_list_input_adds_cache_control_to_first_block(self) -> None:
        from tok.compression._system_prompt_cache import apply_system_cache_hint, paragraph_fingerprints

        system_str = f"{_LONG_PARAGRAPH}\n\n{_ANOTHER_LONG_PARAGRAPH}"
        prior_fps = paragraph_fingerprints(system_str)
        # Pass as list input (pre-structured)
        system_list = [{"type": "text", "text": system_str}]
        result, _, static_chars = apply_system_cache_hint(system_list, prior_fps)
        assert isinstance(result, list)
        if static_chars > 0:
            first = result[0]
            assert "cache_control" in first

    def test_dynamic_suffix_no_cache_control(self) -> None:
        from tok.compression._system_prompt_cache import apply_system_cache_hint

        static_part = _LONG_PARAGRAPH
        dynamic_part = "D" * 300
        system = f"{static_part}\n\n{dynamic_part}"
        # First turn — build prior
        _, prior_fps, _ = apply_system_cache_hint(f"{static_part}\n\nOLD_DYNAMIC", None)
        # Second turn — static prefix same, dynamic differs
        result, _, static_chars = apply_system_cache_hint(system, prior_fps)
        if isinstance(result, list) and len(result) > 1:
            # Last block (dynamic suffix) must not have cache_control
            last = result[-1]
            assert "cache_control" not in last

    def test_estimated_chars_accurate(self) -> None:
        from tok.compression._system_prompt_cache import apply_system_cache_hint

        system = f"{_LONG_PARAGRAPH}\n\n{_ANOTHER_LONG_PARAGRAPH}"
        _, prior_fps, _ = apply_system_cache_hint(system, None)
        _, _, static_chars = apply_system_cache_hint(system, prior_fps)
        # static_chars should be at least as large as one long paragraph
        assert static_chars >= len(_LONG_PARAGRAPH)

    def test_first_turn_static_chars_is_zero(self) -> None:
        from tok.compression._system_prompt_cache import apply_system_cache_hint

        _, _, static_chars = apply_system_cache_hint(_LONG_PARAGRAPH, None)
        assert static_chars == 0
