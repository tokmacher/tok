from __future__ import annotations

import re

from tok.compression import truncate_large_result
from tok.runtime.repeat_targets import build_file_summary, build_search_summary


def test_truncate_large_result_chooses_a_safe_line_boundary() -> None:
    lines = [
        "class Example:",
        "    def build(self):",
        '        payload = call_very_long_function("alpha", "beta", "gamma", "delta", "epsilon")',
        "        return payload",
        "",
        "class NextExample:",
    ]
    lines.extend(f"    filler line {idx}" for idx in range(1, 90))
    large_text = "\n".join(lines)

    truncated = truncate_large_result(large_text, limit=420)

    assert "... [TRUNCATED" in truncated
    assert "continue at line" in truncated
    assert "\n... [TRUNCATED" in truncated
    assert (
        'call_very_long_function("alpha", "beta", "gamma", "delta", "epsilon")' in truncated
        or "call_very_long_function(" not in truncated
    )


def test_truncate_large_result_reports_clear_continuation_metadata() -> None:
    large_text = "\n".join(f"line {idx}: value {idx * 11}" for idx in range(1, 160))

    truncated = truncate_large_result(large_text, limit=500)

    assert re.search(r"omitted lines \d+-\d+; continue at line \d+", truncated)
    assert "... [TRUNCATED" in truncated
    assert "line 1:" in truncated
    assert "line 159:" in truncated


def test_build_file_summary_anchors_to_decisive_assignment_over_boilerplate() -> None:
    text = "class Example:\n    def resolve(self):\n        logger.debug('noise')\n        intermediate = compute_noise()\n        value = compute_value()\n        return value\n\n# repeated boilerplate\npass"

    summary = build_file_summary(text, max_chars=280, max_lines=6)

    assert "class Example:" in summary
    assert "def resolve(self):" in summary
    assert "value = compute_value()" in summary or "return value" in summary
    assert "logger.debug" not in summary
    assert "repeated boilerplate" not in summary


def test_build_search_summary_anchors_to_decisive_match_line_over_boilerplate() -> None:
    text = "src/example.py:3:        logger.debug('noise')\nsrc/example.py:4:        intermediate = compute_noise()\nsrc/example.py:5:        result = compute_value()\nsrc/example.py:6:        return result\nsrc/example.py:7:        pass"

    summary = build_search_summary(text, max_chars=280, max_lines=4)

    assert "result = compute_value()" in summary or "return result" in summary
    assert "logger.debug" not in summary
    assert "pass" not in summary
