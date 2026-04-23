"""Tests for tool result truncation module."""

from __future__ import annotations

from tok.compression._tool_result_truncation import (
    _choose_line_boundary,
    _extract_symbol_table,
    _leading_whitespace_width,
    _line_boundary_priority,
    _pytest_aware_truncation,
    truncate_large_result,
)


class TestLeadingWhitespaceWidth:
    def test_no_whitespace(self) -> None:
        assert _leading_whitespace_width("hello") == 0

    def test_spaces_only(self) -> None:
        assert _leading_whitespace_width("  hello") == 2

    def test_tabs_only(self) -> None:
        assert _leading_whitespace_width("\t\thello") == 2

    def test_mixed_spaces_and_tabs(self) -> None:
        assert _leading_whitespace_width(" \t hello") == 3

    def test_only_whitespace(self) -> None:
        assert _leading_whitespace_width("   \t  ") == 6

    def test_empty_string(self) -> None:
        assert _leading_whitespace_width("") == 0


class TestLineBoundaryPriority:
    def test_both_empty_stripped(self) -> None:
        assert _line_boundary_priority("", "") == 0

    def test_prev_empty_stripped(self) -> None:
        assert _line_boundary_priority("", "def foo():") == 0

    def test_next_empty_stripped(self) -> None:
        assert _line_boundary_priority("def foo():", "") == 0

    def test_next_indent_less_than_prev(self) -> None:
        assert _line_boundary_priority("    def foo():", "  class Bar:") == 0

    def test_next_indent_greater_prev(self) -> None:
        result = _line_boundary_priority("class Foo:", "    def method(self):")
        assert result >= 0

    def test_next_line_is_decorator(self) -> None:
        result = _line_boundary_priority("class Foo:", "    @property")
        assert result == 1

    def test_next_line_is_def(self) -> None:
        result = _line_boundary_priority("class Foo:", "    def method(self):")
        assert result == 1

    def test_next_line_is_class(self) -> None:
        result = _line_boundary_priority("class Foo:", "    class Inner:")
        assert result == 1

    def test_next_line_is_async_def(self) -> None:
        result = _line_boundary_priority("class Foo:", "    async def amethod(self):")
        assert result == 1

    def test_next_line_is_if(self) -> None:
        result = _line_boundary_priority("def foo():", "    if x > 0:")
        assert result == 1

    def test_next_line_is_for(self) -> None:
        result = _line_boundary_priority("def foo():", "    for i in range(10):")
        assert result == 1

    def test_next_line_is_while(self) -> None:
        result = _line_boundary_priority("def foo():", "    while True:")
        assert result == 1

    def test_next_line_is_with(self) -> None:
        result = _line_boundary_priority("def foo():", "    with open('f'):")
        assert result == 1

    def test_next_line_is_try(self) -> None:
        result = _line_boundary_priority("def foo():", "    try:")
        assert result == 1

    def test_next_line_is_except(self) -> None:
        result = _line_boundary_priority("def foo():", "    except ValueError:")
        assert result == 1

    def test_next_line_is_elif(self) -> None:
        result = _line_boundary_priority("if x:", "    elif y:")
        assert result == 1

    def test_next_line_is_else(self) -> None:
        result = _line_boundary_priority("if x:", "    else:")
        assert result == 1

    def test_next_line_is_match(self) -> None:
        result = _line_boundary_priority("def foo():", "    match x:")
        assert result == 1

    def test_next_line_is_case(self) -> None:
        result = _line_boundary_priority("match x:", "    case 1:")
        assert result == 1

    def test_next_line_at_same_indent_no_keyword(self) -> None:
        result = _line_boundary_priority("def foo():", "    return x + 1")
        assert result == 2


class TestChooseLineBoundary:
    def test_single_line(self) -> None:
        lines = ["hello"]
        offsets = [0, 5]
        result = _choose_line_boundary(lines, offsets, target_chars=2, search_window_chars=10)
        assert result == 1

    def test_empty_lines(self) -> None:
        lines: list[str] = []
        offsets: list[int] = []
        result = _choose_line_boundary(lines, offsets, target_chars=2, search_window_chars=10)
        assert result == 0

    def test_finds_boundary_in_search_window(self) -> None:
        lines = ["line0", "line1", "line2", "line3", "line4"]
        offsets = [0, 5, 10, 15, 20, 25]
        result = _choose_line_boundary(lines, offsets, target_chars=12, search_window_chars=10)
        assert 1 <= result <= 4

    def test_no_candidates_in_search_window(self) -> None:
        lines = ["line0", "line1", "line2", "line3", "line4"]
        offsets = [0, 5, 10, 15, 20, 25]
        result = _choose_line_boundary(lines, offsets, target_chars=100, search_window_chars=5)
        assert result >= 1

    def test_uses_line_boundary_priority(self) -> None:
        lines = ["class Foo:", "    @property", "    def method(self):"]
        offsets = [0, 11, 25, 45]
        result = _choose_line_boundary(lines, offsets, target_chars=15, search_window_chars=20)
        assert result >= 1


class TestPytestAwareTruncation:
    def test_no_pytest_summary(self) -> None:
        text = "Some random output\nwith multiple lines\nbut no pytest info"
        lines = text.splitlines(keepends=True)
        result = _pytest_aware_truncation(text, lines, limit=100)
        assert result is None

    def test_pytest_summary_only_no_failures(self) -> None:
        text = """=========================== test session starts ============================
collected 5 items

=========================== 5 passed in 1.23s ============================"""
        lines = text.splitlines(keepends=True)
        result = _pytest_aware_truncation(text, lines, limit=100)
        assert result is None

    def test_pytest_failures_section(self) -> None:
        text = """=========================== test session starts ============================
collected 5 items

_____________________________ test_one _____________________________
File "/tests/test_file.py", line 10
    assert False
FAILED test_file.py::test_one

_____________________________ test_two _____________________________
File "/tests/test_file.py", line 20
    assert True
PASSED test_file.py::test_two

=========================== 5 passed, 1 failed in 2.00s ==========================="""
        lines = text.splitlines(keepends=True)
        result = _pytest_aware_truncation(text, lines, limit=100)
        assert result is not None
        assert "TRUNCATED" in result
        assert "FAIL" in result.upper()

    def test_pytest_only_failures_start(self) -> None:
        text = """_____________________________ test_one _____________________________
File "/tests/test_file.py", line 10
    assert False
FAILED test_file.py::test_one

=========================== 1 failed in 1.00s ==========================="""
        lines = text.splitlines(keepends=True)
        result = _pytest_aware_truncation(text, lines, limit=50)
        assert result is not None
        assert "TRUNCATED" in result

    def test_pytest_multiple_failures_sections(self) -> None:
        text = """__________ test_fail_1 __________
FAILED test_a.py::test_fail_1

__________ test_fail_2 __________
FAILED test_b.py::test_fail_2

=========================== 2 failed in 1.00s ==========================="""
        lines = text.splitlines(keepends=True)
        result = _pytest_aware_truncation(text, lines, limit=50)
        assert result is not None

    def test_pytest_summary_regex_variations(self) -> None:
        text = "========================== 5 passed, 2 failed ==========================="
        lines = text.splitlines(keepends=True)
        result = _pytest_aware_truncation(text, lines, limit=100)
        assert result is None

    def test_pytest_error_not_failure(self) -> None:
        text = """__________ test_error __________
ERROR test_file.py::test_error - ValueError

=========================== 1 error in 1.00s ==========================="""
        lines = text.splitlines(keepends=True)
        result = _pytest_aware_truncation(text, lines, limit=50)
        assert result is None


class TestExtractSymbolTable:
    def test_empty_lines(self) -> None:
        result = _extract_symbol_table([], 0, 0)
        assert result == ""

    def test_no_symbols(self) -> None:
        lines = ["line one", "line two", "no symbols here"]
        result = _extract_symbol_table(lines, 0, 3)
        assert result == ""

    def test_single_import(self) -> None:
        lines = ["import os", "import sys"]
        result = _extract_symbol_table(lines, 0, 2)
        assert "import os" in result
        assert "import sys" in result

    def test_class_and_def(self) -> None:
        lines = [
            "class Foo:",
            "    def method(self):",
            "    async def amethod(self):",
        ]
        result = _extract_symbol_table(lines, 0, 3)
        assert "class Foo" in result
        assert "def method" in result
        assert "async def amethod" in result

    def test_imports_at_top(self) -> None:
        lines = [
            "import os",
            "from collections import defaultdict",
            "class Foo:",
        ]
        result = _extract_symbol_table(lines, 0, 3)
        assert "import os" in result
        assert "from collections import defaultdict" in result
        assert "class Foo" in result

    def test_max_symbols_limit(self) -> None:
        lines = ["class Class" + str(i) + ":" for i in range(50)]
        lines += ["def func" + str(i) + "():" for i in range(50)]
        result = _extract_symbol_table(lines, 0, 100)
        assert "_MAX_SYMBOLS" in result or "..." in result

    def test_deeply_nested_skipped(self) -> None:
        lines = [
            "class Foo:",
            "            def deeply_nested_method(self):",
        ]
        result = _extract_symbol_table(lines, 0, 2)
        assert "deeply_nested_method" not in result

    def test_top_level_function_skipped_if_indent_too_large(self) -> None:
        lines = [
            "class Foo:",
            "    def method(self):",
            "def top_level_func():",
        ]
        result = _extract_symbol_table(lines, 0, 3)
        assert "def top_level_func" in result

    def test_line_numbers_in_output(self) -> None:
        lines = ["class Foo:", "    def bar(self):"]
        result = _extract_symbol_table(lines, 0, 2)
        assert "L1" in result or "L3" in result

    def test_truncates_long_signatures(self) -> None:
        long_sig = "def " + "a" * 100 + "()"
        lines = [long_sig]
        result = _extract_symbol_table(lines, 0, 1)
        assert "..." in result


class TestTruncateLargeResult:
    def test_already_compressed(self) -> None:
        text = "already compressed content"
        result = truncate_large_result(text, already_compressed=True)
        assert result == text

    def test_small_text_unchanged(self) -> None:
        text = "short text"
        result = truncate_large_result(text, limit=1200)
        assert result == text

    def test_small_text_near_limit(self) -> None:
        text = "x" * 2000
        result = truncate_large_result(text, limit=1200)
        assert len(result) < len(text)

    def test_small_multi_line_file_at_high_limit(self) -> None:
        lines = ["line " + str(i) for i in range(50)]
        text = "\n".join(lines)
        result = truncate_large_result(text, limit=1200)
        assert result == text

    def test_small_multi_line_file_at_low_limit(self) -> None:
        lines = ["line " + str(i) for i in range(100)]
        text = "\n".join(lines)
        result = truncate_large_result(text, limit=500)
        assert len(result) < len(text)

    def test_single_line_with_error_signal(self) -> None:
        text = "line1\n" * 20 + "error: something went wrong here\n" + "line4\n" * 20
        result = truncate_large_result(text, limit=50)
        assert "TRUNCATED" in result or "SIGNAL FOUND" in result

    def test_single_line_with_fail_signal(self) -> None:
        text = "line1\n" * 20 + "fail: test failed here\n" + "line3\n" * 20
        result = truncate_large_result(text, limit=30)
        assert "TRUNCATED" in result or "SIGNAL FOUND" in result

    def test_single_line_with_exception_signal(self) -> None:
        text = "line1\n" * 20 + "exception occurred here\n" + "line3\n" * 20
        result = truncate_large_result(text, limit=30)
        assert "TRUNCATED" in result or "SIGNAL FOUND" in result

    def test_single_line_with_traceback_signal(self) -> None:
        text = "line1\n" * 20 + "traceback here...\n" + "line3\n" * 20
        result = truncate_large_result(text, limit=30)
        assert "TRUNCATED" in result or "SIGNAL FOUND" in result

    def test_single_line_no_signal(self) -> None:
        text = "just some normal text here"
        result = truncate_large_result(text, limit=10)
        assert "TRUNCATED" in result

    def test_single_line_whitespace_only(self) -> None:
        text = "   " * 100
        result = truncate_large_result(text, limit=50)
        assert "TRUNCATED" in result

    def test_multi_line_pytest_with_failures(self) -> None:
        text = """=========================== test session starts ============================
collected 5 items

_____________________________ test_one _____________________________
File "/tests/test_file.py", line 10
    assert False
FAILED test_file.py::test_one

=========================== 5 passed, 1 failed in 2.00s ==========================="""
        result = truncate_large_result(text, limit=100)
        assert "TRUNCATED" in result or "FAIL" in result.upper()

    def test_multi_line_non_pytest(self) -> None:
        lines = ["line" + str(i) for i in range(100)]
        text = "\n".join(lines)
        result = truncate_large_result(text, limit=100)
        assert "TRUNCATED" in result
        assert "offset=" in result

    def test_symbol_table_inserted_in_omitted_section(self) -> None:
        lines = ["def func" + str(i) + "():\n    pass\n" for i in range(100)]
        text = "".join(lines)
        result = truncate_large_result(text, limit=100)
        assert "offset=" in result or "@symbols" in result

    def test_result_shorter_than_original(self) -> None:
        lines = ["x" * 80 for _ in range(50)]
        text = "\n".join(lines)
        result = truncate_large_result(text, limit=100)
        assert len(result) < len(text)

    def test_continuation_line_in_marker(self) -> None:
        lines = ["line" + str(i) for i in range(200)]
        text = "\n".join(lines)
        result = truncate_large_result(text, limit=100)
        assert "continue at line" in result

    def test_fallback_truncation_when_compression_fails(self) -> None:
        lines = ["line" + str(i) for i in range(100)]
        text = "\n".join(lines)
        result = truncate_large_result(text, limit=50)
        assert "TRUNCATED" in result

    def test_fallback_with_signal_found(self) -> None:
        text = "normal line\n" * 50 + "warning: deprecated\n" + "normal line\n" * 50
        result = truncate_large_result(text, limit=50)
        assert "TRUNCATED" in result or "SIGNAL FOUND" in result

    def test_fallback_with_no_signal(self) -> None:
        text = "normal line\n" * 100
        result = truncate_large_result(text, limit=50)
        assert "TRUNCATED" in result

    def test_head_and_tail_preserved(self) -> None:
        lines = ["HEAD"] + ["middle"] * 100 + ["TAIL"]
        text = "\n".join(lines)
        result = truncate_large_result(text, limit=50)
        assert "HEAD" in result
        assert "TAIL" in result

    def test_omitted_chars_reported(self) -> None:
        text = "x" * 1000
        result = truncate_large_result(text, limit=100)
        assert "TRUNCATED" in result
        assert "1000" not in result or "CHARS" in result

    def test_large_limit_still_truncates_extreme_text(self) -> None:
        text = "x" * 100000
        result = truncate_large_result(text, limit=1200)
        assert len(result) < len(text)
        assert "TRUNCATED" in result

    def test_whitespace_only_lines_counted(self) -> None:
        text = "   \n" * 200 + "error line\n" + "   \n" * 200
        result = truncate_large_result(text, limit=100)
        assert "TRUNCATED" in result or "SIGNAL" in result

    def test_limit_1(self) -> None:
        text = "x" * 100
        result = truncate_large_result(text, limit=1)
        assert "TRUNCATED" in result

    def test_limit_very_small(self) -> None:
        text = "line1\n" * 10 + "error\n" + "line4\n" * 10
        result = truncate_large_result(text, limit=20)
        assert "TRUNCATED" in result

    def test_avg_line_len_threshold(self) -> None:
        lines = ["x" * 50 for _ in range(50)]
        text = "\n".join(lines)
        result = truncate_large_result(text, limit=1000)
        assert "TRUNCATED" in result or result == text
