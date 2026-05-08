"""Tests for file-read compression codecs."""

from __future__ import annotations

from tok.compression._tool_result_file_read import (
    _SIGNATURE_CONTINUATION_RE,
    _build_section_map,
    _compress_file_read,
    _extract_python_skeleton,
    _is_python_file,
    _is_signature_continuation,
)


class TestIsSignatureContinuation:
    def test_prior_unclosed_parens(self) -> None:
        assert _is_signature_continuation(1, "line") is True

    def test_empty_stripped(self) -> None:
        assert _is_signature_continuation(0, "") is False

    def test_stripped_starts_with_close_paren(self) -> None:
        assert _is_signature_continuation(0, ")") is True
        assert _is_signature_continuation(0, ")") is True

    def test_stripped_starts_with_close_bracket(self) -> None:
        assert _is_signature_continuation(0, "]") is True

    def test_stripped_starts_with_close_brace(self) -> None:
        assert _is_signature_continuation(0, "}") is True

    def test_stripped_ends_with_comma(self) -> None:
        assert _is_signature_continuation(0, "arg,") is True

    def test_stripped_ends_with_backslash(self) -> None:
        assert _is_signature_continuation(0, "arg \\") is True

    def test_matches_continuation_regex(self) -> None:
        assert _is_signature_continuation(0, "x,") is True
        assert _is_signature_continuation(0, "])") is True
        assert _is_signature_continuation(0, "],  # comment") is True

    def test_no_match(self) -> None:
        assert _is_signature_continuation(0, "def foo():") is False
        assert _is_signature_continuation(0, "class Bar:") is False


class TestBuildSectionMap:
    def test_empty_lines(self) -> None:
        result = _build_section_map([])
        assert result == ""

    def test_no_matches(self) -> None:
        lines = ["import os", "x = 1", "y = 2"]
        result = _build_section_map(lines)
        assert result == ""

    def test_single_class(self) -> None:
        lines = ["class Foo:"]
        result = _build_section_map(lines)
        assert "Foo:L1" in result

    def test_single_function(self) -> None:
        lines = ["", "def bar():"]
        result = _build_section_map(lines)
        assert "bar:L2" in result

    def test_async_function(self) -> None:
        lines = ["async def amethod():"]
        result = _build_section_map(lines)
        assert "amethod:L1" in result

    def test_multiple_sections(self) -> None:
        lines = ["class Foo:", "def bar():", "async def amethod():", "class Baz:"]
        result = _build_section_map(lines)
        assert "Foo:L1" in result
        assert "bar:L2" in result
        assert "amethod:L3" in result
        assert "Baz:L4" in result

    def test_limit_12_sections(self) -> None:
        lines = [f"def func{i}():" for i in range(20)]
        result = _build_section_map(lines)
        sections = result.split(",")
        assert len(sections) == 12


class TestIsPythonFile:
    def test_py_extension(self) -> None:
        context = {"args": {"path": "foo.py"}}
        assert _is_python_file("", context) is True

    def test_pyi_extension(self) -> None:
        context = {"args": {"path": "foo.pyi"}}
        assert _is_python_file("", context) is True

    def test_non_py_extension(self) -> None:
        context = {"args": {"path": "foo.txt"}}
        assert _is_python_file("", context) is False

    def test_file_path_variants(self) -> None:
        assert _is_python_file("", {"args": {"file_path": "foo.py"}}) is True
        assert _is_python_file("", {"args": {"AbsolutePath": "foo.py"}}) is True
        assert _is_python_file("", {"args": {"TargetFile": "foo.py"}}) is True

    def test_empty_args(self) -> None:
        assert _is_python_file("def foo():\n    pass", {}) is False

    def test_def_keyword(self) -> None:
        text = "def foo():\n    pass\nclass Bar:\n    pass\nimport os"
        assert _is_python_file(text) is True

    def test_class_keyword(self) -> None:
        text = "class Foo:\n    pass\ndef bar():\n    pass\nimport sys"
        assert _is_python_file(text) is True

    def test_import_statement(self) -> None:
        text = "import os\nfrom sys import path\nclass Foo:\n    pass"
        assert _is_python_file(text) is True

    def test_from_import(self) -> None:
        text = "from collections import defaultdict\ndef foo():\n    pass\nclass Bar:\n    pass"
        assert _is_python_file(text) is True

    def test_async_def(self) -> None:
        text = "async def foo():\n    await bar()\ndef main():\n    pass\nclass Foo:\n    pass"
        assert _is_python_file(text) is True

    def test_decorator(self) -> None:
        text = "@property\ndef foo(self):\n    return self._foo\ndef bar():\n    pass\nclass Baz:\n    pass"
        assert _is_python_file(text) is True

    def test_main_check(self) -> None:
        text = "if __name__ == '__main__':\n    main()\ndef foo():\n    pass\nclass Bar:\n    pass"
        assert _is_python_file(text) is True

    def test_needs_three_matches(self) -> None:
        text = "x = 1\ny = 2\nz = 3"
        assert _is_python_file(text) is False

    def test_content_overrides_extension(self) -> None:
        context = {"args": {"path": "foo.txt"}}
        text = "def foo():\nclass Bar:\nimport os"
        assert _is_python_file(text, context) is True

    def test_javascript_not_python(self) -> None:
        text = "function foo() { return 1; }\nconst x = 2;"
        assert _is_python_file(text) is False


class TestExtractPythonSkeleton:
    def test_invalid_syntax(self) -> None:
        result = _extract_python_skeleton("def foo( {")
        assert result is None

    def test_empty_file(self) -> None:
        result = _extract_python_skeleton("")
        assert result is None

    def test_simple_function(self) -> None:
        text = "def foo():\n    return 1"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "def foo():" in result

    def test_function_with_args(self) -> None:
        text = "def foo(x, y):\n    return x + y"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "def foo(x, y):" in result

    def test_function_with_type_annotations(self) -> None:
        text = "def foo(x: int, y: str) -> bool:\n    return True"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "def foo(x: int, y: str) -> bool:" in result

    def test_async_function(self) -> None:
        text = "async def foo():\n    await bar()"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "async def foo():" in result

    def test_function_with_default_args(self) -> None:
        text = "CONST = 1\ndef foo(x=1, y='hello'):\n    pass"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "def foo(" in result

    def test_function_with_long_default(self) -> None:
        text = "CONST = 1\ndef foo(x='this is a very long default value that should be truncated'):\n    pass"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "def foo(" in result

    def test_function_with_varargs(self) -> None:
        text = "def foo(*args, **kwargs):\n    pass"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "*args" in result
        assert "**kwargs" in result

    def test_function_with_kwonly_args(self) -> None:
        text = "def foo(*, key, value=None):\n    pass"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "key" in result

    def test_class_definition(self) -> None:
        text = "class Foo:\n    pass"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "class Foo:" in result

    def test_class_with_base(self) -> None:
        text = "class Foo(Bar, Baz):\n    pass"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "class Foo(Bar, Baz):" in result

    def test_class_with_methods(self) -> None:
        text = """class Foo:
    def method(self):
        pass"""
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "class Foo:" in result
        assert "def method(" in result

    def test_class_with_async_method(self) -> None:
        text = """class Foo:
    async def amethod(self):
        pass"""
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "async def amethod(" in result

    def test_decorated_class_method(self) -> None:
        text = """class Foo:
    @classmethod
    def bar(cls):
        pass"""
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "@classmethod" in result
        assert "def bar():" in result or "def bar(cls):" in result

    def test_nested_class(self) -> None:
        text = """class Outer:
    class Inner:
        pass"""
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "class Outer:" in result
        assert "class Inner:" in result

    def test_module_level_import(self) -> None:
        text = "import os\nimport sys"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "import os" in result
        assert "import sys" in result

    def test_from_import_statement(self) -> None:
        text = "from collections import defaultdict"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "from collections import defaultdict" in result

    def test_annotated_assignment(self) -> None:
        text = "x: int = 1"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "x: int" in result

    def test_annotated_assignment_with_long_value(self) -> None:
        text = "x: int = 1" + "0" * 100
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "x: int =" in result

    def test_constant_assignment(self) -> None:
        text = "MAX_SIZE = 1000"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "MAX_SIZE = 1000" in result

    def test_constant_dict(self) -> None:
        text = "CONFIG = {'a': 1, 'b': 2}"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "CONFIG = {'a': 1, 'b': 2}" in result

    def test_constant_list(self) -> None:
        text = "ITEMS = [1, 2, 3]"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "ITEMS = [1, 2, 3]" in result

    def test_dataclass_style_field(self) -> None:
        text = """class Foo:
    x: int = 1
    y: str = 'hello'"""
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "x: int = 1" in result

    def test_method_return_statement(self) -> None:
        text = """def foo():
    return 42"""
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "return 42" in result

    def test_method_yield_statement(self) -> None:
        text = """def foo():
    yield 1"""
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "def foo():" in result

    def test_method_raise_statement(self) -> None:
        text = """def foo():
    raise ValueError("bad")"""
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "raise ValueError" in result

    def test_class_method_return_yield(self) -> None:
        text = """class Foo:
    def bar(self):
        return self.x"""
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "return self.x" in result

    def test_no_code_only_comments(self) -> None:
        text = "# just a comment"
        result = _extract_python_skeleton(text)
        assert result is None

    def test_pass_only_class(self) -> None:
        text = "class Empty:\n    pass"
        result = _extract_python_skeleton(text)
        assert result is not None
        assert "pass" in result


class TestCompressFileRead:
    def test_small_file_returned_verbatim(self) -> None:
        text = "x = 1\ny = 2"
        result = _compress_file_read(text)
        assert result == text

    def test_small_file_with_many_lines(self) -> None:
        text = "\n".join([f"line {i}" for i in range(50)])
        result = _compress_file_read(text)
        assert result == text

    def test_verbatim_flag_bypasses_compression(self) -> None:
        text = "x = 1\n" * 200
        context = {"args": {"verbatim": True}}
        result = _compress_file_read(text, tool_context=context)
        assert result == text

    def test_offset_arg_bypasses_compression(self) -> None:
        text = "x = 1\n" * 200
        context = {"args": {"offset": 10}}
        result = _compress_file_read(text, tool_context=context)
        assert result == text

    def test_limit_arg_bypasses_compression(self) -> None:
        text = "x = 1\n" * 200
        context = {"args": {"limit": 50}}
        result = _compress_file_read(text, tool_context=context)
        assert result == text

    def test_start_arg_bypasses_compression(self) -> None:
        text = "x = 1\n" * 200
        context = {"args": {"start": 1}}
        result = _compress_file_read(text, tool_context=context)
        assert result == text

    def test_end_arg_bypasses_compression(self) -> None:
        text = "x = 1\n" * 200
        context = {"args": {"end": 10}}
        result = _compress_file_read(text, tool_context=context)
        assert result == text

    def test_file_heat_zero_returns_text(self) -> None:
        text = "x = 1\n" * 200
        context = {"args": {"path": "test.py"}, "file_heat": {"test.py": 0.0}}
        result = _compress_file_read(text, tool_context=context)
        assert result == text

    def test_file_heat_positive_triggers_skeletonization(self) -> None:
        text = "x = 1\n" * 200
        context = {"args": {"path": "test.py"}, "file_heat": {"test.py": 1.0}}
        result = _compress_file_read(text, tool_context=context)
        assert "skeleton" in result or "ast_skeleton" in result.lower() or result != text

    def test_aggressive_compression_small_file(self) -> None:
        text = "x = 1\ny = 2"

        class MockProfile:
            compression_aggressiveness = 0.4

        context = {"_model_profile": MockProfile()}
        result = _compress_file_read(text, tool_context=context)
        assert result == text

    def test_aggressive_compression_medium_file(self) -> None:
        text = "x = 1\n" * 100

        class MockProfile:
            compression_aggressiveness = 0.6

        context = {"_model_profile": MockProfile()}
        result = _compress_file_read(text, tool_context=context)
        assert result == text

    def test_session_model_profile(self) -> None:
        text = "x = 1\n" * 200

        class MockProfile:
            compression_aggressiveness = 0.9

        class MockSession:
            model_profile = MockProfile()

        result = _compress_file_read(text, session=MockSession())
        assert "skeleton" in result or "ast_skeleton" in result.lower() or result != text

    def test_large_python_file_uses_ast_skeleton(self) -> None:
        text = "def foo():\n    return 1\n\n" + "pass\n" * 100
        context = {"args": {"path": "test.py"}}
        result = _compress_file_read(text, tool_context=context)
        assert "ast_skeleton" in result.lower() or "skeleton" in result

    def test_non_python_file_uses_heuristic_skeleton(self) -> None:
        text = "[section1]\nkey1=value1\nkey2=value2\n\n[section2]\nkey3=value3\n" + "\n".join(
            ["x = " + str(i) for i in range(200)]
        )
        context = {"args": {"path": "test.txt"}}
        result = _compress_file_read(text, tool_context=context)
        assert "skeleton" in result or "TRUNCATED" in result or result != text

    def test_session_skeleton_delivered_paths_tracked(self) -> None:
        text = "def foo():\n    return 1\n\n" + "pass\n" * 100

        class MockSession:
            _skeleton_delivered_paths: set[str] = set()
            model_profile = None

        context = {"args": {"path": "test.py"}}
        result = _compress_file_read(text, tool_context=context, session=MockSession())
        assert "test.py" in MockSession._skeleton_delivered_paths or result != text

    def test_result_longer_than_original_returns_original(self) -> None:
        lines = ["short line"] * 5
        text = "\n".join(lines)

        class MockSession:
            model_profile = None
            _skeleton_delivered_paths: set[str] = set()

        result = _compress_file_read(text, session=MockSession())
        assert result == text

    def test_result_trimmed_to_head_tail(self) -> None:
        lines = ["def func" + str(i) + "():\n    pass\n" for i in range(100)]
        text = "".join(lines)

        class MockSession:
            model_profile = None
            _skeleton_delivered_paths: set[str] = set()

        result = _compress_file_read(text, session=MockSession())
        assert len(result) < len(text)
        assert "omitted" in result.lower() or "skeleton" in result.lower()

    def test_section_map_included(self) -> None:
        text = "def foo():\n    pass\n\nclass Bar:\n    pass\n\n" + "x = 1\n" * 100

        class MockSession:
            model_profile = None
            _skeleton_delivered_paths: set[str] = set()

        context = {"args": {"path": "test.py"}}
        result = _compress_file_read(text, tool_context=context, session=MockSession())
        assert "sections:" in result or "skeleton" in result

    def test_original_chars_in_header(self) -> None:
        text = "def foo():\n    return 1\n\n" + "pass\n" * 100

        class MockSession:
            model_profile = None
            _skeleton_delivered_paths: set[str] = set()

        context = {"args": {"path": "test.py"}}
        result = _compress_file_read(text, tool_context=context, session=MockSession())
        assert "original_chars:" in result

    def test_multiple_tools_context_args_formats(self) -> None:
        text = "def foo():\n    return 1\n\n" + "pass\n" * 100

        class MockSession:
            model_profile = None
            _skeleton_delivered_paths: set[str] = set()

        for path_key in ["path", "file_path", "AbsolutePath", "TargetFile"]:
            context = {"args": {path_key: "test.py"}}
            result = _compress_file_read(text, tool_context=context, session=MockSession())
            assert "skeleton" in result or result == text

    def test_empty_file(self) -> None:
        result = _compress_file_read("")
        assert result == ""

    def test_large_file_all_constants(self) -> None:
        text = "\n".join([f"CONST{i} = {i}" for i in range(200)])
        context = {"args": {"path": "test.py"}}
        result = _compress_file_read(text, tool_context=context)
        assert "skeleton" in result or result == text


_SMALL_PYTHON = (
    "def greet(name: str) -> str:\n"
    "    return f'hello {name}'\n"
    "\n"
    "def farewell(name: str) -> str:\n"
    "    return f'goodbye {name}'\n"
)  # < 100 lines, < 10k chars — normally returned verbatim


class TestHotFileSkeletonPromotion:
    """Files with heat >= threshold must return skeleton even when below small-file size limits."""

    def test_small_python_file_below_heat_threshold_returned_verbatim(self) -> None:
        context = {"args": {"path": "greet.py"}, "file_heat": {"greet.py": 1.0}}
        result = _compress_file_read(_SMALL_PYTHON, tool_context=context)
        assert result == _SMALL_PYTHON

    def test_small_python_file_at_heat_threshold_returns_skeleton(self) -> None:
        context = {"args": {"path": "greet.py"}, "file_heat": {"greet.py": 3.0}}
        result = _compress_file_read(_SMALL_PYTHON, tool_context=context)
        assert result != _SMALL_PYTHON
        assert "is_skeleton:true" in result

    def test_small_python_file_above_heat_threshold_returns_skeleton(self) -> None:
        context = {"args": {"path": "greet.py"}, "file_heat": {"greet.py": 6.0}}
        result = _compress_file_read(_SMALL_PYTHON, tool_context=context)
        assert "is_skeleton:true" in result

    def test_skeleton_header_present_for_hot_small_file(self) -> None:
        context = {"args": {"path": "greet.py"}, "file_heat": {"greet.py": 3.0}}
        result = _compress_file_read(_SMALL_PYTHON, tool_context=context)
        assert ">>> tool:file_read" in result
        assert "ast_skeleton:true" in result

    def test_hot_file_with_offset_arg_still_bypasses_skeleton(self) -> None:
        context = {"args": {"path": "greet.py", "offset": 1}, "file_heat": {"greet.py": 5.0}}
        result = _compress_file_read(_SMALL_PYTHON, tool_context=context)
        assert result == _SMALL_PYTHON

    def test_hot_file_with_verbatim_arg_still_bypasses_skeleton(self) -> None:
        context = {"args": {"path": "greet.py", "verbatim": True}, "file_heat": {"greet.py": 5.0}}
        result = _compress_file_read(_SMALL_PYTHON, tool_context=context)
        assert result == _SMALL_PYTHON

    def test_hot_small_non_python_file_is_returned_verbatim(self) -> None:
        text = "name: tok\nversion: 1\n"
        context = {"args": {"path": "config.yaml"}, "file_heat": {"config.yaml": 8.0}}
        result = _compress_file_read(text, tool_context=context)
        assert result == text

    def test_hot_invalid_python_file_is_returned_verbatim(self) -> None:
        text = "def broken(:\n    pass\n"
        context = {"args": {"path": "broken.py"}, "file_heat": {"broken.py": 8.0}}
        result = _compress_file_read(text, tool_context=context)
        assert result == text

    def test_hot_small_python_marks_skeleton_delivered_path(self) -> None:
        class MockSession:
            _skeleton_delivered_paths: set[str] = set()

        session = MockSession()
        context = {"args": {"path": "./greet.py"}, "file_heat": {"greet.py": 3.0}}
        result = _compress_file_read(_SMALL_PYTHON, tool_context=context, session=session)

        assert "is_skeleton:true" in result
        assert "greet.py" in session._skeleton_delivered_paths


class TestSignatureContinuationRegex:
    def test_comma_at_end(self) -> None:
        assert _SIGNATURE_CONTINUATION_RE.match(",") is not None

    def test_close_paren_only(self) -> None:
        assert _SIGNATURE_CONTINUATION_RE.match(")") is not None

    def test_close_bracket_only(self) -> None:
        assert _SIGNATURE_CONTINUATION_RE.match("]") is not None

    def test_comma_with_whitespace(self) -> None:
        assert _SIGNATURE_CONTINUATION_RE.match(", ") is not None

    def test_non_continuation(self) -> None:
        assert _SIGNATURE_CONTINUATION_RE.match("def foo():") is None
        assert _SIGNATURE_CONTINUATION_RE.match("class Bar:") is None
