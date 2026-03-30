"""Tests for tok.translator — output-side Tok -> readable English."""

from tok.translator import (
    postprocess_response,
    strip_markdown_fallback,
    tok_to_readable,
)


class TestTokToReadable:
    def test_basic_msg_extraction(self):
        text = """\
>>> t:3|usr:test|agt:reply|state:done
@thought
  |> Internal reasoning here
@msg role:assistant
  |> Hello, this is the response.
  |> Second line of response."""
        result = tok_to_readable(text)
        assert "Hello, this is the response." in result
        assert "Second line of response." in result
        assert "Internal reasoning" not in result
        assert ">>>" not in result

    def test_thought_blocks_stripped(self):
        text = """\
>>> t:1|usr:x|agt:y|state:z
@thought
  |> Secret reasoning
  |> More secret reasoning
@msg role:assistant
  |> Visible output"""
        result = tok_to_readable(text)
        assert "Secret reasoning" not in result
        assert "Visible output" in result

    def test_code_fences_preserved(self):
        text = """\
@msg role:assistant
  |> Here is the code:
```python
def hello():
    print("world")
```
  |> That should work."""
        result = tok_to_readable(text)
        assert "def hello():" in result
        assert "That should work." in result

    def test_empty_response(self):
        text = ">>> t:1|usr:x|agt:y|state:z\n@thought\n  |> only thoughts"
        result = tok_to_readable(text)
        assert result == ""

    def test_non_tok_lines_in_msg(self):
        text = """\
@msg role:assistant
  |> Start
Some raw line
  |> End"""
        result = tok_to_readable(text)
        assert "Start" in result
        assert "Some raw line" in result
        assert "End" in result


class TestStripMarkdownFallback:
    def test_strips_headers(self):
        text = "## Hello\nWorld"
        result = strip_markdown_fallback(text)
        assert result == "Hello\nWorld"

    def test_strips_bold(self):
        text = "This is **bold** text"
        result = strip_markdown_fallback(text)
        assert result == "This is bold text"

    def test_preserves_code_blocks(self):
        text = "Text\n```python\ndef foo():\n    pass\n```\nMore text"
        result = strip_markdown_fallback(text)
        assert "```python" in result
        assert "def foo():" in result

    def test_preserves_inline_code(self):
        text = "Use `print()` to output"
        result = strip_markdown_fallback(text)
        assert "`print()`" in result

    def test_strips_horizontal_rules(self):
        text = "Above\n---\nBelow"
        result = strip_markdown_fallback(text)
        assert "---" not in result


class TestPostprocessResponse:
    def test_tok_response(self):
        text = ">>> t:1|usr:x|agt:y|state:z\n@msg role:assistant\n  |> Hello"
        result, mode = postprocess_response(text)
        assert mode == "tok-native"
        assert "Hello" in result

    def test_markdown_response(self):
        text = "## Hello\nThis is a normal response."
        result, mode = postprocess_response(text)
        assert mode == "markdown"
        assert "Hello" in result

    def test_tok_empty_falls_back(self):
        text = ">>> t:1|usr:x|agt:y|state:z\n@thought\n  |> only thoughts"
        result, mode = postprocess_response(text)
        assert mode == "tok-empty"
