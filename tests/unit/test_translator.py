"""Tests for tok.translator — output-side Tok -> readable English."""

from tok.runtime.policy.translator import (
    _is_likely_tok,
    postprocess_response,
    strip_markdown_fallback,
    tok_to_readable,
)


class TestTokToReadable:
    def test_basic_msg_extraction(self) -> None:
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

    def test_thought_blocks_stripped(self) -> None:
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

    def test_code_fences_preserved(self) -> None:
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

    def test_empty_response(self) -> None:
        text = ">>> t:1|usr:x|agt:y|state:z\n@thought\n  |> only thoughts"
        result = tok_to_readable(text)
        assert result == ""

    def test_non_tok_lines_in_msg(self) -> None:
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
    def test_strips_headers(self) -> None:
        text = "## Hello\nWorld"
        result = strip_markdown_fallback(text)
        assert result == "Hello\nWorld"

    def test_strips_bold(self) -> None:
        text = "This is **bold** text"
        result = strip_markdown_fallback(text)
        assert result == "This is bold text"

    def test_preserves_code_blocks(self) -> None:
        text = "Text\n```python\ndef foo():\n    pass\n```\nMore text"
        result = strip_markdown_fallback(text)
        assert "```python" in result
        assert "def foo():" in result

    def test_preserves_inline_code(self) -> None:
        text = "Use `print()` to output"
        result = strip_markdown_fallback(text)
        assert "`print()`" in result

    def test_strips_horizontal_rules(self) -> None:
        text = "Above\n---\nBelow"
        result = strip_markdown_fallback(text)
        assert "---" not in result


class TestPostprocessResponse:
    def test_tok_response(self) -> None:
        text = ">>> t:1|usr:x|agt:y|state:z\n@msg role:assistant\n  |> Hello"
        result, mode = postprocess_response(text)
        assert mode == "tok-native"
        assert "Hello" in result

    def test_markdown_response(self) -> None:
        text = "## Hello\nThis is a normal response."
        result, mode = postprocess_response(text)
        assert mode == "markdown"
        assert "Hello" in result

    def test_tok_empty_falls_back(self) -> None:
        text = ">>> t:1|usr:x|agt:y|state:z\n@thought\n  |> only thoughts"
        _result, mode = postprocess_response(text)
        assert mode == "tok-empty"


class TestIsLikelyTokRegression:
    def test_at_thought_without_header(self) -> None:
        assert _is_likely_tok("@thought\nTesting")

    def test_at_thought_with_pipe_content(self) -> None:
        assert _is_likely_tok("@thought\n  |> hidden reasoning")

    def test_bare_triple_chevron_header(self) -> None:
        assert _is_likely_tok(">>> t:1|usr:test|agt:reply|state:active")

    def test_triple_chevron_with_markdown_body(self) -> None:
        assert _is_likely_tok(">>> t:1|usr:test|agt:reply|state:active\n## Result\nPlain markdown")

    def test_triple_chevron_with_known_at_block(self) -> None:
        assert _is_likely_tok(">>> t:1|s:d|agt:r|state:active\n@msg role:assistant\n  |> done")

    def test_plain_text_rejected(self) -> None:
        assert not _is_likely_tok("Just some plain text")

    def test_plain_markdown_rejected(self) -> None:
        assert not _is_likely_tok("## Hello\nThis is a normal response.")

    def test_elixir_pipe_in_code_block_rejected(self) -> None:
        text = "```elixir\n[1, 2, 3]\n|> Enum.map(fn x -> x * 2 end)\n|> Enum.filter(fn x -> x > 2 end)\n```"
        assert not _is_likely_tok(text)

    def test_fsharp_pipe_in_code_block_rejected(self) -> None:
        text = "```fsharp\n[1; 2; 3]\n|> List.map (fun x -> x * 2)\n```"
        assert not _is_likely_tok(text)

    def test_at_thought_postprocess_mode_is_tok_empty(self) -> None:
        _result, mode = postprocess_response("@thought\nTesting")
        assert mode == "tok-empty"

    def test_triple_chevron_with_markdown_postprocess_mode(self) -> None:
        _result, mode = postprocess_response(">>> t:1|usr:test|agt:reply|state:active\n## Result\nPlain markdown")
        assert mode == "tok-native"
