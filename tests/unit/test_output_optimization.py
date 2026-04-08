from tok.protocol.parser import TokParser
from tok.runtime.policy.translator import tok_to_readable
from tok.universal_runtime import (
    RuntimeSession,
    parse_tok_response,
    translate_response_tools,
)


def test_parser_inlined_content() -> None:
    parser = TokParser()
    # Test @msg |> content
    nodes = parser.parse("@msg |> Hello world")
    assert len(nodes) == 1
    assert nodes[0].type == "msg"
    assert nodes[0].text.strip() == "Hello world"

    # Test @Tool name=val
    nodes = parser.parse("@Tool name=write_to_file TargetFile=foo.txt")
    assert len(nodes) == 1
    assert nodes[0].type == "Tool"
    # It might be in label OR in attrs if used with name=
    assert (nodes[0].label == "write_to_file") or (nodes[0].attrs.get("name") == "write_to_file")
    assert nodes[0].attrs["TargetFile"] == "foo.txt"


def test_lazy_tok_injection() -> None:
    # Response starting with @Tool without >>> or @msg
    text = "@Tool name=view_file path=src/main.py"
    blocks = translate_response_tools(text)
    # Since we auto-inject @msg role:assistant, we expect two blocks:
    # one for @msg and one for @Tool (or they might be nested depending on parser)
    # Actually, translator_response_tools returns a list of content blocks (dicts)
    assert any(b["type"] == "tool_use" and b["name"] == "view_file" for b in blocks)

    # Response starting with |>
    text = "|> Hello from lazy mode"
    blocks = translate_response_tools(text)
    assert any(b["type"] == "text" and "Hello from lazy mode" in b["text"] for b in blocks)


def test_mode_inference() -> None:
    session = RuntimeSession()
    session._last_mode = "tok-native"

    # Response without >>> header
    text = "@msg role:assistant\n  |> Content"
    tok_blocks, _signals, mode = parse_tok_response(text, session=session)

    assert mode == "tok-native"
    assert any(b["type"] == "text" and "Content" in b["text"] for b in tok_blocks)


def test_translator_lazy_tok() -> None:
    text = "|> This is a lazy response"
    readable = tok_to_readable(text)
    assert readable == "This is a lazy response"

    text = "@Tool name=foo |> arg\n|> Message"
    readable = tok_to_readable(text)
    assert readable == "Message"
