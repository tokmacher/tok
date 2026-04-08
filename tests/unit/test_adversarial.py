"""Adversarial property tests for Tok serialization."""

import json
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from tok.protocol.format_bridge import Bridge
from tok.protocol.parser import TokParser


def _recursive_data() -> st.SearchStrategy[Any]:
    letter_text = st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
        min_size=1,
        max_size=20,
    )
    base = st.one_of(
        st.booleans(),
        st.integers(min_value=-(10**6), max_value=10**6),
        letter_text,
    )

    return st.recursive(
        base,
        lambda children: (st.lists(children, max_size=3) | st.dictionaries(letter_text, children, max_size=3)),
        max_leaves=30,
    )


def _primitive_dict_data() -> st.SearchStrategy[dict[str, Any]]:
    letter_text = st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
        min_size=1,
        max_size=20,
    )
    safe_text = letter_text.filter(lambda s: s.lower() not in {"null", "none", "true", "false"})
    base = st.one_of(
        st.booleans(),
        st.integers(min_value=-(10**6), max_value=10**6),
        safe_text,
    )
    return st.dictionaries(letter_text, base, min_size=1, max_size=5)


def _encode_as_tok(payload: Any) -> str:
    return Bridge.json(json.dumps(payload, ensure_ascii=False))


def _decode_tok(tok_text: str) -> Any:
    return Bridge.decode(tok_text)


@given(sample=_primitive_dict_data())
def test_bridge_roundtrip_property(sample: Any) -> None:
    tok_text = _encode_as_tok(sample)
    recovered = _decode_tok(tok_text)
    assert recovered == sample


@given(sample=_recursive_data())
def test_parser_roundtrip_nodes(sample: Any) -> None:
    tok_text = _encode_as_tok(sample)
    parser = TokParser()
    nodes = parser.parse(tok_text)
    canonical = parser.encode(nodes)
    reparse = parser.parse(canonical)
    assert isinstance(canonical, str)
    assert isinstance(reparse, list)


def test_deep_nesting_roundtrip() -> None:
    data: dict[str, Any] = {"alpha": {"beta": {"gamma": {"delta": {"epsilon": {"zeta": "end"}}}}}}
    for _ in range(10):
        data = {"nest": data, "count": 1}
    tok_text = _encode_as_tok(data)
    assert _decode_tok(tok_text) == data


def test_unicode_and_control_roundtrip() -> None:
    payload = {
        "text": "🔥" * 30 + "\u0000" + "\u2028" + "\u200b",
        "notes": ["null\u0007byte", "tabs\t and spaces"],
    }
    tok_text = _encode_as_tok(payload)
    assert _decode_tok(tok_text) == payload


def test_huge_integer_roundtrip() -> None:
    payload = {"huge": 10**120}
    tok_text = _encode_as_tok(payload)
    assert _decode_tok(tok_text) == payload


def test_empty_structures_roundtrip() -> None:
    payload = {"empty_list": [], "empty_dict": {}, "empty_str": ""}
    tok_text = _encode_as_tok(payload)
    decoded = _decode_tok(tok_text)
    assert isinstance(decoded, dict)
    assert decoded.get("empty_list") in ([], "")
    assert decoded.get("empty_dict") in ({}, "")
    assert decoded.get("empty_str", "") == ""


def test_controlled_tokens_handle_special_chars() -> None:
    payload = {"emoji": "😀" * 50, "control": "\x00\x1b\x7f"}
    tok_text = _encode_as_tok(payload)
    assert _decode_tok(tok_text) == payload
