from tok.universal_runtime import (
    normalize_tool_events,
    response_contract_for_mode,
)


def test_normalize_tool_events_classification() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "view_file",
                    "input": {"path": "foo.py"},
                },
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "read_url_content",
                    "input": {"Url": "http://ex.com"},
                },
                {
                    "type": "tool_use",
                    "id": "t3",
                    "name": "bash",
                    "input": {"command": "ls"},
                },
                {
                    "type": "tool_use",
                    "id": "t4",
                    "name": "grep_search",
                    "input": {"query": "pattern"},
                },
                {
                    "type": "tool_use",
                    "id": "t5",
                    "name": "custom",
                    "input": {"arg": 1},
                },
            ],
        }
    ]

    events = normalize_tool_events(messages)

    assert len(events) == 5
    # view_file IS in FILE_LIKE_TOOLS
    assert events[0].compressibility_class == "file_read"
    assert events[0].path == "foo.py"
    assert events[0].fidelity_requirement == "high"

    # read_url_content is NOT in FILE_LIKE_TOOLS (case sensitive or missing)
    # Actually FILE_LIKE_TOOLS is {'view', 'view_file', 'read', 'read_file',
    # 'cat', 'open_file', 'get_file'}
    assert events[1].compressibility_class == "tool_result"  # Default fallback

    assert events[2].compressibility_class == "command"
    assert events[2].command == "ls"
    assert events[2].fidelity_requirement == "high"

    assert events[3].compressibility_class == "search"
    assert events[3].query == "pattern"
    assert events[3].fidelity_requirement == "default"

    assert events[4].compressibility_class == "tool_result"


def test_normalize_tool_events_path_variations() -> None:
    cases = [
        ({"path": "p1"}, "p1"),
        ({"file_path": "p2"}, "p2"),
        ({"AbsolutePath": "p3"}, "p3"),
        ({"TargetFile": "p4"}, "p4"),
    ]

    for input_dict, expected_path in cases:
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "x",
                        "name": "read",
                        "input": input_dict,
                    }
                ],
            }
        ]
        events = normalize_tool_events(messages)
        assert events[0].path == expected_path


def test_response_contract_malformed_tok_cases() -> None:
    # Incomplete @ block
    text = "@thought\nTesting"
    contract = response_contract_for_mode(text, tool_compatible=False)
    # Since IS_TOK matches but has_visible_content_block is False (thought only)
    # it returns tok-empty (postprocess_response:103)
    assert contract.mode == "tok-empty"
    assert contract.behavior_signals["malformed_tok_response"] == 1

    # Text outside @ blocks in strict mode
    # TokParser treats "Visible text" as content of the last node or a new node if it wraps.
    # In this case "@thought\n|> hidden\nVisible text"
    # TokParser will wrap "Visible text" in a text block.
    text = "@thought\n  |> hidden\n@msg role:assistant\n  |> Visible text"
    contract = response_contract_for_mode(text, tool_compatible=False)
    assert contract.mode == "tok-native"
    assert contract.content_blocks[0]["text"].strip() == "Visible text"


def test_response_contract_rejects_hybrid_tool_json_text_pattern() -> None:
    text = (
        '>>> t:1|usr:test|agt:reply|state:active\n@Tool(json={"command": "pytest -q"})\n@msg role:assistant\n  |> done'
    )
    contract = response_contract_for_mode(text, tool_compatible=False)

    assert contract.mode == "tok"
    assert contract.behavior_signals["malformed_tok_response"] == 1
    assert contract.behavior_signals["fail_open_compat_response"] == 1
    assert contract.behavior_signals["malformed_tok_hybrid_tool"] == 1


def test_response_contract_rejects_non_inverted_assistant_message() -> None:
    text = ">>> t:1|usr:test|agt:reply|state:active\n@msg role:assistant\nPlain text without inversion"
    contract = response_contract_for_mode(text, tool_compatible=False)

    assert contract.mode == "tok"
    assert contract.behavior_signals["malformed_tok_response"] == 1
    assert contract.behavior_signals["malformed_tok_non_inverted_msg"] == 1
    assert contract.behavior_signals["fail_open_compat_response"] == 1


def test_response_contract_rejects_markdown_after_tok_header() -> None:
    text = ">>> t:1|usr:test|agt:reply|state:active\n## Result\nPlain markdown"
    contract = response_contract_for_mode(text, tool_compatible=False)

    assert contract.mode == "tok-empty"
    assert contract.behavior_signals["malformed_tok_response"] == 1
    assert contract.behavior_signals["malformed_tok_markdown_fallback"] == 1
    assert contract.behavior_signals["fail_open_compat_response"] == 1


def test_response_contract_rejects_bad_tok_header_shape() -> None:
    text = ">>> turns|goal:fix\n@msg role:assistant\n  |> ok"
    contract = response_contract_for_mode(text, tool_compatible=False)

    assert contract.mode == "tok"
    assert contract.behavior_signals["malformed_tok_response"] == 1
    assert contract.behavior_signals["malformed_tok_bad_header"] == 1
    assert contract.behavior_signals["fail_open_compat_response"] == 1


def test_response_contract_tool_compatible_preserves_malformed_signals() -> None:
    """Malformed Tok signals must not be silently dropped in tool-compatible mode."""
    text = (
        '>>> t:1|usr:test|agt:reply|state:active\n@Tool(json={"command": "pytest -q"})\n@msg role:assistant\n  |> done'
    )
    contract = response_contract_for_mode(text, tool_compatible=True)

    assert contract.mode == "tool-compatible"
    assert contract.behavior_signals.get("malformed_tok_response") == 1
    assert contract.behavior_signals.get("malformed_tok_hybrid_tool") == 1
    assert contract.behavior_signals.get("fail_open_compat_response") == 1


def test_translate_response_tools_complex_nesting() -> None:
    from tok.universal_runtime import translate_response_tools

    # Correct Tok tool syntax: @Tool label {attrs}
    text = """@Tool edit
  path: foo.py
  @Search
    |> old
  @Replace
    |> new
"""
    blocks = translate_response_tools(text)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["name"] == "edit"
    assert blocks[0]["input"]["path"] == "foo.py"
    # Search/Replace are children nodes, check how they are handled.
    # Actually universal_runtime.py:406 extracts children as attrs if they are
    # search/replace.
    assert blocks[0]["input"]["search"].strip() == "old"
    assert blocks[0]["input"]["replace"].strip() == "new"
