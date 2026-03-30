from tok.bridge_memory import BridgeMemoryState
from tok.universal_runtime import (
    RuntimeSession,
    normalize_tool_events,
    response_contract_for_mode,
)


def test_runtime_contract_prefers_structured_memory_on_cold_start(tmp_path):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "bridge_memory.tok").write_text(
        "@mem v:b1 t:3\n@h\n@f goal\n  |> ship_runtime|score:3|last:3\n"
    )
    (memory_dir / "memory.tok").write_text(
        ">>> turns:9|goal:legacy_fallback|files:old.py\n"
    )

    session = RuntimeSession(memory_dir=memory_dir)
    memory = session.load_memory(model="claude-sonnet-4")

    assert "ship_runtime" in memory
    signals = session.consume_behavior_signals()
    assert signals.get("cold_start_structured_memory", 0) == 1
    assert signals.get("cold_start_wire_fallback", 0) == 0


def test_runtime_contract_uses_wire_fallback_when_structured_missing(tmp_path):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    fallback_line = ">>> turns:4|goal:wire_only|files:main.py"
    (memory_dir / "memory.tok").write_text(fallback_line + "\n")

    session = RuntimeSession(memory_dir=memory_dir)
    # Simulate empty structured memory so the fallback path is exercised.
    session.bridge_memory = BridgeMemoryState()
    memory = session.load_memory(model="claude-sonnet-4")

    assert fallback_line in memory
    signals = session.consume_behavior_signals()
    assert signals.get("cold_start_wire_fallback", 0) == 1
    assert signals.get("cold_start_structured_memory", 0) == 0


def test_response_contract_classifies_tok_native_vs_fail_open():
    tok_text = ">>> turns:2|goal:fix\n@msg role:assistant\n  |> done"
    markdown_text = "## heading\nPlain response"

    native = response_contract_for_mode(tok_text, tool_compatible=False)
    degraded = response_contract_for_mode(markdown_text, tool_compatible=False)

    assert native.mode == "tok-native"
    assert native.behavior_signals.get("tok_native_response", 0) == 1
    assert degraded.mode == "markdown"
    assert degraded.behavior_signals.get("non_tok_response", 0) == 1
    assert degraded.behavior_signals.get("fail_open_compat_response", 0) == 1


def test_response_contract_handles_tool_compatible_plain_text():
    text = "Plain response"

    contract = response_contract_for_mode(text, tool_compatible=True)

    assert contract.mode == "tool-compatible"
    assert contract.behavior_signals.get("tool_compatible_response", 0) == 1
    assert "non_tok_response" not in contract.behavior_signals


def test_response_contract_marks_malformed_hybrid_tool_blocks():
    text = (
        ">>> turns:3|goal:tests\n"
        '@Tool(json={"command": "pytest"})\n'
        "@msg role:assistant\n"
        "  |> done"
    )

    contract = response_contract_for_mode(text, tool_compatible=False)

    assert contract.mode in {"tok", "tok-empty"}
    assert contract.behavior_signals.get("malformed_tok_response", 0) == 1
    assert contract.behavior_signals.get("malformed_tok_hybrid_tool", 0) == 1
    assert contract.behavior_signals.get("fail_open_compat_response", 0) == 1


def test_normalize_tool_events_captures_file_command_search_classes():
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "view_file",
                    "input": {"path": "src/main.py"},
                },
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "bash",
                    "input": {"command": "pytest -q"},
                },
                {
                    "type": "tool_use",
                    "id": "t3",
                    "name": "grep_search",
                    "input": {"query": "pattern"},
                },
            ],
        }
    ]

    events = normalize_tool_events(messages)

    assert events[0].compressibility_class == "file_read"
    assert events[0].path == "src/main.py"
    assert events[1].compressibility_class == "command"
    assert events[1].command == "pytest -q"
    assert events[2].compressibility_class == "search"
    assert events[2].query == "pattern"
