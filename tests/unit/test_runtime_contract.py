from tok.runtime.memory.bridge_memory import BridgeMemoryState
from tok.runtime.pipeline.response_processing import response_behavior_signals
from tok.universal_runtime import (
    RuntimeSession,
    normalize_tool_events,
    response_contract_for_mode,
)


def test_runtime_contract_prefers_structured_memory_on_cold_start(
    tmp_path,
) -> None:
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "bridge_memory.tok").write_text("@mem v:b1 t:3\n@h\n@f goal\n  |> ship_runtime|score:3|last:3\n")
    (memory_dir / "memory.tok").write_text(">>> turns:9|goal:legacy_fallback|files:old.py\n")

    session = RuntimeSession(memory_dir=memory_dir)
    memory = session.load_memory(model="claude-sonnet-4")

    assert "ship_runtime" in memory
    signals = session.consume_behavior_signals()
    assert signals.get("cold_start_structured_memory", 0) == 1
    assert signals.get("cold_start_wire_fallback", 0) == 0


def test_runtime_contract_uses_wire_fallback_when_structured_missing(
    tmp_path,
) -> None:
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


def test_response_contract_classifies_tok_native_vs_fail_open() -> None:
    tok_text = ">>> turns:2|goal:fix\n@msg role:assistant\n  |> done"
    markdown_text = "## heading\nPlain response"

    native = response_contract_for_mode(tok_text, tool_compatible=False)
    degraded = response_contract_for_mode(markdown_text, tool_compatible=False)

    assert native.mode == "tok-native"
    assert native.behavior_signals.get("tok_native_response", 0) == 1
    assert degraded.mode == "markdown"
    assert degraded.behavior_signals.get("non_tok_response", 0) == 1
    assert degraded.behavior_signals.get("fail_open_compat_response", 0) == 1


def test_response_contract_plain_text_does_not_reuse_previous_tok_mode(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._last_mode = "tok-native"

    contract = response_contract_for_mode("Plain response", tool_compatible=False, session=session)

    assert contract.mode == "markdown"
    assert session._last_mode == "markdown"
    assert contract.behavior_signals.get("non_tok_response", 0) == 1
    assert contract.behavior_signals.get("tok_native_response", 0) == 0


def test_response_contract_handles_tool_compatible_plain_text() -> None:
    text = "Plain response"

    contract = response_contract_for_mode(text, tool_compatible=True)

    assert contract.mode == "tool-compatible"
    assert contract.behavior_signals.get("tool_compatible_response", 0) == 1
    assert "non_tok_response" not in contract.behavior_signals


def test_response_contract_marks_malformed_hybrid_tool_blocks() -> None:
    text = '>>> turns:3|goal:tests\n@Tool(json={"command": "pytest"})\n@msg role:assistant\n  |> done'

    contract = response_contract_for_mode(text, tool_compatible=False)

    assert contract.mode in {"tok", "tok-empty"}
    assert contract.behavior_signals.get("malformed_tok_response", 0) == 1
    assert contract.behavior_signals.get("malformed_tok_hybrid_tool", 0) == 1
    assert contract.behavior_signals.get("fail_open_compat_response", 0) == 1


def test_normalize_tool_events_captures_file_command_search_classes() -> None:
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


def test_response_contract_repairs_structured_answer_from_session_anchors(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/parser.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:parse_error",
        score_delta=3,
    )
    session._last_user_prompt_text = (
        "Based on the conversation so far, respond in exactly two lines:\n"
        "File=<the primary file that answered the question>\n"
        "Verification=<the function, class, or finding that supports the answer>"
    )
    session._last_user_prompt_labels = ("file", "verification")

    contract = response_contract_for_mode(
        "Verification=Parser.parse() method is empty (pass) and pytest fails",
        tool_compatible=False,
        session=session,
    )

    visible = "\n".join(block.get("text", "") for block in contract.content_blocks if block.get("type") == "text")
    assert "File=src/tok/parser.py" in visible
    assert "Verification=parse_error" in visible
    assert contract.behavior_signals.get("structured_answer_repaired", 0) == 1
    assert contract.behavior_signals.get("structured_answer_backfilled", 0) == 1


def test_response_behavior_signals_exempts_expected_strict_structured_answer(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._last_user_prompt_labels = ("file", "verification")
    session._last_user_prompt_text = "File=<...> Verification=<...>"

    signals = response_behavior_signals(
        "File=src/tok/parser.py\nVerification=parse_error",
        tool_compatible=False,
        session=session,
    )

    assert signals == {}


def test_response_contract_repairs_research_verification_with_anchor_inference(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/runtime/policy/smart_policy.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:BridgeMemoryState",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:compress_history",
        score_delta=3,
    )
    session._last_user_prompt_labels = ("file", "verification")
    session._last_user_prompt_text = "File=<...>\nVerification=<...>"

    contract = response_contract_for_mode(
        "File=src/tok/runtime/policy/smart_policy.py\n"
        "Verification=class SmartPolicy, BridgeMemoryState, and the history compression methods within",
        tool_compatible=False,
        session=session,
    )

    visible = "\n".join(block.get("text", "") for block in contract.content_blocks if block.get("type") == "text")
    assert "File=src/tok/runtime/policy/smart_policy.py" in visible
    assert "Verification=BridgeMemoryState, compress_history" in visible
    assert contract.behavior_signals.get("structured_answer_repaired", 0) == 1


def test_response_contract_repair_ignores_low_quality_verification_anchor(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/runtime/policy/smart_policy.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:and",
        score_delta=5,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:BridgeMemoryState",
        score_delta=3,
    )
    session._last_user_prompt_labels = ("file", "verification")
    session._last_user_prompt_text = "File=<...>\nVerification=<...>"

    contract = response_contract_for_mode(
        "File=src/tok/runtime/policy/smart_policy.py\n"
        "Verification=class SmartPolicy, BridgeMemoryState, and the history compression methods within",
        tool_compatible=False,
        session=session,
    )

    visible = "\n".join(block.get("text", "") for block in contract.content_blocks if block.get("type") == "text")
    assert "Verification=and" not in visible
    assert "BridgeMemoryState" in visible


def test_response_contract_canonicalizes_module_file_to_package_init(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/compression/__init__.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:compress_history",
        score_delta=3,
    )
    session._last_user_prompt_labels = ("file", "verification")
    session._last_user_prompt_text = "File=<...>\nVerification=<...>"

    contract = response_contract_for_mode(
        "File=src/tok/compression.py\nVerification=compress_history",
        tool_compatible=False,
        session=session,
    )

    visible = "\n".join(block.get("text", "") for block in contract.content_blocks if block.get("type") == "text")
    assert "File=src/tok/compression.py (src/tok/compression/__init__.py)" in visible


def test_response_contract_prefers_existing_package_file_over_stale_module_anchor(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/compression.py",
        score_delta=5,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/compression/__init__.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:compress_history",
        score_delta=3,
    )
    session._last_user_prompt_labels = ("file", "verification")
    session._last_user_prompt_text = "File=<...>\nVerification=<...>"

    contract = response_contract_for_mode(
        "File=src/tok/compression.py\nVerification=compress_history",
        tool_compatible=False,
        session=session,
    )

    visible = "\n".join(block.get("text", "") for block in contract.content_blocks if block.get("type") == "text")
    assert "File=src/tok/compression.py (src/tok/compression/__init__.py)" in visible


def test_response_contract_quarantines_tool_intent_during_answer_phase(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/gateway.py",
        score_delta=4,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:_response_contract_for_mode",
        score_delta=4,
    )
    session._answer_phase_expected_this_turn = True
    session._request_has_tools = False

    contract = response_contract_for_mode(
        "@tool view_file id:call_1 path:src/tok/gateway.py",
        tool_compatible=True,
        session=session,
    )

    assert contract.behavior_signals.get("answer_phase_fallback_failed_no_anchor", 0) == 1


def test_response_contract_applies_non_labeled_answer_phase_fallback(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/runtime/pipeline/response_processing.py",
        score_delta=4,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:_repair_structured_answer_text",
        score_delta=4,
    )
    session._answer_phase_expected_this_turn = True
    session._request_has_tools = False
    session._last_user_prompt_text = "Summarize the final answer now."
    session._last_user_prompt_labels = ()

    contract = response_contract_for_mode(
        "I will inspect one more area before answering.",
        tool_compatible=True,
        session=session,
    )

    assert contract.behavior_signals.get("answer_phase_fallback_failed_no_anchor", 0) == 1


def test_response_contract_answer_phase_fallback_fails_without_anchors(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._answer_phase_expected_this_turn = True
    session._request_has_tools = False

    contract = response_contract_for_mode(
        "@tool view_file id:call_1 path:src/tok/gateway.py",
        tool_compatible=True,
        session=session,
    )

    assert contract.behavior_signals.get("answer_phase_fallback_failed_no_anchor", 0) == 1
    assert any(block.get("type") == "tool_use" for block in contract.content_blocks)


def test_response_contract_does_not_quarantine_when_tools_are_expected(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/gateway.py",
        score_delta=4,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:_response_contract_for_mode",
        score_delta=4,
    )
    session._answer_phase_expected_this_turn = True
    session._request_has_tools = True

    contract = response_contract_for_mode(
        "@tool view_file id:call_1 path:src/tok/gateway.py",
        tool_compatible=True,
        session=session,
    )

    assert contract.behavior_signals.get("answer_phase_tool_intent_quarantined", 0) == 0
    assert any(block.get("type") == "tool_use" for block in contract.content_blocks)
