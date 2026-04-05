from typing import Any

from tok.runtime.types import (
    RuntimeRequest,
)
from tok.runtime.pipeline.request_validation import (
    validate_anthropic_request_body,
)
from tok.runtime.pipeline.request_preparation import apply_schema_adaptations
from tok.runtime.pipeline.response_processing import (
    response_behavior_signals,
    response_contract,
    response_contract_for_mode,
)
from tok.runtime.core import (
    RuntimeSession,
    TOOL_COMPAT_MEMORY_PROFILE,
    UniversalTokRuntime,
    compact_structured_answer_memory,
    extract_structured_answer_memory,
    ground_structured_answer_memory,
    reinforce_structured_answer_memory,
    calculate_invisible_pressure,
    calculate_semantic_regression_score,
    collect_transient_error_snippets,
    evaluate_replay_gate,
)
from tok.runtime.memory.tok_state import (
    _apply_tool_compatible_sticky_fields,
    _canonicalize_tool_compatible_state_fields,
    _delta_tok_state_fields,
    _parse_tok_state_fields,
)
from tok.runtime.pipeline.tool_processing import (
    build_tool_use_id_to_context,
    collect_behavior_signals,
)
from tok.runtime.config import TOK_READ_PLAN_HINT


def test_runtime_prepare_request_injects_memory_and_collects_signals():
    runtime = UniversalTokRuntime()
    session = RuntimeSession(
        fallback_memory=">>> g:fix_gateway|t:3",
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "a1",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "a2",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    },
                ],
            }
        ],
        system="Existing system prompt",
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.body["system"].startswith("Existing system prompt")
    assert "=== MODE: TOK-NATIVE ===" in prepared.body["system"]
    assert ">>>" in prepared.body["system"]
    assert prepared.behavior_signals["repeat_file_read"] == 1
    assert (
        prepared.normalized_tool_events[0].compressibility_class == "file_read"
    )


def test_runtime_prepare_request_injects_read_plan_hint_for_file_burst():
    runtime = UniversalTokRuntime()
    session = RuntimeSession()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Inspect the bridge request path."}
    ]
    for idx in range(6):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"file_{idx}",
                        "name": "read_file",
                        "input": {"path": f"src/file_{idx}.py"},
                    }
                ],
            }
        )

    request = RuntimeRequest(
        model="claude-sonnet-4-6",
        adapter_kind="claude-bridge",
        tool_compatible=True,
        messages=messages,
    )

    prepared = runtime.prepare_request(request, session)

    assert TOK_READ_PLAN_HINT in prepared.body["system"]
    assert prepared.behavior_signals.get("read_plan_hint_injected", 0) == 1


def test_runtime_prepare_request_translates_result_blocks():
    runtime = UniversalTokRuntime()
    session = RuntimeSession()
    request = RuntimeRequest(
        model="claude-sonnet-4",
        messages=[
            {
                "role": "user",
                "content": "@Result id:tool_1\n  |> done",
            }
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.body["messages"][0]["content"][0]["type"] == "tool_result"
    assert (
        prepared.body["messages"][0]["content"][0]["tool_use_id"] == "tool_1"
    )


def test_runtime_prepare_request_preserves_bridge_tail_when_adaptive_history_hits_zero():
    runtime = UniversalTokRuntime()
    session = RuntimeSession()
    session._step_count = 11
    request = RuntimeRequest(
        model="claude-sonnet-4-6",
        adapter_kind="claude-bridge",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "Inspect the flaky runtime failure."},
            {
                "role": "assistant",
                "content": "I will inspect the failing path.",
            },
            {
                "role": "user",
                "content": "Now verify the bridge request still has a tail.",
            },
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.body["messages"]
    assert "empty_messages" not in validate_anthropic_request_body(
        prepared.body
    )


def test_validate_anthropic_request_body_rejects_invalid_message_shape():
    failures = validate_anthropic_request_body(
        {
            "model": "claude-sonnet-4",
            "messages": [{"role": "assistant", "content": 7}],
        }
    )

    assert "invalid_message_content" in failures


def test_apply_schema_adaptations_preserves_non_empty_content_guarantee():
    adapted = apply_schema_adaptations(
        [
            {"role": "user", "content": None},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": []},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "   "}],
            },
        ]
    )

    assert [msg["content"] for msg in adapted] == [" ", " ", " ", " "]


def test_validate_anthropic_request_body_rejects_empty_messages():
    failures = validate_anthropic_request_body(
        {
            "model": "claude-sonnet-4",
            "messages": [],
        }
    )

    assert "empty_messages" in failures


def test_runtime_prepare_request_skips_history_for_tool_heavy_payload(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t0",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_0.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_1.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_2.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t3",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_3.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t4",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_4.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t5",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_5.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t6",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_6.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t7",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_7.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t8",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_8.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t9",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_9.py"},
                    },
                ],
            }
        ],
    )

    prepared = runtime.prepare_request(request, session)

    # Tool-heavy bridge turns should stay compressible in tool-compatible mode.
    assert (
        prepared.behavior_signals.get("tok_history_compression_skipped", 0)
        == 0
    )
    assert (
        prepared.behavior_signals.get("tok_soft_tool_use_count_high", 0) == 1
    )
    assert (
        "Plain text. Tool calls only. Omit all headers."
        in prepared.body["system"]
    )


def test_runtime_prepare_request_preserves_prompt_cached_system_list(
    monkeypatch,
):
    import tok.compression as compression

    runtime = UniversalTokRuntime()
    session = RuntimeSession()
    request = RuntimeRequest(
        model="claude-sonnet-4-6",
        adapter_kind="claude-bridge",
        tool_compatible=True,
        system=[
            {
                "type": "text",
                "text": "Repair the bridge.\n" + ("noise\n" * 600),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {"role": "user", "content": "Inspect the bridge request path."}
        ],
    )

    monkeypatch.setattr(
        compression,
        "compress_user_prompt",
        lambda prompt: "g:repair_bridge|constraints:no_revert",
    )

    prepared = runtime.prepare_request(request, session)

    assert isinstance(prepared.body["system"], list)
    assert prepared.body["system"][0]["cache_control"]["type"] == "ephemeral"
    assert "### Optimized Task Context" in prepared.body["system"][0]["text"]
    assert any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and "Plain text. Tool calls only. Omit all headers."
        in block.get("text", "")
        for block in prepared.body["system"][1:]
    )
    assert prepared.behavior_signals.get("tok_prompt_optimized", 0) == 1


def test_runtime_prepare_request_still_skips_extreme_tool_volume_in_tool_compatible_mode(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    huge_result = "trace line\n" * 600
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Audit the oversized outputs."}
    ]
    for idx in range(6):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"hv{idx}",
                        "name": "bash",
                        "input": {"cmd": f"cmd {idx}"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool_result",
                "tool_use_id": f"hv{idx}",
                "content": huge_result,
            }
        )

    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=messages,
    )

    prepared = runtime.prepare_request(request, session)

    assert (
        prepared.behavior_signals.get("tok_history_compression_skipped", 0)
        == 1
    )
    assert prepared.behavior_signals.get("tok_skip_tool_volume_heavy", 0) == 1


def test_runtime_tool_heavy_bridge_body_with_thinking_blocks_passes_validation(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Inspect these files."}
    ]
    for idx in range(4):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": f"planning step {idx}"},
                    {
                        "type": "tool_use",
                        "id": f"tu{idx}",
                        "name": "view_file",
                        "input": {"path": f"src/tok/file_{idx}.py"},
                    },
                ],
            }
        )
        messages.append(
            {
                "role": "tool_result",
                "tool_use_id": f"tu{idx}",
                "content": f"file {idx} content",
            }
        )
    messages.append({"role": "user", "content": "Now summarize."})

    request = RuntimeRequest(
        model="claude-sonnet-4",
        adapter_kind="claude-bridge",
        tool_compatible=True,
        messages=messages,
    )

    from tok.runtime.pipeline.request_validation import (
        canonicalize_anthropic_bridge_body,
        validate_anthropic_bridge_body,
    )

    prepared = runtime.prepare_request(request, session)
    canonical, _, signals = canonicalize_anthropic_bridge_body(prepared.body)

    assert validate_anthropic_bridge_body(canonical) == []
    for msg in canonical["messages"]:
        for block in msg.get("content", []):
            if isinstance(block, dict):
                assert block.get("type") not in {
                    "redacted_thinking",
                }


def test_runtime_prepare_request_compression_in_tool_compatible_mode(tmp_path):
    """Test that compression runs in tool-compatible mode with moderate tool usage."""
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    # Create a longer conversation history to trigger compression
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Help me with these files"}
    ]

    # First turn
    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t0",
                    "name": "view_file",
                    "input": {"path": "src/tok/file_0.py"},
                },
            ],
        }
    )
    messages.append(
        {
            "role": "tool_result",
            "tool_use_id": "t0",
            "content": "file 0 content" * 100,
        }
    )
    messages.append({"role": "user", "content": "Thanks, now help with this"})

    # Second turn
    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "view_file",
                    "input": {"path": "src/tok/file_1.py"},
                },
            ],
        }
    )
    messages.append(
        {
            "role": "tool_result",
            "tool_use_id": "t1",
            "content": "file 1 content" * 100,
        }
    )
    messages.append({"role": "user", "content": "Great, one more"})

    # Third turn with multiple tools
    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "view_file",
                    "input": {"path": "src/tok/file_2.py"},
                },
                {
                    "type": "tool_use",
                    "id": "t3",
                    "name": "view_file",
                    "input": {"path": "src/tok/file_3.py"},
                },
            ],
        }
    )
    messages.append(
        {
            "role": "tool_result",
            "tool_use_id": "t2",
            "content": "file 2 content" * 100,
        }
    )
    messages.append(
        {
            "role": "tool_result",
            "tool_use_id": "t3",
            "content": "file 3 content" * 100,
        }
    )
    messages.append({"role": "user", "content": "Now help with more"})

    # Fourth turn
    messages.append(
        {
            "role": "assistant",
            "content": "I'll help with more files.",
        }
    )

    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=messages,
    )

    prepared = runtime.prepare_request(request, session)

    # With 4 tool uses and enough history, compression should run in tool-compatible mode
    assert prepared.behavior_signals.get("tool_compatible_compression", 0) == 1
    assert (
        "Plain text. Tool calls only. Omit all headers."
        in prepared.body["system"]
    )
    # Compression should have happened
    assert len(prepared.body["messages"]) < len(request.messages)


def test_natural_first_preserves_semantic_dedup_without_tool_compatible_mode():
    runtime = UniversalTokRuntime()
    session = RuntimeSession()
    large_output = "file content line\n" * 50
    request = RuntimeRequest(
        model="claude-sonnet-4-6",
        request_policy="natural_first",
        tool_compatible=True,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "view_file",
                        "input": {"path": "src/tok/runtime/core.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": large_output,
                    }
                ],
            },
        ],
    )

    runtime.prepare_request(request, session)
    prepared = runtime.prepare_request(request, session)
    system_text = str(prepared.body.get("system", ""))

    assert prepared.request_policy == "natural_first"
    assert prepared.effective_tool_compatible is False
    assert prepared.request_policy_escalated is False
    assert (
        prepared.behavior_signals.get("request_policy_natural_first", 0) == 1
    )
    assert prepared.behavior_signals.get("semantic_dedup_hit", 0) == 1
    assert "Plain text. Tool calls only. Omit all headers." not in system_text
    assert "@stable_result(hash:...)" in system_text


def test_natural_first_keeps_tool_dense_history_on_natural_path(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4-6",
        request_policy="natural_first",
        request_has_tools=True,
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "Inspect the runtime path."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "view_file",
                        "input": {"path": "src/tok/runtime/core.py"},
                    }
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "t1",
                "content": "runtime core" * 80,
            },
            {"role": "user", "content": "Inspect the gateway path too."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway/_app_factory.py"},
                    }
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "t2",
                "content": "gateway app factory" * 80,
            },
            {"role": "user", "content": "Summarize the current evidence."},
        ],
    )

    prepared = runtime.prepare_request(request, session)
    system_text = str(prepared.body.get("system", ""))

    assert prepared.effective_tool_compatible is False
    assert prepared.request_policy_escalated is False
    assert (
        prepared.behavior_signals.get("request_policy_natural_first", 0) == 1
    )
    assert (
        prepared.behavior_signals.get("request_policy_tool_compatible", 0) == 0
    )
    assert "Plain text. Tool calls only. Omit all headers." not in system_text


def test_natural_first_sticky_escalation_decays_after_quiet_window(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    dense_request = RuntimeRequest(
        model="claude-sonnet-4-6",
        request_policy="natural_first",
        request_has_tools=True,
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "Inspect the runtime path."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "view_file",
                        "input": {"path": "src/tok/runtime/core.py"},
                    }
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "t1",
                "content": "runtime core" * 80,
            },
            {"role": "user", "content": "Inspect the gateway path too."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway/_app_factory.py"},
                    }
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "t2",
                "content": "gateway app factory" * 80,
            },
            {"role": "user", "content": "Summarize the current evidence."},
        ],
    )
    quiet_request = RuntimeRequest(
        model="claude-sonnet-4-6",
        request_policy="natural_first",
        tool_compatible=True,
        messages=[
            {
                "role": "user",
                "content": "Summarize what you already know without more tool calls.",
            }
        ],
    )

    session.note_request_policy_tool_mode_recovery()
    first = runtime.prepare_request(dense_request, session)
    assert first.effective_tool_compatible is True

    sticky_turn = runtime.prepare_request(quiet_request, session)
    deescalated = runtime.prepare_request(quiet_request, session)

    assert sticky_turn.effective_tool_compatible is True
    assert (
        sticky_turn.behavior_signals.get("request_policy_held_by_recovery", 0)
        == 1
    )
    assert (
        sticky_turn.behavior_signals.get(
            "request_policy_recovery_sticky_continuations", 0
        )
        == 1
    )
    assert deescalated.effective_tool_compatible is False
    assert (
        deescalated.behavior_signals.get("request_policy_deescalations", 0)
        == 1
    )


def test_natural_first_escalates_on_stream_recovery_watch(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.note_request_policy_stream_recovery()
    request = RuntimeRequest(
        model="claude-sonnet-4-6",
        request_policy="natural_first",
        tool_compatible=True,
        messages=[
            {
                "role": "user",
                "content": "Retry safely after the earlier stream recovery.",
            }
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.effective_tool_compatible is True
    assert prepared.request_policy_escalated is True
    assert (
        prepared.behavior_signals.get(
            "request_policy_reason_stream_recovery", 0
        )
        == 1
    )
    assert (
        prepared.behavior_signals.get(
            "request_policy_escalation_source_stream_recovery", 0
        )
        == 1
    )
    assert (
        prepared.behavior_signals.get("request_policy_tool_compatible", 0) == 1
    )


def test_request_policy_is_not_held_by_recovery_when_only_cooldown_is_active(
    tmp_path,
):
    """Request policy should not be held by recovery when only cooldown is active."""
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")

    session._stream_recovery_cooldown_remaining = 1
    session._stream_recovery_cooldown_suppressed = True
    session._stream_recovery_reacquisition_budget = 0
    session._stream_recovery_history_floor_budget = 0

    request = RuntimeRequest(
        model="claude-sonnet-4-6",
        request_policy="natural_first",
        tool_compatible=True,
        messages=[
            {
                "role": "user",
                "content": "Test cooldown-only suppression.",
            }
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert (
        prepared.behavior_signals.get(
            "request_policy_reason_stream_recovery", 0
        )
        == 0
    )
    assert (
        prepared.behavior_signals.get("request_policy_held_by_recovery", 0)
        == 0
    )
    assert (
        prepared.behavior_signals.get(
            "request_policy_recovery_cooldown_suppressed", 0
        )
        == 1
    )

    assert session._stream_recovery_cooldown_suppressed is False


def test_natural_first_escalates_on_invalid_tool_history_recovery(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.note_request_policy_tool_mode_recovery()
    request = RuntimeRequest(
        model="claude-sonnet-4-6",
        request_policy="natural_first",
        tool_compatible=True,
        messages=[
            {
                "role": "user",
                "content": "Recover from the earlier tool-history issue.",
            }
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.effective_tool_compatible is True
    assert prepared.request_policy_escalated is True
    assert (
        prepared.behavior_signals.get("request_policy_reason_tool_recovery", 0)
        == 1
    )
    assert (
        prepared.behavior_signals.get(
            "request_policy_escalation_source_tool_recovery", 0
        )
        == 1
    )


def test_record_invalid_tool_history_recovery_tracks_blocked_flag():
    """Test that blocked parameter is reflected in emitted signals."""
    session = RuntimeSession()

    # Successful recovery (not blocked)
    signals = session.record_invalid_tool_history_recovery(blocked=False)
    assert signals["tok_bridge_invalid_tool_history_recovery"] == 1
    assert signals["tok_bridge_invalid_tool_history_blocked"] == 0

    # Failed recovery (blocked)
    signals = session.record_invalid_tool_history_recovery(blocked=True)
    assert signals["tok_bridge_invalid_tool_history_recovery"] == 1
    assert signals["tok_bridge_invalid_tool_history_blocked"] == 1


def test_record_invalid_tool_history_recovery_clears_hot_state_on_repeat():
    """Test that repeated recoveries clear specific hot state keys to prevent loops."""
    session = RuntimeSession()
    session.bridge_memory._upsert(
        session.bridge_memory.hot, "turns", "2", score_delta=3
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot, "cmds", "some_cmd", score_delta=1
    )
    session._last_tool_compatible_state = "previous_state"

    # First recovery - no reset
    signals = session.record_invalid_tool_history_recovery(blocked=False)
    assert "tok_bridge_invalid_tool_history_session_reset" not in signals
    assert session._invalid_tool_history_recovery_count == 1

    # Second recovery - triggers reset of specific keys
    signals = session.record_invalid_tool_history_recovery(blocked=True)
    assert signals["tok_bridge_invalid_tool_history_session_reset"] == 1
    assert session._last_tool_compatible_state == ""
    # Only specific keys are cleared: turns, next, cmds, errs, blockers
    assert session.bridge_memory.hot.get("turns", []) == []
    assert session.bridge_memory.hot.get("cmds", []) == []


def test_reset_invalid_tool_history_recovery_clears_counter():
    """Test that reset clears the recovery counter."""
    session = RuntimeSession()
    session.record_invalid_tool_history_recovery(blocked=False)
    session.record_invalid_tool_history_recovery(blocked=False)
    assert session._invalid_tool_history_recovery_count == 2

    session.reset_invalid_tool_history_recovery()
    assert session._invalid_tool_history_recovery_count == 0


def test_natural_first_escalates_on_repeated_tool_loop_signal(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4-6",
        request_policy="natural_first",
        tool_compatible=True,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "view_file",
                        "input": {"path": "src/tok/runtime/core.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "view_file",
                        "input": {"path": "src/tok/runtime/core.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "runtime core",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "t2",
                        "content": "runtime core",
                    },
                ],
            },
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.effective_tool_compatible is True
    assert prepared.request_policy_escalated is True
    assert (
        prepared.behavior_signals.get(
            "request_policy_reason_structured_tool_loop", 0
        )
        == 1
    )
    assert (
        prepared.behavior_signals.get(
            "request_policy_escalation_source_structured_tool_loop", 0
        )
        == 1
    )


def test_runtime_prepare_request_suppresses_unchanged_tool_compatible_state(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "Help me with these files"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t0",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_0.py"},
                    },
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "t0",
                "content": "file 0 content" * 100,
            },
            {"role": "user", "content": "Thanks, now help with this"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_1.py"},
                    },
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "t1",
                "content": "file 1 content" * 100,
            },
            {"role": "user", "content": "Great, one more"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_2.py"},
                    },
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "t2",
                "content": "file 2 content" * 100,
            },
            {"role": "user", "content": "Now what changed?"},
        ],
    )

    first = runtime.prepare_request(request, session)
    second = runtime.prepare_request(request, session)

    assert (
        "Plain text. Tool calls only. Omit all headers."
        in first.body["system"]
    )
    assert (
        first.behavior_signals.get("state_resend_full_turn", 0)
        + first.behavior_signals.get("state_resend_delta_turn", 0)
        == 1
    )
    assert (
        second.behavior_signals.get("state_resend_suppressed_turn", 0)
        + second.behavior_signals.get("state_resend_delta_turn", 0)
        >= 1
    )
    assert len(second.body["system"]) <= len(first.body["system"])


def test_tool_compatible_state_fields_are_canonicalized_for_shorter_wire_state():
    fields = _canonicalize_tool_compatible_state_fields(
        {
            "turns": ["13"],
            "goal": ["Tool_result_(c5):_1_passed_in_0.05s"],
            "files": ["*B", "src/tok/gateway.py", "src/tok/app.py"],
            "tests": [
                "Tool_result_(c3):_FAILED_tests/unit/test_gateway.py::test_route"
            ],
            "errs": [
                "Tool_result_(c3):_FAILED_tests/unit/test_gateway.py::test_route_-_NotImplementedError"
            ],
            "constraints": ["do_not_write"],
        }
    )

    assert fields == {
        "turns": ["13"],
        "goal": ["1_passed"],
        "files": ["src/tok/gateway.py", "src/tok/app.py"],
        "tests": ["tests/unit/test_gateway.py::test_route"],
        "errs": ["tests/unit/test_gateway.py::test_route_NotImplementedError"],
    }


def test_tool_compatible_goal_strips_fix_prefix_but_keeps_path_anchor():
    fields = _canonicalize_tool_compatible_state_fields(
        {
            "turns": ["1"],
            "goal": ["Fix_failing_tests_in_src/tok/gateway.py"],
            "files": ["src/tok/gateway.py"],
        }
    )

    assert fields["goal"] == ["src/tok/gateway.py"]


def test_tool_compatible_canonicalizes_answer_facts():
    fields = _canonicalize_tool_compatible_state_fields(
        {
            "turns": ["3"],
            "files": ["src/tok/compression.py"],
            "facts": [
                "answer_verification:compress_history function in src/tok/compression.py",
                "answer_related:src/tok/bridge_memory.py:14: class BridgeMemoryState",
            ],
        }
    )

    assert fields["facts"] == [
        "answer_verification:compress_history",
        "answer_related:src/tok/bridge_memory.py",
    ]


def test_parse_tok_state_fields_folds_unknown_answer_keys_into_facts():
    fields = _parse_tok_state_fields(
        ">>> files:*A|answer_file:src/tok/compression.py|answer_verification:compress_history function|turns:5"
    )

    assert fields["turns"] == ["5"]
    assert fields["files"] == ["*A"]
    assert fields["facts"] == [
        "answer_file:src/tok/compression.py",
        "answer_verification:compress_history function",
    ]


def test_tool_compatible_sticky_fields_are_carried_forward():
    previous = {
        "turns": ["10"],
        "files": ["src/tok/gateway.py"],
        "tests": ["1_passed"],
    }
    current = {
        "turns": ["13"],
        "goal": ["1_passed"],
    }

    merged = _apply_tool_compatible_sticky_fields(previous, current)
    delta = _delta_tok_state_fields(previous, merged)

    # Sticky fields should be carried forward
    assert merged["files"] == ["src/tok/gateway.py"]
    assert merged["tests"] == ["1_passed"]
    # But they should not be in delta if they haven't changed
    assert "f:src/tok/gateway.py" not in delta
    assert "s:1_passed" not in delta


def test_tool_compatible_delta_resends_answer_facts_and_files():
    previous = {
        "turns": ["4"],
        "files": ["src/tok/compression.py"],
        "facts": [
            "answer_file:src/tok/compression.py",
            "answer_verification:compress_history",
        ],
    }
    current = {
        "turns": ["5"],
        "goal": ["related_prompt"],
        "files": ["src/tok/compression.py"],
        "facts": [
            "answer_file:src/tok/compression.py",
            "answer_verification:compress_history",
        ],
    }

    delta = _delta_tok_state_fields(previous, current)

    assert "f:src/tok/compression.py" in delta
    assert (
        "x:answer_file:src/tok/compression.py,answer_verification:compress_history"
        in delta
    )


def test_tool_compatible_state_with_answer_facts_suppresses_when_unchanged(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t0",
                        "name": "view_file",
                        "input": {"path": "src/tok/compression.py"},
                    },
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "t0",
                "content": "def compress_history(messages, keep_turns=2, profile=None):",
            },
            {
                "role": "user",
                "content": "What is the main implementation entry point?",
            },
        ],
    )

    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/compression.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:compress_history",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "files",
        "src/tok/compression.py",
        score_delta=3,
    )

    first = runtime.prepare_request(request, session)
    second = runtime.prepare_request(request, session)

    assert first.behavior_signals.get("state_resend_full_turn", 0) == 1
    assert second.behavior_signals.get("state_resend_suppressed_turn", 0) == 1
    assert second.behavior_signals.get("answer_anchor_present", 0) == 1
    assert (
        second.behavior_signals.get("answer_anchor_verified_current", 0) == 1
    )
    assert (
        "Reuse existing File=/Verification= facts" not in second.body["system"]
    )


def test_tool_compatible_sticky_files_choose_primary_file_from_test_anchor():
    previous = {
        "turns": ["10"],
        "goal": ["src/tok/app.py"],
        "files": ["src/tok/app.py", "src/tok/gateway.py"],
        "tests": ["FAILED_tests/unit/test_gateway"],
    }
    current = {
        "turns": ["13"],
        "goal": ["1_passed"],
        "tests": ["1_passed"],
    }

    merged = _apply_tool_compatible_sticky_fields(previous, current)

    assert merged["files"] == ["src/tok/gateway.py"]


def test_tool_compatible_sticky_files_keep_primary_context_without_strong_anchor():
    previous = {
        "turns": ["4"],
        "goal": ["src/tok/compression.py"],
        "files": ["src/tok/compression.py"],
    }
    current = {
        "turns": ["5"],
        "goal": ["src/tok/bridge_memory.py"],
        "files": ["src/tok/bridge_memory.py"],
    }

    merged = _apply_tool_compatible_sticky_fields(previous, current)

    assert merged["files"] == [
        "src/tok/compression.py",
        "src/tok/bridge_memory.py",
    ]


def test_runtime_prepare_request_reverts_when_mutated_body_fails_preflight(
    tmp_path, monkeypatch
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "hello"}],
    )

    def _bad_injection(
        body,
        tok_state=None,
        tool_compatible=False,
        grammar=None,
        todo=None,
        deltas=None,
        pressure=0,
        behavior_signals=None,
        runtime_hints=None,
    ):
        del (
            tok_state,
            tool_compatible,
            grammar,
            todo,
            deltas,
            pressure,
            behavior_signals,
            runtime_hints,
        )
        body["messages"] = "not-a-list"
        return body

    monkeypatch.setattr(
        "tok.runtime._request_preparation.inject_system_additions",
        _bad_injection,
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.body == {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert prepared.compressed is False
    assert prepared.input_saved_tokens == 0
    assert prepared.behavior_signals["tok_preflight_rejected"] >= 1


def test_runtime_prepare_request_captures_file_snapshots(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f1",
                        "name": "view_file",
                        "input": {"path": "src/tok/config.py"},
                    }
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "f1",
                "content": "class Config:\n    VALUE = 42",
            },
        ],
    )

    runtime.prepare_request(request, session)

    facts = session.bridge_memory.hot.get("facts", [])
    assert any("file[src/tok/config.py]:" in entry.value for entry in facts)


def test_runtime_prepare_request_captures_search_snapshots(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "s1",
                        "name": "grep_search",
                        "input": {"query": "config", "search_path": "src"},
                    }
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "s1",
                "content": "src/tok/config.py: FOUND config",
            },
        ],
    )

    runtime.prepare_request(request, session)

    facts = session.bridge_memory.hot.get("facts", [])
    assert any("search[config]:" in entry.value for entry in facts)


def test_runtime_process_response_updates_memory_and_family_mode():
    runtime = UniversalTokRuntime()
    session = RuntimeSession()

    processed = runtime.process_response(
        ">>> g:fix|t:1\n@msg role:assistant\n  |> ok",
        model="google/gemini-2.0-flash",
        session=session,
        behavior_signals={"repeat_file_read": 1, "repeat_search": 1},
    )

    assert processed.mode == "tok-native"
    assert processed.behavior_signals["tok_native_response"] == 1
    assert processed.updated_memory == ">>> g:fix|t:1"
    assert processed.family_mode == "tok-universal"


def test_response_contract_for_mode_flags_mixed_answer_tool_event():
    contract = response_contract_for_mode(
        "@tool view_file id:call_1 path:src/tok/gateway.py\n\n"
        "@msg role:assistant\n"
        "  |> File=src/tok/gateway.py\n"
        "  |> Verification=health\n",
        tool_compatible=True,
    )

    assert any(
        block.get("type") == "tool_use" for block in contract.content_blocks
    )
    assert contract.behavior_signals["mixed_tool_visible_text"] == 1
    assert contract.behavior_signals["mixed_answer_tool_event"] == 1
    assert contract.behavior_signals["tool_contract_failure"] == 1


def test_response_contract_for_mode_keeps_non_answer_tool_text_telemetry_only():
    contract = response_contract_for_mode(
        "@tool view_file id:call_1 path:src/tok/gateway.py\n\n"
        "@msg role:assistant\n"
        "  |> I will inspect src/tok/gateway.py first.\n",
        tool_compatible=True,
    )

    assert any(
        block.get("type") == "tool_use" for block in contract.content_blocks
    )
    assert contract.behavior_signals["mixed_tool_visible_text"] == 1
    assert "mixed_answer_tool_event" not in contract.behavior_signals
    assert "tool_contract_failure" not in contract.behavior_signals


def test_runtime_process_response_treats_mixed_answer_tool_as_contract_failure(
    monkeypatch,
):
    emitted: list[tuple[str, dict, str]] = []

    def _capture(name: str, payload: dict, *, model: str = "") -> None:
        emitted.append((name, payload, model))

    monkeypatch.setattr("tok.runtime.metrics.emit_event_sync", _capture)

    runtime = UniversalTokRuntime()
    session = RuntimeSession()
    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py\n\n"
        "@msg role:assistant\n"
        "  |> File=src/tok/gateway.py\n"
        "  |> Verification=health\n",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
    )

    assert processed.behavior_signals["tool_contract_failure"] == 1
    assert processed.behavior_signals["mixed_answer_tool_event"] == 1
    assert processed.family_mode == "tok-universal"

    protocol_events = [
        event for event in emitted if event[0] == "protocol_drift"
    ]
    assert protocol_events
    assert protocol_events[-1][1]["signals"]["tool_contract_failure"] == 1


def test_runtime_process_response_marks_answer_ready_tool_violation():
    runtime = UniversalTokRuntime()
    session = RuntimeSession()

    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={"answer_ready_turn": 1},
    )

    assert processed.behavior_signals["answer_ready_turn"] == 1
    assert processed.behavior_signals["answer_ready_tool_violation"] == 1
    assert (
        processed.behavior_signals.get("answer_ready_mixed_turn_violation", 0)
        == 0
    )
    assert processed.behavior_signals.get("tool_contract_failure", 0) == 0


def test_runtime_process_response_marks_answer_ready_mixed_turn_violation(
    monkeypatch,
):
    emitted: list[tuple[str, dict, str]] = []

    def _capture(name: str, payload: dict, *, model: str = "") -> None:
        emitted.append((name, payload, model))

    monkeypatch.setattr("tok.runtime.metrics.emit_event_sync", _capture)

    runtime = UniversalTokRuntime()
    session = RuntimeSession()

    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py\n\n"
        "@msg role:assistant\n"
        "  |> File=src/tok/gateway.py\n"
        "  |> Verification=health\n",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={"answer_ready_turn": 1},
    )

    assert processed.behavior_signals["answer_ready_turn"] == 1
    assert processed.behavior_signals["answer_ready_tool_violation"] == 1
    assert processed.behavior_signals["answer_ready_mixed_turn_violation"] == 1
    assert processed.behavior_signals["tool_contract_failure"] == 1

    protocol_events = [
        event for event in emitted if event[0] == "protocol_drift"
    ]
    assert protocol_events
    assert (
        protocol_events[-1][1]["signals"]["answer_ready_mixed_turn_violation"]
        == 1
    )


def test_runtime_process_response_marks_answer_ready_failed_to_answer():
    runtime = UniversalTokRuntime()
    session = RuntimeSession()

    processed = runtime.process_response(
        "@msg role:assistant\n  |> I will think about it.\n",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={"answer_ready_turn": 1},
    )

    assert processed.behavior_signals["answer_ready_turn"] == 1
    assert processed.behavior_signals["answer_ready_failed_to_answer"] == 1
    assert (
        processed.behavior_signals.get("answer_ready_tool_violation", 0) == 0
    )


def test_runtime_process_response_answer_ready_clean_answer_has_no_violation():
    runtime = UniversalTokRuntime()
    session = RuntimeSession()

    processed = runtime.process_response(
        "File=src/tok/gateway.py\nVerification=health",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={"answer_ready_turn": 1},
    )

    assert processed.behavior_signals["answer_ready_turn"] == 1
    assert (
        processed.behavior_signals.get("answer_ready_tool_violation", 0) == 0
    )
    assert (
        processed.behavior_signals.get("answer_ready_mixed_turn_violation", 0)
        == 0
    )
    assert (
        processed.behavior_signals.get("answer_ready_failed_to_answer", 0) == 0
    )


def test_answer_ready_mixed_turn_requests_repair_for_next_turn(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")

    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py\n\n"
        "@msg role:assistant\n"
        "  |> File=src/tok/gateway.py\n"
        "  |> Verification=health\n",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={"answer_ready_turn": 1},
    )

    assert processed.behavior_signals["answer_ready_repair_requested"] == 1
    assert session._answer_ready_repair_pending is True


def test_prepare_request_activates_answer_ready_repair_once(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._answer_ready_repair_pending = True
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.behavior_signals.get("answer_ready_repair_active", 0) == 1
    assert "Previous turn failed answer assembly." in prepared.body["system"]
    assert session._answer_ready_repair_active is True


def test_late_toolless_fresh_answer_requests_late_repair(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")

    processed = runtime.process_response(
        "File=src/tok/gateway.py\nVerification=health",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={
            "payload_pressure_ready": 1,
            "toolless_fresh_answer_event": 1,
            "late_freshness_signal_promoted": 1,
        },
    )

    assert (
        processed.behavior_signals["late_answer_assembly_repair_requested"]
        == 1
    )
    assert (
        processed.behavior_signals["late_freshness_signal_consumed_by_tok"]
        == 1
    )
    assert session._late_answer_assembly_repair_pending is True
    assert session._late_answer_assembly_repair_mode_pending == "tool_only"


def test_prepare_request_activates_late_tool_only_repair_once(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._late_answer_assembly_repair_pending = True
    session._late_answer_assembly_repair_mode_pending = "tool_only"
    session._answer_ready_repair_pending = True
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert (
        prepared.behavior_signals.get("late_answer_assembly_repair_active", 0)
        == 1
    )
    assert (
        prepared.behavior_signals.get(
            "late_answer_assembly_repair_tool_only", 0
        )
        == 1
    )
    assert prepared.behavior_signals.get("answer_ready_repair_active", 0) == 0
    assert (
        "Previous turn tried to answer before satisfying the late fresh-evidence contract."
        in prepared.body["system"]
    )
    assert (
        "Previous turn failed answer assembly." not in prepared.body["system"]
    )


def test_late_mixed_answer_tool_requests_answer_only_repair(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")

    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py\n\n"
        "@msg role:assistant\n"
        "  |> File=src/tok/gateway.py\n"
        "  |> Verification=health\n",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={
            "late_staged_retry_context": 1,
            "late_mixed_signal_promoted": 1,
        },
    )

    assert (
        processed.behavior_signals["late_answer_assembly_repair_requested"]
        == 1
    )
    assert (
        processed.behavior_signals[
            "late_answer_assembly_repair_answer_only_requested"
        ]
        == 1
    )
    assert processed.behavior_signals["late_mixed_signal_consumed_by_tok"] == 1
    assert session._late_answer_assembly_repair_pending is True
    assert session._late_answer_assembly_repair_mode_pending == "answer_only"


def test_prepare_request_activates_late_answer_only_repair_once(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._late_answer_assembly_repair_pending = True
    session._late_answer_assembly_repair_mode_pending = "answer_only"
    session._answer_ready_repair_pending = True
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert (
        prepared.behavior_signals.get("late_answer_assembly_repair_active", 0)
        == 1
    )
    assert (
        prepared.behavior_signals.get(
            "late_answer_assembly_repair_answer_only", 0
        )
        == 1
    )
    assert prepared.behavior_signals.get("answer_ready_repair_active", 0) == 0
    assert (
        "Previous turn failed final answer assembly after evidence was available."
        in prepared.body["system"]
    )
    assert (
        "Previous turn failed answer assembly." not in prepared.body["system"]
    )


def test_late_answer_only_repair_resolves_after_clean_answer(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._late_answer_assembly_repair_pending = True
    session._late_answer_assembly_repair_mode_pending = "answer_only"
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    runtime.prepare_request(request, session)
    processed = runtime.process_response(
        "File=src/tok/gateway.py\nVerification=health",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={"late_staged_retry_context": 1},
    )

    assert (
        processed.behavior_signals.get(
            "late_answer_assembly_repair_resolved", 0
        )
        == 1
    )
    assert (
        processed.behavior_signals.get(
            "late_answer_assembly_repair_answer_only_resolved", 0
        )
        == 1
    )
    assert session._late_answer_assembly_repair_pending is False


def test_late_answer_only_repair_failure_is_tracked(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._late_answer_assembly_repair_pending = True
    session._late_answer_assembly_repair_mode_pending = "answer_only"
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    runtime.prepare_request(request, session)
    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py\n\n"
        "@msg role:assistant\n"
        "  |> File=src/tok/gateway.py\n"
        "  |> Verification=health\n",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={
            "late_staged_retry_context": 1,
            "late_mixed_signal_promoted": 1,
        },
    )

    assert (
        processed.behavior_signals.get(
            "late_answer_assembly_repair_requested", 0
        )
        == 1
    )
    assert (
        processed.behavior_signals.get("late_answer_assembly_repair_failed", 0)
        == 1
    )
    assert (
        processed.behavior_signals.get(
            "late_answer_assembly_repair_answer_only_failed", 0
        )
        == 1
    )
    assert session._late_answer_assembly_repair_pending is False
    assert session._late_answer_assembly_repair_mode_pending == ""


def test_genuine_freshness_miss_still_wins_over_mixed_for_late_repair_mode(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")

    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py\n\n"
        "@msg role:assistant\n"
        "  |> File=src/tok/gateway.py\n"
        "  |> Verification=health\n",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={
            "payload_pressure_ready": 1,
            "toolless_fresh_answer_event": 1,
            "late_freshness_signal_promoted": 1,
            "late_mixed_signal_promoted": 1,
        },
    )

    assert (
        processed.behavior_signals["late_answer_assembly_repair_requested"]
        == 1
    )
    assert (
        processed.behavior_signals.get(
            "late_answer_assembly_repair_answer_only_requested", 0
        )
        == 0
    )
    assert session._late_answer_assembly_repair_mode_pending == "tool_only"


def test_answer_ready_repair_resolved_after_clean_answer(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._answer_ready_repair_pending = True
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    runtime.prepare_request(request, session)
    processed = runtime.process_response(
        "File=src/tok/gateway.py\nVerification=health",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={"answer_ready_turn": 1},
    )

    assert (
        processed.behavior_signals.get("answer_ready_repair_resolved", 0) == 1
    )
    assert session._answer_ready_repair_pending is False


def test_late_answer_assembly_repair_resolved_after_clean_tool_only_turn(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._late_answer_assembly_repair_pending = True
    session._late_answer_assembly_repair_mode_pending = "tool_only"
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    runtime.prepare_request(request, session)
    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={"late_staged_retry_context": 1},
    )

    assert (
        processed.behavior_signals.get(
            "late_answer_assembly_repair_resolved", 0
        )
        == 1
    )
    assert session._late_answer_assembly_repair_pending is False
    assert session._late_answer_followthrough_pending is False


def test_prepare_request_activates_late_answer_followthrough_once(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._late_answer_followthrough_pending = True
    session._late_answer_assembly_repair_pending = True
    session._late_answer_assembly_repair_mode_pending = "tool_only"
    session._answer_ready_repair_pending = True
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert (
        prepared.behavior_signals.get("late_answer_followthrough_active", 0)
        == 1
    )
    assert (
        prepared.behavior_signals.get("late_answer_assembly_repair_active", 0)
        == 0
    )
    assert prepared.behavior_signals.get("answer_ready_repair_active", 0) == 0
    assert (
        "Previous turn gathered the required evidence. In this turn, do not call tools."
        in prepared.body["system"]
    )
    assert (
        "Previous turn tried to answer before satisfying the late fresh-evidence contract."
        not in prepared.body["system"]
    )


def test_late_answer_followthrough_resolves_after_clean_answer(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._late_answer_followthrough_pending = True
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    runtime.prepare_request(request, session)
    processed = runtime.process_response(
        "File=src/tok/gateway.py\nVerification=health",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={"late_staged_retry_context": 1},
    )

    assert (
        processed.behavior_signals.get("late_answer_followthrough_resolved", 0)
        == 1
    )
    assert (
        processed.behavior_signals.get("late_answer_followthrough_failed", 0)
        == 0
    )
    assert session._late_answer_followthrough_pending is False


def test_late_answer_followthrough_failure_is_tracked(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._late_answer_followthrough_pending = True
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    runtime.prepare_request(request, session)
    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={"late_staged_retry_context": 1},
    )

    assert (
        processed.behavior_signals.get("late_answer_followthrough_failed", 0)
        == 1
    )
    assert (
        processed.behavior_signals.get("late_answer_followthrough_resolved", 0)
        == 0
    )
    assert (
        processed.behavior_signals.get(
            "late_answer_assembly_repair_requested", 0
        )
        == 0
    )
    assert session._late_answer_followthrough_pending is False


def test_answer_ready_repair_failed_on_second_answer_ready_miss(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._answer_ready_repair_pending = True
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    runtime.prepare_request(request, session)
    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={"answer_ready_turn": 1},
    )

    assert processed.behavior_signals.get("answer_ready_repair_failed", 0) == 1
    assert session._answer_ready_repair_pending is False


def test_late_answer_assembly_repair_failed_on_second_late_miss(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._late_answer_assembly_repair_pending = True
    session._late_answer_assembly_repair_mode_pending = "tool_only"
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    runtime.prepare_request(request, session)
    processed = runtime.process_response(
        "File=src/tok/gateway.py\nVerification=health",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={
            "payload_pressure_ready": 1,
            "toolless_fresh_answer_event": 1,
            "late_freshness_signal_promoted": 1,
        },
    )

    assert (
        processed.behavior_signals.get("late_answer_assembly_repair_failed", 0)
        == 1
    )
    assert session._late_answer_assembly_repair_pending is False
    assert session._late_answer_assembly_repair_mode_pending == ""


def test_answer_ready_repair_not_triggered_for_non_answer_ready_tool_turn(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")

    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
    )

    assert (
        processed.behavior_signals.get("answer_ready_repair_requested", 0) == 0
    )
    assert session._answer_ready_repair_pending is False


def test_late_answer_assembly_repair_not_triggered_for_unsupported_or_bad_args(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")

    processed = runtime.process_response(
        "File=src/tok/gateway.py\nVerification=health",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
        behavior_signals={
            "payload_pressure_ready": 1,
            "toolless_fresh_answer_event": 1,
            "late_freshness_signal_promoted": 1,
            "unsupported_tool_event": 1,
            "bad_tool_args_event": 1,
        },
    )

    assert (
        processed.behavior_signals.get(
            "late_answer_assembly_repair_requested", 0
        )
        == 0
    )
    assert (
        processed.behavior_signals.get(
            "late_freshness_signal_consumed_by_tok", 0
        )
        == 0
    )
    assert session._late_answer_assembly_repair_pending is False


def test_mixed_turn_keeps_telemetry_without_requesting_repair(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")

    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py\n\n"
        "@msg role:assistant\n"
        "  |> File=src/tok/gateway.py\n"
        "  |> Verification=health\n",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
    )

    assert processed.behavior_signals.get("mixed_answer_tool_event", 0) == 1
    assert (
        processed.behavior_signals.get("mixed_turn_repair_requested", 0) == 0
    )


def test_prepare_request_does_not_activate_mixed_turn_repair(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.behavior_signals.get("mixed_turn_repair_active", 0) == 0
    assert (
        "Previous turn mixed tool use with a final answer."
        not in prepared.body["system"]
    )


def test_answer_ready_repair_keeps_single_repair_block(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._answer_ready_repair_pending = True
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.behavior_signals.get("answer_ready_repair_active", 0) == 1
    assert (
        prepared.body["system"].count("Previous turn failed answer assembly.")
        == 1
    )
    assert (
        "Previous turn mixed tool use with a final answer."
        not in prepared.body["system"]
    )


def test_mixed_turn_telemetry_not_triggered_for_plain_tool_turn(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    processed = runtime.process_response(
        "@tool view_file id:call_1 path:src/tok/gateway.py",
        model="google/gemini-2.0-flash",
        session=session,
        tool_compatible=True,
    )

    assert processed.behavior_signals.get("mixed_answer_tool_event", 0) == 0


def test_tool_contract_failure_contributes_to_runtime_pressure_metrics():
    assert calculate_invisible_pressure({"tool_contract_failure": 1}) == 1
    assert (
        calculate_semantic_regression_score({"tool_contract_failure": 1}) == 1
    )


def test_extract_structured_answer_memory_promotes_file_and_verification():
    fields = extract_structured_answer_memory(
        "File=src/tok/compression.py\n"
        "Verification=compress_history function with keep_turns support\n"
        "Related=src/tok/bridge_memory.py:14: class BridgeMemoryState"
    )

    assert fields["files"] == ["src/tok/compression.py"]
    assert "answer_file:src/tok/compression.py" in fields["facts"]
    assert any(
        fact.startswith("answer_verification:compress_history function")
        for fact in fields["facts"]
    )
    assert any(
        fact.startswith("answer_related:src/tok/bridge_memory.py")
        for fact in fields["facts"]
    )


def test_extract_structured_answer_memory_parses_freeform_entrypoint_reply():
    fields = extract_structured_answer_memory(
        "The main implementation entry is `compress_history` in `src/tok/compression.py` (line 30)."
    )

    assert fields["files"] == ["src/tok/compression.py"]
    assert "answer_file:src/tok/compression.py" in fields["facts"]
    assert "answer_verification:compress_history" in fields["facts"]


def test_ground_structured_answer_memory_rejects_unguarded_file_guess(
    tmp_path,
):
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory.record_file_snapshot(
        "src/tok/compression.py",
        "def compress_history(messages, keep_turns=2):",
    )

    grounded = ground_structured_answer_memory(
        session,
        {
            "files": ["src/agents/agent.py", "src/tok/compression.py"],
            "facts": [
                "answer_file:src/agents/agent.py",
                "answer_file:src/tok/compression.py",
                "answer_verification:compress_history function",
            ],
        },
    )

    assert grounded["files"] == ["src/tok/compression.py"]
    assert "answer_file:src/tok/compression.py" in grounded["facts"]
    assert "answer_file:src/agents/agent.py" not in grounded["facts"]
    assert "answer_verification:compress_history function" in grounded["facts"]


def test_ground_structured_answer_memory_normalizes_to_exact_seen_path(
    tmp_path,
):
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory.record_file_snapshot(
        "src/tok/compression.py",
        "def compress_history(messages, keep_turns=2):",
    )

    grounded = ground_structured_answer_memory(
        session,
        {
            "files": ["src/compression.py"],
            "facts": ["answer_file:src/compression.py"],
        },
    )

    assert grounded["files"] == ["src/tok/compression.py"]
    assert grounded["facts"] == ["answer_file:src/tok/compression.py"]


def test_reinforce_structured_answer_memory_carries_forward_prior_file(
    tmp_path,
):
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/compression.py",
        score_delta=3,
    )

    reinforced = reinforce_structured_answer_memory(
        session,
        {"facts": ["answer_verification:compress_history function"]},
    )

    assert reinforced["files"] == ["src/tok/compression.py"]
    assert "answer_file:src/tok/compression.py" in reinforced["facts"]


def test_reinforce_structured_answer_memory_prefers_more_specific_verification(
    tmp_path,
):
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:compress_history function",
        score_delta=3,
    )

    reinforced = reinforce_structured_answer_memory(
        session,
        {"facts": ["answer_verification:compress function"]},
    )

    assert (
        reinforced["facts"][0]
        == "answer_verification:compress_history function"
    )


def test_compact_structured_answer_memory_trims_answer_payloads():
    compacted = compact_structured_answer_memory(
        {
            "files": ["src/tok/compression.py"],
            "facts": [
                "answer_file:src/tok/compression.py",
                "answer_verification:The main entry point is the `compress_history` function in `src/tok/compression.py`.",
            ],
        }
    )

    assert compacted["files"] == ["src/tok/compression.py"]
    assert compacted["facts"] == [
        "answer_file:src/tok/compression.py",
        "answer_verification:compress_history",
    ]


def test_compact_structured_answer_memory_preserves_symbol_after_def_keyword():
    compacted = compact_structured_answer_memory(
        {
            "facts": [
                "answer_verification:def compress_history(messages, keep_turns=2, profile=None): in src/tok/compression.py",
            ],
        }
    )

    assert compacted["facts"] == [
        "answer_verification:compress_history",
    ]


def test_runtime_process_response_stores_structured_tool_compatible_answers(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory.record_search_snapshot(
        "compress_history", "src/tok/compression.py:305: def compress_history("
    )

    processed = runtime.process_response(
        "File=src/tok/compression.py\n"
        "Verification=compress_history function with keep_turns support",
        model="deepseek/deepseek-v3.2",
        session=session,
        tool_compatible=True,
    )

    assert processed.mode == "tool-compatible"
    assert "src/tok/compression.py" in [
        entry.value for entry in session.bridge_memory.hot.get("files", [])
    ]
    assert any(
        entry.value == "answer_verification:compress_history"
        for entry in session.bridge_memory.hot.get("facts", [])
    )
    assert "src/tok/compression.py" in [
        entry.value for entry in session.bridge_memory.durable.get("files", [])
    ]
    assert any(
        entry.value == "answer_verification:compress_history"
        for entry in session.bridge_memory.durable.get("facts", [])
    )


def test_structured_answers_survive_hot_replacement_via_durable_memory(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory.record_search_snapshot(
        "compress_history", "src/tok/compression.py:305: def compress_history("
    )

    runtime.process_response(
        "Verification=The main entry point is the `compress_history` function in `src/tok/compression.py`.",
        model="deepseek/deepseek-v3.2",
        session=session,
        tool_compatible=True,
    )

    session.bridge_memory.replace_hot_from_wire_state(
        ">>> g:related_prompt|t:2"
    )
    wire = session.bridge_memory.wire_state(TOOL_COMPAT_MEMORY_PROFILE)

    assert "answer_file:src/tok/compression.py" in wire
    assert "answer_verification:" in wire
    assert "compress_history" in wire


def test_runtime_process_response_does_not_write_healed_tool_compatible_memory(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")

    processed = runtime.process_response(
        "Verification=compress_history function in the codebase\n"
        "I have confirmed the result.",
        model="deepseek/deepseek-v3.2",
        session=session,
        tool_compatible=True,
    )

    assert processed.updated_memory == ""
    assert session.fallback_memory == ""


def test_runtime_session_loads_persisted_fallback_memory_on_startup(tmp_path):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()
    (memory_dir / "memory.tok").write_text(">>> g:stale_raw|t:1\n")
    (memory_dir / "bridge_memory.tok").write_text(
        "@mem v:b1 t:2\n@h\n@f turns\n  |> 2|score:3|last:2\n@f goal\n  |> fresh_hot|score:3|last:2\n"
    )

    session = RuntimeSession(memory_dir=memory_dir)

    assert (
        session.load_memory(model="claude-sonnet-4") == ">>> t:2|g:fresh_hot"
    )
    signals = session.consume_behavior_signals()
    assert signals.get("cold_start_structured_memory", 0) == 1
    assert signals.get("cold_start_wire_fallback", 0) == 0


def test_runtime_session_write_memory_persists_fallback_memory(tmp_path):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()

    session = RuntimeSession(memory_dir=memory_dir)

    updated = session.write_memory(
        ">>> g:keep_bridge_warm|t:4\n@msg role:assistant\n  |> ok"
    )

    assert updated == ">>> g:keep_bridge_warm|t:4"
    assert (
        memory_dir / "memory.tok"
    ).read_text() == ">>> g:keep_bridge_warm|t:4\n"


def test_collect_behavior_signals_matches_bridge_contract():
    import hashlib
    import json

    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r1",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                },
                {
                    "type": "tool_use",
                    "id": "r2",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                },
                {
                    "type": "tool_use",
                    "id": "s1",
                    "name": "grep",
                    "input": {"query": "create_app"},
                },
                {
                    "type": "tool_use",
                    "id": "s2",
                    "name": "grep",
                    "input": {"query": "create_app"},
                },
                {
                    "type": "tool_use",
                    "id": "b1",
                    "name": "bash",
                    "input": {"command": "python -c 'print(1)' >&2"},
                },
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "r1",
            "content": "def route(request):\n    return request\n",
        },
        {
            "role": "tool_result",
            "tool_use_id": "r2",
            "content": "def route(request):\n    return request\n",
        },
        {
            "role": "tool_result",
            "tool_use_id": "s1",
            "content": "src/tok/app.py:10: app = create_app()",
        },
        {
            "role": "tool_result",
            "tool_use_id": "s2",
            "content": "src/tok/app.py:10: app = create_app()",
        },
    ]

    def _make_cache_key(tool_name: str, args: dict) -> str:
        args_str = json.dumps(args, sort_keys=True)
        raw_key = f"{tool_name}:{args_str}"
        return hashlib.sha256(raw_key.encode()).hexdigest()[:12]

    file_cache_key = _make_cache_key(
        "view_file", {"path": "src/tok/gateway.py"}
    )
    search_cache_key = _make_cache_key("grep", {"query": "create_app"})

    file_content = "def route(request):\n    return request\n"
    search_content = "src/tok/app.py:10: app = create_app()"
    file_hash = hashlib.sha256(file_content.encode()).hexdigest()[:8]
    search_hash = hashlib.sha256(search_content.encode()).hexdigest()[:8]

    result_cache = {
        file_cache_key: (file_hash, file_content),
        search_cache_key: (search_hash, search_content),
    }

    signals = collect_behavior_signals(
        messages,
        build_tool_use_id_to_context(messages),
        result_cache,  # type: ignore[arg-type]
    )

    assert signals["cached_file_read"] == 1
    assert signals["cached_search"] == 1
    assert signals["python_c_workaround"] == 1
    assert signals["stderr_workaround"] == 1
    # When content is cached, no reacquisition cost tokens are added
    assert signals.get("reacquisition_cost_tokens", 0) == 0
    assert signals.get("file_reacquisition_cost_tokens", 0) == 0
    assert signals.get("search_reacquisition_cost_tokens", 0) == 0


def test_prepare_request_distinguishes_answer_ready_reacquisition(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/gateway.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:health",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f1",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f2",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {"role": "user", "content": "confirm the gateway entry point"},
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert (
        prepared.behavior_signals.get("answer_anchor_reacquisition_attempt", 0)
        == 1
    )
    assert (
        prepared.behavior_signals.get("answer_ready_reacquisition_attempt", 0)
        == 1
    )
    assert (
        prepared.behavior_signals.get("repair_phase_reacquisition_attempt", 0)
        == 0
    )
    assert (
        prepared.behavior_signals.get("benign_reverification_attempt", 0) == 0
    )


def test_prepare_request_distinguishes_benign_reverification(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/gateway.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:health",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f1",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f2",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": (
                    "If fresh evidence is required, use the read-only tools first. "
                    "Confirm the gateway entry point."
                ),
            },
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert (
        prepared.behavior_signals.get("answer_anchor_reacquisition_attempt", 0)
        == 1
    )
    assert (
        prepared.behavior_signals.get("benign_reverification_attempt", 0) == 1
    )
    assert (
        prepared.behavior_signals.get("answer_ready_reacquisition_attempt", 0)
        == 0
    )


def test_prepare_request_distinguishes_repair_phase_reacquisition(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._answer_ready_repair_pending = True
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/gateway.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:health",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f1",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f2",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": (
                    "If fresh evidence is required, use the read-only tools first. "
                    "Confirm the gateway entry point."
                ),
            },
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert (
        prepared.behavior_signals.get("answer_anchor_reacquisition_attempt", 0)
        == 1
    )
    assert (
        prepared.behavior_signals.get("repair_phase_reacquisition_attempt", 0)
        == 1
    )
    assert (
        prepared.behavior_signals.get("benign_reverification_attempt", 0) == 0
    )


def test_response_contract_detects_tok_native_and_fail_open_modes():
    native = response_contract(
        ">>> goal:fix|turns:1\n@msg role:assistant\n  |> ok"
    )
    compat = response_contract("## heading\nPlain response")

    assert native.mode == "tok-native"
    assert native.behavior_signals == {"tok_native_response": 1}
    assert native.content_blocks == [{"type": "text", "text": "ok"}]

    assert compat.mode == "markdown"
    assert compat.behavior_signals["non_tok_response"] == 1
    assert compat.behavior_signals["fail_open_compat_response"] == 1


def test_response_behavior_signals_detect_non_tok_output():
    assert response_behavior_signals("## hello\nplain markdown") == {
        "non_tok_response": 1
    }
    assert response_behavior_signals("@msg role:assistant\n  |> ok") == {}


def test_process_response_no_drift_in_tool_compatible_mode():
    runtime = UniversalTokRuntime()
    session = RuntimeSession()
    prose = (
        "Here is the updated implementation. I have examined the codebase "
        "and explored the relevant modules to understand the architecture."
    )
    result = runtime.process_response(
        prose,
        model="claude-sonnet-4",
        session=session,
        tool_compatible=True,
    )
    assert "semantic_drift_detected" not in result.behavior_signals
    assert "non_tok_response" not in result.behavior_signals


def test_process_response_still_detects_drift_in_tok_native_mode():
    runtime = UniversalTokRuntime()
    session = RuntimeSession()
    prose = (
        "Here is the updated implementation. I have examined the codebase "
        "and explored the relevant modules to understand the architecture."
    )
    result = runtime.process_response(
        prose,
        model="claude-sonnet-4",
        session=session,
        tool_compatible=False,
    )
    assert result.behavior_signals.get("semantic_drift_detected") == 1


def test_replay_gate_evaluates_pressure_and_failures():
    result = evaluate_replay_gate(
        {
            "min_savings_pct": 20.0,
            "max_invisible_pressure": 3,
            "max_repeat_file_read": 0,
            "max_repeat_search": 0,
            "max_non_tok_response": 0,
            "max_fail_open_compat_response": 0,
            "max_malformed_tok_response": 0,
            "max_blocker_rediscovery": 0,
        },
        savings_pct=100.0,
        behavior_signals={
            "repeat_file_read": 1,
            "repeat_search": 1,
            "python_c_workaround": 1,
        },
    )

    assert (
        calculate_invisible_pressure(
            {
                "repeat_file_read": 1,
                "repeat_search": 1,
                "python_c_workaround": 1,
            }
        )
        == 3
    )
    assert result.invisible_pressure == 3
    assert result.passed is False
    assert "max_repeat_file_read" in result.failed_checks


def test_select_resend_strategy():
    """_select_resend_strategy: verified-current suppresses; new answer anchor full; else delta."""
    from tok.runtime.core import _select_resend_reason, _select_resend_strategy

    fields = {"goal": ["fix bug"], "files": ["src/tok/gateway.py"]}
    answer_fields = {
        "goal": ["fix bug"],
        "facts": ["answer_file:src/tok/gateway.py"],
    }
    changed_answer_fields = {
        "goal": ["fix bug"],
        "facts": [
            "answer_file:src/tok/gateway.py",
            "answer_verification:health",
        ],
    }

    # unchanged answer-bearing state → suppress
    assert (
        _select_resend_strategy(answer_fields, answer_fields, True)
        == "suppress"
    )
    assert (
        _select_resend_reason(answer_fields, answer_fields, True)
        == "verified_current_state"
    )

    # new answer-bearing state → full
    assert _select_resend_strategy(answer_fields, {}, True) == "full"
    assert (
        _select_resend_reason(answer_fields, {}, True) == "new_answer_anchor"
    )

    # identical non-answer fields → suppress
    assert _select_resend_strategy(fields, fields, False) == "suppress"
    assert (
        _select_resend_reason(fields, fields, False)
        == "verified_current_state"
    )

    # changed fields, no answer facts → delta
    assert _select_resend_strategy(fields, {}, False) == "delta"
    assert _select_resend_reason(fields, {}, False) == "changed_state_delta"
    changed = {"goal": ["different goal"], "files": ["src/tok/gateway.py"]}
    assert _select_resend_strategy(fields, changed, False) == "delta"

    # empty previous with non-empty current → delta (not suppress)
    assert _select_resend_strategy(fields, {}, False) == "delta"
    # changed answer-bearing state after first appearance → delta
    assert (
        _select_resend_strategy(changed_answer_fields, answer_fields, True)
        == "delta"
    )
    assert (
        _select_resend_reason(changed_answer_fields, answer_fields, True)
        == "changed_state_delta"
    )


def test_should_persist_to_durable():
    """_should_persist_to_durable returns True for files and answer_ facts only."""
    from tok.runtime.core import _should_persist_to_durable

    assert _should_persist_to_durable("files", "src/tok/gateway.py") is True
    assert (
        _should_persist_to_durable("facts", "answer_file:src/tok/gateway.py")
        is True
    )
    assert (
        _should_persist_to_durable(
            "facts", "answer_verification:compress_history"
        )
        is True
    )
    assert _should_persist_to_durable("facts", "goal:fix the bug") is False
    assert (
        _should_persist_to_durable("facts", "test:tests/unit/test_gateway.py")
        is False
    )
    assert _should_persist_to_durable("goal", "fix the bug") is False
    assert _should_persist_to_durable("turns", "5") is False


def test_answer_anchor_present_signal_set_when_answer_facts_in_state(tmp_path):
    """answer_anchor_present=1 when state contains answer_file or answer_verification."""
    from tok.runtime.core import (
        RuntimeRequest,
        RuntimeSession,
        UniversalTokRuntime,
    )

    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/compression.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:compress_history",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[{"role": "user", "content": "what is the entry point?"}],
    )
    prepared = runtime.prepare_request(request, session)
    assert prepared.behavior_signals.get("answer_anchor_present", 0) == 1, (
        "answer_anchor_present must be set when answer facts are in state"
    )
    assert (
        prepared.behavior_signals.get(
            "state_resend_reason_answer_anchor_present_kept_full", 0
        )
        == 1
    )


def test_answer_anchor_present_can_suppress_when_state_is_unchanged(tmp_path):
    from tok.runtime.core import (
        RuntimeRequest,
        RuntimeSession,
        UniversalTokRuntime,
    )

    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/compression.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:compress_history",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[{"role": "user", "content": "what is the entry point?"}],
    )

    first = runtime.prepare_request(request, session)
    second = runtime.prepare_request(request, session)

    assert first.behavior_signals.get("state_resend_full_turn", 0) == 1
    assert second.behavior_signals.get("answer_anchor_present", 0) == 1
    assert second.behavior_signals.get("state_resend_suppressed_turn", 0) == 1
    assert (
        second.behavior_signals.get("answer_anchor_verified_current", 0) == 1
    )
    assert (
        second.behavior_signals.get(
            "state_resend_reason_answer_anchor_present_kept_full", 0
        )
        == 0
    )


def test_answer_anchor_present_can_delta_when_state_changes(tmp_path):
    from tok.runtime.core import (
        RuntimeRequest,
        RuntimeSession,
        UniversalTokRuntime,
    )

    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/compression.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "files",
        "src/tok/compression.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "tests",
        "tests/unit/test_compression.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "errs",
        "Compression entry point unclear",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[{"role": "user", "content": "what is the entry point?"}],
    )

    first = runtime.prepare_request(request, session)
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:compress_history",
        score_delta=3,
    )
    second = runtime.prepare_request(request, session)

    assert first.behavior_signals.get("state_resend_full_turn", 0) == 1
    assert second.behavior_signals.get("answer_anchor_present", 0) == 1
    assert second.behavior_signals.get("state_resend_delta_turn", 0) == 1
    assert second.behavior_signals.get("answer_anchor_delta_allowed", 0) == 1
    assert (
        second.behavior_signals.get(
            "state_resend_reason_answer_anchor_present_kept_full", 0
        )
        == 0
    )


def test_answer_anchor_present_signal_absent_without_answer_facts(tmp_path):
    """answer_anchor_present is not set when no answer facts are present."""
    from tok.runtime.core import (
        RuntimeRequest,
        RuntimeSession,
        UniversalTokRuntime,
    )

    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "files",
        "src/tok/gateway.py",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[{"role": "user", "content": "hello"}],
    )
    prepared = runtime.prepare_request(request, session)
    assert prepared.behavior_signals.get("answer_anchor_present", 0) == 0, (
        "answer_anchor_present must not be set when no answer facts are in state"
    )


def test_prepare_request_emits_resend_reason_for_suppressed_or_delta(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[{"role": "user", "content": "hello"}],
    )

    runtime.prepare_request(request, session)
    prepared = runtime.prepare_request(request, session)

    if prepared.behavior_signals.get("state_resend_delta_turn", 0):
        assert (
            prepared.behavior_signals.get(
                "state_resend_reason_delta_selected", 0
            )
            == 1
        )
    if prepared.behavior_signals.get("state_resend_suppressed_turn", 0):
        assert (
            prepared.behavior_signals.get(
                "state_resend_reason_state_suppressed", 0
            )
            == 1
        )


def test_prepare_request_marks_full_resend_when_delta_not_smaller(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    # Previous state has minimal content
    session._last_tool_compatible_state_fields = {
        "goal": ["alpha"],
    }
    # Current state has much more content (files, tests added)
    # The delta will include all the new content, making it similar in size to full state
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "goal",
        "beta",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "files",
        "src/tok/gateway.py,src/tok/cli.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "tests",
        "test_gateway_passed,test_cli_passed",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[{"role": "user", "content": "hello"}],
    )

    prepared = runtime.prepare_request(request, session)

    # When delta is not smaller (most content is new), use full resend
    assert prepared.behavior_signals.get("state_resend_full_turn", 0) == 1
    assert (
        prepared.behavior_signals.get(
            "state_resend_reason_delta_not_smaller", 0
        )
        == 1
    )


def test_prepare_request_emits_answer_anchor_reacquisition_hint_and_signal(
    tmp_path,
):
    from tok.runtime.core import (
        RuntimeRequest,
        RuntimeSession,
        UniversalTokRuntime,
    )

    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/gateway.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:health",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f1",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f2",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "s1",
                        "name": "grep_search",
                        "input": {"query": "health", "search_path": "src/tok"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "s2",
                        "name": "grep_search",
                        "input": {"query": "health", "search_path": "src/tok"},
                    }
                ],
            },
            {"role": "user", "content": "confirm the gateway entry point"},
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.behavior_signals.get("answer_anchor_present", 0) == 1
    assert (
        prepared.behavior_signals.get("answer_anchor_reacquisition_attempt", 0)
        == 1
    )
    assert (
        "Reuse existing File=/Verification= facts"
        not in prepared.body["system"]
    )


def test_prepare_request_emits_answer_now_directive_when_answer_ready_from_anchor(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/gateway.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:health",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.behavior_signals.get("answer_ready_turn", 0) == 1
    assert (
        "Answer now using the existing File=/Verification= evidence."
        in prepared.body["system"]
    )


def test_prepare_request_does_not_emit_answer_now_directive_when_fresh_evidence_still_required(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/gateway.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:health",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "user",
                "content": (
                    "If fresh evidence is required, use the read-only tools first. "
                    "Confirm the gateway entry point."
                ),
            }
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.behavior_signals.get("answer_ready_turn", 0) == 0
    assert (
        "Answer now using the existing File=/Verification= evidence."
        not in prepared.body["system"]
    )


def test_prepare_request_emits_answer_now_directive_after_fresh_tool_results(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "src/tok/gateway.py: health endpoint handler",
                    },
                    {
                        "type": "text",
                        "text": "Use the evidence you just retrieved.",
                    },
                ],
            }
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.behavior_signals.get("answer_ready_turn", 0) == 1
    assert (
        "Answer now using the existing File=/Verification= evidence."
        in prepared.body["system"]
    )


def test_prepare_request_tool_results_only_do_not_trigger_answer_ready(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "src/tok/gateway.py: health endpoint handler",
                    }
                ],
            }
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.behavior_signals.get("answer_ready_turn", 0) == 0
    assert session._late_answer_followthrough_pending is False
    assert (
        "Answer now using the existing File=/Verification= evidence."
        not in prepared.body["system"]
    )


def test_prepare_request_read_only_audit_turn_suppresses_answer_repairs(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/protocol/schema.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:canonical_idl",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "user",
                "content": (
                    "Read-only audit. No edits, no tests, no installs, no network. "
                    "Determine whether the protocol surface is coherent, but do not answer yet."
                ),
            }
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.behavior_signals.get("answer_ready_turn", 0) == 0
    assert prepared.behavior_signals.get("answer_ready_repair_active", 0) == 0
    assert (
        prepared.behavior_signals.get("late_answer_followthrough_active", 0)
        == 0
    )
    assert session._answer_ready_repair_pending is False
    assert session._late_answer_followthrough_pending is False
    assert (
        "Answer now using the existing File=/Verification= evidence."
        not in prepared.body["system"]
    )


def test_prepare_request_does_not_emit_answer_anchor_reacquisition_without_anchor(
    tmp_path,
):
    from tok.runtime.core import (
        RuntimeRequest,
        RuntimeSession,
        UniversalTokRuntime,
    )

    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f1",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f2",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {"role": "user", "content": "confirm the gateway entry point"},
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.behavior_signals.get("answer_anchor_present", 0) == 0
    assert (
        prepared.behavior_signals.get("answer_anchor_reacquisition_attempt", 0)
        == 0
    )
    assert (
        "Reuse existing File=/Verification= facts"
        not in prepared.body["system"]
    )


def test_prepare_request_does_not_emit_answer_anchor_hint_on_clean_turn(
    tmp_path,
):
    from tok.runtime.core import (
        RuntimeRequest,
        RuntimeSession,
        UniversalTokRuntime,
    )

    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_file:src/tok/gateway.py",
        score_delta=3,
    )
    session.bridge_memory._upsert(
        session.bridge_memory.hot,
        "facts",
        "answer_verification:health",
        score_delta=3,
    )
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {"role": "user", "content": "confirm the gateway entry point"}
        ],
    )

    prepared = runtime.prepare_request(request, session)

    assert prepared.behavior_signals.get("answer_anchor_present", 0) == 1
    assert (
        prepared.behavior_signals.get("answer_anchor_reacquisition_attempt", 0)
        == 0
    )
    assert (
        "Reuse existing File=/Verification= facts"
        not in prepared.body["system"]
    )


def test_record_fallback_event_increments_and_degrades():
    """record_fallback_event increments counter and sets _baseline_only at threshold."""
    from tok.runtime.core import RuntimeSession, _FALLBACK_THRESHOLD

    session = RuntimeSession()
    assert session._consecutive_fallback_count == 0
    assert not session._baseline_only

    for i in range(_FALLBACK_THRESHOLD - 1):
        session.record_fallback_event()
        assert not session._baseline_only, (
            f"should not degrade before threshold at i={i}"
        )

    session.record_fallback_event()
    assert session._baseline_only, "should degrade at threshold"
    assert session._consecutive_fallback_count == _FALLBACK_THRESHOLD


def test_reset_fallback_count_clears_counter():
    """reset_fallback_count resets the consecutive counter but not _baseline_only."""
    from tok.runtime.core import RuntimeSession

    session = RuntimeSession()
    session._consecutive_fallback_count = 2
    session.reset_fallback_count()
    assert session._consecutive_fallback_count == 0


def test_cost_usd_computed_when_pricing_provided():
    """LiveBenchmarkRunner computes cost_usd when pricing dict is provided."""
    from unittest.mock import MagicMock, patch

    from tok.testing.live_benchmark import (
        LiveBenchmarkRunner,
        load_benchmark_definition,
    )

    pricing = {"prompt": 1.0, "completion": 3.0}
    runner = LiveBenchmarkRunner(
        model="test/model", pricing=pricing, client=MagicMock()
    )

    definition = load_benchmark_definition("coding-loop-5")

    fake_usage = MagicMock()
    fake_usage.prompt_tokens = 1000
    fake_usage.completion_tokens = 500
    fake_usage.total_tokens = 1500

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "Answer: src/compress.py"
    fake_response.usage = fake_usage

    with patch.object(
        runner.client.chat.completions,  # type: ignore[attr-defined]
        "create",
        return_value=fake_response,
    ):
        result = runner.run(definition, mode="baseline", turns=1)

    assert result.provider_usage.cost_usd is not None
    # 1000 * 1.0 / 1e6 + 500 * 3.0 / 1e6 = 0.000001 + 0.0000015 = 0.0000025
    # 1000 * 1.0 / 1e6 + 500 * 3.0 / 1e6 = 0.001 + 0.0015 = 0.0025\n    assert abs(result.provider_usage.cost_usd - 0.0025) < 1e-10


def test_cost_usd_none_when_no_pricing():
    """LiveBenchmarkRunner leaves cost_usd as None when no pricing is provided."""
    from unittest.mock import MagicMock, patch

    from tok.testing.live_benchmark import (
        LiveBenchmarkRunner,
        load_benchmark_definition,
    )

    runner = LiveBenchmarkRunner(model="test/model", client=MagicMock())
    definition = load_benchmark_definition("coding-loop-5")

    fake_usage = MagicMock()
    fake_usage.prompt_tokens = 100
    fake_usage.completion_tokens = 50
    fake_usage.total_tokens = 150

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "Answer: src/compress.py"
    fake_response.usage = fake_usage

    with patch.object(
        runner.client.chat.completions,  # type: ignore[attr-defined]
        "create",
        return_value=fake_response,
    ):
        result = runner.run(definition, mode="baseline", turns=1)

    assert result.provider_usage.cost_usd is None


def test_collect_transient_error_snippets_returns_transient_errors():
    """collect_transient_error_snippets must return snippets for ImportError etc."""
    messages = [
        {"role": "user", "content": "ImportError: no module named 'tok'"},
        {"role": "user", "content": "SyntaxError in adapters.py line 42"},
        {
            "role": "user",
            "content": "blocked on test failure",
        },  # hard blocker, NOT transient
    ]

    errs = collect_transient_error_snippets(messages)

    assert isinstance(errs, list)
    assert len(errs) >= 1, (
        f"Expected at least 1 transient error snippet, got: {errs}"
    )
    assert not any("blocked on" in e.lower() for e in errs), (
        f"Hard blocker phrase leaked into errs: {errs}"
    )


def test_collect_transient_error_snippets_empty_when_no_transient_errors():
    """collect_transient_error_snippets must return [] when no transient-error phrases present."""
    messages = [
        {"role": "user", "content": "everything looks good, running tests"},
        {"role": "assistant", "content": ">>> goal:fix_bridge|turns:1"},
    ]

    errs = collect_transient_error_snippets(messages)
    assert errs == [], f"Expected empty list for clean messages, got: {errs}"


def test_collect_behavior_signals_values_are_all_ints():
    """collect_behavior_signals must only return int-valued keys (safe for arithmetic)."""
    messages = [
        {"role": "user", "content": "ImportError: no module named 'tok'"},
    ]

    signals = collect_behavior_signals(messages)
    for key, value in signals.items():
        assert isinstance(value, int), (
            f"collect_behavior_signals returned non-int value for '{key}': {value!r}"
        )


def test_prepare_request_injects_transient_errors_into_hot_errs(tmp_path):
    """prepare_request must upsert transient error snippets into session.bridge_memory.hot['errs']."""
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=[
            {
                "role": "user",
                "content": "ModuleNotFoundError: No module named 'tok.neuro'",
            },
        ],
    )

    runtime.prepare_request(request, session)

    errs = [e.value for e in session.bridge_memory.hot.get("errs", [])]
    assert any(
        "ModuleNotFoundError" in e or "no module named" in e.lower()
        for e in errs
    ), f"Expected transient error in hot['errs'], got: {errs}"


def test_no_cut_session_continues_functioning(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t0",
                    "content": "output 0",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t0",
                    "name": "bash",
                    "input": {"cmd": "ls"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "output 1",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "bash",
                    "input": {"cmd": "pwd"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": "output 2",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "bash",
                    "input": {"cmd": "whoami"},
                }
            ],
        },
    ]

    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=messages,
    )

    prepared = runtime.prepare_request(request, session)

    assert (
        prepared.behavior_signals.get("tok_history_cut_point_missing", 0) == 1
    )
    assert "tok_history_cut_blocked_tool_result" in prepared.behavior_signals
    assert (
        prepared.behavior_signals.get("tok_history_compression_skipped", 0)
        == 0
    )
    assert len(prepared.body["messages"]) == len(messages)
    assert "empty_messages" not in validate_anthropic_request_body(
        prepared.body
    )


def test_skip_history_and_cut_point_missing_are_distinguishable(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t0",
                    "content": "output 0",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t0",
                    "name": "bash",
                    "input": {"cmd": "ls"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "output 1",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "bash",
                    "input": {"cmd": "pwd"},
                }
            ],
        },
    ]

    request = RuntimeRequest(
        model="claude-sonnet-4",
        tool_compatible=True,
        messages=messages,
    )

    prepared = runtime.prepare_request(request, session)

    assert (
        prepared.behavior_signals.get("tok_history_compression_skipped", 0)
        == 0
    )
    assert (
        prepared.behavior_signals.get("tok_history_cut_point_missing", 0) == 1
    )
    assert "tok_history_cut_blocked_tool_result" in prepared.behavior_signals
