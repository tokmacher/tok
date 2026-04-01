from tok.runtime.core import RuntimeSession, UniversalTokRuntime
from tok.runtime.pipeline.request_validation import (
    canonicalize_anthropic_bridge_body,
    canonicalize_anthropic_bridge_messages,
    summarize_message_structure,
    validate_anthropic_bridge_body,
)
from tok.runtime.types import RuntimeRequest


def test_top_level_tool_result_is_rewritten_and_merged_into_user_blocks():
    messages = [
        {"role": "user", "content": "Inspect the bridge."},
        {
            "role": "tool_result",
            "tool_use_id": "tool_1",
            "content": "compressed log",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Summarize the failure."}],
        },
    ]

    canonical, changed, signals = canonicalize_anthropic_bridge_messages(
        messages
    )

    assert changed is True
    assert canonical == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect the bridge."},
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": "compressed log",
                },
                {"type": "text", "text": "Summarize the failure."},
            ],
        }
    ]


def test_assistant_tool_use_blocks_remain_assistant_messages():
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    },
                    {"type": "text", "text": "   "},
                ],
            }
        ],
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert changed is True
    assert canonical["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                }
            ],
        }
    ]


def test_strict_bridge_validation_rejects_empty_messages_after_canonicalization():
    canonical, _, _ = canonicalize_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "   "}],
        }
    )

    failures = validate_anthropic_bridge_body(canonical)

    assert failures == ["empty_messages"]


def test_canonicalization_falls_back_to_original_on_validation_failure():
    body = {
        "model": "claude-sonnet-4",
        "system": 123,
        "messages": [
            {"role": "user", "content": "Inspect the bridge."},
            {
                "role": "tool_result",
                "tool_use_id": "tool_1",
                "content": "compressed log",
            },
        ],
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert changed is False
    assert canonical == body
    assert signals.get("tok_bridge_canonical_validation_failed", 0) == 1


def test_strict_bridge_validation_rejects_cross_role_tool_shapes():
    failures = validate_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "view_file",
                            "input": {"path": "src/tok/gateway.py"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file contents",
                        }
                    ],
                },
            ],
        }
    )

    assert "user_contains_tool_use" in failures
    assert "assistant_contains_tool_result" in failures


def test_strict_bridge_validation_rejects_text_before_tool_result_in_user_message():
    failures = validate_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "view_file",
                            "input": {"path": "src/tok/gateway.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is the result"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file contents",
                        },
                        {"type": "text", "text": "Continue."},
                    ],
                },
            ],
        }
    )

    assert "user_tool_result_after_text" in failures


def test_strict_bridge_validation_rejects_unknown_tool_result_ids():
    failures = validate_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "missing_tool",
                            "content": "file contents",
                        }
                    ],
                }
            ],
        }
    )

    assert "tool_result_unknown_tool_use_id" in failures
    assert "tool_result_not_immediately_after_assistant_tool_use" in failures


def test_strict_bridge_validation_rejects_tool_results_separated_from_tool_use():
    failures = validate_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "view_file",
                            "input": {"path": "src/tok/gateway.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": "First let me explain the problem.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file contents",
                        }
                    ],
                },
            ],
        }
    )

    assert "tool_result_not_immediately_after_assistant_tool_use" in failures


def test_bridge_tail_with_adaptive_keep_zero_still_strictly_valid(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._step_count = 11
    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4-6",
            adapter_kind="claude-bridge",
            tool_compatible=True,
            messages=[
                {
                    "role": "user",
                    "content": "Inspect the flaky runtime failure.",
                },
                {
                    "role": "assistant",
                    "content": "I will inspect the failing path.",
                },
                {
                    "role": "user",
                    "content": "Now verify the bridge request still has a tail.",
                },
            ],
        ),
        session,
    )

    canonical, _, _ = canonicalize_anthropic_bridge_body(prepared.body)

    assert canonical["messages"]
    failures = validate_anthropic_bridge_body(canonical)
    assert failures == []


def test_tool_heavy_bridge_request_passes_canonicalization_and_strict_validation(
    tmp_path,
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    messages = [{"role": "user", "content": "Help me with these files"}]
    messages.extend(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t0",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_0.py"},
                    }
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "t0",
                "content": "file 0 " * 120,
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
                    }
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "t1",
                "content": "file 1 " * 120,
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
                    {
                        "type": "tool_use",
                        "id": "t3",
                        "name": "view_file",
                        "input": {"path": "src/tok/file_3.py"},
                    },
                ],
            },
            {
                "role": "tool_result",
                "tool_use_id": "t2",
                "content": "file 2 " * 120,
            },
            {
                "role": "tool_result",
                "tool_use_id": "t3",
                "content": "file 3 " * 120,
            },
            {"role": "user", "content": "Now help with more"},
            {"role": "assistant", "content": "I'll help with more files."},
        ]
    )

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            adapter_kind="claude-bridge",
            tool_compatible=True,
            messages=messages,
        ),
        session,
    )
    canonical, changed, signals = canonicalize_anthropic_bridge_body(
        prepared.body
    )

    assert prepared.behavior_signals.get("tool_compatible_compression", 0) == 1
    assert prepared.behavior_signals.get("tok_bridge_canonicalized", 0) == 1
    assert changed is False
    assert validate_anthropic_bridge_body(canonical) == []
    assert {message["role"] for message in canonical["messages"]} <= {
        "user",
        "assistant",
    }
    assert any(
        any(block.get("type") == "tool_result" for block in message["content"])
        for message in canonical["messages"]
        if message["role"] == "user"
    )


def test_canonicalization_strips_thinking_blocks_from_assistant_message():
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze this..."},
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    },
                ],
            }
        ],
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert changed is True
    assert canonical["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                }
            ],
        }
    ]
    assert signals.get("tok_bridge_thinking_block_dropped", 0) == 1
    assert signals.get("tok_bridge_unsupported_block_dropped", 0) == 1


def test_assistant_message_with_only_unsupported_blocks_is_dropped():
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "internal thought"},
                    {"type": "redacted_thinking", "data": "..."},
                ],
            },
            {"role": "user", "content": "Continue."},
        ],
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert changed is True
    assert len(canonical["messages"]) == 1
    assert canonical["messages"][0]["role"] == "user"
    assert signals.get("tok_bridge_thinking_block_dropped", 0) == 2
    assert signals.get("tok_bridge_unsupported_block_dropped", 0) == 2


def test_user_message_retains_supported_blocks_after_filtering():
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the result"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "file contents",
                    },
                    {"type": "thinking", "thinking": "should not be here"},
                ],
            },
        ],
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert changed is True
    assert canonical["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the result"},
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": "file contents",
                },
            ],
        }
    ]
    assert signals.get("tok_bridge_thinking_block_dropped", 0) == 1
    assert signals.get("tok_bridge_unsupported_block_dropped", 0) == 1


def test_canonicalized_mixed_user_message_is_rejected_by_strict_bridge_validation():
    canonical, _, _ = canonicalize_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "messages": [
                {"role": "user", "content": "Inspect the bridge."},
                {
                    "role": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": "compressed log",
                },
                {"role": "user", "content": "Summarize the failure."},
            ],
        }
    )

    failures = validate_anthropic_bridge_body(canonical)

    assert "user_tool_result_after_text" in failures
    assert "tool_result_unknown_tool_use_id" in failures
    assert "tool_result_not_immediately_after_assistant_tool_use" in failures


def test_strict_bridge_validation_rejects_unsupported_block_types():
    failures = validate_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "this should have been removed",
                        }
                    ],
                }
            ],
        }
    )

    assert "unsupported_block_type" in failures


def test_summarize_message_structure_includes_unsupported_blocks():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "redacted_thinking", "data": "..."},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "bash",
                    "input": {},
                },
            ],
        },
        {"role": "user", "content": "ok"},
    ]

    summary = summarize_message_structure(messages)

    assert summary["unsupported_blocks"] == {
        "thinking": 1,
        "redacted_thinking": 1,
    }
    assert summary["tool_use_blocks"] == 1
