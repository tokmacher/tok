import re
from typing import Any, cast

import tok.runtime._request_preparation as request_preparation_module
from tok.runtime.core import RuntimeSession, UniversalTokRuntime
from tok.runtime.pipeline.request_validation import (
    bridge_strict_failure_signals,
    canonicalize_anthropic_bridge_body,
    canonicalize_anthropic_bridge_messages,
    quarantine_invalid_tool_history_messages,
    summarize_bridge_pairing,
    summarize_message_structure,
    validate_anthropic_bridge_body,
    validate_anthropic_outgoing_bridge_body,
)
from tok.runtime.types import RuntimeRequest


def _provider_sensitive_large_tool_batch_messages() -> list[dict[str, Any]]:
    tool_uses = [
        {
            "type": "tool_use",
            "id": f"toolu_batch_{index + 1}",
            "name": "read_file",
            "input": {"path": f"file_{index + 1}.py"},
        }
        for index in range(18)
    ]
    assistant_content = (
        tool_uses[:9]
        + [{"type": "text", "text": "Collecting evidence."}]
        + tool_uses[9:]
    )
    tool_results = [
        {
            "type": "tool_result",
            "tool_use_id": f"toolu_batch_{index + 1}",
            "content": f"result {index + 1}",
        }
        for index in range(18)
    ]
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Inspect concurrency path."}],
        },
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": tool_results},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_tail_1",
                    "name": "grep_search",
                    "input": {"pattern": "stream_recovery"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_tail_1",
                    "content": "tail result",
                }
            ],
        },
    ]


def test_top_level_tool_result_is_rewritten_and_merged_into_user_blocks():
    messages: list[dict[str, Any]] = [
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
            "content": [{"type": "text", "text": "Inspect the bridge."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": "compressed log",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Summarize the failure."}],
        },
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


def test_strict_bridge_validation_rejects_missing_next_turn_tool_results():
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
                    "content": [{"type": "text", "text": "Continue."}],
                },
            ],
        }
    )

    assert "assistant_tool_use_missing_next_tool_result" in failures


def test_strict_bridge_validation_rejects_incomplete_next_turn_tool_results():
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
                        },
                        {
                            "type": "tool_use",
                            "id": "tool_2",
                            "name": "grep",
                            "input": {"pattern": "TODO"},
                        },
                    ],
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

    assert (
        "assistant_tool_use_incomplete_next_tool_result_coverage" in failures
    )


def test_strict_bridge_validation_rejects_out_of_order_next_turn_tool_results():
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
                        },
                        {
                            "type": "tool_use",
                            "id": "tool_2",
                            "name": "grep",
                            "input": {"pattern": "TODO"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_2",
                            "content": "grep output",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file output",
                        },
                    ],
                },
            ],
        }
    )

    assert "tool_result_not_immediately_after_assistant_tool_use" in failures
    assert (
        "assistant_tool_use_incomplete_next_tool_result_coverage" in failures
    )


def test_strict_bridge_validation_accepts_alternating_multi_turn_tool_pairs():
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect concurrency path."}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_a1",
                        "name": "read_file",
                        "input": {"path": "a.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tool_a2",
                        "name": "read_file",
                        "input": {"path": "b.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tool_a3",
                        "name": "read_file",
                        "input": {"path": "c.py"},
                    },
                    {"type": "text", "text": "Collecting evidence."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_a1",
                        "content": "A",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_a2",
                        "content": "B",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_a3",
                        "content": "C",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_b1",
                        "name": "grep",
                        "input": {"pattern": "TODO"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_b1",
                        "content": "D",
                    }
                ],
            },
        ],
    }

    assert validate_anthropic_bridge_body(body) == []
    timeline = summarize_bridge_pairing(body["messages"])
    assert timeline[0]["tool_use_ids"] == ["tool_a1", "tool_a2", "tool_a3"]
    assert timeline[0]["next_tool_result_ids"] == [
        "tool_a1",
        "tool_a2",
        "tool_a3",
    ]
    assert timeline[1]["tool_use_ids"] == ["tool_b1"]
    assert timeline[1]["next_tool_result_ids"] == ["tool_b1"]


def test_outgoing_bridge_validation_flags_large_interleaved_tool_use_batch():
    body = {
        "model": "claude-sonnet-4",
        "messages": _provider_sensitive_large_tool_batch_messages(),
    }

    assert validate_anthropic_bridge_body(body) == []
    assert set(validate_anthropic_outgoing_bridge_body(body)) == {
        "provider_sensitive_large_tool_use_text_interleaving",
        "provider_sensitive_assistant_tool_use_text_interleaving",
    }
    summary = summarize_message_structure(body["messages"])
    assert summary["provider_sensitivity_risks"] == {
        "assistant_large_tool_use_batch": 1,
        "assistant_tool_use_text_interleaving": 1,
        "assistant_large_tool_use_text_interleaving": 1,
        "provider_sensitive_large_tool_use_text_interleaving": 1,
    }


def test_outgoing_bridge_validation_flags_small_interleaved_tool_use_batch():
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect concurrency path."}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"path": "a.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "Read",
                        "input": {"path": "b.py"},
                    },
                    {"type": "text", "text": "Collecting evidence."},
                    {
                        "type": "tool_use",
                        "id": "toolu_3",
                        "name": "Read",
                        "input": {"path": "c.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "a",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": "b",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_3",
                        "content": "c",
                    },
                ],
            },
        ],
    }

    assert validate_anthropic_bridge_body(body) == []
    assert validate_anthropic_outgoing_bridge_body(body) == [
        "provider_sensitive_assistant_tool_use_text_interleaving",
    ]
    summary = summarize_message_structure(body["messages"])
    assert summary["provider_sensitivity_risks"] == {
        "assistant_tool_use_text_interleaving": 1
    }


def test_strict_bridge_validation_rejects_alternating_multi_turn_reordered_first_exchange():
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect concurrency path."}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_a1",
                        "name": "read_file",
                        "input": {"path": "a.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tool_a2",
                        "name": "read_file",
                        "input": {"path": "b.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tool_a3",
                        "name": "read_file",
                        "input": {"path": "c.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_a1",
                        "content": "A",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_a3",
                        "content": "C",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_a2",
                        "content": "B",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_b1",
                        "name": "grep",
                        "input": {"pattern": "TODO"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_b1",
                        "content": "D",
                    }
                ],
            },
        ],
    }

    failures = validate_anthropic_bridge_body(body)
    assert "tool_result_not_immediately_after_assistant_tool_use" in failures
    assert (
        "assistant_tool_use_incomplete_next_tool_result_coverage" in failures
    )


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
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Help me with these files"}
    ]
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

    assert (
        prepared.behavior_signals.get("tok_history_pairing_safety_degraded", 0)
        == 1
    )
    assert prepared.behavior_signals.get("tool_compatible_compression", 0) == 0
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


def test_prepare_request_discards_history_rewrite_that_breaks_pairing(
    tmp_path, monkeypatch
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    original_messages = [
        {"role": "user", "content": "Inspect the bridge."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                },
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "grep",
                    "input": {"pattern": "TODO"},
                },
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "tool_1",
            "content": "file contents",
        },
        {
            "role": "tool_result",
            "tool_use_id": "tool_2",
            "content": "grep contents",
        },
        {"role": "user", "content": "Summarize the results."},
    ]

    invalid_recent = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                },
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "grep",
                    "input": {"pattern": "TODO"},
                },
            ],
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
    ]

    def _fake_compress_history(
        messages, keep_turns=2, profile=None, prune_tool_results=False
    ):
        del messages, keep_turns, profile, prune_tool_results
        return invalid_recent, "compressed tail"

    def _fake_compress_recent_window(
        messages,
        tool_use_id_to_context=None,
        threshold=0,
        tool_compatible=False,
    ):
        del tool_use_id_to_context, threshold, tool_compatible
        return messages, {"file": 128}

    monkeypatch.setattr(
        request_preparation_module,
        "compress_history",
        _fake_compress_history,
    )
    monkeypatch.setattr(
        request_preparation_module,
        "compress_recent_window",
        _fake_compress_recent_window,
    )

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4-6",
            adapter_kind="claude-bridge",
            tool_compatible=True,
            messages=original_messages,
        ),
        session,
    )

    assert (
        prepared.behavior_signals["tok_history_pairing_safety_degraded"] == 1
    )
    assert prepared.body["messages"] != invalid_recent


def test_prepare_request_applies_stream_recovery_history_floor(
    tmp_path, monkeypatch
):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._stream_recovery_history_floor_budget = 1
    original_messages = [
        {"role": "user", "content": "Earlier context"},
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
            "role": "tool_result",
            "tool_use_id": "tool_1",
            "content": "file contents",
        },
        {"role": "user", "content": "Please summarize."},
    ]

    called = {"compress_history": 0}

    def _fail_if_called(*args, **kwargs):
        del args, kwargs
        called["compress_history"] += 1
        raise AssertionError("compress_history should not run with floor")

    monkeypatch.setattr(
        request_preparation_module,
        "compress_history",
        _fail_if_called,
    )

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4-6",
            adapter_kind="claude-bridge",
            tool_compatible=True,
            messages=original_messages,
        ),
        session,
    )

    assert called["compress_history"] == 0
    assert (
        prepared.behavior_signals["stream_recovery_history_floor_applied"] == 1
    )
    assert (
        prepared.behavior_signals["stream_recovery_history_floor_kept_context"]
        == 1
    )


def test_stream_recovery_history_floor_leaves_safe_outgoing_body(
    tmp_path,
):
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    session._stream_recovery_history_floor_budget = 1
    runtime = UniversalTokRuntime()
    messages = _provider_sensitive_large_tool_batch_messages()

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            messages=messages,
            system="System context",
            adapter_kind="claude-bridge",
            tool_compatible=True,
        ),
        session,
    )

    assert (
        prepared.behavior_signals["stream_recovery_history_floor_applied"] == 1
    )
    assert validate_anthropic_outgoing_bridge_body(prepared.body) == []
    kept = prepared.body["messages"]
    assert len(kept) == 3
    assert kept[0]["role"] == "user"
    assert kept[1]["role"] == "assistant"
    assert any(
        isinstance(block, dict)
        and block.get("type") == "tool_use"
        and block.get("id") == "toolu_tail_1"
        for block in kept[1].get("content", [])
    )
    assert kept[2]["role"] == "user"
    assert any(
        isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id") == "toolu_tail_1"
        for block in kept[2].get("content", [])
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


def test_canonicalization_sanitizes_invalid_tool_ids_and_matching_results():
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01 bad/id",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01 bad/id",
                        "content": "file contents",
                    }
                ],
            },
        ],
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert changed is True
    rewritten_id = canonical["messages"][0]["content"][0]["id"]
    assert re.fullmatch(r"[A-Za-z0-9_-]+", rewritten_id)
    assert (
        canonical["messages"][1]["content"][0]["tool_use_id"] == rewritten_id
    )
    assert signals["tok_bridge_tool_id_sanitized"] == 1
    assert signals["tok_bridge_invalid_tool_id_seen"] == 1
    assert signals["tok_bridge_tool_result_id_rewritten"] == 1
    assert signals["tok_bridge_tool_result_rewrite_complete"] == 1
    assert validate_anthropic_bridge_body(canonical) == []


def test_canonicalization_synthesizes_blank_tool_ids_and_repairs_pairing():
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    },
                    {
                        "type": "tool_use",
                        "name": "view_file",
                        "input": {"path": "src/tok/runtime.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "",
                        "content": "first",
                    },
                    {
                        "type": "tool_result",
                        "content": "second",
                    },
                ],
            },
        ],
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert changed is True
    ids = [block["id"] for block in canonical["messages"][0]["content"]]
    assert len(ids) == 2
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in ids)
    assert canonical["messages"][1]["content"][0]["tool_use_id"] == ids[0]
    assert canonical["messages"][1]["content"][1]["tool_use_id"] == ids[1]
    assert signals["tok_bridge_blank_tool_id_synthesized"] == 2
    assert signals["tok_bridge_tool_result_id_rewritten"] == 2
    assert signals["tok_bridge_tool_result_pairing_repaired"] == 2
    assert validate_anthropic_bridge_body(canonical) == []


def test_canonicalization_generates_unique_ids_for_multiple_invalid_tool_ids():
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "bad id",
                        "name": "view_file",
                        "input": {"path": "src/tok/gateway.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "bad/id",
                        "name": "view_file",
                        "input": {"path": "src/tok/runtime.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "bad id",
                        "content": "first",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "bad/id",
                        "content": "second",
                    },
                ],
            },
        ],
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert changed is True
    ids = [block["id"] for block in canonical["messages"][0]["content"]]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in ids)
    assert canonical["messages"][1]["content"][0]["tool_use_id"] == ids[0]
    assert canonical["messages"][1]["content"][1]["tool_use_id"] == ids[1]
    assert signals["tok_bridge_tool_id_sanitized"] == 2
    assert signals["tok_bridge_tool_result_id_rewritten"] == 2
    assert signals["tok_bridge_invalid_tool_id_seen"] == 2
    assert signals["tok_bridge_tool_result_rewrite_complete"] == 1


def test_canonicalization_reorders_recoverable_tool_results_to_assistant_order():
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
                    {
                        "type": "tool_use",
                        "id": "tool_2",
                        "name": "grep",
                        "input": {"pattern": "TODO"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_2",
                        "content": "grep output",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "file output",
                    },
                    {"type": "text", "text": "Continue."},
                ],
            },
        ],
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert changed is True
    assert canonical["messages"][1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "tool_1",
            "content": "file output",
        },
        {
            "type": "tool_result",
            "tool_use_id": "tool_2",
            "content": "grep output",
        },
    ]
    assert canonical["messages"][2]["content"] == [
        {"type": "text", "text": "Continue."}
    ]
    assert signals["tok_bridge_tool_result_order_repaired"] == 1
    assert signals["tok_bridge_tool_result_pairing_repaired"] == 1
    assert signals["tok_bridge_user_tool_result_text_split"] == 1
    assert validate_anthropic_bridge_body(canonical) == []


def test_canonicalization_leaves_unrecoverable_nonempty_tool_result_ids_invalid():
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
                    {
                        "type": "tool_use",
                        "id": "tool_2",
                        "name": "grep",
                        "input": {"pattern": "TODO"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_2",
                        "content": "grep output",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "unknown_tool",
                        "content": "unknown output",
                    },
                ],
            },
        ],
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert changed is False
    assert signals == {}
    failures = validate_anthropic_bridge_body(canonical)
    assert "tool_result_unknown_tool_use_id" in failures
    assert "tool_result_not_immediately_after_assistant_tool_use" in failures
    assert (
        "assistant_tool_use_missing_next_tool_result" in failures
        or "assistant_tool_use_incomplete_next_tool_result_coverage"
        in failures
    )


def test_quarantine_invalid_tool_history_preserves_surrounding_text():
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I inspected the repo."},
                {
                    "type": "tool_use",
                    "id": "",
                    "name": "",
                    "input": {},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "",
                    "content": "bad payload",
                },
                {"type": "text", "text": "Continue from preserved context."},
            ],
        },
    ]

    quarantined, changed, signals = quarantine_invalid_tool_history_messages(
        messages
    )

    assert changed is True
    assert quarantined == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "I inspected the repo."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Continue from preserved context.",
                }
            ],
        },
    ]
    assert signals["tok_bridge_invalid_tool_history_quarantined"] == 1
    assert signals["tok_bridge_quarantined_tool_use_blocks"] == 1
    assert signals["tok_bridge_quarantined_tool_result_blocks"] == 1


def test_strict_bridge_validation_rejects_invalid_tool_id_shapes():
    failures = validate_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "bad id",
                            "name": "view_file",
                            "input": {"path": "src/tok/gateway.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "bad id",
                            "content": "file contents",
                        }
                    ],
                },
            ],
        }
    )

    assert "invalid_tool_use_block" in failures
    assert "invalid_tool_result_block" in failures


def test_bridge_strict_failure_signals_classify_invalid_and_pairing_failures():
    signals = bridge_strict_failure_signals(
        [
            "invalid_tool_use_block",
            "invalid_tool_result_block",
            "tool_result_unknown_tool_use_id",
        ]
    )

    assert signals["tok_bridge_strict_failure"] == 1
    assert signals["tok_bridge_strict_invalid_tool_use_block"] == 1
    assert signals["tok_bridge_strict_invalid_tool_result_block"] == 1
    assert signals["tok_bridge_strict_tool_result_unknown_tool_use_id"] == 1
    assert signals["tok_bridge_strict_pairing_or_ordering_failure"] == 1


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
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": "file contents",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the result"},
            ],
        },
    ]
    assert signals.get("tok_bridge_thinking_block_dropped", 0) == 1
    assert signals.get("tok_bridge_unsupported_block_dropped", 0) == 1
    assert signals.get("tok_bridge_user_tool_result_text_split", 0) == 1


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

    assert "tool_result_unknown_tool_use_id" in failures
    assert "tool_result_not_immediately_after_assistant_tool_use" in failures


def test_canonicalization_splits_mixed_user_message_after_tool_results():
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
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "file contents",
                    },
                    {"type": "text", "text": "Summarize it."},
                ],
            },
        ],
    }

    canonical, changed, signals = canonicalize_anthropic_bridge_body(body)

    assert changed is True
    assert signals.get("tok_bridge_user_tool_result_text_split", 0) == 1
    assert canonical["messages"] == [
        body["messages"][0],
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
        {
            "role": "user",
            "content": [{"type": "text", "text": "Summarize it."}],
        },
    ]
    assert validate_anthropic_bridge_body(canonical) == []


def test_strict_bridge_validation_rejects_unsplit_mixed_tool_result_and_text():
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
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file contents",
                        },
                        {"type": "text", "text": "Summarize it."},
                    ],
                },
            ],
        }
    )

    assert "tool_result_not_immediately_after_assistant_tool_use" in failures


def test_canonicalization_splits_interleaved_user_text_and_tool_result_deterministically():
    canonical, changed, signals = canonicalize_anthropic_bridge_body(
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
                        {"type": "text", "text": "Before"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file contents",
                        },
                        {"type": "text", "text": "After"},
                    ],
                },
            ],
        }
    )

    assert changed is True
    assert signals.get("tok_bridge_user_tool_result_text_split", 0) == 1
    assert canonical["messages"][1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tool_1",
                "content": "file contents",
            }
        ],
    }
    assert canonical["messages"][2] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Before"},
            {"type": "text", "text": "After"},
        ],
    }
    assert validate_anthropic_bridge_body(canonical) == []


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
    messages: list[dict[str, Any]] = [
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

    summary = cast(dict[str, Any], summarize_message_structure(messages))

    assert summary["unsupported_blocks"] == {
        "thinking": 1,
        "redacted_thinking": 1,
    }
    assert summary["tool_use_blocks"] == 1
