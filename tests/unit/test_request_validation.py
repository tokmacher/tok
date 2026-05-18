import re
from typing import Any, NoReturn, cast

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
    assistant_content = [
        *tool_uses[:9],
        {"type": "text", "text": "Collecting evidence."},
        *tool_uses[9:],
    ]
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


def _broad_audit_file_read_messages(count: int = 8) -> list[dict[str, Any]]:
    tool_uses = [
        {
            "type": "tool_use",
            "id": f"toolu_audit_{index}",
            "name": "read_file",
            "input": {"path": f"src/tok/audit_{index}.py"},
        }
        for index in range(count)
    ]
    tool_results = [
        {
            "type": "tool_result",
            "tool_use_id": f"toolu_audit_{index}",
            "content": f"def marker_{index}():\n    return {index}\n",
        }
        for index in range(count)
    ]
    return [
        {"role": "user", "content": [{"type": "text", "text": "Audit the bridge implementation."}]},
        {"role": "assistant", "content": tool_uses},
        {"role": "user", "content": tool_results},
    ]


def test_top_level_tool_result_is_rewritten_and_merged_into_user_blocks() -> None:
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

    canonical, changed, _signals = canonicalize_anthropic_bridge_messages(messages)

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


def test_assistant_tool_use_blocks_remain_assistant_messages() -> None:
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

    canonical, changed, _signals = canonicalize_anthropic_bridge_body(body)

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


def test_strict_bridge_validation_rejects_empty_messages_after_canonicalization() -> None:
    canonical, _, _ = canonicalize_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "   "}],
        }
    )

    failures = validate_anthropic_bridge_body(canonical)

    assert failures == ["empty_messages"]


def test_canonicalization_falls_back_to_original_on_validation_failure() -> None:
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


def test_strict_bridge_validation_rejects_cross_role_tool_shapes() -> None:
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


def test_strict_bridge_validation_rejects_text_before_tool_result_in_user_message() -> None:
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


def test_strict_bridge_validation_rejects_unknown_tool_result_ids() -> None:
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


def test_strict_bridge_validation_rejects_tool_results_separated_from_tool_use() -> None:
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


def test_strict_bridge_validation_rejects_missing_next_turn_tool_results() -> None:
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


def test_strict_bridge_validation_rejects_incomplete_next_turn_tool_results() -> None:
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

    assert "assistant_tool_use_incomplete_next_tool_result_coverage" in failures


def test_strict_bridge_validation_rejects_out_of_order_next_turn_tool_results() -> None:
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
    assert "assistant_tool_use_incomplete_next_tool_result_coverage" in failures


def test_strict_bridge_validation_accepts_alternating_multi_turn_tool_pairs() -> None:
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Inspect concurrency path."}],
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


def test_collect_provider_sensitivity_risks_flags_terminal_large_tool_batch() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Inspect concurrency path."}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.py"}},
                {"type": "text", "text": "Reading"},
                {"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {"path": "b.py"}},
            ],
        },
    ]

    from tok.runtime.pipeline.bridge_message_validation import (
        collect_provider_sensitivity_risks,
    )

    risks = collect_provider_sensitivity_risks(messages, large_tool_batch_threshold=2)
    assert risks.get("provider_sensitive_large_tool_use_batch_unterminated", 0) == 1
    assert risks.get("provider_sensitive_large_tool_use_text_interleaving", 0) == 0


def test_outgoing_bridge_validation_flags_large_interleaved_tool_use_batch() -> None:
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": _provider_sensitive_large_tool_batch_messages(),
    }

    assert validate_anthropic_bridge_body(body) == []
    assert set(validate_anthropic_outgoing_bridge_body(body)) == {
        "provider_sensitive_large_tool_use_text_interleaving",
        "provider_sensitive_assistant_tool_use_text_interleaving",
    }
    summary = summarize_message_structure(body["messages"])
    assert cast("dict[str, Any]", summary)["provider_sensitivity_risks"] == {
        "assistant_large_tool_use_batch": 1,
        "assistant_tool_use_text_interleaving": 1,
        "assistant_large_tool_use_text_interleaving": 1,
        "provider_sensitive_large_tool_use_text_interleaving": 1,
    }


def test_outgoing_bridge_validation_flags_small_interleaved_tool_use_batch() -> None:
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 8192,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Inspect concurrency path."}],
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
    assert cast("dict[str, Any]", summary)["provider_sensitivity_risks"] == {"assistant_tool_use_text_interleaving": 1}


def test_strict_bridge_validation_rejects_alternating_multi_turn_reordered_first_exchange() -> None:
    body = {
        "model": "claude-sonnet-4",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Inspect concurrency path."}],
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
    assert "assistant_tool_use_incomplete_next_tool_result_coverage" in failures


def test_bridge_tail_with_adaptive_keep_zero_still_strictly_valid(
    tmp_path,
) -> None:
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


def test_tool_heavy_bridge_request_compresses_without_pairing_degradation(
    tmp_path,
) -> None:
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Help me with these files"},
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
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t0",
                    "content": "file 0 " * 120,
                }
            ],
        },
        {"role": "assistant", "content": "I found the issue."},
        {"role": "user", "content": "Now continue"},
        {"role": "assistant", "content": "Done."},
    ]

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            adapter_kind="claude-bridge",
            tool_compatible=True,
            messages=messages,
        ),
        session,
    )
    canonical, _changed, _signals = canonicalize_anthropic_bridge_body(prepared.body)

    assert prepared.compressed is True
    assert prepared.behavior_signals.get("bridge_history_cut_search_used", 0) == 1
    assert prepared.behavior_signals.get("tok_history_pairing_safety_degraded", 0) == 0
    assert prepared.behavior_signals.get("tool_compatible_compression", 0) == 1
    assert len(prepared.body["messages"]) < len(messages)
    assert validate_anthropic_bridge_body(canonical) == []
    assert {message["role"] for message in canonical["messages"]} <= {
        "user",
        "assistant",
    }


def test_prepare_request_bridge_cut_search_extends_to_valid_suffix(tmp_path, monkeypatch) -> None:
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    original_messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Help me with these files and keep going until you find the cleanest safe cut.",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_0",
                    "name": "view_file",
                    "input": {"path": "src/tok/file_0.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_0",
                    "content": "file 0 " * 80,
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "view_file",
                    "input": {"path": "src/tok/file_1.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": "file 1 " * 80,
                }
            ],
        },
        {"role": "assistant", "content": "I found the issue."},
        {"role": "user", "content": "Now continue"},
        {"role": "assistant", "content": "Done."},
    ]

    invalid_recent = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "view_file",
                    "input": {"path": "src/tok/file_1.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": "file 1 " * 80,
                }
            ],
        },
    ]

    valid_recent = [
        {
            "content": [
                {
                    "type": "text",
                    "text": "Now continue",
                }
            ],
            "role": "user",
        },
        {
            "content": [
                {
                    "type": "text",
                    "text": "Done.",
                }
            ],
            "role": "assistant",
        },
    ]

    compress_history_calls: list[int] = []

    def _fake_compress_history(messages, keep_turns=2, profile=None, prune_tool_results=False):
        del messages, prune_tool_results
        compress_history_calls.append(keep_turns)
        bridge_search_enabled = bool(profile and profile.get("_bridge_cut_search"))
        assert bridge_search_enabled is True
        if keep_turns == 2:
            return invalid_recent, "compressed tail", frozenset()
        if keep_turns == 1:
            return valid_recent, "compressed tail", frozenset()
        return invalid_recent, "compressed tail", frozenset()

    def _fake_compress_recent_window(
        messages,
        tool_use_id_to_context=None,
        threshold=0,
        tool_compatible=False,
        first_exact_evidence_seen=None,
        preserve_exact_search_evidence=False,
        session_files_read=None,
        model_profile=None,
    ):
        del (
            tool_use_id_to_context,
            threshold,
            tool_compatible,
            first_exact_evidence_seen,
            preserve_exact_search_evidence,
            session_files_read,
            model_profile,
        )
        return messages, {}

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

    assert compress_history_calls[:2] == [4, 3]
    assert prepared.behavior_signals.get("bridge_history_cut_search_used", 0) == 1
    assert prepared.behavior_signals.get("bridge_history_cut_search_extended", 0) == 1
    assert prepared.behavior_signals.get("tok_history_pairing_safety_degraded", 0) == 0
    assert prepared.body["messages"] == valid_recent


def test_prepare_request_bridge_cut_preflight_advances_to_safe_suffix(tmp_path, monkeypatch) -> None:
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    original_messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Inspect the bridge."},
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
                    "content": "file contents " * 80,
                }
            ],
        },
        {"role": "user", "content": "Now continue"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "grep_search",
                    "input": {"pattern": "pairing"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_2",
                    "content": "grep results " * 80,
                }
            ],
        },
        {"role": "assistant", "content": "Done."},
    ]

    invalid_then_safe_recent = [
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
                    "content": "file contents " * 80,
                }
            ],
        },
        {"role": "user", "content": "Now continue"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "grep_search",
                    "input": {"pattern": "pairing"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_2",
                    "content": "grep results " * 80,
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done."}],
        },
    ]

    def _fake_compress_history(messages, keep_turns=2, profile=None, prune_tool_results=False):
        del messages, keep_turns, profile, prune_tool_results
        return invalid_then_safe_recent, "compressed tail", frozenset()

    def _fake_compress_recent_window(
        messages,
        tool_use_id_to_context=None,
        threshold=0,
        tool_compatible=False,
        first_exact_evidence_seen=None,
        preserve_exact_search_evidence=False,
        session_files_read=None,
        model_profile=None,
    ):
        del (
            tool_use_id_to_context,
            threshold,
            tool_compatible,
            first_exact_evidence_seen,
            preserve_exact_search_evidence,
            session_files_read,
            model_profile,
        )
        return messages, {}

    monkeypatch.setattr(request_preparation_module, "compress_history", _fake_compress_history)
    monkeypatch.setattr(request_preparation_module, "compress_recent_window", _fake_compress_recent_window)

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4-6",
            adapter_kind="claude-bridge",
            tool_compatible=True,
            messages=original_messages,
        ),
        session,
    )

    assert prepared.behavior_signals.get("tok_history_pairing_safety_degraded", 0) == 0
    assert prepared.body["messages"][0]["role"] == "user"
    assert prepared.body["messages"][0]["content"][0]["text"] == "Now continue"
    canonical, _changed, _signals = canonicalize_anthropic_bridge_body(prepared.body)
    assert validate_anthropic_bridge_body(canonical) == []


def test_prepare_request_discards_history_rewrite_that_breaks_pairing(tmp_path, monkeypatch, caplog) -> None:
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

    def _fake_compress_history(messages, keep_turns=2, profile=None, prune_tool_results=False):
        del messages, keep_turns, profile, prune_tool_results
        return invalid_recent, "compressed tail", frozenset()

    def _fake_compress_recent_window(
        messages,
        tool_use_id_to_context=None,
        threshold=0,
        tool_compatible=False,
        first_exact_evidence_seen=None,
        preserve_exact_search_evidence=False,
        session_files_read=None,
        model_profile=None,
    ):
        del (
            tool_use_id_to_context,
            threshold,
            tool_compatible,
            first_exact_evidence_seen,
            preserve_exact_search_evidence,
            session_files_read,
            model_profile,
        )
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

    assert prepared.behavior_signals["tok_history_pairing_safety_degraded"] == 1
    assert prepared.body["messages"] != invalid_recent
    warnings = [
        rec.message
        for rec in caplog.records
        if "tok_history_pairing_safety_degraded: rejecting compressed history" in rec.message
    ]
    assert len(warnings) == 1


def test_prepare_request_applies_stream_recovery_history_floor(tmp_path, monkeypatch) -> None:
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

    def _fail_if_called(*args, **kwargs) -> NoReturn:
        del args, kwargs
        called["compress_history"] += 1
        msg = "compress_history should not run with floor"
        raise AssertionError(msg)

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
    assert prepared.behavior_signals["stream_recovery_history_floor_applied"] == 1
    assert prepared.behavior_signals["stream_recovery_history_floor_kept_context"] == 1


def test_stream_recovery_history_floor_leaves_safe_outgoing_body(
    tmp_path,
) -> None:
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

    assert prepared.behavior_signals["stream_recovery_history_floor_applied"] == 1
    assert validate_anthropic_outgoing_bridge_body(prepared.body) == [
        "missing_max_tokens",
    ]
    kept = prepared.body["messages"]
    assert len(kept) == 3
    assert kept[0]["role"] == "user"
    assert kept[1]["role"] == "assistant"
    assert any(
        isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == "toolu_tail_1"
        for block in kept[1].get("content", [])
    )
    assert kept[2]["role"] == "user"
    assert any(
        isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") == "toolu_tail_1"
        for block in kept[2].get("content", [])
    )


def test_first_turn_broad_audit_suppresses_tok_additions_and_preserves_tool_results(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    runtime = UniversalTokRuntime()
    messages = _broad_audit_file_read_messages()

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            messages=messages,
            system="System context",
            adapter_kind="claude-bridge",
            tool_compatible=True,
            request_policy="natural_first",
            request_has_tools=True,
        ),
        session,
    )

    assert prepared.behavior_signals["broad_audit_tok_additions_suppressed"] == 1
    assert prepared.behavior_signals["broad_audit_tool_result_compression_skipped"] == 1
    assert prepared.behavior_signals["broad_audit_history_skipped"] == 1
    assert prepared.behavior_signals["broad_audit_system_additions_skipped"] == 1
    assert prepared.body["system"] == "System context"
    assert prepared.body["messages"] == messages
    assert prepared.compressed is False


def test_canonicalization_preserves_thinking_blocks_in_assistant_message() -> None:
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

    assert changed is False
    assert canonical["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me analyze this...",
                },
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "view_file",
                    "input": {"path": "src/tok/gateway.py"},
                },
            ],
        }
    ]
    assert signals.get("tok_bridge_thinking_block_dropped", 0) == 0
    assert signals.get("tok_bridge_unsupported_block_dropped", 0) == 0


def test_canonicalization_sanitizes_invalid_tool_ids_and_matching_results() -> None:
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
    assert canonical["messages"][1]["content"][0]["tool_use_id"] == rewritten_id
    assert signals["tok_bridge_tool_id_sanitized"] == 1
    assert signals["tok_bridge_invalid_tool_id_seen"] == 1
    assert signals["tok_bridge_tool_result_id_rewritten"] == 1
    assert signals["tok_bridge_tool_result_rewrite_complete"] == 1
    assert validate_anthropic_bridge_body(canonical) == ["first_message_not_user"]


def test_canonicalization_synthesizes_blank_tool_ids_and_repairs_pairing() -> None:
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
    assert validate_anthropic_bridge_body(canonical) == ["first_message_not_user"]


def test_canonicalization_generates_unique_ids_for_multiple_invalid_tool_ids() -> None:
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


def test_canonicalization_reorders_recoverable_tool_results_to_assistant_order() -> None:
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
    assert canonical["messages"][2]["content"] == [{"type": "text", "text": "Continue."}]
    assert signals["tok_bridge_tool_result_order_repaired"] == 1
    assert signals["tok_bridge_tool_result_pairing_repaired"] == 1
    assert signals["tok_bridge_user_tool_result_text_split"] == 1
    assert validate_anthropic_bridge_body(canonical) == ["first_message_not_user"]


def test_canonicalization_leaves_unrecoverable_nonempty_tool_result_ids_invalid() -> None:
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
        or "assistant_tool_use_incomplete_next_tool_result_coverage" in failures
    )


def test_quarantine_invalid_tool_history_preserves_surrounding_text() -> None:
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

    quarantined, changed, signals = quarantine_invalid_tool_history_messages(messages)

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


def test_quarantine_invalid_tool_history_removes_partial_tool_result_exchange() -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I inspected the repo."},
                {"type": "tool_use", "id": "tool_1", "name": "view_file", "input": {"path": "a.py"}},
                {"type": "tool_use", "id": "tool_2", "name": "grep_search", "input": {"pattern": "TODO"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": "file contents"},
                {"type": "text", "text": "Continue from preserved context."},
            ],
        },
    ]

    quarantined, changed, signals = quarantine_invalid_tool_history_messages(messages)

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
    assert signals["tok_bridge_quarantined_tool_use_blocks"] == 2
    assert signals["tok_bridge_quarantined_tool_result_blocks"] == 1


def test_strict_bridge_validation_rejects_invalid_tool_id_shapes() -> None:
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


def test_bridge_strict_failure_signals_classify_invalid_and_pairing_failures() -> None:
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


def test_assistant_message_with_only_thinking_blocks_is_preserved() -> None:
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
    assert len(canonical["messages"]) == 2
    assert canonical["messages"][0]["role"] == "assistant"
    assert canonical["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "internal thought"},
        {"type": "redacted_thinking", "data": "..."},
    ]
    assert canonical["messages"][1]["role"] == "user"
    assert signals.get("tok_bridge_thinking_block_dropped", 0) == 0
    assert signals.get("tok_bridge_unsupported_block_dropped", 0) == 0


def test_user_message_retains_supported_blocks_after_filtering() -> None:
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
    assert signals.get("tok_bridge_thinking_block_dropped", 0) == 0
    assert signals.get("tok_bridge_unsupported_block_dropped", 0) == 1
    assert signals.get("tok_bridge_user_tool_result_text_split", 0) == 1


def test_canonicalized_mixed_user_message_is_rejected_by_strict_bridge_validation() -> None:
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


def test_canonicalization_splits_mixed_user_message_after_tool_results() -> None:
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
    assert validate_anthropic_bridge_body(canonical) == ["first_message_not_user"]


def test_canonicalization_splits_interleaved_user_text_and_tool_result_deterministically() -> None:
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
    assert validate_anthropic_bridge_body(canonical) == ["first_message_not_user"]


def test_strict_bridge_validation_rejects_unsupported_block_types() -> None:
    failures = validate_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "unknown_block",
                            "text": "this should have been removed",
                        }
                    ],
                }
            ],
        }
    )

    assert "unsupported_block_type" in failures


def test_strict_bridge_validation_rejects_invalid_system_blocks() -> None:
    """Test that invalid system blocks are caught gracefully."""
    failures = validate_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "system": [
                {"type": "text", "text": "Valid block"},
                {
                    "type": "invalid_block",
                    "data": "This should fail validation",
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                }
            ],
        }
    )
    assert failures == ["invalid_system_block"]


def test_strict_bridge_validation_rejects_malformed_system_blocks() -> None:
    """Test that malformed system blocks are caught gracefully."""
    failures = validate_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "system": [
                {
                    "type": "tool_use",
                    "id": "",  # Empty ID should fail validation
                    "name": "test_tool",
                    "input": {"key": "value"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                }
            ],
        }
    )
    assert failures == ["invalid_system_block"]


def test_strict_bridge_validation_rejects_malformed_system_blocks_v2() -> None:
    """Test that malformed system blocks are caught gracefully."""
    failures = validate_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "system": [
                {
                    "type": "tool_use",
                    "id": "",  # Empty ID should fail validation
                    "name": "test_tool",
                    "input": {"key": "value"},
                }
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )
    assert failures == ["invalid_system_block"]
    """Test that valid system blocks pass validation."""
    failures = validate_anthropic_bridge_body(
        {
            "model": "claude-sonnet-4",
            "system": [
                {"type": "text", "text": "You are a helpful assistant."},
                {"type": "thinking", "thinking": "Let me think about this..."},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                }
            ],
        }
    )
    assert failures == []


def test_summarize_message_structure_includes_unsupported_blocks() -> None:
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

    summary = cast("dict[str, Any]", summarize_message_structure(messages))

    assert summary["unsupported_blocks"] == {}
    assert summary["tool_use_blocks"] == 1


def test_bridge_preflight_safe_recent_suffix_with_tool_result_only_start(tmp_path) -> None:
    """Parallel agents produce tool_result-only user messages at the start of the recent suffix.

    The safe suffix finder should still find a valid candidate when the
    tool_result IDs all match tool_use IDs from preceding assistant messages.
    """
    from tok.runtime._request_preparation import _bridge_preflight_safe_recent_suffix

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "agent_1_t1", "content": "result 1"},
                {"type": "tool_result", "tool_use_id": "agent_1_t2", "content": "result 2"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I found something."},
                {"type": "tool_use", "id": "agent_1_t3", "name": "bash", "input": {"command": "ls"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "agent_1_t3", "content": "file list"},
            ],
        },
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Next step"},
        {"role": "assistant", "content": "Continuing."},
    ]

    result = _bridge_preflight_safe_recent_suffix(messages)
    assert result is not None
    assert str(result[0].get("content", "")).strip() == "Next step"


def test_bridge_preflight_safe_recent_suffix_tool_result_only_with_complete_pairing(tmp_path) -> None:
    """When tool_result-only user messages have complete pairing within the suffix,
    the fallback should select the earliest one as the safe starting point.

    Dangling tool_results (whose tool_uses were cut from history) are allowed
    as long as every tool_use within the suffix has a matching tool_result.
    """
    from tok.runtime._request_preparation import _bridge_preflight_safe_recent_suffix

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "dangling_t0", "content": "old result"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "pa_t1", "name": "view_file", "input": {"path": "x.py"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "pa_t1", "content": "file content"},
            ],
        },
        {"role": "assistant", "content": "Done."},
    ]

    result = _bridge_preflight_safe_recent_suffix(messages)
    assert result is not None
    assert len(result) == len(messages)
    assert result[0] is messages[0]


def test_bridge_preflight_safe_recent_suffix_all_tool_results_no_safe_pairing(tmp_path) -> None:
    """When tool_result IDs don't match any tool_use in the suffix, return None."""
    from tok.runtime._request_preparation import _bridge_preflight_safe_recent_suffix

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "orphan_id", "content": "orphan result"},
            ],
        },
        {"role": "assistant", "content": "No matching tool use."},
    ]

    result = _bridge_preflight_safe_recent_suffix(messages)
    assert result is None


def test_tool_result_only_suffix_safe_pairing_valid() -> None:
    """Paired tool_result-only suffix should validate successfully."""
    from tok.runtime._request_preparation import _tool_result_only_suffix_has_safe_pairing

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t3", "name": "bash", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t3", "content": "r3"},
            ],
        },
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Next"},
        {"role": "assistant", "content": "Continuing."},
    ]

    assert _tool_result_only_suffix_has_safe_pairing(messages) is True

    messages_no_tool_use: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "orphan", "content": "r"},
            ],
        },
        {"role": "assistant", "content": "No tool use here."},
    ]

    assert _tool_result_only_suffix_has_safe_pairing(messages_no_tool_use) is False

    messages_unmatched_tool_use: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t2", "name": "bash", "input": {}},
            ],
        },
    ]

    assert _tool_result_only_suffix_has_safe_pairing(messages_unmatched_tool_use) is False


def test_parallel_agent_bridge_compression_does_not_degrade(tmp_path) -> None:
    """Full bridge request with parallel agent interleaving should compress without fallback.

    This reproduces the scenario where plan mode + parallel agents produce
    rapid-fire tool_use/tool_result pairs that previously caused:
    - tok_history_cut_point_missing_with_tools
    - tok_history_cut_blocked_tool_result
    - tok_fallback_activated cascade
    """
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")

    tool_calls = 8
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Analyze all files"},
    ]
    for i in range(tool_calls):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"agent_t{i}",
                        "name": "view_file",
                        "input": {"path": f"src/tok/file_{i}.py"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"agent_t{i}",
                        "content": f"file {i} content " * 50,
                    }
                ],
            }
        )

    messages.append({"role": "assistant", "content": "Analysis complete."})
    messages.append({"role": "user", "content": "Now write tests"})
    messages.append({"role": "assistant", "content": "Writing tests."})

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            adapter_kind="claude-bridge",
            tool_compatible=True,
            messages=messages,
        ),
        session,
    )
    canonical, _changed, _signals = canonicalize_anthropic_bridge_body(prepared.body)

    assert prepared.behavior_signals.get("tok_history_pairing_safety_degraded", 0) == 0
    if prepared.compressed:
        assert validate_anthropic_bridge_body(canonical) == []
        assert len(prepared.body["messages"]) < len(messages)


def test_bridge_recent_suffix_rejects_orphaned_tool_result_no_assistant() -> None:
    """Suffix with tool_result but no matching tool_use assistant must be rejected."""
    from tok.runtime._request_preparation import _bridge_recent_suffix_has_safe_pairing

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "orphan"},
            ],
        },
    ]
    assert _bridge_recent_suffix_has_safe_pairing(messages) is False


def test_bridge_recent_suffix_allows_paired_tool_result_in_suffix() -> None:
    """Suffix where tool_result has matching tool_use in the suffix should pass."""
    from tok.runtime._request_preparation import _bridge_recent_suffix_has_safe_pairing

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            ],
        },
    ]
    assert _bridge_recent_suffix_has_safe_pairing(messages) is True


def test_bridge_preflight_safe_recent_suffix_rejects_assistant_tool_use_in_prefix() -> None:
    """When assistant with tool_use is before all user messages, no safe suffix exists."""
    from tok.runtime._request_preparation import _bridge_preflight_safe_recent_suffix

    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            ],
        },
    ]
    result = _bridge_preflight_safe_recent_suffix(messages)
    assert result is None


def test_bridge_preflight_safe_recent_suffix_preserves_skill_result_tail() -> None:
    """Do not cut between a skill/tool result and the following skill instructions."""
    from tok.runtime._request_preparation import _bridge_preflight_safe_recent_suffix

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "we were running /simplify"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "skill_1", "name": "Skill", "input": {"skill": "simplify"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "skill_1",
                    "content": "Successfully loaded skill",
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Simplify the current branch by reviewing the diff."}],
        },
    ]

    result = _bridge_preflight_safe_recent_suffix(messages)

    assert result is not None
    assert result[0] is messages[0]
    assert len(result) == len(messages)


def test_bridge_preflight_safe_recent_suffix_preserves_multi_skill_result_tail() -> None:
    """Do not start after only the last of multiple freshly loaded skills."""
    from tok.runtime._request_preparation import _bridge_preflight_safe_recent_suffix

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Use simplification and security review skills."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "skill_simplify", "name": "Skill", "input": {"skill": "simplify"}},
                {"type": "tool_use", "id": "skill_security", "name": "Skill", "input": {"skill": "security"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "skill_simplify",
                    "content": "Successfully loaded skill: simplify",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "skill_security",
                    "content": "Successfully loaded skill: security",
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Apply both loaded skills to the current diff."}],
        },
    ]

    result = _bridge_preflight_safe_recent_suffix(messages)

    assert result is not None
    assert result[0] is messages[0]
    assert len(result) == len(messages)
