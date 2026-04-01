from typing import Any, cast

from tok.runtime.pipeline.request_validation import (
    canonicalize_anthropic_bridge_messages,
    validate_anthropic_bridge_body,
    summarize_message_structure,
)


def test_top_level_tool_result_rewritten_and_merged():
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Start work."},
        {"role": "tool_result", "tool_use_id": "t1", "content": "result 1"},
        {"role": "tool_result", "tool_use_id": "t2", "content": "result 2"},
        {"role": "user", "content": "Continue work."},
    ]
    canonical, changed, signals = canonicalize_anthropic_bridge_messages(
        messages
    )

    assert changed is True
    assert signals["tok_bridge_top_level_tool_result_rewritten"] == 2
    assert signals["tok_bridge_adjacent_user_merged"] == 1
    assert len(canonical) == 3
    assert canonical[0]["role"] == "user"
    assert len(canonical[1]["content"]) == 2
    assert canonical[0]["content"][0] == {
        "type": "text",
        "text": "Start work.",
    }
    assert canonical[1]["content"][0]["type"] == "tool_result"
    assert canonical[1]["content"][1]["type"] == "tool_result"
    assert canonical[2]["content"][0] == {
        "type": "text",
        "text": "Continue work.",
    }


def test_assistant_tool_use_preserved():
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Action!"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "test", "input": {}}
            ],
        },
    ]
    canonical, changed, signals = canonicalize_anthropic_bridge_messages(
        messages
    )
    assert changed is True
    assert len(canonical) == 2
    assert canonical[1]["role"] == "assistant"


def test_invalid_tool_ids_are_sanitized_after_adjacent_user_merge():
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Start work."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "bad/id",
                    "name": "test",
                    "input": {},
                }
            ],
        },
        {
            "role": "tool_result",
            "tool_use_id": "bad/id",
            "content": "result 1",
        },
        {"role": "user", "content": "Continue work."},
    ]
    canonical, changed, signals = canonicalize_anthropic_bridge_messages(
        messages
    )

    assert changed is True
    assert signals["tok_bridge_tool_id_sanitized"] == 1
    assert signals["tok_bridge_tool_result_id_rewritten"] == 1
    rewritten_id = canonical[1]["content"][0]["id"]
    assert rewritten_id == canonical[2]["content"][0]["tool_use_id"]


def test_validator_rejects_invalid_roles():
    body = {
        "model": "claude-3-5",
        "messages": [{"role": "system", "content": "oops"}],
    }
    failures = validate_anthropic_bridge_body(body)
    assert "invalid_top_level_role" in failures


def test_validator_rejects_cross_role_blocks():
    # User message with tool_use
    body1 = {
        "model": "claude-3-5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "test",
                        "input": {},
                    }
                ],
            }
        ],
    }
    failures1 = validate_anthropic_bridge_body(body1)
    assert "user_contains_tool_use" in failures1

    # Assistant message with tool_result
    body2 = {
        "model": "claude-3-5",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "ok",
                    }
                ],
            }
        ],
    }
    failures2 = validate_anthropic_bridge_body(body2)
    assert "assistant_contains_tool_result" in failures2


def test_validator_rejects_empty_content():
    body: dict[str, Any] = {
        "model": "claude-3-5",
        "messages": [{"role": "user", "content": []}],
    }
    assert "empty_content_blocks" in validate_anthropic_bridge_body(body)


def test_validator_rejects_user_tool_result_after_text():
    body = {
        "model": "claude-3-5",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "test",
                        "input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the result"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "res",
                    },
                ],
            },
        ],
    }

    failures = validate_anthropic_bridge_body(body)

    assert "user_tool_result_after_text" in failures


def test_validator_rejects_unknown_tool_use_ids():
    body = {
        "model": "claude-3-5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "missing",
                        "content": "res",
                    }
                ],
            }
        ],
    }

    failures = validate_anthropic_bridge_body(body)

    assert "tool_result_unknown_tool_use_id" in failures
    assert "tool_result_not_immediately_after_assistant_tool_use" in failures


def test_validator_rejects_tool_results_not_immediately_after_assistant_tool_use():
    body = {
        "model": "claude-3-5",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "test",
                        "input": {},
                    }
                ],
            },
            {"role": "user", "content": "Let me think first."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "res",
                    }
                ],
            },
        ],
    }

    failures = validate_anthropic_bridge_body(body)

    assert "tool_result_not_immediately_after_assistant_tool_use" in failures


def test_summarize_structure():
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "t", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "res"}
            ],
        },
    ]
    summary = cast(dict[str, Any], summarize_message_structure(messages))
    assert summary["count"] == 3
    assert summary["user_msgs"] == 2
    assert summary["assistant_msgs"] == 1
    assert summary["tool_use_blocks"] == 1
    assert summary["tool_result_blocks"] == 1
    assert summary["sequence"][0] == "user[str]"
    assert summary["sequence"][1] == "assistant[tool_use]"
    assert summary["sequence"][2] == "user[tool_result]"
    assert summary["field_shape_risks"] == {}
