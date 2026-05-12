"""Constants defining Tok bridge strict failure signals and registry access.

Packet 08 (0.1.9): centralize the strict failure → signal mapping used by
request validation without changing any names or behavior.
"""

from __future__ import annotations

_STRICT_FAILURE_SIGNAL_MAP: dict[str, str] = {
    "invalid_tool_use_block": "tok_bridge_strict_invalid_tool_use_block",
    "invalid_tool_result_block": "tok_bridge_strict_invalid_tool_result_block",
    "assistant_tool_use_missing_next_tool_result": "tok_bridge_strict_missing_next_tool_result",
    "assistant_tool_use_incomplete_next_tool_result_coverage": "tok_bridge_strict_incomplete_next_tool_result_coverage",
    "tool_result_unknown_tool_use_id": "tok_bridge_strict_tool_result_unknown_tool_use_id",
    "tool_result_not_immediately_after_assistant_tool_use": "tok_bridge_strict_tool_result_ordering_failure",
    "user_tool_result_after_text": "tok_bridge_strict_user_tool_result_after_text",
    "bridge_wire_model_invalid": "tok_bridge_strict_bridge_wire_model_invalid",
    "provider_sensitive_large_tool_use_text_interleaving": (
        "tok_bridge_strict_provider_sensitive_large_tool_use_text_interleaving"
    ),
    "first_message_not_user": "tok_bridge_strict_first_message_not_user",
    "missing_max_tokens": "tok_bridge_strict_missing_max_tokens",
    "invalid_max_tokens": "tok_bridge_strict_invalid_max_tokens",
}

_INVALID_TOOL_HISTORY_FAILURES = frozenset(
    {
        "invalid_tool_use_block",
        "invalid_tool_result_block",
        "assistant_tool_use_missing_next_tool_result",
        "assistant_tool_use_incomplete_next_tool_result_coverage",
        "tool_result_unknown_tool_use_id",
        "tool_result_not_immediately_after_assistant_tool_use",
        "user_tool_result_after_text",
        "bridge_wire_model_invalid",
    }
)

_RECOVERABLE_IMMEDIATE_PAIRING_FAILURES = frozenset(
    {
        "assistant_tool_use_missing_next_tool_result",
        "assistant_tool_use_incomplete_next_tool_result_coverage",
        "tool_result_not_immediately_after_assistant_tool_use",
        "user_tool_result_after_text",
        "tool_results_fragmented_across_user_messages",
        "empty_message_content",
        "empty_content_blocks",
        "missing_max_tokens",
        "invalid_max_tokens",
    }
)

_NON_BLOCKING_OUTGOING_FAILURES = frozenset(
    {
        "missing_max_tokens",
        "invalid_max_tokens",
        "first_message_not_user",
    }
)

_PROVIDER_SENSITIVE_FAILURES = frozenset(
    {
        "provider_sensitive_large_tool_use_text_interleaving",
        "provider_sensitive_large_tool_use_batch_unterminated",
        "provider_sensitive_assistant_tool_use_text_interleaving",
    }
)
