"""Audit / dry-run mode for the Tok compression pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageHit:
    """Reserved for future per-message stage attribution."""

    key: str
    tokens_saved: int


@dataclass(frozen=True)
class MessageAudit:
    index: int
    role: str
    tokens_before: int
    tokens_after: int  # 0 for messages dropped by compress_history
    tokens_saved: int  # = tokens_before - tokens_after


@dataclass(frozen=True)
class AuditReport:
    total_tokens_before: int
    total_tokens_after: int
    total_tokens_saved: int
    messages: tuple[MessageAudit, ...]
    type_breakdown: dict[str, int]  # approximate token savings keyed by strategy


def _message_tokens(msg: dict[str, Any]) -> int:
    """Count tokens in a message dict. Mirrors replay_metrics._msg_text logic."""
    from tok.utils.token_utils import count_tokens

    content = msg.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_use":
                    parts.append(str(block.get("input", "")))
                elif block.get("type") == "tool_result":
                    parts.append(str(block.get("content", "")))
            else:
                parts.append(str(block))
        text = "\n".join(parts)
    else:
        text = str(content)
    return count_tokens(text)


def audit_messages(
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    keep_turns: int = 6,
) -> AuditReport:
    """Run the compression pipeline in dry-run mode and return an AuditReport.

    Does NOT make any API calls. Operates on a deep copy of messages so the
    original list is never mutated.

    Args:
        messages: Conversation messages in Claude API format.
        system_prompt: Optional system prompt (reserved; not currently threaded
            into compress_history).
        keep_turns: Number of recent turns to preserve verbatim. Default 6
            reflects a typical mid-session context.

    Returns:
        AuditReport with per-message token deltas and aggregate token savings.
    """
    from tok.compression import compress_history, compress_tool_results

    if not messages:
        return AuditReport(
            total_tokens_before=0,
            total_tokens_after=0,
            total_tokens_saved=0,
            messages=(),
            type_breakdown={},
        )

    # Snapshot token counts BEFORE compression on the original (unmodified) list.
    tokens_before = [_message_tokens(m) for m in messages]
    total_before = sum(tokens_before)

    # Work on a deep copy so the original is never modified.
    msgs_copy = copy.deepcopy(messages)

    # Stage 1: tool-result compression (mutates msgs_copy in place).
    # bypass_result_cache=True keeps each audit() call stateless.
    msgs_copy, breakdown_chars = compress_tool_results(msgs_copy, bypass_result_cache=True)

    # Stage 2: history truncation.
    recent, _tok_state, _suppressed = compress_history(msgs_copy, keep_turns=keep_turns)

    # Align recent (a suffix of msgs_copy) back to original message indices.
    # compress_history always cuts from the front: recent == msgs_copy[cut_index:]
    n_original = len(messages)
    n_recent = len(recent)
    dropped_count = n_original - n_recent

    tokens_after_list: list[int] = []
    for i in range(n_original):
        if i < dropped_count:
            tokens_after_list.append(0)
        else:
            tokens_after_list.append(_message_tokens(recent[i - dropped_count]))

    total_after = sum(tokens_after_list)

    # breakdown_chars values are character savings; convert to approximate tokens.
    # ~4 chars per token (cl100k_base). Matches replay_metrics.py line 143.
    type_breakdown = {k: max(0, v // 4) for k, v in breakdown_chars.items()}

    message_audits = tuple(
        MessageAudit(
            index=i,
            role=messages[i].get("role", ""),
            tokens_before=tokens_before[i],
            tokens_after=tokens_after_list[i],
            tokens_saved=tokens_before[i] - tokens_after_list[i],
        )
        for i in range(n_original)
    )

    return AuditReport(
        total_tokens_before=total_before,
        total_tokens_after=total_after,
        total_tokens_saved=total_before - total_after,
        messages=message_audits,
        type_breakdown=type_breakdown,
    )
