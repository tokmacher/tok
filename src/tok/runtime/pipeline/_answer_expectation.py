from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from tok.compression import text_of
from tok.runtime._context_fidelity import extract_requested_answer_labels
from tok.runtime.core import RuntimeSession


def record_structured_answer_expectation(
    session: RuntimeSession,
    body: dict[str, Any],
) -> None:
    latest_user_prompt = ""
    messages = body.get("messages", [])
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).strip() != "user":
                continue
            latest_user_prompt = text_of(cast("Any", message.get("content", ""))).strip()
            if latest_user_prompt:
                break
    session._last_user_prompt_text = latest_user_prompt
    session._last_user_prompt_labels = extract_requested_answer_labels(latest_user_prompt)


def restore_latest_assistant_thinking(
    messages: list[dict[str, Any]],
    snapshot: str | None,
) -> bool:
    if snapshot is None:
        return False
    try:
        snapshot_data = json.loads(snapshot)
        original_content = snapshot_data.get("full_content")
        original_hash = snapshot_data.get("content_hash")
        original_block_types = snapshot_data.get("block_types")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False
    if not original_content or not original_hash or not original_block_types:
        return False

    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if not any(isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"} for b in content):
            continue

        msg["content"] = original_content
        restored_content_json = json.dumps(original_content, ensure_ascii=False, sort_keys=True)
        restored_hash = hashlib.sha256(restored_content_json.encode()).hexdigest()
        return bool(restored_hash == original_hash)
    return False
