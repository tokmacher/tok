from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, cast

from tok.compression import text_of
from tok.runtime.core import RuntimeSession
from tok.runtime.types import RuntimeRequest

logger = logging.getLogger("tok.runtime")


def _snapshot_latest_assistant_thinking(
    messages: list[dict[str, Any]],
) -> str | None:
    import hashlib
    import json

    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        thinking_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
        ]
        if not thinking_blocks:
            continue
        content_json = json.dumps(content, ensure_ascii=False, sort_keys=True)
        content_hash = hashlib.sha256(content_json.encode()).hexdigest()
        block_types = [str(b.get("type", "")) for b in content if isinstance(b, dict)]
        snapshot = json.dumps(
            {
                "full_content": content,
                "content_hash": content_hash,
                "block_types": block_types,
            },
            ensure_ascii=False,
        )
        return snapshot
    return None


def _has_exact_search_evidence(evidence_keys: set[str]) -> bool:
    for key in evidence_keys:
        if str(key).startswith("search|"):
            return True
    return False


@dataclass
class Step1Result:
    body: dict[str, Any] = field(default_factory=dict)
    original_body: dict[str, Any] = field(default_factory=dict)
    thinking_snapshot: str | None = None
    last_user_msg: str = ""
    is_bridge_adapter: bool = False
    initial_answer_facts_present: bool = False
    initial_exact_search_evidence_present: bool = False
    compressed: bool = False
    pre_existing_session_signals: dict[str, int] = field(default_factory=dict)
    seen_mutation_pairs: set[tuple[str, str]] = field(default_factory=set)


def run_step_1(
    request: RuntimeRequest,
    session: RuntimeSession,
) -> Step1Result:
    session._request_has_tools = bool(request.request_has_tools)
    session._answer_phase_expected_this_turn = False
    session._natural_response_acceptable_this_turn = False

    initial_answer_facts_present = any(
        entry.value.startswith("answer_")
        for bucket in (session.bridge_memory.hot, session.bridge_memory.durable)
        for entry in bucket.get("facts", [])
    )
    initial_exact_search_evidence_present = _has_exact_search_evidence(
        session._first_exact_evidence_seen | session._pending_exact_evidence_keys
    )

    body: dict[str, Any] = {
        "model": request.model,
        "messages": copy.deepcopy(request.messages),
    }
    if request.system is not None:
        body["system"] = copy.deepcopy(request.system)
    original_body = copy.deepcopy(body)

    _thinking_snapshot = _snapshot_latest_assistant_thinking(request.messages)

    _current_message_count = len(body.get("messages", []))
    _prev_message_count = session._last_request_message_count
    if _prev_message_count > 10 and _current_message_count > 0 and _current_message_count < _prev_message_count * 0.7:
        session.pending_behavior_signals["tok_context_compression_detected"] = 1
        logger.info(
            "tok_context_compression_detected: messages %d -> %d (%.0f%% shrinkage)",
            _prev_message_count,
            _current_message_count,
            (1 - _current_message_count / _prev_message_count) * 100,
        )
    session._last_request_message_count = _current_message_count

    last_user_msg = ""
    if request.messages:
        for m in reversed(request.messages):
            if m.get("role") == "user":
                last_user_msg = text_of(cast("Any", m.get("content", "")))
                break

    is_bridge_adapter = request.uses_bridge_profile

    return Step1Result(
        body=body,
        original_body=original_body,
        thinking_snapshot=_thinking_snapshot,
        last_user_msg=last_user_msg,
        is_bridge_adapter=is_bridge_adapter,
        initial_answer_facts_present=initial_answer_facts_present,
        initial_exact_search_evidence_present=initial_exact_search_evidence_present,
        compressed=False,
        pre_existing_session_signals=dict(session.pending_behavior_signals),
        seen_mutation_pairs=set(),
    )
