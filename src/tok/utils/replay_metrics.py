"""Replay fixture metrics utilities for regression tests and CLI tooling."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

from ..compression import compress_history, compress_tool_results
from ..runtime.memory.session_helpers import get_adaptive_keep_turns
from ..runtime.policy.smart_policy import (
    advance_state,
    initial_state,
    policy_for_model,
)
from ..runtime.pipeline.response_processing import response_contract_for_mode
from ..runtime.pipeline.tool_processing import (
    build_tool_use_id_to_context,
    collect_behavior_signals,
)

try:  # Token counting when tiktoken is available
    import tiktoken
except (
    Exception
):  # pragma: no cover - fallback path is covered via tests when missing
    tiktoken = None  # type: ignore[assignment]

__all__ = ["ReplayFixtureMetrics", "analyze_replay_fixture"]


@dataclass(frozen=True)
class ReplayFixtureMetrics:
    """Aggregated telemetry for a replay fixture."""

    fixture: Path
    lines: int
    total_before_tokens: int
    total_after_tokens: int
    type_savings_tokens: dict[str, int] = field(default_factory=dict)
    behavior_totals: dict[str, int] = field(default_factory=dict)

    @property
    def input_saved_tokens(self) -> int:
        return max(0, self.total_before_tokens - self.total_after_tokens)

    @property
    def savings_pct(self) -> float:
        if self.total_before_tokens == 0:
            return 0.0
        return (self.input_saved_tokens / self.total_before_tokens) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture": str(self.fixture),
            "lines": self.lines,
            "total_before_tokens": self.total_before_tokens,
            "total_after_tokens": self.total_after_tokens,
            "input_saved_tokens": self.input_saved_tokens,
            "savings_pct": self.savings_pct,
            "type_savings_tokens": dict(self.type_savings_tokens),
            "behavior_totals": dict(self.behavior_totals),
        }


def _process_replay_turn(
    record: dict[str, Any],
    token_counter: Callable[[str], int],
    replay_policy: Any,
    replay_state: Any,
    tool_compatible: bool,
    cumulative_user_turns: int,
    behavior_totals: dict[str, int],
    type_savings_tokens: dict[str, int],
    file_cache: dict[
        str, tuple[str, str, float] | tuple[str, str] | tuple[str]
    ]
    | None = None,
) -> tuple[int, int, int, dict[str, int], dict[str, int], int, Any]:
    messages = _record_messages(record)
    if not messages:
        return (
            0,
            0,
            0,
            behavior_totals,
            type_savings_tokens,
            cumulative_user_turns,
            replay_state,
        )

    user_turn_count = sum(1 for m in messages if m.get("role") == "user")
    cumulative_user_turns += user_turn_count

    before_tokens = sum(token_counter(_msg_text(m)) for m in messages)
    msgs_copy = copy.deepcopy(messages)
    id_to_context = build_tool_use_id_to_context(messages)
    compression_level = "balanced"
    history_profile = None

    dummy_session = type(
        "_ReplaySession",
        (),
        {"_step_count": cumulative_user_turns},
    )()
    keep_turns = get_adaptive_keep_turns(dummy_session)

    if replay_policy is not None and replay_state is not None:
        compression_level = replay_policy.tool_levels[replay_state.mode]
        history_profile = replay_policy.history_profiles[replay_state.mode]

    msgs_copy, breakdown_chars = compress_tool_results(
        msgs_copy,
        result_cache=file_cache,
        tool_use_id_to_context=id_to_context,
        compression_level=compression_level,
    )

    for key, value in collect_behavior_signals(
        messages, id_to_context, result_cache=file_cache
    ).items():
        behavior_totals[key] = behavior_totals.get(key, 0) + value

    turn_output_saved = 0
    from ..runtime.policy.semantic_validation import SemanticValidator

    validator = SemanticValidator()
    for assistant_text in _assistant_texts(messages):
        processed = response_contract_for_mode(
            assistant_text, tool_compatible=tool_compatible
        )
        drift_signals = validator.validate_drift(
            assistant_text, processed.behavior_signals
        )
        for key, value in {
            **processed.behavior_signals,
            **drift_signals,
        }.items():
            behavior_totals[key] = behavior_totals.get(key, 0) + value

        visible_text = "\n".join(
            b.get("text", "")
            for b in processed.content_blocks
            if b.get("type") == "text"
        ).strip()
        if (
            visible_text
            and "tok_native_response" in processed.behavior_signals
        ):
            prose_baseline = token_counter(visible_text) + 22
            tok_actual = token_counter(assistant_text)
            turn_output_saved += max(0, prose_baseline - tok_actual)

    for kind, chars in breakdown_chars.items():
        type_savings_tokens[kind] = type_savings_tokens.get(kind, 0) + max(
            0, chars // 4
        )

    after_tool_tokens = sum(token_counter(_msg_text(m)) for m in msgs_copy)
    recent_msgs, tok_state = compress_history(
        msgs_copy,
        keep_turns=keep_turns,
        profile=history_profile,
        prune_tool_results=True,
    )

    if tok_state:
        after_tokens = sum(
            token_counter(_msg_text(m)) for m in recent_msgs
        ) + token_counter(tok_state)
    else:
        after_tokens = after_tool_tokens

    type_savings_tokens["output_minimalist"] = (
        type_savings_tokens.get("output_minimalist", 0) + turn_output_saved
    )
    after_tokens -= turn_output_saved

    if replay_policy is not None and replay_state is not None:
        replay_state = advance_state(
            replay_policy, replay_state, behavior_totals
        )

    return (
        before_tokens,
        after_tokens,
        turn_output_saved,
        behavior_totals,
        type_savings_tokens,
        cumulative_user_turns,
        replay_state,
    )


def analyze_replay_fixture(session_file: str | Path) -> ReplayFixtureMetrics:
    """Produce deterministic metrics for a replay capture."""

    path = Path(session_file)
    if not path.exists():
        raise FileNotFoundError(f"Replay capture not found: {path}")

    meta_path = path.with_suffix(path.suffix + ".meta.json")
    replay_meta = (
        json.loads(meta_path.read_text()) if meta_path.exists() else {}
    )
    replay_model = str(replay_meta.get("model", ""))
    response_mode = str(replay_meta.get("response_mode", "tok-native")).strip()
    tool_compatible = response_mode == "tool-compatible"
    replay_policy = policy_for_model(replay_model) if replay_model else None
    replay_state = initial_state(replay_policy) if replay_policy else None

    token_counter = _token_counter()
    behavior_totals: dict[str, int] = defaultdict(int)
    type_savings_tokens: dict[str, int] = defaultdict(int)
    file_cache: dict[
        str, tuple[str, str, float] | tuple[str, str] | tuple[str]
    ] = {}

    total_before_tokens = 0
    total_after_tokens = 0
    lines_read = 0
    cumulative_user_turns = 0

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            lines_read += 1
            record = json.loads(line)

            (
                before_tokens,
                after_tokens,
                turn_output_saved,
                behavior_totals,
                type_savings_tokens,
                cumulative_user_turns,
                replay_state,
            ) = _process_replay_turn(
                record,
                token_counter,
                replay_policy,
                replay_state,
                tool_compatible,
                cumulative_user_turns,
                behavior_totals,
                type_savings_tokens,
                file_cache,
            )

            total_before_tokens += before_tokens
            total_after_tokens += after_tokens

    return ReplayFixtureMetrics(
        fixture=path,
        lines=lines_read,
        total_before_tokens=total_before_tokens,
        total_after_tokens=total_after_tokens,
        type_savings_tokens=dict(type_savings_tokens),
        behavior_totals=dict(behavior_totals),
    )


def _msg_text(msg: dict[str, Any]) -> str:
    """Extract normalized text content from a message."""

    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
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
        return "\n".join(parts)
    return str(content)


def _record_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if isinstance(messages, list):
        return messages
    if "role" in record and "content" in record:
        return [record]
    return []


def _assistant_texts(messages: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if content.strip():
                texts.append(content)
            continue
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text", ""))
                    if text.strip():
                        parts.append(text)
            if parts:
                texts.append("\n".join(parts))
    return texts


def _token_counter() -> Callable[[str], int]:
    if tiktoken is None:
        return lambda text: len(text) // 4

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover
        return lambda text: len(text) // 4

    return lambda text: len(encoding.encode(text))
