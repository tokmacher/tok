"""Validation utilities for Tok runtime requests."""

import copy
import hashlib
import json
import logging
import os
import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger("tok.runtime.validation")

_ALLOWED_BLOCK_TYPES = frozenset({"text", "tool_use", "tool_result", "thinking", "redacted_thinking"})
_PROVIDER_SAFE_TOOL_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_STRICT_FAILURE_SIGNAL_MAP = {
    "invalid_tool_use_block": "tok_bridge_strict_invalid_tool_use_block",
    "invalid_tool_result_block": "tok_bridge_strict_invalid_tool_result_block",
    "assistant_tool_use_missing_next_tool_result": ("tok_bridge_strict_missing_next_tool_result"),
    "assistant_tool_use_incomplete_next_tool_result_coverage": (
        "tok_bridge_strict_incomplete_next_tool_result_coverage"
    ),
    "tool_result_unknown_tool_use_id": ("tok_bridge_strict_tool_result_unknown_tool_use_id"),
    "tool_result_not_immediately_after_assistant_tool_use": ("tok_bridge_strict_tool_result_ordering_failure"),
    "user_tool_result_after_text": ("tok_bridge_strict_user_tool_result_after_text"),
    "bridge_wire_model_invalid": ("tok_bridge_strict_bridge_wire_model_invalid"),
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
_PROVIDER_SENSITIVE_LARGE_TOOL_BATCH_THRESHOLD = 16
_DEFAULT_PROMPT_BLOAT_THRESHOLD_CHARS = 2000
_DEFAULT_PROMPT_OPTIMIZE_LIMIT_CHARS = 2500
_USER_PROMPT_LEAK_MIN_CHARS = 200
_USER_PROMPT_LEAK_SNIPPET_CHARS = 100
_PROVIDER_SENSITIVE_FAILURES = frozenset(
    {
        "provider_sensitive_large_tool_use_text_interleaving",
        "provider_sensitive_assistant_tool_use_text_interleaving",
    }
)


class _CanonicalTextBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["text"]
    text: str

    @field_validator("text")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            msg = "blank text block"
            raise ValueError(msg)
        return value


class _CanonicalToolUseBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]

    @model_validator(mode="after")
    def _validate_tool_use(self) -> "_CanonicalToolUseBlock":
        if not self.id.strip() or not self.name.strip() or not _is_provider_safe_tool_id(self.id):
            msg = "invalid tool_use block"
            raise ValueError(msg)
        return self


class _CanonicalToolResultBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[dict[str, Any]]

    @model_validator(mode="after")
    def _validate_tool_result(self) -> "_CanonicalToolResultBlock":
        if not self.tool_use_id.strip() or not _is_provider_safe_tool_id(self.tool_use_id):
            msg = "invalid tool_result block"
            raise ValueError(msg)
        return self


class _CanonicalThinkingBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["thinking"]
    thinking: str
    signature: str | None = None


class _CanonicalRedactedThinkingBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["redacted_thinking"]
    data: Any = None


def _is_provider_safe_tool_id(value: str) -> bool:
    return bool(_PROVIDER_SAFE_TOOL_ID_RE.fullmatch(value.strip()))


def _provider_safe_tool_id_seed(value: str) -> str:
    seed = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return seed or "tool"


def _dedupe_provider_safe_tool_id(seed: str, occupied_ids: set[str]) -> str:
    candidate = seed
    suffix = 1
    while candidate in occupied_ids:
        candidate = f"{seed}_{suffix}"
        suffix += 1
    occupied_ids.add(candidate)
    return candidate


def _tool_id_seed_hint(*, msg_index: int, block_index: int, occurrence: int, prefix: str = "toolu") -> str:
    return f"{prefix}_m{msg_index + 1}_b{block_index + 1}_{occurrence}"


def _normalize_or_synthesize_tool_id(
    raw_id: str | None,
    occupied_ids: set[str],
    *,
    seed_hint: str,
) -> tuple[str, str]:
    stripped = str(raw_id or "").strip()
    if stripped and _is_provider_safe_tool_id(stripped):
        if stripped not in occupied_ids:
            occupied_ids.add(stripped)
            return stripped, "unchanged"
        deduped = _dedupe_provider_safe_tool_id(
            _provider_safe_tool_id_seed(f"{stripped}_{seed_hint}"),
            occupied_ids,
        )
        return deduped, "deduped"
    if not stripped:
        synthesized = _dedupe_provider_safe_tool_id(_provider_safe_tool_id_seed(seed_hint), occupied_ids)
        return synthesized, "synthesized"
    sanitized = _dedupe_provider_safe_tool_id(_provider_safe_tool_id_seed(stripped), occupied_ids)
    return sanitized, "sanitized"


def normalize_tool_use_blocks(
    blocks: list[dict[str, Any]],
    *,
    seed_prefix: str = "toolu",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Normalize tool_use block IDs so they are non-blank and provider safe."""
    normalized_blocks = copy.deepcopy(blocks)
    occupied_ids: set[str] = set()
    signals: dict[str, int] = {}

    for block_index, block in enumerate(normalized_blocks):
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        normalized_id, disposition = _normalize_or_synthesize_tool_id(
            block.get("id", ""),
            occupied_ids,
            seed_hint=f"{seed_prefix}_{block_index + 1}",
        )
        block["id"] = normalized_id
        if disposition == "sanitized":
            signals["tool_use_id_sanitized"] = signals.get("tool_use_id_sanitized", 0) + 1
        elif disposition == "synthesized":
            signals["tool_use_blank_id_synthesized"] = signals.get("tool_use_blank_id_synthesized", 0) + 1
        elif disposition == "deduped":
            signals["tool_use_id_deduped"] = signals.get("tool_use_id_deduped", 0) + 1

    return normalized_blocks, signals


_CanonicalContentBlock = Annotated[
    (
        _CanonicalTextBlock
        | _CanonicalToolUseBlock
        | _CanonicalToolResultBlock
        | _CanonicalThinkingBlock
        | _CanonicalRedactedThinkingBlock
    ),
    Field(discriminator="type"),
]
_CANONICAL_CONTENT_ADAPTER = TypeAdapter(list[_CanonicalContentBlock])


class _CanonicalBridgeMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant"]
    content: list[_CanonicalContentBlock]

    @field_validator("content")
    @classmethod
    def _content_must_not_be_empty(cls, value: list[_CanonicalContentBlock]) -> list[_CanonicalContentBlock]:
        if not value:
            msg = "empty content blocks"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _enforce_cross_role_shapes(self) -> "_CanonicalBridgeMessage":
        for block in self.content:
            if self.role == "user" and block.type == "tool_use":
                msg = "user contains tool_use"
                raise ValueError(msg)
            if self.role == "assistant" and block.type == "tool_result":
                msg = "assistant contains tool_result"
                raise ValueError(msg)
        return self


class _CanonicalBridgeBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[_CanonicalBridgeMessage]
    system: str | list[dict[str, Any]] | None = None

    @field_validator("model")
    @classmethod
    def _model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            msg = "missing model"
            raise ValueError(msg)
        return value

    @field_validator("messages")
    @classmethod
    def _messages_must_not_be_empty(cls, value: list[_CanonicalBridgeMessage]) -> list[_CanonicalBridgeMessage]:
        if not value:
            msg = "empty messages"
            raise ValueError(msg)
        return value

    @field_validator("system")
    @classmethod
    def _validate_system_blocks(cls, value: str | list[dict[str, Any]] | None) -> str | list[dict[str, Any]] | None:
        if isinstance(value, list):
            try:
                _CANONICAL_CONTENT_ADAPTER.validate_python(value)
            except ValidationError:
                msg = "invalid_system_block"
                raise ValueError(msg) from None
        return value


def _validate_canonical_bridge_body_model(body: dict[str, Any]) -> list[str]:
    """Validate the canonical bridge body with Pydantic for shape drift."""
    try:
        _CanonicalBridgeBody.model_validate(body)
        return []
    except ValidationError as exc:
        errors = exc.errors()
        if errors and all(err.get("loc") == ("messages",) for err in errors):
            return []
        if errors and any(
            err.get("loc") == ("system",) and "invalid_system_block" in str(err.get("msg", "")) for err in errors
        ):
            return ["invalid_system_block"]
        # Boundary contract: malformed bridge wire shape must block before upstream execution.
        return ["bridge_wire_model_invalid"]


def _all_failures_are_non_blocking(validation_failures: list[str]) -> bool:
    return all(failure in _NON_BLOCKING_OUTGOING_FAILURES for failure in validation_failures)


def _normalize_message_content_to_blocks(
    content: str | list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Normalize message content to a list of Anthropic-style blocks.

    Returns (blocks, drops) where drops maps dropped block type names to
    the count of blocks of that type that were removed because they are
    not in the outbound Anthropic allowlist.
    """
    if isinstance(content, str):
        text = content.strip()
        return ([{"type": "text", "text": text}] if text else []), {}
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        drops: dict[str, int] = {}
        for b in content:
            if not isinstance(b, dict):
                continue
            b_type = b.get("type", "")
            if b_type == "text":
                text = str(b.get("text", "")).strip()
                if text:
                    blocks.append({"type": "text", "text": text})
            elif b_type in _ALLOWED_BLOCK_TYPES:
                if b_type in {"thinking", "redacted_thinking"}:
                    blocks.append(b)
                else:
                    blocks.append(copy.deepcopy(b))
            else:
                drops[b_type] = drops.get(b_type, 0) + 1
        return blocks, drops
    return [], {}


def _check_changed_content(
    canonical_message: dict[str, Any],
    original_content: str | list[dict[str, Any]],
    role: str,
) -> bool:
    """Return True when normalization changed the message content."""
    if role == "tool_result":
        return False
    canonical_blocks = canonical_message.get("content")
    if not isinstance(canonical_blocks, list):
        canonical_blocks = []
    if isinstance(original_content, str):
        normalized_blocks, _ = _normalize_message_content_to_blocks(original_content)
        return bool(normalized_blocks or original_content != "")
    if isinstance(original_content, list):
        original_blocks = [block for block in original_content if isinstance(block, dict)]
        if len(original_blocks) != len(canonical_blocks):
            return True
    normalized_blocks, _ = _normalize_message_content_to_blocks(original_content)
    return bool(canonical_blocks != normalized_blocks)


def _canonicalize_bridge_message(
    message: dict[str, Any],
    *,
    preserve_content: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    """
    Rewrite a single message into a canonical user or assistant role with blocks.

    Returns (canonical_message, drops) where drops maps dropped block type
    names to counts of blocks removed during normalization.

    When preserve_content is True, the message content is passed through
    unchanged. This is used for the latest assistant message containing
    thinking blocks, which Anthropic requires to be returned unmodified
    during tool-use turns.
    """
    role = str(message.get("role", "")).strip()
    if role == "tool_result":
        block = {
            "type": "tool_result",
            "tool_use_id": message.get("tool_use_id", ""),
            "content": copy.deepcopy(message.get("content", "")),
        }
        if "is_error" in message:
            block["is_error"] = bool(message.get("is_error"))
        if "cache_control" in message:
            block["cache_control"] = copy.deepcopy(message.get("cache_control"))

        return {"role": "user", "content": [block]}, {}

    if preserve_content:
        content = message.get("content")
        if isinstance(content, list):
            preserved_blocks: list[dict[str, Any]] = []
            drops: dict[str, int] = {}
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type", "")).strip()
                if block_type in _ALLOWED_BLOCK_TYPES:
                    preserved_blocks.append(copy.deepcopy(block))
                elif block_type:
                    drops[block_type] = drops.get(block_type, 0) + 1
            return {"role": role, "content": preserved_blocks}, drops
        if isinstance(content, str):
            text = content.strip()
            if text:
                return {
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                }, {}
            return {"role": role, "content": []}, {}
        return {"role": role, "content": []}, {}

    canonical_role = "assistant" if role == "assistant" else "user"
    raw_content = message.get("content")
    blocks, drops = _normalize_message_content_to_blocks(raw_content if isinstance(raw_content, (str, list)) else "")
    if canonical_role == "user":
        filtered_blocks: list[dict[str, Any]] = []
        for block in blocks:
            if block.get("type") in {"thinking", "redacted_thinking"}:
                drops[block["type"]] = drops.get(block["type"], 0) + 1
                continue
            filtered_blocks.append(block)
        blocks = filtered_blocks
    return {"role": canonical_role, "content": blocks}, drops


def _merge_adjacent_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge adjacent messages of the same role (specifically for user messages)."""
    if not messages:
        return [], {}

    merged: list[dict[str, Any]] = []
    signals: dict[str, int] = {}

    for msg in messages:
        if not merged:
            merged.append(msg)
            continue

        prev = merged[-1]
        # Only merge user messages; assistant messages must not be merged across tool boundaries.
        # Preserve tool_result-only boundaries so canonical splitting cannot be re-collapsed into
        # mixed user messages that upstream may reject.
        if msg["role"] == "user" and prev["role"] == "user" and _user_messages_are_merge_compatible(prev, msg):
            prev["content"].extend(msg["content"])
            signals["tok_bridge_adjacent_user_merged"] = signals.get("tok_bridge_adjacent_user_merged", 0) + 1
        else:
            merged.append(msg)

    return merged, signals


def _is_tool_result_block(block: dict[str, Any]) -> bool:
    return bool(isinstance(block, dict) and block.get("type") == "tool_result")


def _user_message_tool_result_shape(
    message: dict[str, Any],
) -> tuple[bool, bool]:
    """Return (has_tool_result, has_non_tool_result) for a user message."""
    content = message.get("content")
    if not isinstance(content, list):
        return False, False
    has_tool_result = any(_is_tool_result_block(block) for block in content)
    has_non_tool_result = any(isinstance(block, dict) and block.get("type") != "tool_result" for block in content)
    return has_tool_result, has_non_tool_result


def _user_messages_are_merge_compatible(prev: dict[str, Any], current: dict[str, Any]) -> bool:
    (
        prev_has_tool_result,
        prev_has_non_tool_result,
    ) = _user_message_tool_result_shape(prev)
    (
        curr_has_tool_result,
        curr_has_non_tool_result,
    ) = _user_message_tool_result_shape(current)

    # Only merge adjacent user messages when both are tool_result-only.
    # This preserves prompt/instruction boundaries for text-bearing user turns.
    if prev_has_tool_result and not prev_has_non_tool_result:
        return curr_has_tool_result and not curr_has_non_tool_result
    return False


def _split_mixed_user_tool_result_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    """Split mixed user messages into tool_result-only then residual content."""
    rewritten: list[dict[str, Any]] = []
    changed = False
    split_count = 0

    for message in messages:
        if not isinstance(message, dict):
            rewritten.append(message)
            continue
        if str(message.get("role", "")).strip() != "user":
            rewritten.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            rewritten.append(message)
            continue

        tool_result_blocks = [copy.deepcopy(block) for block in content if _is_tool_result_block(block)]
        non_tool_result_blocks = [
            copy.deepcopy(block) for block in content if isinstance(block, dict) and block.get("type") != "tool_result"
        ]
        if not tool_result_blocks or not non_tool_result_blocks:
            rewritten.append(message)
            continue

        split_count += 1
        changed = True
        left_message = copy.deepcopy(message)
        left_message["content"] = tool_result_blocks
        right_message = copy.deepcopy(message)
        right_message["content"] = non_tool_result_blocks
        rewritten.append(left_message)
        rewritten.append(right_message)

    signals: dict[str, int] = {}
    if split_count:
        signals["tok_bridge_user_tool_result_text_split"] = split_count
    return rewritten, changed, signals


def _rewrite_provider_safe_tool_ids(
    messages: list[dict[str, Any]],
    *,
    protected_content_identities: set[int] | None = None,
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    """
    Rewrite invalid tool IDs to provider-safe IDs and preserve pairings.

    Messages whose content list id() is in protected_content_identities will
    have their content list preserved (not deep-copied). This is used to
    protect the latest assistant message with thinking blocks.
    """
    if protected_content_identities is None:
        protected_content_identities = set()

    rewritten_messages: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            rewritten_messages.append(msg)
            continue
        content = msg.get("content")
        if id(content) in protected_content_identities and isinstance(content, list):
            new_msg = {k: copy.deepcopy(v) if k != "content" else v for k, v in msg.items()}
            rewritten_messages.append(new_msg)
        else:
            rewritten_messages.append(copy.deepcopy(msg))

    occupied_ids: set[str] = set()
    signals: dict[str, int] = {}
    changed = False
    invalid_tool_ids_seen = 0
    pending_pairs: list[dict[str, Any]] = []

    for msg_index, message in enumerate(rewritten_messages):
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if not isinstance(content, list):
            pending_pairs = []
            continue

        if role == "assistant":
            # Skip rewriting tool IDs for protected messages to preserve
            # thinking block integrity (prevents false mutation warnings)
            if id(content) in protected_content_identities:
                # Still track tool IDs for pairing, but don't modify
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        raw_tool_use_id = str(block.get("id", "")).strip()
                        if raw_tool_use_id:
                            occupied_ids.add(raw_tool_use_id)
                            pending_pairs.append(
                                {
                                    "raw_id": raw_tool_use_id,
                                    "new_id": raw_tool_use_id,
                                    "consumed": False,
                                }
                            )
                continue
            (
                assistant_changed,
                assistant_invalid,
            ) = _rewrite_assistant_tool_message(
                message,
                msg_index,
                content,
                occupied_ids,
                pending_pairs,
                signals,
            )
            changed |= assistant_changed
            invalid_tool_ids_seen += assistant_invalid
            continue

        if role == "user":
            if not pending_pairs:
                continue
            user_changed = _rewrite_user_tool_message(
                message,
                content,
                pending_pairs,
                signals,
            )
            changed |= user_changed
            pending_pairs = []
            continue

        pending_pairs = []

    if invalid_tool_ids_seen:
        signals["tok_bridge_invalid_tool_id_seen"] = invalid_tool_ids_seen
    if signals.get("tok_bridge_tool_result_pairing_unrepaired", 0):
        signals["tok_bridge_tool_result_rewrite_incomplete"] = 1
    elif signals.get("tok_bridge_tool_result_id_rewritten", 0):
        signals["tok_bridge_tool_result_rewrite_complete"] = 1

    return rewritten_messages, changed, signals


def _rewrite_assistant_tool_message(
    _message: dict[str, Any],
    msg_index: int,
    content: list[Any],
    occupied_ids: set[str],
    pending_pairs: list[dict[str, Any]],
    signals: dict[str, int],
) -> tuple[bool, int]:
    changed = False
    invalid_tool_ids_seen = 0
    tool_occurrence = 1
    for block_index, block in enumerate(content):
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        raw_tool_use_id = str(block.get("id", "")).strip()
        normalized_id, disposition = _normalize_or_synthesize_tool_id(
            raw_tool_use_id,
            occupied_ids,
            seed_hint=_tool_id_seed_hint(
                msg_index=msg_index,
                block_index=block_index,
                occurrence=tool_occurrence,
            ),
        )
        tool_occurrence += 1
        if raw_tool_use_id and not _is_provider_safe_tool_id(raw_tool_use_id):
            invalid_tool_ids_seen += 1
        if block.get("id") != normalized_id:
            block["id"] = normalized_id
            changed = True
        if disposition == "sanitized":
            signals["tok_bridge_tool_id_sanitized"] = signals.get("tok_bridge_tool_id_sanitized", 0) + 1
        elif disposition == "synthesized":
            signals["tok_bridge_blank_tool_id_synthesized"] = signals.get("tok_bridge_blank_tool_id_synthesized", 0) + 1
        elif disposition == "deduped":
            signals["tok_bridge_tool_id_deduped"] = signals.get("tok_bridge_tool_id_deduped", 0) + 1
        pending_pairs.append(
            {
                "raw_id": raw_tool_use_id,
                "new_id": normalized_id,
                "consumed": False,
            }
        )
    return changed, invalid_tool_ids_seen


def _rewrite_user_tool_message(
    message: dict[str, Any],
    content: list[Any],
    pending_pairs: list[dict[str, Any]],
    signals: dict[str, int],
) -> bool:
    changed = False
    reordered_tool_results: list[tuple[int, dict[str, Any]]] = []
    matched_indices_in_encounter_order: list[int] = []
    unmatched_tool_results: list[dict[str, Any]] = []
    non_tool_blocks: list[dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            if isinstance(block, dict):
                non_tool_blocks.append(block)
            continue
        raw_tool_use_id = str(block.get("tool_use_id", "")).strip()
        match_index: int | None = None
        if raw_tool_use_id:
            exact_matches = [
                index
                for index, pair in enumerate(pending_pairs)
                if (not pair["consumed"] and pair["raw_id"] == raw_tool_use_id)
            ]
            if len(exact_matches) == 1:
                match_index = exact_matches[0]
            elif len(exact_matches) > 1:
                signals["tok_bridge_tool_result_pairing_ambiguous"] = (
                    signals.get("tok_bridge_tool_result_pairing_ambiguous", 0) + 1
                )
        else:
            for index, pair in enumerate(pending_pairs):
                if not pair["consumed"]:
                    match_index = index
                    break
            if match_index is not None:
                signals["tok_bridge_tool_result_pairing_repaired"] = (
                    signals.get("tok_bridge_tool_result_pairing_repaired", 0) + 1
                )
        if match_index is None:
            signals["tok_bridge_tool_result_pairing_unrepaired"] = (
                signals.get("tok_bridge_tool_result_pairing_unrepaired", 0) + 1
            )
            unmatched_tool_results.append(block)
            continue
        pending_pairs[match_index]["consumed"] = True
        normalized_id = str(pending_pairs[match_index]["new_id"])
        if block.get("tool_use_id") != normalized_id:
            block["tool_use_id"] = normalized_id
            signals["tok_bridge_tool_result_id_rewritten"] = signals.get("tok_bridge_tool_result_id_rewritten", 0) + 1
            changed = True
        matched_indices_in_encounter_order.append(match_index)
        reordered_tool_results.append((match_index, block))

    sorted_tool_results = [block for _index, block in sorted(reordered_tool_results, key=lambda item: item[0])]
    reordered_content = sorted_tool_results + unmatched_tool_results + non_tool_blocks
    if reordered_content != content:
        message["content"] = reordered_content
        changed = True
        signals["tok_bridge_tool_result_order_repaired"] = signals.get("tok_bridge_tool_result_order_repaired", 0) + 1
    if matched_indices_in_encounter_order and (
        matched_indices_in_encounter_order != sorted(matched_indices_in_encounter_order)
    ):
        signals["tok_bridge_tool_result_pairing_repaired"] = (
            signals.get("tok_bridge_tool_result_pairing_repaired", 0) + 1
        )
    return changed


def bridge_strict_failure_signals(failures: list[str]) -> dict[str, int]:
    """Convert strict bridge failures into stable behavior signals."""
    signals: dict[str, int] = {}
    if failures:
        signals["tok_bridge_strict_failure"] = 1
    for failure in failures:
        signal = _STRICT_FAILURE_SIGNAL_MAP.get(failure)
        if signal is not None:
            signals[signal] = 1
    if any(
        failure in failures
        for failure in (
            "tool_result_unknown_tool_use_id",
            "tool_result_not_immediately_after_assistant_tool_use",
            "user_tool_result_after_text",
        )
    ):
        signals["tok_bridge_strict_pairing_or_ordering_failure"] = 1
    return signals


def _message_content_without_block_type(content: list[dict[str, Any]], block_type: str) -> list[dict[str, Any]]:
    return [copy.deepcopy(block) for block in content if isinstance(block, dict) and block.get("type") != block_type]


def _assistant_user_tool_exchange_is_broken(
    assistant_content: list[dict[str, Any]],
    user_content: list[dict[str, Any]],
) -> bool:
    assistant_tool_blocks = [
        block for block in assistant_content if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    user_tool_result_blocks = [
        block for block in user_content if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    if not assistant_tool_blocks and not user_tool_result_blocks:
        return False
    if any(
        not str(block.get("id", "")).strip()
        or not _is_provider_safe_tool_id(str(block.get("id", "")))
        or not str(block.get("name", "")).strip()
        or not isinstance(block.get("input", {}), dict)
        for block in assistant_tool_blocks
    ):
        return True
    if any(
        not str(block.get("tool_use_id", "")).strip()
        or not _is_provider_safe_tool_id(str(block.get("tool_use_id", "")))
        or not isinstance(block.get("content", ""), str | list)
        for block in user_tool_result_blocks
    ):
        return True
    pending_tool_use_ids = [str(block.get("id", "")).strip() for block in assistant_tool_blocks]
    seen_tool_use_ids = set(pending_tool_use_ids)
    risks: dict[str, int] = {}
    tool_result_count, matched_pending_tool_use_count = _process_user_tool_results(
        user_content,
        seen_tool_use_ids,
        pending_tool_use_ids,
        risks,
    )
    _evaluate_pending_coverage(
        risks,
        pending_tool_use_ids,
        tool_result_count,
        matched_pending_tool_use_count,
    )
    return bool(risks)


def quarantine_invalid_tool_history_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    """Drop only broken tool exchanges while preserving surrounding text."""
    quarantined_messages = copy.deepcopy(messages)
    changed = False
    signals: dict[str, int] = {}
    index = 0

    while index < len(quarantined_messages):
        message = quarantined_messages[index]
        if not isinstance(message, dict):
            index += 1
            continue
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if role != "assistant" or not isinstance(content, list):
            index += 1
            continue
        assistant_has_tool_use = any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content)
        if not assistant_has_tool_use:
            index += 1
            continue

        next_message = quarantined_messages[index + 1] if index + 1 < len(quarantined_messages) else None
        next_content = next_message.get("content") if isinstance(next_message, dict) else None
        if (
            isinstance(next_message, dict)
            and str(next_message.get("role", "")).strip() == "user"
            and isinstance(next_content, list)
            and _assistant_user_tool_exchange_is_broken(content, next_content)
        ):
            tool_use_count = sum(1 for block in content if isinstance(block, dict) and block.get("type") == "tool_use")
            tool_result_count = sum(
                1 for block in next_content if (isinstance(block, dict) and block.get("type") == "tool_result")
            )
            message["content"] = _message_content_without_block_type(content, "tool_use")
            next_message["content"] = _message_content_without_block_type(next_content, "tool_result")
            signals["tok_bridge_invalid_tool_history_quarantined"] = (
                signals.get("tok_bridge_invalid_tool_history_quarantined", 0) + 1
            )
            signals["tok_bridge_quarantined_tool_use_blocks"] = (
                signals.get("tok_bridge_quarantined_tool_use_blocks", 0) + tool_use_count
            )
            signals["tok_bridge_quarantined_tool_result_blocks"] = (
                signals.get("tok_bridge_quarantined_tool_result_blocks", 0) + tool_result_count
            )
            changed = True
            index += 2
            continue

        if _assistant_user_tool_exchange_is_broken(content, []):
            tool_use_count = sum(1 for block in content if isinstance(block, dict) and block.get("type") == "tool_use")
            message["content"] = _message_content_without_block_type(content, "tool_use")
            signals["tok_bridge_invalid_tool_history_quarantined"] = (
                signals.get("tok_bridge_invalid_tool_history_quarantined", 0) + 1
            )
            signals["tok_bridge_quarantined_tool_use_blocks"] = (
                signals.get("tok_bridge_quarantined_tool_use_blocks", 0) + tool_use_count
            )
            changed = True
        index += 1

    if not changed:
        return quarantined_messages, False, {}

    filtered_messages: list[dict[str, Any]] = []
    dropped_messages = 0
    for message in quarantined_messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list) and not content:
            dropped_messages += 1
            continue
        filtered_messages.append(message)

    if dropped_messages:
        signals["tok_bridge_quarantined_empty_messages_dropped"] = dropped_messages
    if not filtered_messages:
        signals["tok_bridge_invalid_tool_history_quarantine_exhausted"] = 1

    return filtered_messages, True, signals


def _process_bridged_message(
    raw_message: dict[str, Any],
    signals: dict[str, int],
    total_drops: dict[str, int],
    *,
    preserve_content: bool = False,
) -> tuple[dict[str, Any] | None, bool]:
    """
    Process a single message for canonicalization.

    When preserve_content is True, the message content is passed through
    unchanged (for the latest assistant message with thinking blocks).
    """
    if not isinstance(raw_message, dict):
        return None, False

    role = str(raw_message.get("role", "")).strip()
    changed = False

    if role == "tool_result":
        signals["tok_bridge_top_level_tool_result_rewritten"] = (
            signals.get("tok_bridge_top_level_tool_result_rewritten", 0) + 1
        )
        changed = True

    orig_content = raw_message.get("content")
    msg, msg_drops = _canonicalize_bridge_message(raw_message, preserve_content=preserve_content)

    for b_type, count in msg_drops.items():
        total_drops[b_type] = total_drops.get(b_type, 0) + count
        changed = True

    if not msg["content"]:
        return None, True

    if not changed and not preserve_content:
        changed = _check_changed_content(msg, orig_content if isinstance(orig_content, (str, list)) else "", role)

    return msg, changed


def _normalize_assistant_block_order(
    messages: list[dict[str, Any]],
    *,
    skip_identities: set[int] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Reorder assistant message blocks: non-tool_use first, tool_use contiguous at end.

    Enforces Anthropic's actual constraint: in every assistant message all
    non-tool_use blocks (text, thinking, redacted_thinking, etc.) must come
    before all tool_use blocks, and tool_use blocks must be contiguous.

    Messages whose id() is in skip_identities are not reordered. This is used
    to preserve the latest assistant message with thinking blocks unchanged.
    """
    changed = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        if skip_identities is not None and id(message) in skip_identities:
            continue
        if str(message.get("role", "")).strip() != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        non_tool_use: list[dict[str, Any]] = []
        tool_use: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_use.append(block)
            else:
                non_tool_use.append(block)

        reordered = non_tool_use + tool_use
        if reordered != content:
            message["content"] = reordered
            changed = True

    return messages, changed


def _content_hash(content: list[dict[str, Any]]) -> str:
    if not isinstance(content, list):
        return ""
    try:
        return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    except (TypeError, ValueError):
        return ""


def _block_type_sequence(content: list[dict[str, Any]]) -> list[str]:
    if not isinstance(content, list):
        return []
    return [b.get("type", "?") if isinstance(b, dict) else "?" for b in content]


def _has_thinking_with_signature(content: list[dict[str, Any]]) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"} and bool(b.get("signature"))
        for b in content
    )


def _find_protected_message(
    messages: list[dict[str, Any]],
) -> tuple[int | None, int | None]:
    """Find the latest assistant message with thinking blocks."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).strip() != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if any(isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"} for b in content):
            return id(msg), id(content)
    return None, None


def _get_protected_message_info(
    messages: list[dict[str, Any]], protected_original_identity: int | None
) -> tuple[str | None, list[str], bool, int | None]:
    """Get hash, block types, signature status, and index of protected message."""
    if protected_original_identity is None:
        return None, [], False, None
    for i, msg in enumerate(messages):
        if id(msg) == protected_original_identity:
            content = msg.get("content")
            typed_content = content if isinstance(content, list) else []
            return (
                _content_hash(typed_content),
                _block_type_sequence(typed_content),
                _has_thinking_with_signature(typed_content),
                i,
            )
    return None, [], False, None


def _check_thinking_block_mutation(
    merged_messages: list[dict[str, Any]],
    before_hash: str | None,
    before_block_types: list[str],
    before_has_signature: bool,
    protected_msg_index: int | None,
    signals: dict[str, int],
    *,
    protected_content_identity: int | None = None,
    seen_mutation_pairs: set[tuple[str, str]] | None = None,
) -> None:
    """Check if thinking blocks were mutated and log warning if so."""
    if before_hash is None or protected_msg_index is None:
        return
    after_content: Any = None
    if protected_content_identity is not None:
        for msg in merged_messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list) and id(content) == protected_content_identity:
                after_content = content
                break
    if after_content is None:
        for msg in merged_messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            if any(isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"} for b in content):
                after_content = content
                break
    if after_content is not None:
        after_hash = _content_hash(after_content)
        after_block_types = _block_type_sequence(after_content)
        mutation_pair = (before_hash, after_hash)
        if before_hash != after_hash:
            if seen_mutation_pairs is not None:
                if mutation_pair in seen_mutation_pairs:
                    logger.debug(
                        "THINKING_BLOCK_MUTATION_DEDUPLICATED | msg_index=%d | before_hash=%s | after_hash=%s",
                        protected_msg_index,
                        before_hash,
                        after_hash,
                    )
                    return
                seen_mutation_pairs.add(mutation_pair)
            logger.warning(
                "THINKING_BLOCK_MUTATION_DETECTED | "
                "msg_index=%d | "
                "before_hash=%s | after_hash=%s | "
                "before_types=%s | after_types=%s | "
                "has_signature=%s",
                protected_msg_index,
                before_hash,
                after_hash,
                ",".join(before_block_types),
                ",".join(after_block_types),
                before_has_signature,
            )
            signals["thinking_block_mutated"] = 1
            signals["thinking_block_mutated_msg_index"] = protected_msg_index
            signals["thinking_block_mutated_has_signature"] = 1 if before_has_signature else 0
        else:
            logger.debug(
                "thinking_block_preserved | msg_index=%d | hash=%s | types=%s | has_signature=%s",
                protected_msg_index,
                before_hash,
                ",".join(before_block_types),
                before_has_signature,
            )


def _process_messages_into_canonical_path(
    messages: list[dict[str, Any]],
    protected_original_identity: int | None,
    signals: dict[str, int],
    total_drops: dict[str, int],
) -> tuple[list[dict[str, Any]], bool, int | None]:
    """Process raw messages into canonical path."""
    canonical_path: list[dict[str, Any]] = []
    changed = False
    protected_canonical_identity: int | None = None

    for raw_message in messages:
        is_protected = id(raw_message) == protected_original_identity
        msg, message_changed = _process_bridged_message(
            raw_message, signals, total_drops, preserve_content=is_protected
        )
        if is_protected and msg is not None:
            protected_canonical_identity = id(msg)
        if message_changed:
            changed = True
        if msg is not None:
            canonical_path.append(msg)

    return canonical_path, changed, protected_canonical_identity


def _apply_canonicalization_pipeline(
    canonical_path: list[dict[str, Any]],
    protected_canonical_identity: int | None,
    protected_content_identity: int | None,
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    """Apply the full canonicalization pipeline to messages."""
    signals: dict[str, int] = {}
    changed = False

    skip_identities: set[int] | None = None
    if protected_canonical_identity is not None:
        skip_identities = {protected_canonical_identity}

    canonical_path, order_changed = _normalize_assistant_block_order(canonical_path, skip_identities=skip_identities)
    if order_changed:
        changed = True
        signals["tok_bridge_assistant_block_order_normalized"] = (
            signals.get("tok_bridge_assistant_block_order_normalized", 0) + 1
        )

    (
        canonical_path,
        split_changed,
        split_signals,
    ) = _split_mixed_user_tool_result_messages(canonical_path)
    if split_changed:
        changed = True
    if split_signals:
        signals.update(split_signals)

    merged_messages, merge_signals = _merge_adjacent_anthropic_messages(canonical_path)
    signals.update(merge_signals)

    protected_content_identities: set[int] | None = None
    if protected_content_identity is not None:
        protected_content_identities = {protected_content_identity}

    merged_messages, id_changed, id_signals = _rewrite_provider_safe_tool_ids(
        merged_messages,
        protected_content_identities=protected_content_identities,
    )
    if id_changed:
        changed = True
    if id_signals:
        signals.update(id_signals)

    return merged_messages, changed, signals


def canonicalize_anthropic_bridge_messages(
    messages: list[dict[str, Any]],
    *,
    seen_mutation_pairs: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    """
    Canonicalize bridge messages to the Anthropic wire shape.

    Returns (canonical_messages, changed, signals).

    Drops unsupported block types (anything outside {text, tool_use,
    tool_result, thinking, redacted_thinking}) and emits
    ``tok_bridge_unsupported_block_dropped`` when blocks are removed.

    This function removes unsupported block types (e.g., top-level tool_result),
    rewrites top-level tool_results (emitting ``tok_bridge_top_level_tool_result_rewritten``),
    merges adjacent messages of same role (``tok_bridge_adjacent_messages_merged``),
    and preserves assistant thinking blocks so Claude-native reasoning remains
    visible to the upstream provider and to the client stream.

    CRITICAL: The latest assistant message containing thinking/redacted_thinking
    blocks is preserved unchanged (no normalization, no reordering). This is
    required by Anthropic's protocol for multi-turn tool-use continuity.
    """
    if not isinstance(messages, list):
        return messages, False, {}

    (
        protected_original_identity,
        protected_content_identity,
    ) = _find_protected_message(messages)

    (
        before_hash,
        before_block_types,
        before_has_signature,
        protected_msg_index,
    ) = _get_protected_message_info(messages, protected_original_identity)

    signals: dict[str, int] = {}
    total_drops: dict[str, int] = {}

    (
        canonical_path,
        changed,
        protected_canonical_identity,
    ) = _process_messages_into_canonical_path(messages, protected_original_identity, signals, total_drops)

    (
        merged_messages,
        pipeline_changed,
        pipeline_signals,
    ) = _apply_canonicalization_pipeline(
        canonical_path,
        protected_canonical_identity,
        protected_content_identity,
    )
    changed = changed or pipeline_changed
    signals.update(pipeline_signals)

    if not changed and (len(merged_messages) != len(messages)):
        changed = True

    if total_drops:
        total_drop_count = sum(total_drops.values())
        signals["tok_bridge_unsupported_block_dropped"] = total_drop_count

    if seen_mutation_pairs is None:
        seen_mutation_pairs = set()

    if before_hash is not None and protected_msg_index is not None:
        _check_thinking_block_mutation(
            merged_messages,
            before_hash,
            before_block_types,
            before_has_signature,
            protected_msg_index,
            signals,
            protected_content_identity=protected_content_identity,
            seen_mutation_pairs=seen_mutation_pairs,
        )

    if changed:
        signals["tok_bridge_canonicalized"] = 1

    return merged_messages, changed, signals


def canonicalize_anthropic_bridge_body(
    body: dict[str, Any],
    *,
    seen_mutation_pairs: set[tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], bool, dict[str, int]]:
    """Canonicalize a bridge request body for Anthropic before send."""
    if not isinstance(body, dict):
        return body, False, {}
    messages = body.get("messages")
    if not isinstance(messages, list):
        return copy.deepcopy(body), False, {}

    (
        canonical_messages,
        changed,
        signals,
    ) = canonicalize_anthropic_bridge_messages(messages, seen_mutation_pairs=seen_mutation_pairs)
    if not changed:
        return body, False, {}

    new_body = copy.deepcopy(body)
    new_body["messages"] = canonical_messages
    validation_failures = _validate_canonical_bridge_body_model(new_body)
    if validation_failures:
        if _all_failures_are_non_blocking(validation_failures):
            signals["tok_bridge_canonical_validation_nonblocking"] = 1
            return new_body, True, signals
        failed_signals = dict(signals)
        failed_signals["tok_bridge_canonical_validation_failed"] = 1
        return copy.deepcopy(body), False, failed_signals
    return new_body, True, signals


def _process_assistant_tool_ids(
    content: list[dict[str, Any]],
    seen_tool_use_ids: set[str],
) -> list[str]:
    """Extract ordered tool_use IDs from an assistant message."""
    assistant_tool_use_ids: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_use_id = str(block.get("id", "")).strip()
            if tool_use_id:
                assistant_tool_use_ids.append(tool_use_id)
                seen_tool_use_ids.add(tool_use_id)
    return assistant_tool_use_ids


def _process_user_tool_results(
    content: list[dict[str, Any]],
    seen_tool_use_ids: set[str],
    pending_tool_use_ids: list[str],
    risks: dict[str, int],
) -> tuple[int, int]:
    """
    Check user text/tool_result ordering risks.

    Returns `(tool_result_count, ordered_match_count)` for pending tool uses.
    """
    saw_text_block = False
    message_has_order_violation = False
    tool_result_count = 0
    ordered_match_count = 0
    saw_tool_result_block = False
    message_has_mixed_violation = False
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = str(block.get("type", "")).strip()
        if block_type == "text":
            if str(block.get("text", "")).strip():
                saw_text_block = True
                if saw_tool_result_block and not message_has_mixed_violation:
                    risks["tool_result_not_immediately_after_assistant_tool_use"] = (
                        risks.get(
                            "tool_result_not_immediately_after_assistant_tool_use",
                            0,
                        )
                        + 1
                    )
                    message_has_mixed_violation = True
            continue

        if block_type != "tool_result":
            if saw_tool_result_block and not message_has_mixed_violation:
                risks["tool_result_not_immediately_after_assistant_tool_use"] = (
                    risks.get(
                        "tool_result_not_immediately_after_assistant_tool_use",
                        0,
                    )
                    + 1
                )
                message_has_mixed_violation = True
            continue

        saw_tool_result_block = True
        tool_result_count += 1
        if saw_text_block and not message_has_order_violation:
            risks["user_tool_result_after_text"] = risks.get("user_tool_result_after_text", 0) + 1
            message_has_order_violation = True

        tool_use_id = str(block.get("tool_use_id", "")).strip()
        if not tool_use_id:
            continue
        if tool_use_id not in seen_tool_use_ids:
            risks["tool_result_unknown_tool_use_id"] = risks.get("tool_result_unknown_tool_use_id", 0) + 1
        if ordered_match_count >= len(pending_tool_use_ids):
            risks["tool_result_not_immediately_after_assistant_tool_use"] = (
                risks.get("tool_result_not_immediately_after_assistant_tool_use", 0) + 1
            )
            continue
        expected_tool_use_id = pending_tool_use_ids[ordered_match_count]
        if tool_use_id == expected_tool_use_id:
            ordered_match_count += 1
            continue
        risks["tool_result_not_immediately_after_assistant_tool_use"] = (
            risks.get("tool_result_not_immediately_after_assistant_tool_use", 0) + 1
        )
    return tool_result_count, ordered_match_count


def _flush_pending_tool_uses(
    risks: dict[str, int],
    pending_tool_use_ids: list[str],
) -> None:
    """Add missing-tool-result risk for any unresolved pending tool uses."""
    if pending_tool_use_ids:
        risks["assistant_tool_use_missing_next_tool_result"] = risks.get(
            "assistant_tool_use_missing_next_tool_result", 0
        ) + len(pending_tool_use_ids)


def _has_tool_result_blocks(content: list[dict[str, Any]]) -> bool:
    """Return True if any content block is a tool_result."""
    return any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def _check_fragmentation_risk(
    content: list[dict[str, Any]],
    pending_tool_use_ids: list[str],
    awaiting_tool_results_for_message: bool,
    user_already_responded: bool,
    risks: dict[str, int],
) -> None:
    """Detect tool_results split across multiple user messages."""
    if not (awaiting_tool_results_for_message and pending_tool_use_ids and user_already_responded):
        return
    if _has_tool_result_blocks(content):
        risks["tool_results_fragmented_across_user_messages"] = risks.get(
            "tool_results_fragmented_across_user_messages", 0
        ) + len(pending_tool_use_ids)


def _evaluate_pending_coverage(
    risks: dict[str, int],
    pending_tool_use_ids: list[str],
    tool_result_count: int,
    matched_count: int,
) -> None:
    """Check whether pending tool_uses were fully covered by tool_results."""
    if not pending_tool_use_ids:
        return
    if tool_result_count == 0 or matched_count == 0:
        risks["assistant_tool_use_missing_next_tool_result"] = risks.get(
            "assistant_tool_use_missing_next_tool_result", 0
        ) + len(pending_tool_use_ids)
    elif matched_count != len(pending_tool_use_ids):
        risks["assistant_tool_use_incomplete_next_tool_result_coverage"] = (
            risks.get("assistant_tool_use_incomplete_next_tool_result_coverage", 0)
            + len(pending_tool_use_ids)
            - matched_count
        )


def _detect_thinking_between_tool_use(
    content: list[dict[str, Any]],
    risks: dict[str, int],
) -> None:
    """
    Detect thinking/redacted_thinking blocks interleaved between tool_use blocks.

    This is a detection-only signal for telemetry: it catches cases where
    normalization did not run (e.g., raw original body in fail-open path).
    After normalization, this should never fire.
    """
    in_tool_run = False
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            in_tool_run = True
        elif block_type in {"thinking", "redacted_thinking"} and in_tool_run:
            risks["thinking_between_tool_use"] = risks.get("thinking_between_tool_use", 0) + 1


def _collect_bridge_tool_result_shape_risks(
    messages: list[dict[str, Any]],
) -> dict[str, int]:
    """Return Anthropic-specific tool_result ordering/pairing risks."""
    if not isinstance(messages, list):
        return {}

    risks: dict[str, int] = {}
    seen_tool_use_ids: set[str] = set()
    pending_tool_use_ids: list[str] = []
    awaiting_tool_results_for_message: bool = False
    user_already_responded_to_current_assistant: bool = False

    for _i, message in enumerate(messages):
        if not isinstance(message, dict):
            _flush_pending_tool_uses(risks, pending_tool_use_ids)
            pending_tool_use_ids = []
            awaiting_tool_results_for_message = False
            continue

        role = str(message.get("role", "")).strip()
        content = message.get("content")

        if role == "assistant":
            _flush_pending_tool_uses(risks, pending_tool_use_ids)
            if isinstance(content, list):
                _detect_thinking_between_tool_use(content, risks)
            pending_tool_use_ids = _process_assistant_tool_ids(
                content if isinstance(content, list) else [], seen_tool_use_ids
            )
            awaiting_tool_results_for_message = bool(pending_tool_use_ids)
            user_already_responded_to_current_assistant = False
            continue

        if role != "user":
            _flush_pending_tool_uses(risks, pending_tool_use_ids)
            pending_tool_use_ids = []
            awaiting_tool_results_for_message = False
            continue

        if isinstance(content, str) or not isinstance(content, list):
            _flush_pending_tool_uses(risks, pending_tool_use_ids)
            pending_tool_use_ids = []
            awaiting_tool_results_for_message = False
            continue

        _check_fragmentation_risk(
            content,
            pending_tool_use_ids,
            awaiting_tool_results_for_message,
            user_already_responded_to_current_assistant,
            risks,
        )

        if _has_tool_result_blocks(content):
            user_already_responded_to_current_assistant = True

        (
            tool_result_count,
            matched_pending_tool_use_count,
        ) = _process_user_tool_results(content, seen_tool_use_ids, pending_tool_use_ids, risks)
        _evaluate_pending_coverage(
            risks,
            pending_tool_use_ids,
            tool_result_count,
            matched_pending_tool_use_count,
        )
        pending_tool_use_ids = []
        awaiting_tool_results_for_message = False

    _flush_pending_tool_uses(risks, pending_tool_use_ids)

    return risks


def _collect_bridge_provider_sensitivity_risks(
    messages: list[dict[str, Any]],
) -> dict[str, int]:
    """Return provider-sensitive mixed assistant tool/text batch risks."""
    if not isinstance(messages, list):
        return {}

    risks: dict[str, int] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip() != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        tool_positions = [
            block_index
            for block_index, block in enumerate(content)
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if not tool_positions:
            continue
        first_tool = tool_positions[0]
        tool_use_count = len(tool_positions)
        has_text_between_or_after_tool_uses = any(
            isinstance(block, dict) and block.get("type") == "text" and block_index > first_tool
            for block_index, block in enumerate(content)
        )
        if tool_use_count >= _PROVIDER_SENSITIVE_LARGE_TOOL_BATCH_THRESHOLD:
            risks["assistant_large_tool_use_batch"] = risks.get("assistant_large_tool_use_batch", 0) + 1
        if has_text_between_or_after_tool_uses:
            risks["assistant_tool_use_text_interleaving"] = risks.get("assistant_tool_use_text_interleaving", 0) + 1
        if tool_use_count >= _PROVIDER_SENSITIVE_LARGE_TOOL_BATCH_THRESHOLD and has_text_between_or_after_tool_uses:
            next_message = messages[index + 1] if index + 1 < len(messages) else None
            next_content = next_message.get("content") if isinstance(next_message, dict) else None
            next_tool_result_count = 0
            if isinstance(next_content, list):
                next_tool_result_count = sum(
                    1 for block in next_content if isinstance(block, dict) and block.get("type") == "tool_result"
                )
            risks["assistant_large_tool_use_text_interleaving"] = (
                risks.get("assistant_large_tool_use_text_interleaving", 0) + 1
            )
            if next_tool_result_count:
                risks["provider_sensitive_large_tool_use_text_interleaving"] = (
                    risks.get(
                        "provider_sensitive_large_tool_use_text_interleaving",
                        0,
                    )
                    + 1
                )

    return risks


def _summarize_message_blocks(
    content: str | list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[str]:
    """Summarize blocks within a message."""
    blocks_summary = []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                blocks_summary.append("non_dict")
                continue
            b_type = b.get("type", "unknown")
            blocks_summary.append(str(b_type))
            if b_type == "tool_use":
                summary["tool_use_blocks"] += 1
            elif b_type == "tool_result":
                summary["tool_result_blocks"] += 1
            elif b_type not in _ALLOWED_BLOCK_TYPES:
                unsupported = summary["unsupported_blocks"]
                unsupported[b_type] = unsupported.get(b_type, 0) + 1
    elif isinstance(content, str):
        blocks_summary.append("str")
    else:
        blocks_summary.append("empty" if content is None else "unknown")
    return blocks_summary


def summarize_message_structure(
    messages: list[dict[str, Any]],
) -> str | dict[str, Any]:
    """Return a compact structural summary safe for bridge diagnostics."""
    if not isinstance(messages, list):
        return f"invalid_messages_type:{type(messages).__name__}"

    summary: dict[str, Any] = {
        "count": len(messages),
        "sequence": [],
        "user_msgs": 0,
        "assistant_msgs": 0,
        "tool_use_blocks": 0,
        "tool_result_blocks": 0,
        "unsupported_blocks": {},
        "field_shape_risks": {},
    }

    role_seq: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            role_seq.append(f"<{type(msg).__name__}>")
            continue

        role = str(msg.get("role", "none"))
        if role == "user":
            summary["user_msgs"] += 1
        elif role == "assistant":
            summary["assistant_msgs"] += 1

        content = msg.get("content")
        # Pass content as-is if str or list, otherwise empty list
        typed_content: str | list[dict[str, Any]] = content if isinstance(content, (str, list)) else []
        blocks_summary = _summarize_message_blocks(typed_content, summary)

        role_seq.append(f"{role}[{','.join(blocks_summary)}]")

    summary["sequence"] = role_seq
    summary["field_shape_risks"] = _collect_bridge_tool_result_shape_risks(messages)
    summary["provider_sensitivity_risks"] = _collect_bridge_provider_sensitivity_risks(messages)
    return summary


def summarize_bridge_pairing(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return assistant->next-user tool pairing snapshots for bridge diagnostics."""
    if not isinstance(messages, list):
        return []
    timeline: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip() != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        tool_use_ids = [
            str(block.get("id", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if not tool_use_ids:
            continue
        next_message = messages[index + 1] if index + 1 < len(messages) else None
        next_role = str(next_message.get("role", "")).strip() if isinstance(next_message, dict) else "<none>"
        next_content = next_message.get("content") if isinstance(next_message, dict) else None
        next_tool_result_ids: list[str] = []
        if isinstance(next_content, list):
            next_tool_result_ids = [
                str(block.get("tool_use_id", "")).strip()
                for block in next_content
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
        timeline.append(
            {
                "assistant_index": index,
                "next_role": next_role,
                "tool_use_ids": tool_use_ids,
                "next_tool_result_ids": next_tool_result_ids,
            }
        )
    return timeline


def _validate_tool_use_block(block: dict[str, Any], role: str, failures: list[str]) -> None:
    if role == "user":
        failures.append("user_contains_tool_use")
    if (
        not str(block.get("id", "")).strip()
        or not str(block.get("name", "")).strip()
        or not isinstance(block.get("input", {}), dict)
        or not _is_provider_safe_tool_id(str(block.get("id", "")))
    ):
        failures.append("invalid_tool_use_block")


def _validate_tool_result_block(block: dict[str, Any], role: str, failures: list[str]) -> None:
    if role == "assistant":
        failures.append("assistant_contains_tool_result")
    if (
        not str(block.get("tool_use_id", "")).strip()
        or not _is_provider_safe_tool_id(str(block.get("tool_use_id", "")))
        or not isinstance(block.get("content", ""), str | list)
    ):
        failures.append("invalid_tool_result_block")


def _validate_block(
    block: dict[str, str | dict | list],
    role: str,
    msg_index: int,
    block_index: int,
    failures: list[str],
) -> None:
    """Validate a single block within a message."""
    if not isinstance(block, dict):
        failures.append(f"message_{msg_index}_block_{block_index}_not_dict")
        return

    block_type = str(block.get("type", "")).strip()
    if not block_type:
        failures.append(f"message_{msg_index}_block_{block_index}_missing_type")
        return

    if block_type == "text":
        text = block.get("text")
        if not isinstance(text, str) or not text.strip():
            failures.append("empty_message_content")
        return

    if block_type == "tool_use":
        _validate_tool_use_block(block, role, failures)
        return

    if block_type == "tool_result":
        _validate_tool_result_block(block, role, failures)
        return

    if block_type in {"thinking", "redacted_thinking"}:
        if role != "assistant":
            failures.append("unsupported_block_type")
        return

    failures.append("unsupported_block_type")


def _validate_message(message: dict[str, Any], msg_index: int, failures: list[str]) -> None:
    """Validate a single message in the bridge body."""
    if not isinstance(message, dict):
        failures.append(f"message_{msg_index}_not_dict")
        return

    role = str(message.get("role", "")).strip()
    if role not in {"user", "assistant"}:
        failures.append("invalid_top_level_role")
        return

    content = message.get("content")
    if isinstance(content, str):
        if not content.strip():
            failures.append("empty_message_content")
        return

    if not isinstance(content, list):
        failures.append("empty_message_content")
        return
    if not content:
        failures.append("empty_content_blocks")
        return

    for block_index, block in enumerate(content):
        _validate_block(block, role, msg_index, block_index, failures)


def validate_anthropic_bridge_body(body: dict[str, Any]) -> list[str]:
    """Strictly validate the Anthropic bridge wire shape after canonicalization."""
    failures: list[str] = []
    if not isinstance(body, dict):
        return ["body_not_dict"]

    if not str(body.get("model", "")).strip():
        failures.append("missing_model")
        # Missing model means the bridge wire is not valid for upstream send.
        # This signal is consumed by the gateway preflight to block locally.
        failures.append("bridge_wire_model_invalid")

    messages = body.get("messages")
    if not isinstance(messages, list):
        failures.append("messages_not_list")
        # Non-list messages cannot be safely canonicalized or validated further.
        failures.append("bridge_wire_model_invalid")
        return list(set(failures))  # Unique stable codes
    if not messages:
        return ["empty_messages"]

    first_role = str(messages[0].get("role", "")).strip() if isinstance(messages[0], dict) else ""
    if first_role != "user":
        failures.append("first_message_not_user")

    for msg_index, message in enumerate(messages):
        _validate_message(message, msg_index, failures)

    shape_risks = _collect_bridge_tool_result_shape_risks(messages)
    if shape_risks.get("user_tool_result_after_text"):
        failures.append("user_tool_result_after_text")
    if shape_risks.get("assistant_tool_use_missing_next_tool_result"):
        failures.append("assistant_tool_use_missing_next_tool_result")
    if shape_risks.get("assistant_tool_use_incomplete_next_tool_result_coverage"):
        failures.append("assistant_tool_use_incomplete_next_tool_result_coverage")
    if shape_risks.get("tool_result_unknown_tool_use_id"):
        failures.append("tool_result_unknown_tool_use_id")
    if shape_risks.get("tool_result_not_immediately_after_assistant_tool_use"):
        failures.append("tool_result_not_immediately_after_assistant_tool_use")
    if shape_risks.get("tool_results_fragmented_across_user_messages"):
        failures.append("tool_results_fragmented_across_user_messages")

    system = body.get("system")
    if system is not None and not isinstance(system, str | list):
        failures.append("invalid_system_type")
    if not failures:
        failures.extend(_validate_canonical_bridge_body_model(body))
    return list(set(failures))  # Unique stable codes


def validate_anthropic_outgoing_bridge_body(body: dict[str, Any]) -> list[str]:
    """Validate the exact outgoing bridge body with provider-sensitive guards."""
    failures = validate_anthropic_bridge_body(body)
    if failures:
        return failures

    messages = body.get("messages")
    if not isinstance(messages, list):
        failures.append("messages_not_list")
        return failures
    provider_risks = _collect_bridge_provider_sensitivity_risks(messages)
    if provider_risks.get("provider_sensitive_large_tool_use_text_interleaving", 0):
        failures.append("provider_sensitive_large_tool_use_text_interleaving")
    if provider_risks.get("assistant_tool_use_text_interleaving", 0):
        failures.append("provider_sensitive_assistant_tool_use_text_interleaving")

    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        failures.append("missing_max_tokens")
    elif not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        failures.append("invalid_max_tokens")

    return failures


def has_provider_sensitive_failures(failures: list[str]) -> bool:
    return any(failure in _PROVIDER_SENSITIVE_FAILURES for failure in failures)


def has_recoverable_immediate_pairing_failures(
    failures: list[str],
) -> bool:
    return any(failure in _RECOVERABLE_IMMEDIATE_PAIRING_FAILURES for failure in failures)


def has_blocking_outgoing_failures(failures: list[str]) -> bool:
    return any(failure not in _NON_BLOCKING_OUTGOING_FAILURES for failure in failures)


def _is_valid_content_block(block: object) -> bool:
    """Helper for generic runtime validation."""
    if not isinstance(block, dict):
        return False
    block_type = str(block.get("type", "")).strip()
    if not block_type:
        return False
    if block_type == "text":
        return isinstance(block.get("text", ""), str)
    if block_type == "tool_use":
        return isinstance(block.get("name", ""), str) and isinstance(block.get("input", {}), dict)
    if block_type == "tool_result":
        return isinstance(block.get("tool_use_id", ""), str)
    return True


def _validate_message_basic(msg: dict[str, Any], failures: list[str]) -> bool:
    """Validate a single message basic structure. Returns True if failed."""
    if not isinstance(msg, dict):
        failures.append("message_not_dict")
        return True
    if str(msg.get("role", "")).strip() not in {
        "user",
        "assistant",
        "tool_result",
        "system",
    }:
        failures.append("invalid_message_role")
        return True
    content = msg.get("content", "")
    if isinstance(content, list):
        if not all(_is_valid_content_block(block) for block in content):
            failures.append("invalid_content_block")
            return True
    elif not isinstance(content, str):
        failures.append("invalid_message_content")
        return True
    return False


def validate_anthropic_request_body(body: dict[str, Any]) -> list[str]:
    """
    Validate the structure of an Anthropic API request body.

    Returns a list of failure reason strings, or an empty list if valid.
    """
    failures: list[str] = []
    if not isinstance(body, dict):
        return ["body_not_dict"]
    if not str(body.get("model", "")).strip():
        failures.append("missing_model")
    messages = body.get("messages")
    if not isinstance(messages, list):
        failures.append("messages_not_list")
    else:
        if not messages:
            failures.append("empty_messages")
        for msg in messages:
            if _validate_message_basic(msg, failures):
                break
    system = body.get("system")
    if system is not None and not isinstance(system, str | list):
        failures.append("invalid_system_type")
    if isinstance(system, list):
        if not all(_is_valid_content_block(block) for block in system):
            failures.append("invalid_system_block")
    return failures


def detect_prompt_bloat(system_prompt: str | list[dict[str, Any]] | None, user_prompt: str = "") -> bool:
    """
    Identify when system prompts are unusually large or contain leaked user content.

    Returns True if the system prompt exceeds the TOK_PROMPT_BLOAT_THRESHOLD (default 2000)
    or if it appears to contain a substantial portion of the current user prompt.
    """
    if system_prompt is None:
        return False

    # Threshold for automatic optimization (chars)
    bloat_threshold = int(os.getenv("TOK_PROMPT_BLOAT_THRESHOLD", str(_DEFAULT_PROMPT_BLOAT_THRESHOLD_CHARS)))

    system_text = ""
    if isinstance(system_prompt, list):
        system_text = " ".join(
            str(block.get("text", ""))
            for block in system_prompt
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        system_text = str(system_prompt)

    if len(system_text) > bloat_threshold:
        return True

    # Check if user prompt content is leaking into system context (e.g. flattening)
    if user_prompt and len(user_prompt) > _USER_PROMPT_LEAK_MIN_CHARS:
        # Check if a substantial part of the user prompt is in the system prompt
        snippet = user_prompt[:_USER_PROMPT_LEAK_SNIPPET_CHARS].strip()
        if snippet and snippet in system_text:
            return True

    return False


def should_optimize_prompts(
    system_prompt: str | list[dict[str, Any]] | None,
    session_metrics: dict[str, int],
) -> bool:
    """Check if optimization is recommended based on size thresholds or size metrics."""
    # Threshold for intervention (chars)
    size_limit = int(os.getenv("TOK_PROMPT_OPTIMIZE_LIMIT", str(_DEFAULT_PROMPT_OPTIMIZE_LIMIT_CHARS)))

    system_text = ""
    if isinstance(system_prompt, list):
        system_text = " ".join(
            str(block.get("text", ""))
            for block in system_prompt
            if isinstance(block, dict) and block.get("type") == "text"
        )
    elif system_prompt:
        system_text = str(system_prompt)

    if len(system_text) > size_limit:
        return True

    # Check for high growth rate signal if provided in metrics
    if session_metrics.get("tok_prompt_growth_high"):
        return True

    return detect_prompt_bloat(system_prompt)


__all__ = [
    "bridge_strict_failure_signals",
    "canonicalize_anthropic_bridge_body",
    "canonicalize_anthropic_bridge_messages",
    "detect_prompt_bloat",
    "has_provider_sensitive_failures",
    "has_recoverable_immediate_pairing_failures",
    "normalize_tool_use_blocks",
    "quarantine_invalid_tool_history_messages",
    "should_optimize_prompts",
    "summarize_bridge_pairing",
    "summarize_message_structure",
    "validate_anthropic_bridge_body",
    "validate_anthropic_outgoing_bridge_body",
    "validate_anthropic_request_body",
]
