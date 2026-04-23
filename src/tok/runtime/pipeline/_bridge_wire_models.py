"""Pydantic models and helpers for Anthropic-style bridge wire shapes.

This module centralizes the canonical block/message/body models used by
`request_validation` so that the main validation pipeline can stay focused on
policy decisions and error handling.
"""

from __future__ import annotations

import copy
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator

_PROVIDER_SAFE_TOOL_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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
    def _validate_tool_use(self) -> _CanonicalToolUseBlock:
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
    def _validate_tool_result(self) -> _CanonicalToolResultBlock:
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
    def _enforce_cross_role_shapes(self) -> _CanonicalBridgeMessage:
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
