"""Backward-compatible facade for tok.protocol.schema."""

from __future__ import annotations

from .protocol.schema import BlockSchema, DEFAULT_SCHEMA, TokSchema

__all__ = [
    "DEFAULT_SCHEMA",
    "BlockSchema",
    "TokSchema",
]
