"""Provider request-shape canonicalization and validation.

This module is the boundary between provider-neutral runtime/orchestration code
and provider-specific wire-shape details.

Tok 0.2.x supports the Claude Code bridge (Anthropic messages shape). Future
providers should add their own RequestShape implementations here (or under a
provider-specific subpackage) without changing the provider-neutral core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ProviderRequestShape(Protocol):
    """Provider-owned canonicalization/validation for incoming/outgoing request bodies."""

    def canonicalize_bridge_body(
        self,
        body: dict[str, Any],
        *,
        seen_mutation_pairs: set[tuple[str, str]] | None = None,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        raise NotImplementedError

    def validate_bridge_body(self, body: dict[str, Any]) -> list[str]:
        raise NotImplementedError

    def validate_outgoing_bridge_body(self, body: dict[str, Any]) -> list[str]:
        raise NotImplementedError

    def validate_request_body(self, body: dict[str, Any]) -> list[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class AnthropicBridgeRequestShape:
    """Claude Code bridge wire shape (Anthropic messages format)."""

    def canonicalize_bridge_body(
        self,
        body: dict[str, Any],
        *,
        seen_mutation_pairs: set[tuple[str, str]] | None = None,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        from tok.runtime.pipeline.request_validation import canonicalize_anthropic_bridge_body

        return canonicalize_anthropic_bridge_body(body, seen_mutation_pairs=seen_mutation_pairs)

    def validate_bridge_body(self, body: dict[str, Any]) -> list[str]:
        from tok.runtime.pipeline.request_validation import validate_anthropic_bridge_body

        return validate_anthropic_bridge_body(body)

    def validate_outgoing_bridge_body(self, body: dict[str, Any]) -> list[str]:
        from tok.runtime.pipeline.request_validation import validate_anthropic_outgoing_bridge_body

        return validate_anthropic_outgoing_bridge_body(body)

    def validate_request_body(self, body: dict[str, Any]) -> list[str]:
        from tok.runtime.pipeline.request_validation import validate_anthropic_request_body

        return validate_anthropic_request_body(body)


_ANTHROPIC_SHAPE = AnthropicBridgeRequestShape()


def canonicalize_bridge_body(
    body: dict[str, Any],
    *,
    seen_mutation_pairs: set[tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    return _ANTHROPIC_SHAPE.canonicalize_bridge_body(body, seen_mutation_pairs=seen_mutation_pairs)


def validate_bridge_body(body: dict[str, Any]) -> list[str]:
    return _ANTHROPIC_SHAPE.validate_bridge_body(body)


def validate_outgoing_bridge_body(body: dict[str, Any]) -> list[str]:
    return _ANTHROPIC_SHAPE.validate_outgoing_bridge_body(body)


def validate_request_body(body: dict[str, Any]) -> list[str]:
    return _ANTHROPIC_SHAPE.validate_request_body(body)
