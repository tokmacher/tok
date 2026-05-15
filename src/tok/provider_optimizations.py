"""Provider-specific request optimizations behind a provider-neutral entrypoint.

These optimizations are optional and must not be required for correctness. They
exist to reduce token/cost overhead for certain provider surfaces (for example
Anthropic prompt caching behavior).

Core runtime and compression code should not import gateway/provider-specific
modules directly. Callers should route through this module.
"""

from __future__ import annotations

from typing import Any


def apply_provider_optimizations(
    *,
    adapter_kind: str,
    body: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """
    Apply best-effort provider-specific rewrites.

    Returns (body, saved_chars_estimate). For unknown providers this is a no-op.
    """
    if adapter_kind == "claude-bridge":
        from tok.gateway._anthropic_optimizations import apply_anthropic_optimizations

        return apply_anthropic_optimizations(body)
    return body, 0
