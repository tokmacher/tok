"""Compatibility shim for provider request shape helpers.

Tok 0.2.x uses `tok.provider_request_shapes` as the implementation module.
Some older imports still reference `tok.provider_shapes`.
"""

from __future__ import annotations

from typing import Any

from .provider_request_shapes import (
    canonicalize_bridge_body,
    validate_bridge_body,
    validate_outgoing_bridge_body,
    validate_request_body,
)

__all__ = [
    "canonicalize_bridge_body",
    "validate_bridge_body",
    "validate_outgoing_bridge_body",
    "validate_request_body",
]

# Keep a local reference for type checkers; the runtime functions above are the
# supported surface.
_AnyDict = dict[str, Any]
