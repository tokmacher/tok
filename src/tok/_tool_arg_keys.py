"""Single source of truth for tool-call argument keys with cross-layer meaning.

Deliberately a dependency-free leaf module (it imports nothing from ``tok``) so
both the compression layer (``tok.compression``) and the runtime layer
(``tok.runtime``) can import it without creating an import cycle between those
two packages.
"""

from __future__ import annotations

# Argument keys that denote a *bounded* ("precision") file-read window: the agent
# asked for a specific slice rather than the whole file.
#
# Used by the compression layer to exempt such reads from skeletonization / lossy
# truncation and to keep the overlap-delta coverage tracker in sync with what was
# actually delivered, and by the runtime layer to tell a verbatim (window-less)
# read apart from a precision read during skeleton-edit recovery.
#
# Extend the precision definition *here* -- never re-inline a copy elsewhere.
PRECISION_READ_ARG_KEYS: tuple[str, ...] = ("offset", "limit", "start", "end")
