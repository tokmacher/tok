"""Domain-specific exceptions for Tok."""

from __future__ import annotations

__all__ = [
    "BridgeUnavailableError",
    "CompressionError",
    "InvalidSessionStateError",
    "ReplayGateError",
    "SessionError",
    "TokError",
    "TokSafetyError",
]


class TokError(Exception):
    """Base class for all Tok-specific errors."""


class CompressionError(TokError):
    """Signals that a compression or decompression operation failed."""


class SessionError(TokError):
    """Indicates that the current runtime session could not be used."""


class BridgeUnavailableError(TokError):
    """Raised when the bridge or gateway could not be reached."""


class ReplayGateError(TokError):
    """Raised when the replay gate detects an invalid state."""


class InvalidSessionStateError(SessionError):
    """Raised when the session is not in a valid state for the requested action."""


class TokSafetyError(TokError):
    """Raised when a safety check fails and prevents an unsafe operation."""
