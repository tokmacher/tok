"""Tests for tok.exceptions module — exception hierarchy smoke tests."""

from __future__ import annotations

import pytest

from tok.exceptions import (
    BridgeUnavailableError,
    CompressionError,
    InvalidSessionStateError,
    ReplayGateError,
    SessionError,
    TokError,
    TokSafetyError,
)


class TestTokErrorHierarchy:
    """Verify exception hierarchy and inheritance."""

    def test_tok_error_is_base_exception(self) -> None:
        """TokError is the base class for all Tok-specific exceptions."""
        assert issubclass(TokError, Exception)

    def test_compression_error_inherits_from_tok_error(self) -> None:
        """CompressionError signals compression/decompression failures."""
        assert issubclass(CompressionError, TokError)

    def test_session_error_inherits_from_tok_error(self) -> None:
        """SessionError indicates runtime session could not be used."""
        assert issubclass(SessionError, TokError)

    def test_bridge_unavailable_error_inherits_from_tok_error(self) -> None:
        """BridgeUnavailableError raised when bridge/gateway could not be reached."""
        assert issubclass(BridgeUnavailableError, TokError)

    def test_replay_gate_error_inherits_from_tok_error(self) -> None:
        """ReplayGateError raised when replay gate detects invalid state."""
        assert issubclass(ReplayGateError, TokError)

    def test_invalid_session_state_error_inherits_from_session_error(self) -> None:
        """InvalidSessionStateError is a subclass of SessionError."""
        assert issubclass(InvalidSessionStateError, SessionError)
        assert issubclass(InvalidSessionStateError, TokError)

    def test_tok_safety_error_inherits_from_tok_error(self) -> None:
        """TokSafetyError raised when a safety check fails."""
        assert issubclass(TokSafetyError, TokError)


class TestExceptionRaising:
    """Verify each exception can be raised with a message."""

    def test_raise_bridge_unavailable_error(self) -> None:
        """BridgeUnavailableError can be raised with a message."""
        with pytest.raises(BridgeUnavailableError, match="test"):
            raise BridgeUnavailableError("test message")

    def test_raise_compression_error(self) -> None:
        """CompressionError can be raised with a message."""
        with pytest.raises(CompressionError, match="test"):
            raise CompressionError("test message")

    def test_raise_invalid_session_state_error(self) -> None:
        """InvalidSessionStateError can be raised with a message."""
        with pytest.raises(InvalidSessionStateError, match="test"):
            raise InvalidSessionStateError("test message")

    def test_raise_replay_gate_error(self) -> None:
        """ReplayGateError can be raised with a message."""
        with pytest.raises(ReplayGateError, match="test"):
            raise ReplayGateError("test message")

    def test_raise_session_error(self) -> None:
        """SessionError can be raised with a message."""
        with pytest.raises(SessionError, match="test"):
            raise SessionError("test message")

    def test_raise_tok_safety_error(self) -> None:
        """TokSafetyError can be raised with a message."""
        with pytest.raises(TokSafetyError, match="test"):
            raise TokSafetyError("test message")

    def test_raise_tok_error(self) -> None:
        """TokError base exception can be raised with a message."""
        with pytest.raises(TokError, match="test"):
            raise TokError("test message")


class TestExceptionChaining:
    """Verify exceptions support chaining via from syntax."""

    def test_compression_error_supports_chaining(self) -> None:
        """CompressionError can chain an underlying exception."""
        original = ValueError("underlying")
        try:
            raise CompressionError("compression failed") from original
        except CompressionError as e:
            assert e.__cause__ is original

    def test_session_error_supports_chaining(self) -> None:
        """SessionError can chain an underlying exception."""
        original = RuntimeError("session closed")
        try:
            raise SessionError("session unavailable") from original
        except SessionError as e:
            assert e.__cause__ is original

    def test_invalid_session_state_error_supports_chaining(self) -> None:
        """InvalidSessionStateError can chain an underlying exception."""
        original = KeyError("missing key")
        try:
            raise InvalidSessionStateError("invalid state") from original
        except InvalidSessionStateError as e:
            assert e.__cause__ is original
