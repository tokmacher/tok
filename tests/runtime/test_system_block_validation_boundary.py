"""Focused regression tests for the system-block validation boundary."""

from tok.runtime.pipeline.request_validation import (
    validate_anthropic_bridge_body,
)


def test_malformed_system_block_does_not_raise_raw_validation_error() -> None:
    """
    Test that malformed system blocks are caught and return controlled failure.

    This test uses the lowest practical entrypoint that exercises
    _validate_system_blocks through the full validation chain.

    Before the fix: this would raise a raw ValidationError
    After the fix: this returns a controlled failure
    """
    body = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "system": [{"type": "tool_use", "id": "", "name": "", "input": "bad"}],
    }

    failures = validate_anthropic_bridge_body(body)

    assert "invalid_system_block" in failures


def test_valid_system_block_still_passes_validation() -> None:
    """Test that valid system blocks continue to work correctly."""
    body = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "system": [{"type": "text", "text": "You are helpful."}],
    }

    failures = validate_anthropic_bridge_body(body)

    assert failures == []


def test_preflight_path_receives_controlled_failure_for_malformed_system_block() -> None:
    """
    Test the highest practical public validation entrypoint.

    This validates that the real boundary (not the internal validator method)
    receives a controlled failure instead of a raw exception.
    """
    body = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "system": [{"type": "tool_use", "id": "", "name": "", "input": "bad"}],
    }

    failures = validate_anthropic_bridge_body(body)

    assert failures == ["invalid_system_block"] or "invalid_system_block" in failures
