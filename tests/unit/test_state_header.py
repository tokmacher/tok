"""Tests for the optional per-turn Tok state response header.

The bridge tracks compression / baseline / fail-open state per turn, but it was
only visible in logs and `tok stats` -- not to the client/harness mediating the
agent. This opt-in header surfaces that state out-of-band (never injected into
model-visible content), so a harness can show "this turn was served baseline".
"""

from __future__ import annotations

from tok.gateway._bridge_comparison import apply_tok_state_header, tok_state_header_value


def test_state_header_value_reflects_turn_state() -> None:
    assert tok_state_header_value({}, baseline_only=False) == "compressed"
    assert tok_state_header_value({}, baseline_only=True) == "baseline"
    # Fail-open dominates: a degraded/retried turn is the most important to surface.
    assert tok_state_header_value({"tok_fallback_activated": 1}, baseline_only=False) == "fail-open"
    assert tok_state_header_value({"tok_fail_open_retry": 1}, baseline_only=True) == "fail-open"


def test_apply_state_header_is_opt_in() -> None:
    base = {"content-type": "application/json"}

    disabled = apply_tok_state_header(dict(base), {}, baseline_only=True, enabled=False)
    assert "x-tok-state" not in disabled
    assert disabled == base

    enabled = apply_tok_state_header(dict(base), {}, baseline_only=True, enabled=True)
    assert enabled["x-tok-state"] == "baseline"
    # Existing headers are preserved.
    assert enabled["content-type"] == "application/json"
