"""Shared request-policy helpers for gateway and benchmark parity."""

from __future__ import annotations

import os

REQUEST_POLICY_ALIASES = {
    "legacy": "legacy_tool_compatible",
    "legacy_tool_compatible": "legacy_tool_compatible",
    "tool_compatible": "legacy_tool_compatible",
    "tool-compatible": "legacy_tool_compatible",
    "natural": "natural_first",
    "natural_first": "natural_first",
    "natural-first": "natural_first",
    "baseline": "forced_baseline",
    "forced_baseline": "forced_baseline",
    "forced-baseline": "forced_baseline",
}


def normalize_request_policy(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return ""
    return REQUEST_POLICY_ALIASES.get(normalized, "")


def default_request_policy(
    *,
    tok_mode: str | None = None,
    request_policy: str | None = None,
) -> str:
    resolved_tok_mode = (tok_mode if tok_mode is not None else os.getenv("TOK_MODE", "tool-compatible")).strip().lower()
    if resolved_tok_mode == "baseline":
        return "forced_baseline"
    explicit_policy = normalize_request_policy(
        request_policy if request_policy is not None else os.getenv("TOK_REQUEST_POLICY", "")
    )
    if explicit_policy:
        return explicit_policy
    return "natural_first"


def request_policy_mode_label(policy: str) -> str:
    if policy == "forced_baseline":
        return "baseline"
    if policy == "natural_first":
        return "natural-first"
    return "tool-compatible"
