"""Bridge comparison helpers for request/response diffing."""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..runtime.pipeline.request_validation import (
    canonicalize_anthropic_bridge_body,
)

_DROP_RESPONSE_HEADERS = {
    "content-encoding",
    "content-length",
    "transfer-encoding",
    "connection",
}


def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
    """Remove headers that shouldn't be forwarded in responses."""
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in _DROP_RESPONSE_HEADERS
    }


def _payloads_materially_differ(
    content: bytes, original_content: bytes | None
) -> bool:
    """Return True when two JSON payloads differ materially."""
    if original_content is None or original_content == content:
        return False
    try:
        current = json.loads(content)
        original = json.loads(original_content)
    except Exception:
        return True
    if isinstance(current, dict) and isinstance(original, dict):
        current = dict(current)
        original = dict(original)
        if current.get("system", None) == "" and "system" not in original:
            current.pop("system", None)
        if original.get("system", None) == "" and "system" not in current:
            original.pop("system", None)
        current, _, _ = canonicalize_anthropic_bridge_body(current)
        original, _, _ = canonicalize_anthropic_bridge_body(original)
    return bool(current != original)


def _bridge_bodies_materially_differ(
    current: dict[str, Any], original: dict[str, Any]
) -> bool:
    """Return True when two bridge bodies differ after canonical comparison."""
    if not isinstance(current, dict) or not isinstance(original, dict):
        return current != original

    current_cmp = dict(current)
    original_cmp = dict(original)
    if current_cmp.get("system", None) == "" and "system" not in original_cmp:
        current_cmp.pop("system", None)
    if original_cmp.get("system", None) == "" and "system" not in current_cmp:
        original_cmp.pop("system", None)
    current_cmp, _, _ = canonicalize_anthropic_bridge_body(current_cmp)
    original_cmp, _, _ = canonicalize_anthropic_bridge_body(original_cmp)
    return bool(current_cmp != original_cmp)


def _request_fingerprint_diff(
    headers: dict[str, str],
    current_body: dict[str, Any],
    original_body: dict[str, Any],
) -> dict[str, Any]:
    """Compare request topology before and after Tok rewriting."""
    from ._fingerprint import (
        _request_body_fingerprint,
    )

    current_fp = _request_body_fingerprint(headers, current_body)
    original_fp = _request_body_fingerprint(headers, original_body)
    current_cache = current_fp["cache_control"]
    original_cache = original_fp["cache_control"]

    topology_reasons: list[str] = []
    original_system_type = str(original_fp["system"]["type"])
    current_system_type = str(current_fp["system"]["type"])
    if {original_system_type, current_system_type} <= {"list", "str"} and (
        original_system_type != current_system_type
    ):
        topology_reasons.append("system_type_changed")
    if current_cache["total"] != original_cache["total"]:
        topology_reasons.append("cache_control_total_changed")
    if current_cache["system_blocks"] != original_cache["system_blocks"]:
        topology_reasons.append("system_cache_control_changed")
    if (
        current_cache["message_text_blocks"]
        != original_cache["message_text_blocks"]
    ):
        topology_reasons.append("message_text_cache_control_changed")

    original_text_or_system = (
        original_cache["system_blocks"] + original_cache["message_text_blocks"]
    )
    current_text_or_system = (
        current_cache["system_blocks"] + current_cache["message_text_blocks"]
    )
    current_tool_or_tools = (
        current_cache["message_tool_result_blocks"]
        + current_cache["message_tool_use_blocks"]
        + current_cache["tools"]
    )
    if (
        original_text_or_system > 0
        and current_text_or_system == 0
        and current_tool_or_tools > 0
    ):
        topology_reasons.append(
            "text_system_cache_control_removed_only_tool_cache_remains"
        )

    removed_counts = {
        "system_blocks": max(
            0,
            int(original_cache["system_blocks"])
            - int(current_cache["system_blocks"]),
        ),
        "message_text_blocks": max(
            0,
            int(original_cache["message_text_blocks"])
            - int(current_cache["message_text_blocks"]),
        ),
    }

    return {
        "anthropic_beta": current_fp["anthropic_beta"],
        "prompt_caching": bool(
            original_fp["prompt_caching"] or current_fp["prompt_caching"]
        ),
        "body_materially_differs": _bridge_bodies_materially_differ(
            current_body, original_body
        ),
        "messages_changed": current_body.get("messages")
        != original_body.get("messages"),
        "system_changed": original_fp["system"] != current_fp["system"],
        "cache_topology_changed": bool(topology_reasons),
        "cache_topology_reasons": topology_reasons,
        "removed_text_or_system_cache_control": any(removed_counts.values()),
        "removed_cache_control": removed_counts,
        "original": original_fp,
        "rewritten": current_fp,
    }
