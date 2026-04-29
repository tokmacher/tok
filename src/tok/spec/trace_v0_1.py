"""Draft Tok Trace v0.1 fixture validation.

This module validates the docs/spec fixture shape only. It is not runtime bridge
emission and it is not a supported public API for 0.1.x.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRACE_VERSION = "tok-trace/v0.1-draft"
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

ALLOWED_DIRECTIONS = {"request", "response"}
ALLOWED_CLASSES = {"file", "search", "tool", "message", "system", "response"}
ALLOWED_ACTIONS = {
    "pass_through",
    "store",
    "reference",
    "delta",
    "fallback",
    "skeleton_reference",
    "summary_reference",
}
ALLOWED_RESULTS = {"ok", "degraded", "error", "rejected"}
ALLOWED_RESOLVER_STATES = {
    "available_local",
    "resolvable_remote",
    "missing_identifiable",
    "unresolvable_fallback_required",
}
ALLOWED_EXPECTATIONS = {
    "accept_exact",
    "accept_reference",
    "accept_delta",
    "accept_fallback",
    "reject_malformed",
    "accept_non_exact_reference",
}
ALLOWED_DELTA_ALGORITHMS = {"line", "unified_diff", "json_patch", "binary"}


@dataclass(frozen=True)
class AuditResult:
    """Validation result for one draft trace fixture."""

    id: str
    status: str
    errors: tuple[str, ...] = ()
    summary: str = ""


def audit_fixture_file(path: Path) -> list[AuditResult]:
    """Audit a JSON fixture file containing draft trace fixture objects."""
    fixtures = json.loads(path.read_text())
    if not isinstance(fixtures, list):
        return [AuditResult(id=str(path), status="fail", errors=("fixture_file_not_list",))]

    results: list[AuditResult] = []
    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture, dict):
            results.append(AuditResult(id=f"fixture[{index}]", status="fail", errors=("fixture_not_object",)))
            continue

        fixture_id = str(fixture.get("id") or f"fixture[{index}]")
        block = fixture.get("block")
        if not isinstance(block, dict):
            results.append(AuditResult(id=fixture_id, status="fail", errors=("missing_or_invalid_block",)))
            continue

        results.append(audit_block(block, fixture_id=fixture_id))

    return results


def audit_block(block: dict[str, Any], *, fixture_id: str = "trace-block") -> AuditResult:
    """Return pass/warn/fail status for a draft trace block."""
    errors = validate_block(block)
    if errors:
        return AuditResult(id=fixture_id, status="fail", errors=tuple(errors))

    resolver_state = block["audit"]["resolver_state"]
    if resolver_state in {"missing_identifiable", "unresolvable_fallback_required"}:
        return AuditResult(id=fixture_id, status="warn")

    return AuditResult(id=fixture_id, status="pass")


def validate_block(block: dict[str, Any]) -> list[str]:
    """Validate one Tok Trace v0.1 draft block and return error codes."""
    errors: list[str] = []
    envelope = block.get("envelope")
    observation = block.get("observation")
    content = block.get("content")
    audit = block.get("audit")

    for section_name, section in (
        ("envelope", envelope),
        ("observation", observation),
        ("content", content),
        ("audit", audit),
    ):
        if not isinstance(section, dict):
            errors.append(f"missing_or_invalid_section:{section_name}")

    if errors:
        return errors

    _validate_envelope(envelope, errors)
    _validate_observation(observation, errors)
    _validate_content(observation, content, errors)
    _validate_audit(observation, content, audit, errors)
    return errors


def _validate_envelope(envelope: dict[str, Any], errors: list[str]) -> None:
    if envelope.get("trace_version") != TRACE_VERSION:
        errors.append("invalid_trace_version")
    if not isinstance(envelope.get("block_id"), str) or not envelope["block_id"]:
        errors.append("invalid_block_id")
    if not isinstance(envelope.get("session_id"), str) or not envelope["session_id"]:
        errors.append("invalid_session_id")
    if not isinstance(envelope.get("turn"), int) or envelope["turn"] < 0:
        errors.append("invalid_turn")
    if not isinstance(envelope.get("step"), int) or envelope["step"] < 0:
        errors.append("invalid_step")
    if envelope.get("direction") not in ALLOWED_DIRECTIONS:
        errors.append("invalid_direction")

    payload_digest = envelope.get("payload_digest")
    if payload_digest != "draft-uncomputed" and not is_hash(payload_digest):
        errors.append("invalid_payload_digest")


def _validate_observation(observation: dict[str, Any], errors: list[str]) -> None:
    if observation.get("class") not in ALLOWED_CLASSES:
        errors.append("invalid_class")
    if not isinstance(observation.get("key"), str) or not observation["key"]:
        errors.append("invalid_key")
    if observation.get("action") not in ALLOWED_ACTIONS:
        errors.append("invalid_action")
    if observation.get("result") not in ALLOWED_RESULTS:
        errors.append("invalid_result")


def _validate_content(observation: dict[str, Any], content: dict[str, Any], errors: list[str]) -> None:
    action = observation.get("action")
    exact = content.get("exact")

    if not isinstance(exact, bool):
        errors.append("invalid_exact")
    if "hash" in content and not is_hash(content["hash"]):
        errors.append("invalid_hash")
    if "size_bytes" in content and (not isinstance(content["size_bytes"], int) or content["size_bytes"] < 0):
        errors.append("invalid_size_bytes")
    if "resolver_uri" in content and not isinstance(content["resolver_uri"], str):
        errors.append("invalid_resolver_uri")

    if action in {"store", "reference", "fallback"} and exact:
        _require_content_identity(content, errors)
    if action == "delta":
        _require_content_identity(content, errors)
        for field in ("base_hash", "delta_hash"):
            if not is_hash(content.get(field)):
                errors.append(f"missing_or_invalid_{field}")
        if not isinstance(content.get("delta_uri"), str) or not content["delta_uri"]:
            errors.append("missing_or_invalid_delta_uri")
        if content.get("delta_algorithm") not in ALLOWED_DELTA_ALGORITHMS:
            errors.append("missing_or_invalid_delta_algorithm")
    if action in {"skeleton_reference", "summary_reference"} and exact is not False:
        errors.append("non_exact_reference_marked_exact")


def _validate_audit(
    observation: dict[str, Any],
    content: dict[str, Any],
    audit: dict[str, Any],
    errors: list[str],
) -> None:
    action = observation.get("action")
    result = observation.get("result")
    resolver_state = audit.get("resolver_state")
    expectation = audit.get("expectation")

    if resolver_state not in ALLOWED_RESOLVER_STATES:
        errors.append("invalid_resolver_state")
    if expectation not in ALLOWED_EXPECTATIONS:
        errors.append("invalid_expectation")
    if expectation == "reject_malformed" and result != "rejected":
        errors.append("reject_fixture_must_have_rejected_result")

    needs_reason = (
        action == "fallback"
        or result in {"degraded", "error", "rejected"}
        or resolver_state in {"missing_identifiable", "unresolvable_fallback_required"}
    )
    if needs_reason and not audit.get("reason"):
        errors.append("missing_reason")

    if resolver_state == "available_local" and content.get("exact") and not content.get("resolver_uri"):
        errors.append("available_local_missing_resolver")


def _require_content_identity(content: dict[str, Any], errors: list[str]) -> None:
    if not is_hash(content.get("hash")):
        errors.append("missing_or_invalid_content_hash")
    if not isinstance(content.get("size_bytes"), int):
        errors.append("missing_or_invalid_content_size")


def is_hash(value: object) -> bool:
    """Return whether a value is a lowercase sha256 digest string."""
    return isinstance(value, str) and HASH_RE.match(value) is not None
