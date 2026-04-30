"""Draft Tok Trace v0.1 fixture validation.

This module validates the docs/spec fixture shape only. It is not runtime bridge
emission and it is not a supported public API for 0.1.x.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
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
CORE_SECTION_NAMES = {"envelope", "observation", "content", "audit"}


@dataclass(frozen=True)
class AuditResult:
    """Validation result for one draft trace fixture."""

    id: str
    status: str
    errors: tuple[str, ...] = ()
    summary: str = ""


@dataclass(frozen=True)
class AuditContext:
    """Filesystem context for fixture-local audit resolution."""

    fixture_dir: Path | None = None


def audit_fixture_file(path: Path) -> list[AuditResult]:
    """Audit a JSON fixture file containing draft trace fixture objects."""
    try:
        fixtures = json.loads(path.read_text())
    except json.JSONDecodeError:
        return [AuditResult(id=str(path), status="fail", errors=("invalid_fixture_json",))]
    if not isinstance(fixtures, list):
        return [AuditResult(id=str(path), status="fail", errors=("fixture_file_not_list",))]

    results: list[AuditResult] = []
    sequence_blocks: list[dict[str, Any]] = []
    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture, dict):
            results.append(AuditResult(id=f"fixture[{index}]", status="fail", errors=("fixture_not_object",)))
            continue

        fixture_id = str(fixture.get("id") or f"fixture[{index}]")
        block = fixture.get("block")
        if not isinstance(block, dict):
            results.append(AuditResult(id=fixture_id, status="fail", errors=("missing_or_invalid_block",)))
            continue

        results.append(audit_block(block, fixture_id=fixture_id, context=AuditContext(path.parent)))
        sequence_blocks.append(block)

    results.extend(_audit_sequence_consistency(sequence_blocks))
    return results


def audit_trace_file(path: Path) -> list[AuditResult]:
    """Audit either a fixture JSON array or live JSONL trace file."""
    text = path.read_text()
    if text.lstrip().startswith("["):
        return audit_fixture_file(path)

    results: list[AuditResult] = []
    context = AuditContext(path.parent)
    sequence_blocks: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            results.append(AuditResult(id=f"line:{index}", status="fail", errors=("invalid_jsonl_line",)))
            continue
        if not isinstance(record, dict):
            results.append(AuditResult(id=f"line:{index}", status="fail", errors=("jsonl_line_not_object",)))
            continue
        block = record.get("block") if "block" in record else record
        if not isinstance(block, dict):
            results.append(AuditResult(id=f"line:{index}", status="fail", errors=("missing_or_invalid_block",)))
            continue
        block_id = str(record.get("id") or block.get("envelope", {}).get("block_id") or f"line:{index}")
        results.append(audit_block(block, fixture_id=block_id, context=context))
        sequence_blocks.append(block)
    if not results:
        return [AuditResult(id=str(path), status="fail", errors=("trace_file_empty",))]
    results.extend(_audit_sequence_consistency(sequence_blocks))
    return results


def audit_block(
    block: dict[str, Any],
    *,
    fixture_id: str = "trace-block",
    context: AuditContext | None = None,
) -> AuditResult:
    """Return pass/warn/fail status for a draft trace block."""
    errors = validate_block(block)
    if errors:
        return AuditResult(id=fixture_id, status="fail", errors=tuple(errors))

    warnings: list[str] = []
    digest_errors = _audit_payload_digest(block, warnings)
    if digest_errors:
        return AuditResult(id=fixture_id, status="fail", errors=tuple(digest_errors))
    artifact_errors = _audit_artifacts(block, context or AuditContext())
    if artifact_errors:
        return AuditResult(id=fixture_id, status="fail", errors=tuple(artifact_errors))

    resolver_state = block["audit"]["resolver_state"]
    if resolver_state in {"missing_identifiable", "unresolvable_fallback_required"}:
        warnings.append(resolver_state)

    if warnings:
        return AuditResult(id=fixture_id, status="warn", errors=tuple(warnings))

    return AuditResult(id=fixture_id, status="pass")


def canonical_payload_digest(block: dict[str, Any]) -> str:
    """Return the draft canonical digest for stable semantic payload fields."""
    payload = {key: block[key] for key in ("observation", "content", "audit")}
    if "extensions" in block:
        payload["extensions"] = block["extensions"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + sha256(canonical.encode("utf-8")).hexdigest()


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
    _validate_extensions(block.get("extensions"), errors)
    return errors


def _audit_sequence_consistency(blocks: list[dict[str, Any]]) -> list[AuditResult]:
    results: list[AuditResult] = []
    seen_block_ids: set[str] = set()
    last_position_by_session: dict[str, tuple[int, int]] = {}

    for index, block in enumerate(blocks, start=1):
        envelope = block.get("envelope")
        if not isinstance(envelope, dict):
            continue

        block_id = envelope.get("block_id")
        if isinstance(block_id, str) and block_id:
            if block_id in seen_block_ids:
                results.append(AuditResult(id=block_id, status="fail", errors=("duplicate_block_id",)))
            seen_block_ids.add(block_id)

        session_id = envelope.get("session_id")
        turn = envelope.get("turn")
        step = envelope.get("step")
        if not isinstance(session_id, str) or not isinstance(turn, int) or not isinstance(step, int):
            continue

        position = (turn, step)
        previous = last_position_by_session.get(session_id)
        if previous is not None and position < previous:
            results.append(
                AuditResult(
                    id=str(block_id or f"line:{index}"),
                    status="fail",
                    errors=("out_of_order_trace_block",),
                )
            )
        last_position_by_session[session_id] = position

    return results


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
    if expectation == "accept_exact" and content.get("exact") is not True:
        errors.append("accept_exact_requires_exact_content")
    if content.get("exact") is False and expectation == "accept_exact":
        errors.append("non_exact_content_claims_exact_expectation")

    needs_reason = (
        action == "fallback"
        or result in {"degraded", "error", "rejected"}
        or resolver_state in {"missing_identifiable", "unresolvable_fallback_required"}
    )
    if needs_reason and not audit.get("reason"):
        errors.append("missing_reason")

    if resolver_state == "available_local" and content.get("exact") and not content.get("resolver_uri"):
        errors.append("available_local_missing_resolver")


def _validate_extensions(extensions: object, errors: list[str]) -> None:
    if extensions is None:
        return
    if not isinstance(extensions, dict):
        errors.append("invalid_extensions")
        return
    for namespace, value in extensions.items():
        if not isinstance(namespace, str) or not namespace:
            errors.append("invalid_extension_namespace")
            continue
        if namespace in CORE_SECTION_NAMES:
            errors.append("extension_namespace_overrides_core")
        if not isinstance(value, dict):
            errors.append("invalid_extension_payload")


def _require_content_identity(content: dict[str, Any], errors: list[str]) -> None:
    if not is_hash(content.get("hash")):
        errors.append("missing_or_invalid_content_hash")
    if not isinstance(content.get("size_bytes"), int):
        errors.append("missing_or_invalid_content_size")


def is_hash(value: object) -> bool:
    """Return whether a value is a lowercase sha256 digest string."""
    return isinstance(value, str) and HASH_RE.match(value) is not None


def _audit_payload_digest(block: dict[str, Any], warnings: list[str]) -> list[str]:
    payload_digest = block["envelope"].get("payload_digest")
    if payload_digest == "draft-uncomputed":
        warnings.append("draft_payload_digest_uncomputed")
        return []

    if payload_digest != canonical_payload_digest(block):
        return ["payload_digest_mismatch"]
    return []


def _audit_artifacts(block: dict[str, Any], context: AuditContext) -> list[str]:
    errors: list[str] = []
    resolver_state = block["audit"]["resolver_state"]
    content = block["content"]

    if resolver_state != "available_local":
        return errors

    resolved = _resolve_fixture_uri(content.get("resolver_uri"), context)
    if resolved is None:
        errors.append("available_local_unresolved_content_uri")
        return errors

    _verify_artifact_identity(
        resolved,
        expected_hash=content.get("hash"),
        expected_size=content.get("size_bytes"),
        label="content",
        errors=errors,
    )

    if block["observation"]["action"] == "delta":
        _audit_delta(block, context, errors)

    return errors


def _audit_delta(block: dict[str, Any], context: AuditContext, errors: list[str]) -> None:
    content = block["content"]
    if content.get("delta_algorithm") != "unified_diff":
        errors.append("unsupported_delta_algorithm_for_audit")
        return

    base_path = _resolve_fixture_uri(content.get("base_uri"), context)
    delta_path = _resolve_fixture_uri(content.get("delta_uri"), context)
    final_path = _resolve_fixture_uri(content.get("resolver_uri"), context)
    if base_path is None:
        errors.append("missing_or_unresolved_base_uri")
    if delta_path is None:
        errors.append("missing_or_unresolved_delta_uri")
    if final_path is None:
        errors.append("missing_or_unresolved_final_uri")
    if base_path is None or delta_path is None or final_path is None:
        return

    _verify_artifact_identity(
        base_path, expected_hash=content.get("base_hash"), expected_size=None, label="base", errors=errors
    )
    _verify_artifact_identity(
        delta_path,
        expected_hash=content.get("delta_hash"),
        expected_size=None,
        label="delta",
        errors=errors,
    )

    try:
        replayed = _apply_unified_diff(
            base_path.read_text().splitlines(keepends=True), delta_path.read_text().splitlines(keepends=True)
        )
    except ValueError as exc:
        errors.append(f"delta_replay_failed:{exc}")
        return

    final_bytes = final_path.read_bytes()
    if "".join(replayed).encode("utf-8") != final_bytes:
        errors.append("delta_replay_final_mismatch")


def _verify_artifact_identity(
    path: Path,
    *,
    expected_hash: object,
    expected_size: object,
    label: str,
    errors: list[str],
) -> None:
    data = path.read_bytes()
    actual_hash = "sha256:" + sha256(data).hexdigest()
    if expected_hash != actual_hash:
        errors.append(f"{label}_hash_mismatch")
    if expected_size is not None and expected_size != len(data):
        errors.append(f"{label}_size_mismatch")


def _resolve_fixture_uri(uri: object, context: AuditContext) -> Path | None:
    if not isinstance(uri, str) or context.fixture_dir is None:
        return None
    fixture_dir = context.fixture_dir.resolve()
    if uri.startswith("tok-fixture://"):
        relative = uri.removeprefix("tok-fixture://")
        path = (fixture_dir / relative).resolve()
    else:
        path = (fixture_dir / uri).resolve()

    try:
        path.relative_to(fixture_dir)
    except ValueError:
        return None
    return path if path.exists() and path.is_file() else None


def _apply_unified_diff(base_lines: list[str], diff_lines: list[str]) -> list[str]:
    output: list[str] = []
    base_index = 0
    index = 0

    while index < len(diff_lines):
        line = diff_lines[index]
        if line.startswith("--- ") or line.startswith("+++ "):
            index += 1
            continue
        if not line.startswith("@@ "):
            index += 1
            continue

        old_start = _parse_hunk_old_start(line)
        hunk_start = old_start - 1
        output.extend(base_lines[base_index:hunk_start])
        base_index = hunk_start
        index += 1

        while index < len(diff_lines) and not diff_lines[index].startswith("@@ "):
            hunk_line = diff_lines[index]
            if hunk_line in {"\n", "\r\n"}:
                expected = hunk_line
                if base_index >= len(base_lines) or base_lines[base_index] != expected:
                    raise ValueError("context_mismatch")
                output.append(base_lines[base_index])
                base_index += 1
            elif hunk_line.startswith(" "):
                expected = hunk_line[1:]
                if base_index >= len(base_lines) or base_lines[base_index] != expected:
                    raise ValueError("context_mismatch")
                output.append(base_lines[base_index])
                base_index += 1
            elif hunk_line.startswith("-"):
                expected = hunk_line[1:]
                if base_index >= len(base_lines) or base_lines[base_index] != expected:
                    raise ValueError("removal_mismatch")
                base_index += 1
            elif hunk_line.startswith("+"):
                output.append(hunk_line[1:])
            elif hunk_line.startswith("\\"):
                pass
            else:
                raise ValueError("invalid_hunk_line")
            index += 1

    output.extend(base_lines[base_index:])
    return output


def _parse_hunk_old_start(header: str) -> int:
    match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", header)
    if match is None:
        raise ValueError("invalid_hunk_header")
    return int(match.group(1))
