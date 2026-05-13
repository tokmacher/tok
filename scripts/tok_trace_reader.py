from __future__ import annotations

import json
import re
import sys
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
    "accept_pass_through",
    "accept_delta",
    "accept_fallback",
    "reject_malformed",
    "accept_non_exact_reference",
}
ALLOWED_DELTA_ALGORITHMS = {"line", "unified_diff", "json_patch", "binary"}
CORE_SECTION_NAMES = {"envelope", "observation", "content", "audit"}
NON_EXACT_REFERENCE_ACTIONS = {"skeleton_reference", "summary_reference"}
FALLBACK_RESULTS = {"degraded", "error", "rejected"}
FALLBACK_RESOLVER_STATES = {"missing_identifiable", "unresolvable_fallback_required"}


@dataclass(frozen=True)
class AuditResult:
    id: str
    status: str
    errors: tuple[str, ...] = ()
    summary: str = ""


@dataclass(frozen=True)
class AuditContext:
    fixture_dir: Path | None = None


def is_hash(value: object) -> bool:
    return isinstance(value, str) and HASH_RE.match(value) is not None


def canonical_payload_digest(block: dict[str, Any]) -> str:
    payload = {key: block[key] for key in ("observation", "content", "audit")}
    if "extensions" in block:
        payload["extensions"] = block["extensions"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + sha256(canonical.encode("utf-8")).hexdigest()


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _require_content_identity(content: dict[str, Any], errors: list[str]) -> None:
    if not is_hash(content.get("hash")):
        errors.append("missing_or_invalid_content_hash")
    if not _is_nonnegative_int(content.get("size_bytes")):
        errors.append("missing_or_invalid_content_size")


def validate_block(block: dict[str, Any]) -> list[str]:
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

    if envelope.get("trace_version") != TRACE_VERSION:
        errors.append("invalid_trace_version")
    if not isinstance(envelope.get("block_id"), str) or not envelope["block_id"]:
        errors.append("invalid_block_id")
    if not isinstance(envelope.get("session_id"), str) or not envelope["session_id"]:
        errors.append("invalid_session_id")
    if not _is_nonnegative_int(envelope.get("turn")):
        errors.append("invalid_turn")
    if not _is_nonnegative_int(envelope.get("step")):
        errors.append("invalid_step")
    if envelope.get("direction") not in ALLOWED_DIRECTIONS:
        errors.append("invalid_direction")

    payload_digest = envelope.get("payload_digest")
    if payload_digest != "draft-uncomputed" and not is_hash(payload_digest):
        errors.append("invalid_payload_digest")

    if observation.get("class") not in ALLOWED_CLASSES:
        errors.append("invalid_class")
    if not isinstance(observation.get("key"), str) or not observation["key"]:
        errors.append("invalid_key")
    if observation.get("action") not in ALLOWED_ACTIONS:
        errors.append("invalid_action")
    if observation.get("result") not in ALLOWED_RESULTS:
        errors.append("invalid_result")

    action = observation.get("action")
    exact = content.get("exact")
    if not isinstance(exact, bool):
        errors.append("invalid_exact")
    if "hash" in content and not is_hash(content["hash"]):
        errors.append("invalid_hash")
    if "size_bytes" in content and not _is_nonnegative_int(content["size_bytes"]):
        errors.append("invalid_size_bytes")
    if "resolver_uri" in content and not isinstance(content["resolver_uri"], str):
        errors.append("invalid_resolver_uri")

    if exact:
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
    if action in NON_EXACT_REFERENCE_ACTIONS and exact is not False:
        errors.append("non_exact_reference_marked_exact")

    resolver_state = audit.get("resolver_state")
    expectation = audit.get("expectation")
    if resolver_state not in ALLOWED_RESOLVER_STATES:
        errors.append("invalid_resolver_state")
    if expectation not in ALLOWED_EXPECTATIONS:
        errors.append("invalid_expectation")

    if resolver_state == "available_local":
        if not content.get("resolver_uri"):
            errors.append("available_local_missing_resolver")
        _require_content_identity(content, errors)

    if expectation == "reject_malformed" and observation.get("result") != "rejected":
        errors.append("reject_fixture_must_have_rejected_result")

    if expectation == "accept_exact":
        if exact is not True:
            errors.append("accept_exact_requires_exact_content")
        if resolver_state != "available_local":
            errors.append("accept_exact_requires_available_local_content")
        _require_content_identity(content, errors)

    if expectation == "accept_reference":
        if action != "reference":
            errors.append("accept_reference_requires_reference_action")
        if exact is not True:
            errors.append("accept_reference_requires_exact_content")
        _require_content_identity(content, errors)

    if expectation == "accept_non_exact_reference":
        if action not in NON_EXACT_REFERENCE_ACTIONS:
            errors.append("accept_non_exact_reference_requires_non_exact_reference_action")
        if exact is not False:
            errors.append("accept_non_exact_reference_requires_non_exact_content")

    if expectation == "accept_pass_through":
        if action != "pass_through" or observation.get("result") != "ok":
            errors.append("accept_pass_through_requires_ok_pass_through")
        if exact is not False:
            errors.append("accept_pass_through_requires_non_exact_content")

    if expectation == "accept_fallback":
        if not (
            action == "fallback"
            or observation.get("result") in FALLBACK_RESULTS
            or resolver_state in FALLBACK_RESOLVER_STATES
        ):
            errors.append("accept_fallback_requires_fallback_or_degradation")

    if observation.get("result") == "ok" and resolver_state == "unresolvable_fallback_required":
        errors.append("ok_result_cannot_require_unresolvable_fallback")

    if expectation == "accept_delta":
        if action != "delta":
            errors.append("accept_delta_requires_delta_action")
        if exact is not True:
            errors.append("accept_delta_requires_exact_content")
        if resolver_state != "available_local":
            errors.append("accept_delta_requires_available_local_content")

    needs_reason = (
        action == "fallback"
        or observation.get("result") in FALLBACK_RESULTS
        or resolver_state in (FALLBACK_RESOLVER_STATES)
    )
    if needs_reason and not audit.get("reason"):
        errors.append("missing_reason")

    extensions = block.get("extensions")
    if extensions is not None:
        if not isinstance(extensions, dict):
            errors.append("invalid_extensions")
        else:
            for namespace, value in extensions.items():
                if not isinstance(namespace, str) or not namespace:
                    errors.append("invalid_extension_namespace")
                    continue
                if namespace in CORE_SECTION_NAMES:
                    errors.append("extension_namespace_overrides_core")
                if not isinstance(value, dict):
                    errors.append("invalid_extension_payload")

    return errors


def _audit_payload_digest(block: dict[str, Any], warnings: list[str]) -> list[str]:
    payload_digest = block["envelope"].get("payload_digest")
    if payload_digest == "draft-uncomputed":
        warnings.append("draft_payload_digest_uncomputed")
        return []
    if payload_digest != canonical_payload_digest(block):
        return ["payload_digest_mismatch"]
    return []


def _resolver_warnings(block: dict[str, Any]) -> list[str]:
    resolver_state = block["audit"]["resolver_state"]
    expectation = block["audit"]["expectation"]
    if resolver_state in {"missing_identifiable", "unresolvable_fallback_required"}:
        return [resolver_state]
    if resolver_state == "resolvable_remote" and expectation in {
        "accept_reference",
        "accept_non_exact_reference",
        "accept_pass_through",
    }:
        return [resolver_state]
    return []


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


def _verify_artifact_identity(
    path: Path,
    *,
    expected_hash: object,
    expected_size: object,
    label: str,
    errors: list[str],
) -> None:
    try:
        data = path.read_bytes()
    except OSError:
        errors.append(f"{label}_artifact_unreadable")
        return
    actual_hash = "sha256:" + sha256(data).hexdigest()
    if expected_hash is not None and expected_hash != actual_hash:
        errors.append(f"{label}_hash_mismatch")
    if expected_size is not None and expected_size != len(data):
        errors.append(f"{label}_size_mismatch")


def _parse_hunk_old_start(header: str) -> int:
    match = re.match(r"^@@ -(?P<old_start>\d+)(?:,\d+)? \+(?P<new_start>\d+)(?:,\d+)? @@", header)
    if not match:
        raise ValueError("invalid_hunk_header")
    return int(match.group("old_start"))


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
        delta_path, expected_hash=content.get("delta_hash"), expected_size=None, label="delta", errors=errors
    )

    try:
        base_lines = base_path.read_text().splitlines(keepends=True)
        delta_lines = delta_path.read_text().splitlines(keepends=True)
    except OSError:
        errors.append("delta_artifact_unreadable")
        return

    try:
        replayed = _apply_unified_diff(base_lines, delta_lines)
    except ValueError as exc:
        errors.append(f"delta_replay_failed:{exc}")
        return

    try:
        final_bytes = final_path.read_bytes()
    except OSError:
        errors.append("final_artifact_unreadable")
        return
    if "".join(replayed).encode("utf-8") != final_bytes:
        errors.append("delta_replay_final_mismatch")


def _audit_artifacts(block: dict[str, Any], context: AuditContext) -> list[str]:
    errors: list[str] = []
    if block["audit"]["resolver_state"] != "available_local":
        return errors
    content = block["content"]
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


def audit_block(block: dict[str, Any], *, fixture_id: str, context: AuditContext) -> AuditResult:
    errors = validate_block(block)
    if errors:
        return AuditResult(id=fixture_id, status="fail", errors=tuple(errors))
    warnings: list[str] = []
    digest_errors = _audit_payload_digest(block, warnings)
    if digest_errors:
        return AuditResult(id=fixture_id, status="fail", errors=tuple(digest_errors))
    artifact_errors = _audit_artifacts(block, context)
    if artifact_errors:
        return AuditResult(id=fixture_id, status="fail", errors=tuple(artifact_errors))
    for warning in _resolver_warnings(block):
        if warning not in warnings:
            warnings.append(warning)
    if warnings:
        return AuditResult(id=fixture_id, status="warn", errors=tuple(warnings))
    return AuditResult(id=fixture_id, status="pass")


def audit_fixture_file(path: Path) -> list[AuditResult]:
    try:
        fixtures = json.loads(path.read_text())
    except UnicodeDecodeError:
        return [AuditResult(id=str(path), status="fail", errors=("trace_file_not_utf8",))]
    except json.JSONDecodeError:
        return [AuditResult(id=str(path), status="fail", errors=("trace_file_invalid_json",))]

    context = AuditContext(path.parent)
    results: list[AuditResult] = []
    seen_block_ids: set[str] = set()
    last_position_by_session: dict[str, tuple[int, int]] = {}
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            results.append(AuditResult(id="fixture", status="fail", errors=("fixture_not_object",)))
            continue
        fixture_id = str(fixture.get("id") or "trace-block")
        block = fixture.get("block")
        if not isinstance(block, dict):
            results.append(AuditResult(id=fixture_id, status="fail", errors=("missing_or_invalid_block",)))
            continue
        results.append(audit_block(block, fixture_id=fixture_id, context=context))

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
                AuditResult(id=str(block_id or fixture_id), status="fail", errors=("out_of_order_trace_block",))
            )
        last_position_by_session[session_id] = position
    return results


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in {"-h", "--help"}:
        print("Usage: python scripts/tok_trace_reader.py <fixture.json>")
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"Missing file: {path}")
        return 2
    results = audit_fixture_file(path)
    output = []
    for result in results:
        output.append(
            {
                "id": result.id,
                "status": result.status,
                "errors": list(result.errors),
                "summary": result.summary,
            }
        )
    print(json.dumps(output, indent=2, sort_keys=True))
    failed = any(r.status == "fail" for r in results)
    warned = any(r.status == "warn" for r in results)
    if failed:
        return 1
    if warned:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
