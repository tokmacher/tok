from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tok.spec.trace import (
    AuditContext,
    audit_block,
    audit_fixture_file,
    audit_trace_file,
    canonical_payload_digest,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "docs" / "spec" / "fixtures" / "trace_fixtures.json"
FIXTURE_DIR = FIXTURE_PATH.parent
ADVERSARIAL_PACKS = FIXTURE_DIR / "adversarial_packs.json"


def _fixture_block(fixture_id: str) -> dict[str, Any]:
    fixtures = json.loads(FIXTURE_PATH.read_text())
    for fixture in fixtures:
        if fixture["id"] == fixture_id:
            block = deepcopy(fixture["block"])
            assert isinstance(block, dict)
            return block
    raise AssertionError(f"fixture not found: {fixture_id}")


def _audit_mutated(block: dict[str, Any]) -> tuple[str, ...]:
    block["envelope"]["payload_digest"] = canonical_payload_digest(block)
    result = audit_block(block, fixture_id="attack", context=AuditContext(FIXTURE_DIR))
    assert result.status in {"fail", "warn"}
    return result.errors


def _copy_fixture_artifacts(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE_DIR / "artifacts", tmp_path / "artifacts")


def _implemented_pack_cases() -> dict[str, dict[str, str]]:
    manifest = json.loads(ADVERSARIAL_PACKS.read_text())
    for pack in manifest["packs"]:
        if pack["id"] == "trace-l1-l2-core-adversarial":
            return {case["id"]: case for case in pack["cases"]}
    raise AssertionError("trace-l1-l2-core-adversarial pack not found")


@pytest.mark.parametrize("field", ["hash", "size_bytes"])
def test_exact_content_requires_identity_for_every_action(field: str) -> None:
    block = _fixture_block("unresolvable_fallback_required")
    block["content"] = {"exact": True}
    block["audit"]["expectation"] = "accept_exact"
    block["audit"]["resolver_state"] = "missing_identifiable"
    block["content"].pop(field, None)

    errors = _audit_mutated(block)

    assert f"missing_or_invalid_content_{'hash' if field == 'hash' else 'size'}" in errors


def test_exact_available_local_content_requires_resolver_uri() -> None:
    block = _fixture_block("first_exact_file_observation")
    block["content"].pop("resolver_uri")

    errors = _audit_mutated(block)

    assert "available_local_missing_resolver" in errors


@pytest.mark.parametrize("fixture_id", ["summary_reference_non_exact", "skeleton_reference_non_exact"])
def test_non_exact_reference_cannot_claim_exactness(fixture_id: str) -> None:
    block = _fixture_block(fixture_id)
    block["content"]["exact"] = True

    errors = _audit_mutated(block)

    assert "non_exact_reference_marked_exact" in errors


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        (
            lambda block: block.__setitem__("observation", {**block["observation"], "action": "fallback"}),
            "missing_reason",
        ),
        (
            lambda block: block.__setitem__("observation", {**block["observation"], "result": "degraded"}),
            "missing_reason",
        ),
        (lambda block: block.__setitem__("observation", {**block["observation"], "result": "error"}), "missing_reason"),
        (
            lambda block: block.__setitem__("observation", {**block["observation"], "result": "rejected"}),
            "missing_reason",
        ),
    ],
)
def test_fallback_degraded_error_and_rejected_records_require_reason(mutator, expected_error: str) -> None:
    block = _fixture_block("first_exact_file_observation")
    block["audit"].pop("reason", None)
    mutator(block)

    errors = _audit_mutated(block)

    assert expected_error in errors


def test_reject_malformed_expectation_requires_rejected_result() -> None:
    block = _fixture_block("first_exact_file_observation")
    block["audit"]["expectation"] = "reject_malformed"

    errors = _audit_mutated(block)

    assert "reject_fixture_must_have_rejected_result" in errors


@pytest.mark.parametrize(
    ("section", "field", "value", "expected_error"),
    [
        ("envelope", "trace_version", "tok-trace/v9", "invalid_trace_version"),
        ("envelope", "direction", "sideways", "invalid_direction"),
        ("envelope", "payload_digest", "sha256:not-hex", "invalid_payload_digest"),
        ("observation", "class", "network", "invalid_class"),
        ("observation", "action", "teleport", "invalid_action"),
        ("observation", "result", "maybe", "invalid_result"),
        ("audit", "resolver_state", "probably_available", "invalid_resolver_state"),
        ("audit", "expectation", "accept_magic", "invalid_expectation"),
    ],
)
def test_invalid_enum_and_digest_shapes_are_rejected(section: str, field: str, value: str, expected_error: str) -> None:
    block = _fixture_block("first_exact_file_observation")
    block[section][field] = value
    if field != "payload_digest":
        block["envelope"]["payload_digest"] = canonical_payload_digest(block)

    result = audit_block(block, fixture_id="attack", context=AuditContext(FIXTURE_DIR))

    assert result.status == "fail"
    assert expected_error in result.errors


def test_forged_payload_digest_after_semantic_mutation_fails() -> None:
    block = _fixture_block("first_exact_file_observation")
    block["observation"]["key"] = "file:src/other.py"

    result = audit_block(block, fixture_id="attack", context=AuditContext(FIXTURE_DIR))

    assert result.status == "fail"
    assert "payload_digest_mismatch" in result.errors


def test_valid_digest_with_wrong_artifact_hash_fails() -> None:
    block = _fixture_block("first_exact_file_observation")
    block["content"]["hash"] = "sha256:" + "a" * 64

    errors = _audit_mutated(block)

    assert "content_hash_mismatch" in errors


def test_valid_digest_with_wrong_artifact_size_fails() -> None:
    block = _fixture_block("first_exact_file_observation")
    block["content"]["size_bytes"] += 1

    errors = _audit_mutated(block)

    assert "content_size_mismatch" in errors


@pytest.mark.parametrize(
    "resolver_uri",
    [
        "tok-fixture://../outside.py",
        "../outside.py",
        "/tmp/outside.py",
        "tok-fixture:///tmp/outside.py",
    ],
)
def test_resolver_path_escape_variants_fail(resolver_uri: str) -> None:
    block = _fixture_block("first_exact_file_observation")
    block["content"]["resolver_uri"] = resolver_uri

    errors = _audit_mutated(block)

    assert "available_local_unresolved_content_uri" in errors


@pytest.mark.parametrize("resolver_uri", ["tok-fixture://artifacts", "tok-fixture://artifacts/missing.txt"])
def test_available_local_resolver_directory_or_missing_file_fails(resolver_uri: str) -> None:
    block = _fixture_block("first_exact_file_observation")
    block["content"]["resolver_uri"] = resolver_uri

    errors = _audit_mutated(block)

    assert "available_local_unresolved_content_uri" in errors


def test_duplicate_block_ids_fail_for_fixture_json_arrays(tmp_path: Path) -> None:
    _copy_fixture_artifacts(tmp_path)
    block = _fixture_block("first_exact_file_observation")
    fixture_file = tmp_path / "fixtures.json"
    fixture_file.write_text(json.dumps([{"id": "a", "block": block}, {"id": "b", "block": block}]))

    results = audit_fixture_file(fixture_file)

    assert any(result.errors == ("duplicate_block_id",) for result in results)


def test_duplicate_block_ids_fail_for_jsonl_traces(tmp_path: Path) -> None:
    _copy_fixture_artifacts(tmp_path)
    block = _fixture_block("first_exact_file_observation")
    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(json.dumps(block) + "\n" + json.dumps(block) + "\n")

    results = audit_trace_file(trace_file)

    assert any(result.errors == ("duplicate_block_id",) for result in results)


def test_out_of_order_step_within_same_turn_fails(tmp_path: Path) -> None:
    _copy_fixture_artifacts(tmp_path)
    first = _fixture_block("first_exact_file_observation")
    second = _fixture_block("first_exact_file_observation")
    first["envelope"]["block_id"] = "step-2"
    first["envelope"]["turn"] = 1
    first["envelope"]["step"] = 2
    first["envelope"]["payload_digest"] = canonical_payload_digest(first)
    second["envelope"]["block_id"] = "step-1"
    second["envelope"]["turn"] = 1
    second["envelope"]["step"] = 1
    second["envelope"]["payload_digest"] = canonical_payload_digest(second)
    trace_file = tmp_path / "out_of_order_step.jsonl"
    trace_file.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")

    results = audit_trace_file(trace_file)

    assert any(result.errors == ("out_of_order_trace_block",) for result in results)


def test_same_turn_step_order_is_allowed_across_different_sessions(tmp_path: Path) -> None:
    _copy_fixture_artifacts(tmp_path)
    first = _fixture_block("first_exact_file_observation")
    second = _fixture_block("first_exact_file_observation")
    first["envelope"]["block_id"] = "alpha-step-2"
    first["envelope"]["session_id"] = "alpha"
    first["envelope"]["turn"] = 1
    first["envelope"]["step"] = 2
    first["envelope"]["payload_digest"] = canonical_payload_digest(first)
    second["envelope"]["block_id"] = "beta-step-1"
    second["envelope"]["session_id"] = "beta"
    second["envelope"]["turn"] = 1
    second["envelope"]["step"] = 1
    second["envelope"]["payload_digest"] = canonical_payload_digest(second)
    trace_file = tmp_path / "multi_session.jsonl"
    trace_file.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")

    results = audit_trace_file(trace_file)

    assert all(result.status == "pass" for result in results)


def test_malformed_jsonl_line_does_not_skip_valid_neighbors(tmp_path: Path) -> None:
    _copy_fixture_artifacts(tmp_path)
    block = _fixture_block("first_exact_file_observation")
    trace_file = tmp_path / "mixed.jsonl"
    trace_file.write_text(json.dumps(block) + "\n{bad json\n" + json.dumps(block) + "\n")

    results = audit_trace_file(trace_file)

    assert any(result.errors == ("invalid_jsonl_line",) for result in results)
    assert any(result.status == "pass" for result in results)
    assert any(result.errors == ("duplicate_block_id",) for result in results)


@pytest.mark.parametrize(
    ("extensions", "expected_error"),
    [
        ({"envelope": {}}, "extension_namespace_overrides_core"),
        ({"observation": {}}, "extension_namespace_overrides_core"),
        ({"content": {}}, "extension_namespace_overrides_core"),
        ({"audit": {}}, "extension_namespace_overrides_core"),
        ({"tok.live": "not-object"}, "invalid_extension_payload"),
        ({"": {}}, "invalid_extension_namespace"),
    ],
)
def test_extension_attack_shapes_are_rejected(extensions: dict[str, object], expected_error: str) -> None:
    block = _fixture_block("first_exact_file_observation")
    block["extensions"] = extensions

    errors = _audit_mutated(block)

    assert expected_error in errors


def test_extension_semantic_mutation_is_covered_by_payload_digest() -> None:
    block = _fixture_block("first_exact_file_observation")
    block["extensions"] = {"tok.attack": {"claim": "benign"}}
    block["envelope"]["payload_digest"] = canonical_payload_digest(block)
    block["extensions"]["tok.attack"]["claim"] = "mutated"

    result = audit_block(block, fixture_id="attack", context=AuditContext(FIXTURE_DIR))

    assert result.status == "fail"
    assert "payload_digest_mismatch" in result.errors


def test_named_adversarial_pack_expected_errors_are_exercised_by_current_tests() -> None:
    cases = _implemented_pack_cases()
    exercised_errors = {
        "payload_digest_mismatch",
        "available_local_unresolved_content_uri",
        "non_exact_reference_marked_exact",
        "unsupported_delta_algorithm_for_audit",
        "invalid_jsonl_line",
        "duplicate_block_id",
        "out_of_order_trace_block",
        "invalid_trace_version",
        "extension_namespace_overrides_core",
    }

    assert {case["expected_error"] for case in cases.values()} <= exercised_errors
    assert cases["resolver_state_lie"]["expected_error"] == "available_local_unresolved_content_uri"
