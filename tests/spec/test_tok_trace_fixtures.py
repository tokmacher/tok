from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from tok.spec.trace_v0_1 import AuditContext, audit_block, audit_fixture_file, canonical_payload_digest, validate_block

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "docs" / "spec" / "fixtures" / "tok_trace_v0_1_fixtures.json"
EXPECTED_PATH = ROOT / "docs" / "spec" / "fixtures" / "tok_trace_v0_1_expected.json"


def load_fixtures() -> list[dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text())


def load_expected() -> list[dict[str, object]]:
    return json.loads(EXPECTED_PATH.read_text())


def test_trace_fixture_file_has_expected_coverage() -> None:
    fixtures = load_fixtures()
    fixture_ids = {fixture["id"] for fixture in fixtures}

    assert len(fixtures) >= 10
    assert "first_exact_file_observation" in fixture_ids
    assert "unchanged_cached_tool_result" in fixture_ids
    assert "delta_result" in fixture_ids
    assert "fallback_raw_output" in fixture_ids
    assert "missing_resolver_cache" in fixture_ids
    assert "malformed_block_rejection" in fixture_ids
    assert "skeleton_reference_non_exact" in fixture_ids
    assert "summary_reference_non_exact" in fixture_ids


def test_valid_trace_fixtures_match_draft_schema() -> None:
    for fixture in load_fixtures():
        if fixture["valid"]:
            block = fixture["block"]
            assert isinstance(block, dict)
            assert validate_block(block) == [], fixture["id"]


def test_invalid_trace_fixtures_are_rejected() -> None:
    invalid_fixtures = [fixture for fixture in load_fixtures() if not fixture["valid"]]

    assert invalid_fixtures
    for fixture in invalid_fixtures:
        block = fixture["block"]
        assert isinstance(block, dict)
        assert validate_block(block), fixture["id"]


def test_non_exact_fixtures_never_claim_exact_recoverability() -> None:
    non_exact_actions = {"skeleton_reference", "summary_reference"}

    for fixture in load_fixtures():
        block = fixture["block"]
        assert isinstance(block, dict)
        if block["observation"]["action"] in non_exact_actions:
            assert block["content"]["exact"] is False, fixture["id"]
            assert block["audit"]["expectation"] == "accept_non_exact_reference", fixture["id"]


def test_expected_audit_results_match_fixture_ids() -> None:
    fixture_ids = {fixture["id"] for fixture in load_fixtures()}
    expected_ids = {result["id"] for result in load_expected()}

    assert expected_ids == fixture_ids


def test_expected_audit_results_match_draft_validator() -> None:
    fixtures = {fixture["id"]: fixture for fixture in load_fixtures()}

    for result in load_expected():
        fixture = fixtures[result["id"]]
        block = fixture["block"]
        assert isinstance(block, dict)
        assert result["expected_status"] in {"pass", "warn", "fail"}, result["id"]
        assert audit_block(block, context=AuditContext(FIXTURE_PATH.parent)).status == result["expected_status"], (
            result["id"]
        )
        assert isinstance(result["summary"], str) and result["summary"], result["id"]


def test_fixture_file_audit_uses_local_artifacts() -> None:
    results = {result.id: result for result in audit_fixture_file(FIXTURE_PATH)}

    assert results["first_exact_file_observation"].status == "pass"
    assert results["delta_result"].status == "pass"
    assert results["missing_resolver_cache"].status == "warn"
    assert results["malformed_block_rejection"].status == "fail"


def test_payload_digest_matches_canonical_payload() -> None:
    for fixture in load_fixtures():
        block = fixture["block"]
        assert isinstance(block, dict)
        digest = block["envelope"]["payload_digest"]
        assert digest == canonical_payload_digest(block), fixture["id"]


def test_payload_digest_placeholder_is_warning() -> None:
    fixture = next(fixture for fixture in load_fixtures() if fixture["id"] == "first_exact_file_observation")
    block = deepcopy(fixture["block"])
    assert isinstance(block, dict)
    block["envelope"]["payload_digest"] = "draft-uncomputed"

    result = audit_block(block, fixture_id="placeholder", context=AuditContext(FIXTURE_PATH.parent))

    assert result.status == "warn"
    assert "draft_payload_digest_uncomputed" in result.errors


def test_payload_digest_mismatch_fails() -> None:
    fixture = next(fixture for fixture in load_fixtures() if fixture["id"] == "first_exact_file_observation")
    block = deepcopy(fixture["block"])
    assert isinstance(block, dict)
    block["envelope"]["payload_digest"] = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    result = audit_block(block, fixture_id="bad-digest", context=AuditContext(FIXTURE_PATH.parent))

    assert result.status == "fail"
    assert "payload_digest_mismatch" in result.errors


def test_available_local_artifact_hash_mismatch_fails() -> None:
    fixture = next(fixture for fixture in load_fixtures() if fixture["id"] == "first_exact_file_observation")
    block = deepcopy(fixture["block"])
    assert isinstance(block, dict)
    block["content"]["hash"] = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    block["envelope"]["payload_digest"] = canonical_payload_digest(block)

    result = audit_block(block, fixture_id="bad-hash", context=AuditContext(FIXTURE_PATH.parent))

    assert result.status == "fail"
    assert "content_hash_mismatch" in result.errors


def test_non_exact_content_cannot_claim_exact_expectation() -> None:
    fixture = next(fixture for fixture in load_fixtures() if fixture["id"] == "summary_reference_non_exact")
    block = deepcopy(fixture["block"])
    assert isinstance(block, dict)
    block["audit"]["expectation"] = "accept_exact"

    assert "accept_exact_requires_exact_content" in validate_block(block)
