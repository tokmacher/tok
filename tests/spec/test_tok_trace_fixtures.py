from __future__ import annotations

import json
from pathlib import Path

from tok.spec.trace_v0_1 import audit_block, validate_block

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
        assert audit_block(block).status == result["expected_status"], result["id"]
        assert isinstance(result["summary"], str) and result["summary"], result["id"]
