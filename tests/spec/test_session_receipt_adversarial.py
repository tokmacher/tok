from __future__ import annotations

import json
from pathlib import Path

from tok.protocol.session_receipt import verify_session_receipt

ROOT = Path(__file__).resolve().parents[2]
GOOD_FIXTURE = ROOT / "docs/spec/fixtures/session_receipt_good.json"
ADVERSARIAL_FIXTURE = ROOT / "docs/spec/fixtures/session_receipt_adversarial.json"


def test_session_receipt_good_fixture_passes() -> None:
    payload = json.loads(GOOD_FIXTURE.read_text())

    result = verify_session_receipt(payload)

    assert result.passed is True
    assert result.level == "L1_internal_consistency"
    assert result.errors == []


def test_session_receipt_adversarial_cases_fail(tmp_path: Path) -> None:
    pack = json.loads(ADVERSARIAL_FIXTURE.read_text())
    artifact = tmp_path / "receipt-artifact.jsonl"
    artifact.write_text("artifact content\n", encoding="utf-8")

    for case in pack["cases"]:
        receipt = case["receipt"]
        for item in receipt.get("artifacts", []):
            if item.get("path") == "__TEST_ARTIFACT__":
                item["path"] = str(artifact)
        result = verify_session_receipt(receipt)
        expected_error = case["expected_error"]

        assert result.passed is False, case["name"]
        assert any(error.startswith(expected_error) for error in result.errors), (
            case["name"],
            expected_error,
            result.errors,
        )
