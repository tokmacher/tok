from __future__ import annotations

import logging

import pytest


@pytest.fixture()
def debug_caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.DEBUG, logger="tok.evidence_safety")
    return caplog


def test_record_evidence_decision_logged_at_debug(debug_caplog: pytest.LogCaptureFixture) -> None:
    from tok.runtime.evidence_safety import record_evidence_decision

    record_evidence_decision(decision="preserved", reason="first_occurrence_guard", tool="read_file", kind="file")
    assert any("evidence_decision:" in r.message and "decision=preserved" in r.message for r in debug_caplog.records)


def test_all_reason_codes_accepted(debug_caplog: pytest.LogCaptureFixture) -> None:
    from tok.runtime.evidence_safety import EVIDENCE_DECISION_REASON_CODES, record_evidence_decision

    for reason in EVIDENCE_DECISION_REASON_CODES:
        record_evidence_decision(decision="preserved", reason=reason, tool="t")
    assert not any("unknown reason=" in r.message for r in debug_caplog.records)


def test_unknown_reason_code_warns(caplog: pytest.LogCaptureFixture) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.WARNING, logger="tok.evidence_safety")
    from tok.runtime.evidence_safety import record_evidence_decision

    record_evidence_decision(decision="preserved", reason="not_a_real_reason", tool="t")
    assert any("unknown reason=" in r.message for r in caplog.records)
