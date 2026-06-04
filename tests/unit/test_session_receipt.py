from __future__ import annotations

import json
from pathlib import Path

from tok.protocol.session_receipt import (
    SESSION_RECEIPT_SCHEMA,
    AdapterIdentity,
    ArtifactReference,
    EvidenceSummary,
    SavingsSummary,
    SessionIdentity,
    TokSessionReceipt,
    generate_session_receipt,
    verify_session_receipt,
)
from tok.receipt import BridgeReceipt, append_bridge_receipt
from tok.runtime._diagnostics import DiagnosticsSnapshot
from tok.utils.savings_event import SavingsEvent, append_savings_event


def test_session_receipt_model_generates_from_local_records(tmp_path: Path) -> None:
    bridge_path = tmp_path / "receipts.jsonl"
    savings_path = tmp_path / "savings_events.jsonl"
    append_bridge_receipt(
        bridge_path,
        BridgeReceipt(
            receipt_id="bridge-1",
            session_id="session-1",
            turn=2,
            step=1,
            request_policy="legacy_tool_compatible",
            compression_applied=True,
            evidence_summary={"resolver_backed_count": 1},
            savings={"tokens_saved": 7},
            fallback=False,
        ),
    )
    append_savings_event(
        SavingsEvent(
            event_id="event-1",
            session_id="session-1",
            request_id="request-1",
            timestamp="2026-05-26T00:00:00Z",
            model="test-model",
            baseline_input_tokens=100,
            actual_input_tokens=60,
            input_tokens_saved=40,
            actual_output_tokens=20,
            baseline_cost_usd=0.01,
            actual_cost_usd=0.006,
            cost_saved_usd=0.004,
        ),
        savings_path,
    )

    receipt = generate_session_receipt(
        session_id="session-1",
        bridge_receipts_path=bridge_path,
        savings_events_path=savings_path,
        diagnostics=DiagnosticsSnapshot(
            calls=1,
            evidence_exact_observed_count=2,
            evidence_non_exact_reference_count=1,
            savings_source="savings_events",
        ),
        adapter=AdapterIdentity(
            client="codex-cli",
            adapter="codex-cli-probe",
            transport_boundary="stdio",
            capabilities=["fixture-only"],
            unstable=True,
        ),
    )

    assert receipt.schema_ == SESSION_RECEIPT_SCHEMA
    assert receipt.model_dump(by_alias=True)["schema"] == SESSION_RECEIPT_SCHEMA
    assert receipt.session.session_id == "session-1"
    assert receipt.session.turn_count == 2
    assert receipt.adapter.client == "codex-cli"
    assert receipt.savings_summary.confidence == "event_backed"
    assert receipt.savings_summary.baseline_input_tokens == 100
    assert receipt.savings_summary.tok_input_tokens == 60
    assert receipt.savings_summary.gross_tokens_saved == 40
    assert receipt.evidence_summary.exact_count == 2
    assert receipt.evidence_summary.reference_count == 1
    assert receipt.evidence_summary.resolver_backed_count == 1
    assert receipt.bridge_receipts[0].digest.startswith("sha256:")
    assert receipt.savings_events[0].digest.startswith("sha256:")
    assert receipt.artifacts[0].available is True
    assert receipt.validation.level == "L2_digest"


def test_session_receipt_warns_when_metadata_only(tmp_path: Path) -> None:
    receipt = generate_session_receipt(
        session_id="session-2",
        diagnostics=DiagnosticsSnapshot(
            baseline_only=True,
            fallback_count=1,
            fail_open_count=1,
            session_quality="degraded",
            evidence_exact_reacquisition_required_count=1,
        ),
    )

    assert receipt.savings_summary.confidence == "metadata_only"
    assert receipt.savings_summary.degraded_to_baseline is True
    assert "Savings are metadata-only unless backed by savings events." in receipt.warnings
    assert "No bridge receipts were available for this session receipt." in receipt.warnings
    assert "Session includes fallback or degraded baseline behavior." in receipt.warnings
    assert "Exact reacquisition remains required for some non-exact evidence." in receipt.warnings


def test_session_receipt_savings_fields_include_overhead_and_confidence(tmp_path: Path) -> None:
    savings_path = tmp_path / "savings_events.jsonl"
    append_savings_event(
        SavingsEvent(
            event_id="event-overhead-1",
            session_id="session-overhead",
            request_id="request-overhead-1",
            timestamp="2026-05-26T00:00:00Z",
            model="test-model",
            baseline_input_tokens=120,
            actual_input_tokens=70,
            input_tokens_saved=50,
            baseline_output_tokens=35,
            actual_output_tokens=30,
            output_tokens_saved=5,
            cache_read_tokens=11,
            cache_write_tokens=7,
            baseline_cost_usd=0.012,
            actual_cost_usd=0.008,
            cost_saved_usd=0.004,
            fallback=True,
            degraded_to_baseline=True,
            non_headline_estimates={"reacquisition_cost_tokens": 9},
        ),
        savings_path,
    )

    receipt = generate_session_receipt(
        session_id="session-overhead",
        savings_events_path=savings_path,
    )

    summary = receipt.savings_summary
    assert summary.baseline_input_tokens == 120
    assert summary.tok_input_tokens == 70
    assert summary.output_tokens == 30
    assert summary.cache_read_tokens == 11
    assert summary.cache_write_tokens == 7
    assert summary.gross_tokens_saved == 55
    assert summary.reacquisition_tokens_spent == 9
    assert summary.net_tokens_saved == 46
    assert summary.estimated_baseline_cost_usd == 0.012
    assert summary.estimated_tok_cost_usd == 0.008
    assert summary.estimated_cost_saved_usd == 0.004
    assert summary.pricing_source == "tok.utils.pricing"
    assert summary.pricing_freshness == "event_recorded"
    assert summary.fallback_count == 1
    assert summary.degraded_to_baseline is True
    assert summary.compression_bypass_count == 0
    assert summary.confidence == "event_backed"


def test_session_receipt_trace_backed_confidence_from_diagnostics() -> None:
    receipt = generate_session_receipt(
        session_id="session-trace",
        diagnostics=DiagnosticsSnapshot(
            baseline_tokens=200,
            actual_tokens=140,
            session_tokens_saved=60,
            session_net_tokens_saved=52,
            reacquisition_cost_tokens=8,
            baseline_cost_usd=0.02,
            actual_cost_usd=0.014,
            cost_saved_usd=0.006,
            savings_source="trace_reconstruction",
        ),
    )

    assert receipt.savings_summary.confidence == "trace_backed"
    assert receipt.savings_summary.pricing_source == "trace_reconstruction"
    assert receipt.savings_summary.pricing_freshness == "diagnostic_snapshot"
    assert receipt.savings_summary.reacquisition_tokens_spent == 8
    assert receipt.savings_summary.net_tokens_saved == 52


def test_session_receipt_examples_parse() -> None:
    root = Path(__file__).resolve().parents[2]
    for path in (
        root / "docs/spec/tok-session-receipt/v0.1-draft/examples/good_receipt.json",
        root / "docs/spec/tok-session-receipt/v0.1-draft/examples/degraded_receipt.json",
    ):
        payload = json.loads(path.read_text())
        assert payload["schema"] == SESSION_RECEIPT_SCHEMA


def test_session_receipt_cli_emits_json(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from tok.cli import app

    session_dir = tmp_path / "tok" / "sessions" / "session-a"
    session_dir.mkdir(parents=True)
    append_bridge_receipt(
        session_dir / "receipts.jsonl",
        BridgeReceipt(
            receipt_id="bridge-cli-1",
            session_id="session-cli",
            turn=1,
            step=1,
            request_policy="legacy_tool_compatible",
            compression_applied=False,
            fallback=True,
        ),
    )
    monkeypatch.setenv("TOK_DIR", str(tmp_path / "tok"))

    result = CliRunner().invoke(app, ["session-receipt", "--latest", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == SESSION_RECEIPT_SCHEMA
    assert payload["session"]["session_id"] == "session-cli"
    assert payload["session"]["turn_count"] == 1
    assert payload["savings_summary"]["confidence"] == "metadata_only"


def test_verify_session_receipt_accepts_local_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text("local evidence\n", encoding="utf-8")
    receipt = TokSessionReceipt(
        receipt_id="tsr_verify_ok",
        created_at="2026-05-26T00:00:00Z",
        session=SessionIdentity(session_id="session-verify"),
        savings_summary=SavingsSummary(confidence="metadata_only"),
        evidence_summary=EvidenceSummary(reacquisition_required_count=0),
        artifacts=[
            ArtifactReference(
                kind="bridge_receipts",
                path=str(artifact),
                digest="sha256:" + __import__("hashlib").sha256(artifact.read_bytes()).hexdigest(),
                exactness="exact",
                available=True,
            )
        ],
    )

    result = verify_session_receipt(receipt)

    assert result.passed is True
    assert result.level == "L3_local_recovery"


def test_verify_session_receipt_rejects_bad_schema() -> None:
    result = verify_session_receipt({"schema": "tok-session-receipt/v9"})

    assert result.passed is False
    assert result.level == "L0_schema"
    assert result.errors


def test_verify_session_receipt_rejects_digest_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text("changed evidence\n", encoding="utf-8")
    receipt = TokSessionReceipt(
        receipt_id="tsr_verify_bad_digest",
        created_at="2026-05-26T00:00:00Z",
        session=SessionIdentity(session_id="session-verify"),
        artifacts=[
            ArtifactReference(
                kind="bridge_receipts",
                path=str(artifact),
                digest="sha256:" + "0" * 64,
                exactness="exact",
                available=True,
            )
        ],
    )

    result = verify_session_receipt(receipt)

    assert result.passed is False
    assert result.level == "L1_internal_consistency"
    assert result.errors == [f"artifact_digest_mismatch:{artifact}"]


def test_audit_session_receipt_cli_json(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tok.cli import app

    receipt_path = tmp_path / "receipt.json"
    receipt = TokSessionReceipt(
        receipt_id="tsr_audit_cli",
        created_at="2026-05-26T00:00:00Z",
        session=SessionIdentity(session_id="session-audit"),
    )
    receipt_path.write_text(json.dumps(receipt.model_dump(by_alias=True)), encoding="utf-8")

    result = CliRunner().invoke(app, ["audit", "--session-receipt", str(receipt_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["passed"] is True
    assert payload["level"] == "L1_internal_consistency"


def test_default_deferred_levels_not_shared_across_instances() -> None:
    from tok.protocol.session_receipt import ValidationSummary

    a = ValidationSummary()
    b = ValidationSummary()
    a.deferred_levels.append("should_not_mutate_shared_state")
    assert "should_not_mutate_shared_state" not in b.deferred_levels


def test_audit_invalid_json_uses_canonical_deferred_levels(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tok.cli import app
    from tok.protocol.session_receipt import _DEFERRED_L1_THROUGH_L5

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not valid json", encoding="utf-8")

    result = CliRunner().invoke(app, ["audit", "--session-receipt", str(bad_path), "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["passed"] is False
    assert payload["level"] == "L0_schema"
    assert payload["deferred_levels"] == list(_DEFERRED_L1_THROUGH_L5)
