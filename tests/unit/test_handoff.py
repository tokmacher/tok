from __future__ import annotations

import json
from pathlib import Path

from tok.protocol.handoff import HANDOFF_SCHEMA, HandoffPacket, export_handoff, inspect_handoff
from tok.protocol.session_receipt import (
    ArtifactReference,
    SavingsSummary,
    SessionIdentity,
    TokSessionReceipt,
)
from tok.receipt import BridgeReceipt, append_bridge_receipt


def test_export_handoff_from_session_receipt_model() -> None:
    receipt = TokSessionReceipt(
        receipt_id="tsr_handoff_model",
        created_at="2026-05-26T00:00:00Z",
        session=SessionIdentity(session_id="session-handoff"),
        savings_summary=SavingsSummary(fallback_count=1, degraded_to_baseline=True),
        artifacts=[
            ArtifactReference(
                kind="summary",
                path="docs/plans/substrate_execution_prompt.md",
                exactness="summary",
                available=False,
            )
        ],
        warnings=["metadata-only savings"],
    )

    packet = export_handoff(
        session_receipt=receipt,
        source_agent="agent-a",
        target_agent="agent-b",
        goal="Continue bounded task",
        completed=["Generated receipt"],
        next_steps=["Reacquire exact files"],
    )

    assert packet.schema_ == HANDOFF_SCHEMA
    assert packet.model_dump(by_alias=True)["schema"] == HANDOFF_SCHEMA
    assert packet.session_receipt.receipt_id == "tsr_handoff_model"
    assert packet.task_state.goal == "Continue bounded task"
    assert packet.evidence[0].exactness == "summary"
    assert packet.required_reacquisitions[0].path == "docs/plans/substrate_execution_prompt.md"
    assert "Linked session receipt reports fallback or degraded baseline state." in packet.warnings


def test_export_handoff_from_session_receipt_path(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt = TokSessionReceipt(
        receipt_id="tsr_handoff_path",
        created_at="2026-05-26T00:00:00Z",
        session=SessionIdentity(session_id="session-handoff"),
    )
    receipt_path.write_text(json.dumps(receipt.model_dump(by_alias=True)), encoding="utf-8")

    packet = export_handoff(session_receipt_path=receipt_path)

    assert packet.session_receipt.receipt_id == "tsr_handoff_path"
    assert packet.session_receipt.path == str(receipt_path)
    assert packet.session_receipt.digest.startswith("sha256:")
    assert packet.required_reacquisitions == []


def test_handoff_export_cli_emits_json(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from tok.cli import app

    session_dir = tmp_path / "tok" / "sessions" / "session-a"
    session_dir.mkdir(parents=True)
    append_bridge_receipt(
        session_dir / "receipts.jsonl",
        BridgeReceipt(
            receipt_id="bridge-handoff-1",
            session_id="session-handoff-cli",
            turn=1,
            step=1,
            request_policy="legacy_tool_compatible",
            compression_applied=False,
            fallback=True,
        ),
    )
    monkeypatch.setenv("TOK_DIR", str(tmp_path / "tok"))

    result = CliRunner().invoke(app, ["handoff", "export", "--latest", "--json", "--goal", "Continue"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    packet = HandoffPacket.model_validate(payload)
    assert packet.schema_ == HANDOFF_SCHEMA
    assert packet.session_receipt.receipt_id.startswith("tsr_")
    assert packet.task_state.goal == "Continue"


def test_inspect_handoff_reports_required_reacquisitions() -> None:
    receipt = TokSessionReceipt(
        receipt_id="tsr_inspect",
        created_at="2026-05-26T00:00:00Z",
        session=SessionIdentity(session_id="session-inspect"),
        artifacts=[
            ArtifactReference(
                kind="reference",
                path="tok://resolver/missing",
                exactness="reference",
                available=False,
            )
        ],
    )
    packet = export_handoff(session_receipt=receipt, goal="Resume safely")

    result = inspect_handoff(packet)

    assert result.passed is True
    assert result.required_reacquisitions[0].path == "tok://resolver/missing"
    assert "Required exact reacquisitions:" in result.summary
    assert any("non_exact_evidence_requires_reacquisition" in warning for warning in result.warnings)


def test_inspect_handoff_verifies_linked_session_receipt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt = TokSessionReceipt(
        receipt_id="tsr_linked",
        created_at="2026-05-26T00:00:00Z",
        session=SessionIdentity(session_id="session-linked"),
    )
    receipt_path.write_text(json.dumps(receipt.model_dump(by_alias=True)), encoding="utf-8")
    packet = export_handoff(session_receipt_path=receipt_path)

    result = inspect_handoff(packet)

    assert result.passed is True
    assert result.session_receipt_validation_level == "L1_internal_consistency"
    assert "Session receipt validation: L1_internal_consistency" in result.summary


def test_inspect_handoff_rejects_missing_exact_evidence(tmp_path: Path) -> None:
    receipt = TokSessionReceipt(
        receipt_id="tsr_missing_exact",
        created_at="2026-05-26T00:00:00Z",
        session=SessionIdentity(session_id="session-missing"),
        artifacts=[
            ArtifactReference(
                kind="bridge_receipts",
                path=str(tmp_path / "missing.jsonl"),
                exactness="exact",
                available=True,
            )
        ],
    )
    packet = export_handoff(session_receipt=receipt)

    result = inspect_handoff(packet)

    assert result.passed is False
    assert result.errors == [f"exact_evidence_missing:{tmp_path / 'missing.jsonl'}"]


def test_handoff_inspect_cli_json(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tok.cli import app

    receipt = TokSessionReceipt(
        receipt_id="tsr_cli_inspect",
        created_at="2026-05-26T00:00:00Z",
        session=SessionIdentity(session_id="session-cli-inspect"),
    )
    packet = export_handoff(session_receipt=receipt, goal="Inspect packet")
    packet_path = tmp_path / "handoff.json"
    packet_path.write_text(json.dumps(packet.model_dump(by_alias=True)), encoding="utf-8")

    result = CliRunner().invoke(app, ["handoff", "inspect", str(packet_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["passed"] is True
    assert payload["summary"].startswith("Handoff:")
