"""Local Tok handoff packet model and exporter.

Experimental: this module is not part of the defended 0.2.x root API.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from tok.protocol.session_receipt import TokSessionReceipt, generate_session_receipt, verify_session_receipt

HANDOFF_SCHEMA = "tok-handoff/v0.1-draft"
EvidenceForm = Literal["exact", "summary", "skeleton", "reference"]


class HandoffReceiptReference(BaseModel, frozen=True):
    receipt_id: str
    path: str = ""
    digest: str = ""
    model_config = {"extra": "forbid"}


class HandoffTaskState(BaseModel, frozen=True):
    goal: str = ""
    completed: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    model_config = {"extra": "forbid"}


class HandoffEvidence(BaseModel, frozen=True):
    label: str
    exactness: EvidenceForm
    path: str = ""
    digest: str = ""
    model_config = {"extra": "forbid"}


class RequiredReacquisition(BaseModel, frozen=True):
    reason: str
    path: str
    model_config = {"extra": "forbid"}


class HandoffPacket(BaseModel, frozen=True):
    schema_: str = Field(default=HANDOFF_SCHEMA, alias="schema", serialization_alias="schema")
    handoff_id: str
    created_at: str
    source_agent: str = "unknown"
    target_agent: str = "unknown"
    session_receipt: HandoffReceiptReference
    task_state: HandoffTaskState = Field(default_factory=HandoffTaskState)
    evidence: list[HandoffEvidence] = Field(default_factory=list)
    required_reacquisitions: list[RequiredReacquisition] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model_config = {"extra": "forbid", "populate_by_name": True}


class HandoffInspectionResult(BaseModel, frozen=True):
    passed: bool
    summary: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_reacquisitions: list[RequiredReacquisition] = Field(default_factory=list)
    session_receipt_validation_level: str = ""
    model_config = {"extra": "forbid"}


def export_handoff(
    *,
    session_receipt: TokSessionReceipt | None = None,
    session_receipt_path: Path | None = None,
    session_id: str | None = None,
    bridge_receipts_path: Path | None = None,
    savings_events_path: Path | None = None,
    source_agent: str = "unknown",
    target_agent: str = "unknown",
    goal: str = "",
    completed: list[str] | None = None,
    next_steps: list[str] | None = None,
) -> HandoffPacket:
    """Build a local handoff packet from a session receipt or local session records."""
    receipt, receipt_path = _resolve_session_receipt(
        session_receipt=session_receipt,
        session_receipt_path=session_receipt_path,
        session_id=session_id,
        bridge_receipts_path=bridge_receipts_path,
        savings_events_path=savings_events_path,
    )
    evidence = [
        HandoffEvidence(
            label=artifact.kind,
            exactness=artifact.exactness,
            path=artifact.path,
            digest=artifact.digest,
        )
        for artifact in receipt.artifacts
    ]
    required = [
        RequiredReacquisition(
            reason="Exact evidence must be reacquired before edit-like work.",
            path=artifact.path,
        )
        for artifact in receipt.artifacts
        if artifact.exactness != "exact" or not artifact.available
    ]
    warnings = list(receipt.warnings)
    if receipt.savings_summary.degraded_to_baseline or receipt.savings_summary.fallback_count:
        warnings.append("Linked session receipt reports fallback or degraded baseline state.")
    return HandoffPacket(
        handoff_id="handoff_" + uuid.uuid4().hex,
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        source_agent=source_agent,
        target_agent=target_agent,
        session_receipt=HandoffReceiptReference(
            receipt_id=receipt.receipt_id,
            path=str(receipt_path) if receipt_path is not None else "",
            digest=_digest_file(receipt_path) if receipt_path is not None and receipt_path.exists() else "",
        ),
        task_state=HandoffTaskState(
            goal=goal,
            completed=completed or [],
            next_steps=next_steps or [],
        ),
        evidence=evidence,
        required_reacquisitions=required,
        warnings=warnings,
    )


def inspect_handoff(packet: HandoffPacket | dict[str, Any] | Path) -> HandoffInspectionResult:
    """Inspect a local handoff packet and report exact reacquisition requirements."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        parsed = _load_handoff_packet(packet)
    except Exception as exc:
        return HandoffInspectionResult(
            passed=False,
            summary=f"Invalid handoff packet: {exc}",
            errors=[f"schema_invalid:{exc}"],
        )

    if parsed.schema_ != HANDOFF_SCHEMA:
        errors.append(f"schema_version_unsupported:{parsed.schema_}")

    session_level = ""
    receipt_ref = parsed.session_receipt
    if receipt_ref.path:
        receipt_path = Path(receipt_ref.path)
        if not receipt_path.exists():
            errors.append(f"session_receipt_missing:{receipt_ref.path}")
        else:
            if receipt_ref.digest:
                actual = _digest_file(receipt_path)
                if actual != receipt_ref.digest:
                    errors.append(f"session_receipt_digest_mismatch:{receipt_ref.path}")
            try:
                receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt_result = verify_session_receipt(receipt_payload)
                session_level = receipt_result.level
                warnings.extend(receipt_result.warnings)
                errors.extend(f"session_receipt:{error}" for error in receipt_result.errors)
            except Exception as exc:
                errors.append(f"session_receipt_invalid:{exc}")
    else:
        warnings.append("No session receipt path is available for local verification.")

    for evidence in parsed.evidence:
        if evidence.exactness == "exact" and evidence.path:
            path = Path(evidence.path)
            if not path.exists():
                errors.append(f"exact_evidence_missing:{evidence.path}")
            elif evidence.digest and _digest_file(path) != evidence.digest:
                errors.append(f"exact_evidence_digest_mismatch:{evidence.path}")
        elif evidence.exactness != "exact":
            warnings.append(f"non_exact_evidence_requires_reacquisition:{evidence.label}")

    for reacquisition in parsed.required_reacquisitions:
        if not reacquisition.path:
            errors.append("required_reacquisition_missing_path")

    summary = _render_handoff_summary(parsed, session_level=session_level)
    return HandoffInspectionResult(
        passed=not errors,
        summary=summary,
        errors=errors,
        warnings=warnings + list(parsed.warnings),
        required_reacquisitions=list(parsed.required_reacquisitions),
        session_receipt_validation_level=session_level,
    )


def _resolve_session_receipt(
    *,
    session_receipt: TokSessionReceipt | None,
    session_receipt_path: Path | None,
    session_id: str | None,
    bridge_receipts_path: Path | None,
    savings_events_path: Path | None,
) -> tuple[TokSessionReceipt, Path | None]:
    if session_receipt is not None:
        return session_receipt, session_receipt_path
    if session_receipt_path is not None:
        payload = json.loads(session_receipt_path.read_text(encoding="utf-8"))
        return TokSessionReceipt.model_validate(payload), session_receipt_path
    if session_id is None:
        msg = "session_id is required when no session receipt is supplied"
        raise ValueError(msg)
    return (
        generate_session_receipt(
            session_id=session_id,
            bridge_receipts_path=bridge_receipts_path,
            savings_events_path=savings_events_path,
        ),
        None,
    )


def _load_handoff_packet(packet: HandoffPacket | dict[str, Any] | Path) -> HandoffPacket:
    if isinstance(packet, HandoffPacket):
        return packet
    if isinstance(packet, Path):
        payload = json.loads(packet.read_text(encoding="utf-8"))
        return HandoffPacket.model_validate(payload)
    return HandoffPacket.model_validate(packet)


def _render_handoff_summary(packet: HandoffPacket, *, session_level: str) -> str:
    lines = [
        f"Handoff: {packet.handoff_id}",
        f"Session receipt: {packet.session_receipt.receipt_id}",
    ]
    if session_level:
        lines.append(f"Session receipt validation: {session_level}")
    if packet.task_state.goal:
        lines.append(f"Goal: {packet.task_state.goal}")
    if packet.task_state.completed:
        lines.append("Completed:")
        lines.extend(f"- {item}" for item in packet.task_state.completed)
    if packet.task_state.next_steps:
        lines.append("Next steps:")
        lines.extend(f"- {item}" for item in packet.task_state.next_steps)
    if packet.required_reacquisitions:
        lines.append("Required exact reacquisitions:")
        lines.extend(f"- {item.path}: {item.reason}" for item in packet.required_reacquisitions)
    return "\n".join(lines)


def _digest_file(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


__all__ = [
    "HANDOFF_SCHEMA",
    "HandoffEvidence",
    "HandoffPacket",
    "HandoffReceiptReference",
    "HandoffInspectionResult",
    "HandoffTaskState",
    "RequiredReacquisition",
    "export_handoff",
    "inspect_handoff",
]
