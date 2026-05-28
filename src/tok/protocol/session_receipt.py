"""Local Tok session receipt model and generator.

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

from tok.receipt import BridgeReceipt, read_bridge_receipts
from tok.runtime._diagnostics import DiagnosticsSnapshot
from tok.utils.savings_event import SavingsEvent, read_savings_events

SESSION_RECEIPT_SCHEMA = "tok-session-receipt/v0.1-draft"

EvidenceForm = Literal["exact", "summary", "skeleton", "reference"]
SavingsConfidence = Literal["none", "metadata_only", "event_backed", "trace_backed"]
ValidationLevel = Literal[
    "L0_schema",
    "L1_internal_consistency",
    "L2_digest",
    "L3_local_recovery",
    "L4_signed_provenance",
    "L5_remote_verification",
]


def _default_deferred_validation_levels() -> list[ValidationLevel]:
    return ["L4_signed_provenance", "L5_remote_verification"]


class SessionIdentity(BaseModel, frozen=True):
    session_id: str
    started_at: str = ""
    ended_at: str = ""
    turn_count: int = 0
    model_config = {"extra": "forbid"}


class AdapterIdentity(BaseModel, frozen=True):
    client: str = "unknown"
    adapter: str = "unknown"
    transport_boundary: str = "unknown"
    capabilities: list[str] = Field(default_factory=list)
    unstable: bool = False
    model_config = {"extra": "forbid"}


class ProviderIdentity(BaseModel, frozen=True):
    name: str = "unknown"
    model: str = ""
    api_base: str = ""
    model_config = {"extra": "forbid"}


class DiagnosticsSummary(BaseModel, frozen=True):
    status: str = "ok"
    bridge: str = "tok"
    mode: str = "unknown"
    request_policy: str = ""
    baseline_only: bool = False
    fallback_count: int = 0
    fail_open_count: int = 0
    session_quality: str = "clean"
    last_degradation_reason: str = ""
    calls: int = 0
    persistence_failures: int = 0
    evidence_exact_observed_count: int = 0
    evidence_non_exact_reference_count: int = 0
    evidence_non_exact_summary_count: int = 0
    evidence_non_exact_skeleton_count: int = 0
    evidence_exact_reacquisition_required_count: int = 0
    evidence_exact_reacquisition_satisfied_count: int = 0
    evidence_compression_blocked_for_safety_count: int = 0
    savings_source: str = "session_tracker"
    model_config = {"extra": "forbid"}

    @classmethod
    def from_snapshot(cls, snapshot: DiagnosticsSnapshot | None) -> DiagnosticsSummary:
        if snapshot is None:
            return cls()
        return cls(
            status=snapshot.status,
            bridge=snapshot.bridge,
            mode=snapshot.mode,
            request_policy=snapshot.request_policy,
            baseline_only=snapshot.baseline_only,
            fallback_count=snapshot.fallback_count,
            fail_open_count=snapshot.fail_open_count,
            session_quality=snapshot.session_quality,
            last_degradation_reason=snapshot.last_degradation_reason,
            calls=snapshot.calls,
            persistence_failures=snapshot.persistence_failures,
            evidence_exact_observed_count=snapshot.evidence_exact_observed_count,
            evidence_non_exact_reference_count=snapshot.evidence_non_exact_reference_count,
            evidence_non_exact_summary_count=snapshot.evidence_non_exact_summary_count,
            evidence_non_exact_skeleton_count=snapshot.evidence_non_exact_skeleton_count,
            evidence_exact_reacquisition_required_count=snapshot.evidence_exact_reacquisition_required_count,
            evidence_exact_reacquisition_satisfied_count=snapshot.evidence_exact_reacquisition_satisfied_count,
            evidence_compression_blocked_for_safety_count=snapshot.evidence_compression_blocked_for_safety_count,
            savings_source=snapshot.savings_source,
        )


class SavingsSummary(BaseModel, frozen=True):
    baseline_input_tokens: int = 0
    tok_input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    gross_tokens_saved: int = 0
    net_tokens_saved: int = 0
    reacquisition_tokens_spent: int = 0
    estimated_baseline_cost_usd: float = 0.0
    estimated_tok_cost_usd: float = 0.0
    estimated_cost_saved_usd: float = 0.0
    pricing_source: str = ""
    pricing_freshness: str = "unknown"
    fallback_count: int = 0
    degraded_to_baseline: bool = False
    compression_bypass_count: int = 0
    confidence: SavingsConfidence = "none"
    model_config = {"extra": "forbid"}


class EvidenceSummary(BaseModel, frozen=True):
    exact_count: int = 0
    summary_count: int = 0
    skeleton_count: int = 0
    reference_count: int = 0
    reacquisition_required_count: int = 0
    reacquisition_satisfied_count: int = 0
    resolver_backed_count: int = 0
    non_exact_edit_block_count: int = 0
    model_config = {"extra": "forbid"}


class ArtifactReference(BaseModel, frozen=True):
    kind: str
    path: str = ""
    digest: str = ""
    exactness: EvidenceForm = "reference"
    available: bool = False
    model_config = {"extra": "forbid"}


class ReceiptReference(BaseModel, frozen=True):
    receipt_id: str = ""
    event_id: str = ""
    action_id: str = ""
    digest: str = ""
    model_config = {"extra": "forbid"}


class ValidationSummary(BaseModel, frozen=True):
    level: ValidationLevel = "L0_schema"
    passed: bool = True
    deferred_levels: list[ValidationLevel] = Field(default_factory=_default_deferred_validation_levels)
    errors: list[str] = Field(default_factory=list)
    model_config = {"extra": "forbid"}


class TokSessionReceipt(BaseModel, frozen=True):
    schema_: str = Field(default=SESSION_RECEIPT_SCHEMA, alias="schema", serialization_alias="schema")
    receipt_id: str
    created_at: str
    session: SessionIdentity
    adapter: AdapterIdentity = Field(default_factory=AdapterIdentity)
    provider: ProviderIdentity = Field(default_factory=ProviderIdentity)
    diagnostics_summary: DiagnosticsSummary = Field(default_factory=DiagnosticsSummary)
    savings_summary: SavingsSummary = Field(default_factory=SavingsSummary)
    evidence_summary: EvidenceSummary = Field(default_factory=EvidenceSummary)
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    bridge_receipts: list[ReceiptReference] = Field(default_factory=list)
    savings_events: list[ReceiptReference] = Field(default_factory=list)
    deterministic_action_receipts: list[ReceiptReference] = Field(default_factory=list)
    validation: ValidationSummary = Field(default_factory=ValidationSummary)
    warnings: list[str] = Field(default_factory=list)
    model_config = {"extra": "forbid", "populate_by_name": True}


class ReceiptVerificationResult(BaseModel, frozen=True):
    passed: bool
    level: ValidationLevel
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    deferred_levels: list[ValidationLevel] = Field(default_factory=_default_deferred_validation_levels)
    model_config = {"extra": "forbid"}


def generate_session_receipt(
    *,
    session_id: str,
    bridge_receipts_path: Path | None = None,
    savings_events_path: Path | None = None,
    diagnostics: DiagnosticsSnapshot | None = None,
    adapter: AdapterIdentity | None = None,
    provider: ProviderIdentity | None = None,
) -> TokSessionReceipt:
    """Generate a local session receipt from available local records."""
    bridge_receipts = read_bridge_receipts(bridge_receipts_path) if bridge_receipts_path else []
    savings_events = read_savings_events(savings_events_path) if savings_events_path else []
    diagnostics_summary = DiagnosticsSummary.from_snapshot(diagnostics)
    savings_summary = _build_savings_summary(
        bridge_receipts=bridge_receipts,
        savings_events=savings_events,
        diagnostics=diagnostics,
    )
    evidence_summary = _build_evidence_summary(
        bridge_receipts=bridge_receipts,
        diagnostics=diagnostics_summary,
    )
    artifacts = _build_artifacts(
        bridge_receipts_path=bridge_receipts_path,
        savings_events_path=savings_events_path,
    )
    warnings = _build_warnings(
        bridge_receipts=bridge_receipts,
        savings_events=savings_events,
        savings_summary=savings_summary,
        evidence_summary=evidence_summary,
    )
    validation = ValidationSummary(
        level="L2_digest" if artifacts else "L1_internal_consistency",
        passed=True,
        deferred_levels=(
            ["L3_local_recovery", "L4_signed_provenance", "L5_remote_verification"]
            if not artifacts
            else ["L4_signed_provenance", "L5_remote_verification"]
        ),
    )
    return TokSessionReceipt(
        receipt_id="tsr_" + uuid.uuid4().hex,
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        session=SessionIdentity(
            session_id=session_id,
            turn_count=max((receipt.turn for receipt in bridge_receipts), default=0),
        ),
        adapter=adapter or AdapterIdentity(),
        provider=provider or _provider_from_events(savings_events),
        diagnostics_summary=diagnostics_summary,
        savings_summary=savings_summary,
        evidence_summary=evidence_summary,
        artifacts=artifacts,
        bridge_receipts=[
            ReceiptReference(receipt_id=receipt.receipt_id, digest=receipt.digest) for receipt in bridge_receipts
        ],
        savings_events=[
            ReceiptReference(event_id=event.event_id, digest=_digest_payload(event.to_dict()))
            for event in savings_events
        ],
        validation=validation,
        warnings=warnings,
    )


def verify_session_receipt(receipt: TokSessionReceipt | dict[str, Any]) -> ReceiptVerificationResult:
    """Verify a session receipt using local, schema-level evidence."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        parsed = receipt if isinstance(receipt, TokSessionReceipt) else TokSessionReceipt.model_validate(receipt)
    except Exception as exc:
        return ReceiptVerificationResult(
            passed=False,
            level="L0_schema",
            errors=[f"schema_invalid:{exc}"],
            deferred_levels=[
                "L1_internal_consistency",
                "L2_digest",
                "L3_local_recovery",
                "L4_signed_provenance",
                "L5_remote_verification",
            ],
        )

    if parsed.schema_ != SESSION_RECEIPT_SCHEMA:
        errors.append(f"schema_version_unsupported:{parsed.schema_}")
    if not parsed.receipt_id:
        errors.append("missing_receipt_id")
    if not parsed.session.session_id:
        errors.append("missing_session_id")
    if parsed.savings_summary.fallback_count < parsed.diagnostics_summary.fallback_count:
        errors.append("fallback_count_inconsistent")
    if parsed.diagnostics_summary.baseline_only and not parsed.savings_summary.degraded_to_baseline:
        errors.append("degraded_baseline_omitted")
    if parsed.savings_summary.degraded_to_baseline and parsed.diagnostics_summary.session_quality == "clean":
        warnings.append("degraded_savings_with_clean_diagnostics")
    if parsed.adapter.client == "codex-cli" and parsed.adapter.adapter.startswith("claude"):
        errors.append("adapter_profile_mismatch")
    if parsed.adapter.client == "claude-code" and parsed.adapter.adapter.startswith("codex"):
        errors.append("adapter_profile_mismatch")
    if (
        parsed.evidence_summary.reacquisition_satisfied_count > parsed.evidence_summary.reacquisition_required_count
        and parsed.evidence_summary.reacquisition_required_count > 0
    ):
        errors.append("reacquisition_satisfied_exceeds_required")
    if parsed.savings_summary.confidence in {"event_backed", "trace_backed"} and not parsed.savings_events:
        errors.append("savings_confidence_without_events")
    expected_saved = max(0, parsed.savings_summary.baseline_input_tokens - parsed.savings_summary.tok_input_tokens)
    if parsed.savings_summary.gross_tokens_saved and parsed.savings_summary.gross_tokens_saved != expected_saved:
        errors.append("savings_summary_inconsistent")

    digest_checked = False
    local_recovery_checked = False
    for artifact in parsed.artifacts:
        if artifact.kind in {"summary", "skeleton", "reference"} and artifact.exactness == "exact":
            errors.append(f"artifact_exactness_kind_mismatch:{artifact.kind}")
        if artifact.exactness == "exact" and not artifact.available:
            errors.append(f"exact_artifact_unavailable:{artifact.path}")
        if artifact.kind == "resolver" and artifact.path.startswith("tok://") and not artifact.available:
            errors.append(f"resolver_uri_unavailable:{artifact.path}")
        if artifact.digest and artifact.available and artifact.path:
            digest_checked = True
            path = Path(artifact.path)
            if not path.exists():
                errors.append(f"artifact_missing:{artifact.path}")
                continue
            actual = _digest_file(path)
            if actual != artifact.digest:
                errors.append(f"artifact_digest_mismatch:{artifact.path}")
        if artifact.exactness == "exact" and artifact.available:
            local_recovery_checked = True
        if artifact.exactness != "exact" and artifact.kind == "resolver" and artifact.available:
            errors.append(f"resolver_artifact_not_exact:{artifact.path}")
    for reference in parsed.bridge_receipts:
        if not reference.receipt_id or not reference.digest:
            errors.append("invalid_bridge_receipt_reference")
    for reference in parsed.savings_events:
        if not reference.event_id or not reference.digest:
            errors.append("invalid_savings_event_reference")

    if errors:
        return ReceiptVerificationResult(
            passed=False,
            level="L1_internal_consistency",
            errors=errors,
            warnings=warnings,
            deferred_levels=[
                "L2_digest",
                "L3_local_recovery",
                "L4_signed_provenance",
                "L5_remote_verification",
            ],
        )
    if local_recovery_checked and parsed.evidence_summary.reacquisition_required_count == 0:
        return ReceiptVerificationResult(
            passed=True,
            level="L3_local_recovery",
            warnings=warnings,
        )
    if digest_checked:
        return ReceiptVerificationResult(
            passed=True,
            level="L2_digest",
            warnings=warnings,
            deferred_levels=["L3_local_recovery", "L4_signed_provenance", "L5_remote_verification"],
        )
    return ReceiptVerificationResult(
        passed=True,
        level="L1_internal_consistency",
        warnings=warnings,
        deferred_levels=[
            "L2_digest",
            "L3_local_recovery",
            "L4_signed_provenance",
            "L5_remote_verification",
        ],
    )


def _build_savings_summary(
    *,
    bridge_receipts: list[BridgeReceipt],
    savings_events: list[SavingsEvent],
    diagnostics: DiagnosticsSnapshot | None,
) -> SavingsSummary:
    if savings_events:
        baseline_input = sum(event.baseline_input_tokens for event in savings_events)
        actual_input = sum(event.actual_input_tokens for event in savings_events)
        output_tokens = sum(event.actual_output_tokens for event in savings_events)
        gross_saved = sum(event.input_tokens_saved + event.output_tokens_saved for event in savings_events)
        reacquisition_spent = sum(_event_reacquisition_tokens(event) for event in savings_events)
        fallback_count = sum(1 for event in savings_events if event.fallback)
        degraded = any(event.degraded_to_baseline for event in savings_events)
        return SavingsSummary(
            baseline_input_tokens=baseline_input,
            tok_input_tokens=actual_input,
            output_tokens=output_tokens,
            cache_read_tokens=sum(event.cache_read_tokens for event in savings_events),
            cache_write_tokens=sum(event.cache_write_tokens for event in savings_events),
            gross_tokens_saved=gross_saved,
            net_tokens_saved=gross_saved - reacquisition_spent,
            reacquisition_tokens_spent=reacquisition_spent,
            estimated_baseline_cost_usd=sum(event.baseline_cost_usd for event in savings_events),
            estimated_tok_cost_usd=sum(event.actual_cost_usd for event in savings_events),
            estimated_cost_saved_usd=sum(event.cost_saved_usd for event in savings_events),
            pricing_source="tok.utils.pricing",
            pricing_freshness="event_recorded",
            fallback_count=fallback_count,
            degraded_to_baseline=degraded,
            compression_bypass_count=sum(1 for event in savings_events if event.input_tokens_saved <= 0),
            confidence="event_backed",
        )
    if diagnostics is not None:
        return SavingsSummary(
            baseline_input_tokens=diagnostics.baseline_tokens,
            tok_input_tokens=diagnostics.actual_tokens,
            output_tokens=0,
            gross_tokens_saved=diagnostics.session_tokens_saved,
            net_tokens_saved=diagnostics.session_net_tokens_saved,
            reacquisition_tokens_spent=diagnostics.reacquisition_cost_tokens,
            estimated_baseline_cost_usd=diagnostics.baseline_cost_usd,
            estimated_tok_cost_usd=diagnostics.actual_cost_usd,
            estimated_cost_saved_usd=diagnostics.cost_saved_usd,
            pricing_source=diagnostics.savings_source,
            pricing_freshness="diagnostic_snapshot",
            fallback_count=diagnostics.fallback_count,
            degraded_to_baseline=diagnostics.is_degraded,
            compression_bypass_count=sum(1 for receipt in bridge_receipts if not receipt.compression_applied),
            confidence=_diagnostics_savings_confidence(diagnostics),
        )
    return SavingsSummary(
        fallback_count=sum(1 for receipt in bridge_receipts if receipt.fallback),
        compression_bypass_count=sum(1 for receipt in bridge_receipts if not receipt.compression_applied),
        degraded_to_baseline=any(receipt.fallback for receipt in bridge_receipts),
        confidence="metadata_only" if bridge_receipts else "none",
    )


def _build_evidence_summary(
    *,
    bridge_receipts: list[BridgeReceipt],
    diagnostics: DiagnosticsSummary,
) -> EvidenceSummary:
    summary = EvidenceSummary(
        exact_count=diagnostics.evidence_exact_observed_count,
        summary_count=diagnostics.evidence_non_exact_summary_count,
        skeleton_count=diagnostics.evidence_non_exact_skeleton_count,
        reference_count=diagnostics.evidence_non_exact_reference_count,
        reacquisition_required_count=diagnostics.evidence_exact_reacquisition_required_count,
        reacquisition_satisfied_count=diagnostics.evidence_exact_reacquisition_satisfied_count,
        non_exact_edit_block_count=diagnostics.evidence_compression_blocked_for_safety_count,
    )
    if any((receipt.evidence_summary or {}).get("resolver_backed_count") for receipt in bridge_receipts):
        resolver_backed = sum(
            int((receipt.evidence_summary or {}).get("resolver_backed_count", 0)) for receipt in bridge_receipts
        )
        return summary.model_copy(update={"resolver_backed_count": resolver_backed})
    return summary


def _event_reacquisition_tokens(event: SavingsEvent) -> int:
    estimates = event.non_headline_estimates
    for key in ("reacquisition_cost_tokens", "reacquisition_tokens_spent"):
        if key in estimates:
            return max(0, int(estimates[key]))
    return 0


def _diagnostics_savings_confidence(diagnostics: DiagnosticsSnapshot) -> SavingsConfidence:
    trace_sources = {"trace", "trace_events", "trace_reconstruction", "bridge_trace"}
    return "trace_backed" if diagnostics.savings_source in trace_sources else "metadata_only"


def _build_artifacts(
    *,
    bridge_receipts_path: Path | None,
    savings_events_path: Path | None,
) -> list[ArtifactReference]:
    artifacts: list[ArtifactReference] = []
    for kind, path in (
        ("bridge_receipts", bridge_receipts_path),
        ("savings_events", savings_events_path),
    ):
        if path is None:
            continue
        artifacts.append(
            ArtifactReference(
                kind=kind,
                path=str(path),
                digest=_digest_file(path) if path.exists() else "",
                exactness="exact" if path.exists() else "reference",
                available=path.exists(),
            )
        )
    return artifacts


def _build_warnings(
    *,
    bridge_receipts: list[BridgeReceipt],
    savings_events: list[SavingsEvent],
    savings_summary: SavingsSummary,
    evidence_summary: EvidenceSummary,
) -> list[str]:
    warnings: list[str] = []
    if not savings_events:
        warnings.append("Savings are metadata-only unless backed by savings events.")
    if not bridge_receipts:
        warnings.append("No bridge receipts were available for this session receipt.")
    if savings_summary.degraded_to_baseline or savings_summary.fallback_count:
        warnings.append("Session includes fallback or degraded baseline behavior.")
    if evidence_summary.reacquisition_required_count > evidence_summary.reacquisition_satisfied_count:
        warnings.append("Exact reacquisition remains required for some non-exact evidence.")
    return warnings


def _provider_from_events(events: list[SavingsEvent]) -> ProviderIdentity:
    if not events:
        return ProviderIdentity()
    model = next((event.model for event in events if event.model), "")
    return ProviderIdentity(name="unknown", model=model)


def _digest_file(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def _digest_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


__all__ = [
    "AdapterIdentity",
    "ArtifactReference",
    "DiagnosticsSummary",
    "EvidenceSummary",
    "ProviderIdentity",
    "ReceiptReference",
    "SESSION_RECEIPT_SCHEMA",
    "SavingsSummary",
    "SessionIdentity",
    "TokSessionReceipt",
    "ValidationSummary",
    "generate_session_receipt",
    "verify_session_receipt",
]
