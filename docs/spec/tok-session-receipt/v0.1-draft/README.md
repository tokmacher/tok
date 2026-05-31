# Tok Session Receipt v0.1 Draft

> **Status:** Draft local schema for v0.2.2 Workstream B. **Scope:** Portable summary of
> one local Tok session assembled from local receipts, savings events, diagnostics,
> traces, and evidence-safety state. **Non-goal:** This is not Tok Session runtime
> support, not remote resolver routing, not signed provenance, and not a universal
> protocol-compliance certificate.

`TokSessionReceipt` is the portable unit of exchange for a completed or partial local
Tok session. It answers the same questions across client adapters:

- Which client adapter and provider surface handled the session?
- What would the session have cost without Tok?
- What did Tok actually send?
- What was saved?
- Was exactness preserved?
- Can original exact content be recovered locally?
- Did fallback or degraded baseline behavior happen?

The receipt is local-first. It may reference local resolver entries, local trace
sidecars, local bridge receipts, and local savings event files. It must not imply remote
attestation unless a future schema version explicitly adds that layer.

## Relationship To Existing Records

| Existing Record              | Current Scope                                    | Session Receipt Relationship                                  |
| ---------------------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| `BridgeReceipt`              | Per-turn bridge operation outcome in local JSONL | Referenced by digest and summarized across the session        |
| `SavingsEvent`               | Per-request token and cost accounting            | Aggregated into session-level savings fields                  |
| `DeterministicActionReceipt` | Exact local action decision or execution         | Linked when exact actions are part of the handoff/audit trail |
| `DiagnosticsSnapshot`        | Health and stats snapshot                        | Selected fields copied into `diagnostics_summary`             |
| Tok live trace               | Structural trace and sidecar metadata            | Referenced by path/digest when available                      |
| Resolver manifest/cache      | Local exact-content recovery                     | Referenced for exact reacquisition, not embedded by default   |

## Top-Level Shape

```json
{
  "schema": "tok-session-receipt/v0.1-draft",
  "receipt_id": "tsr_...",
  "created_at": "2026-05-26T00:00:00Z",
  "session": {},
  "adapter": {},
  "provider": {},
  "diagnostics_summary": {},
  "savings_summary": {},
  "evidence_summary": {},
  "artifacts": [],
  "bridge_receipts": [],
  "savings_events": [],
  "deterministic_action_receipts": [],
  "validation": {},
  "warnings": []
}
```

## Required Fields

### Identity

- `schema`: fixed string `tok-session-receipt/v0.1-draft`.
- `receipt_id`: unique local receipt id.
- `created_at`: UTC ISO-8601 timestamp.
- `session.session_id`: local Tok session id.
- `session.started_at`: known start timestamp or empty string when unavailable.
- `session.ended_at`: known end timestamp or empty string when unavailable.
- `session.turn_count`: number of known turns.

### Adapter

- `adapter.client`: client runtime name, for example `claude-code` or `codex-cli`.
- `adapter.adapter`: Tok adapter name, for example `claude-bridge` or `codex-cli-probe`.
- `adapter.transport_boundary`: `http-proxy`, `stdio`, `in-process`, or another
  documented boundary.
- `adapter.capabilities`: adapter-reported capability labels.
- `adapter.unstable`: whether use required `TOK_UNSTABLE_ADAPTERS=1`.

### Provider

- `provider.name`: provider family, for example `anthropic`, `openai-compatible`, or
  `unknown`.
- `provider.model`: model name when known.
- `provider.api_base`: upstream API base when safe to report; omit secrets.

### Diagnostics Summary

These fields map directly from `DiagnosticsSnapshot` where available:

- `status`
- `bridge`
- `mode`
- `request_policy`
- `baseline_only`
- `fallback_count`
- `fail_open_count`
- `session_quality`
- `last_degradation_reason`
- `calls`
- `persistence_failures`
- `evidence_exact_observed_count`
- `evidence_non_exact_reference_count`
- `evidence_non_exact_summary_count`
- `evidence_non_exact_skeleton_count`
- `evidence_exact_reacquisition_required_count`
- `evidence_exact_reacquisition_satisfied_count`
- `evidence_compression_blocked_for_safety_count`
- `savings_source`

The receipt may include more diagnostic fields, but consumers should treat these as the
minimum stable draft set.

### Savings Summary

- `baseline_input_tokens`
- `tok_input_tokens`
- `output_tokens`
- `cache_read_tokens`
- `cache_write_tokens`
- `gross_tokens_saved`
- `net_tokens_saved`
- `reacquisition_tokens_spent`
- `estimated_baseline_cost_usd`
- `estimated_tok_cost_usd`
- `estimated_cost_saved_usd`
- `pricing_source`
- `pricing_freshness`
- `fallback_count`
- `degraded_to_baseline`
- `compression_bypass_count`
- `confidence`

`confidence` values:

- `none`: no usable accounting.
- `metadata_only`: receipt has counters but no durable event trail.
- `event_backed`: receipt includes savings event references.
- `trace_backed`: receipt includes trace/receipt references sufficient for local audit.

Savings claims must be reported as estimates unless backed by the required local event
and pricing evidence.

### Evidence Summary

Evidence forms:

- `exact`: verbatim, first-hand observed content.
- `summary`: lossy natural-language summary.
- `skeleton`: structural outline without full content.
- `reference`: pointer or stable stub.

Fields:

- `exact_count`
- `summary_count`
- `skeleton_count`
- `reference_count`
- `reacquisition_required_count`
- `reacquisition_satisfied_count`
- `resolver_backed_count`
- `non_exact_edit_block_count`

Rule: non-exact evidence must not authorize edits or release claims without exact
reacquisition.

### Artifacts

Each artifact reference should contain:

- `kind`: `trace`, `resolver`, `bridge_receipts`, `savings_events`,
  `deterministic_action_receipt`, or `other`.
- `path`: local path when safe to disclose.
- `digest`: `sha256:<hex>` when known.
- `exactness`: one of `exact`, `summary`, `skeleton`, `reference`.
- `available`: whether the artifact was available during receipt generation.

## Validation Levels

Validation reports the strongest level proven by local evidence.

- `L0_schema`: JSON parsed and required top-level fields exist.
- `L1_internal_consistency`: counts, ids, and degraded/fallback flags are internally
  consistent.
- `L2_digest`: referenced local digests match available local files or records.
- `L3_local_recovery`: exact evidence needed for recovery is available through local
  artifact paths or resolver references.
- `L4_signed_provenance`: deferred; not implemented in v0.1 draft.
- `L5_remote_verification`: deferred; not implemented in v0.1 draft.

Receipts must clearly report deferred levels. Passing `L0` through `L3` is not proof of
universal protocol compliance.

## Non-Claims

A session receipt does not claim:

- hosted service behavior
- universal protocol compliance
- stable Python SDK support
- remote resolver availability
- signed provenance
- production readiness
- savings achieved without local `tok stats` or savings event evidence
- bridge health without bridge health output

## Example Files

- `examples/good_receipt.json`: clean local receipt with event-backed savings.
- `examples/degraded_receipt.json`: degraded baseline receipt with explicit fallback and
  weak savings confidence.
