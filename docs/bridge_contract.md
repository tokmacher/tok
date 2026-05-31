# Tok Bridge Contract

> **Status:** Documentation of existing types. Not a new module or abstraction.
> **Created:** 2026-05-26 **Scope:** The bridge/runtime contract between adapters and
> the Tok runtime. This is not Tok Session, not Tok Capability, not remote resolver
> routing, and not agent-to-agent exchange.

This document maps the current adapter and runtime boundary. It records what exists in
the codebase today so future adapter work can stay transport-thin and avoid turning the
Claude Code bridge into the identity of Tok's core runtime.

## Contract Types

| Contract Concept   | Existing Type              | Location                                |
| ------------------ | -------------------------- | --------------------------------------- |
| Runtime identity   | `SurfaceMetadata`          | `src/tok/runtime/types.py`              |
| Inbound request    | `RuntimeRequest`           | `src/tok/runtime/types.py`              |
| Prepared request   | `PreparedRuntimeRequest`   | `src/tok/runtime/types.py`              |
| Processed response | `ProcessedRuntimeResponse` | `src/tok/runtime/types.py`              |
| Safety decision    | `SafetyDecision`           | `src/tok/runtime/types.py`              |
| Signal packet      | `SignalPacket`             | `src/tok/runtime/types.py`              |
| Lifecycle stages   | `RequestLifecycle`         | `src/tok/runtime/_request_lifecycle.py` |
| Health snapshot    | `DiagnosticsSnapshot`      | `src/tok/runtime/_diagnostics.py`       |
| Adapter boundary   | `AdapterProtocol`          | `src/tok/adapters/adapter_protocol.py`  |
| Evidence ledger    | `EvidenceSafetyState`      | `src/tok/runtime/evidence_safety.py`    |

## Runtime Identity

`SurfaceMetadata` is the runtime-neutral identity and wire-shape descriptor. Its current
fields are:

- `runtime`
- `adapter`
- `input_shape`
- `output_shape`
- `supports_tool_pairs`
- `uses_bridge_profile`
- `requires_provider_canonicalization`
- `uses_cut_search`
- `uses_plan_finalization_guard`
- `uses_first_turn_broad_audit_guard`

`RuntimeRequest` carries model input, messages, optional system content, adapter
identity, request policy, tool flags, and optional grammar/todo/delta fields. It exposes
surface-derived properties so adapter-specific behavior can stay explicit.

`PreparedRuntimeRequest` carries the runtime-prepared request body, compression and
token accounting, behavior signals, normalized tool events, request-policy metadata, and
bloat attribution.

`ProcessedRuntimeResponse` carries response content blocks, output savings, behavior
signals, mode fields, and updated memory.

## Lifecycle Stages

`RequestLifecycle` records which bridge/runtime preparation stages completed. The
gateway-stage flags are:

- `initial_preflight`
- `model_extraction`
- `tool_compatibility_check`
- `request_preparation`
- `runtime_preparation`
- `repeat_target_capture`
- `tool_event_normalization`
- `hot_memory_refresh`
- `signals_and_metrics`
- `prepared_preflight`
- `compression_safety_applied`
- `plan_finalization_guard`
- `final_payload_construction`

`response_processing_complete` is reserved for post-response tracking and is not part of
the request-preparation gateway completion check.

## Adapter Boundary

`AdapterProtocol` defines the transport boundary that runtime adapters should expose:

- `identify_runtime()`
- `parse_inbound_request(raw_bytes)`
- `build_outbound_response(processed)`
- `supported_capabilities()`
- `transport_boundary()`

Adapters should map client/provider wire shapes into `RuntimeRequest` and back from
`ProcessedRuntimeResponse`. Compression, exactness, fallback state, resolver behavior,
stats, and audit evidence belong in the core runtime layers, not in adapter-specific
transport code.

## Evidence Safety

`EvidenceSafetyState` is the session-local evidence ledger. It uses these evidence
forms:

- `exact`: verbatim, first-hand observation content.
- `summary`: lossy natural-language summary of an observation.
- `skeleton`: structural outline that is not full content.
- `reference`: pointer or stable stub.

Non-exact evidence must not authorize edit-like behavior without exact reacquisition.
The ledger records exact observations, non-exact emissions, pending reacquisition, and
compact audit counters.

## Diagnostics

`DiagnosticsSnapshot` is the health and stats snapshot. Its fields currently group into
these categories:

- bridge status and configuration
- token and cost savings
- fallback, fail-open, and degradation state
- repeated search/file-read and hot-target signals
- request-policy and tool-history repair signals
- stream recovery signals
- session quality and task/smoothness scores
- evidence-safety counters
- savings attribution source

The snapshot is used for health and stats reporting. It is not a protocol-compliance
certificate.

## Adapter Implementations

| Adapter                 | Scope                        | Transport               | Gating                        |
| ----------------------- | ---------------------------- | ----------------------- | ----------------------------- |
| `ClaudeBridgeAdapter`   | Live defended path           | HTTP proxy              | Always active                 |
| `OpenAIChatAdapter`     | Experimental                 | Chat-message mapping    | Experimental root export only |
| `OrchestratorAdapter`   | Programmatic OpenRouter path | Direct API              | Experimental root export only |
| `TextLoopAdapter`       | Test/utility path            | In-process              | Experimental root export only |
| `CodexAdapter`          | Experimental adapter proof   | Local HTTP proxy helper | `TOK_UNSTABLE_ADAPTERS=1`     |
| `OpenCodeAdapter` probe | Fixture-only                 | TBD                     | `TOK_UNSTABLE_ADAPTERS=1`     |

## OrchestratorAdapter Partial Proof

`OrchestratorAdapter` is a live programmatic adapter that routes turn preparation and
response processing through `UniversalTokRuntime` and `RuntimeSession`. The orchestrator
path also uses session savings tracking.

This proves that some non-Claude-shaped traffic can use the runtime boundary, but it is
not an intercepting proxy for a second coding-agent client.

## Codex Local Proxy Proof

`src/tok/gateway/_adapter_proxy.py` is the gated adapter proof. It is separate from the
defended Claude bridge gateway and routes Codex-shaped request bytes through:

```text
CodexAdapter.parse_inbound_request()
UniversalTokRuntime.prepare_request()
provider forwarder
UniversalTokRuntime.process_response()
CodexAdapter.build_outbound_response()
```

If adapter parsing or runtime preparation fails, it returns the original request body
with raw-passthrough diagnostics. This proves the adapter/runtime boundary without
claiming Codex CLI is a supported public workflow.

## Explicit Non-Goals

This document does not define:

- Tok Session
- Tok Capability
- remote resolver routing
- signed provenance
- agent-to-agent exchange
- universal protocol compliance
- a stable Python SDK

## Gap Analysis

The Codex proxy proof is local and gated. A real Codex CLI version still needs a
verified transport override before it can be described as a live user workflow.
