# Tok 0.1.9 Architecture Target

## Vocabulary

This document uses the following architectural vocabulary:

- **Module**: A Python file or package providing a related set of capabilities. e.g.,
  `gateway/`, `runtime/core.py`
- **Interface**: The public API surface of a module, defined by exported symbols in
  `__all__` or public class method signatures
- **Implementation**: The internal logic within a module that fulfills the interface
  contract
- **Depth**: The number of abstraction layers between the user-facing surface and the
  core logic. Shallower is simpler.
- **Seam**: A boundary between modules where behavior can be substituted without
  changing adjacent modules
- **Adapter**: A module that translates between external protocols (Claude Code, OpenAI)
  and the internal runtime
- **Leverage**: The ratio of behavioral change to code change — abstractions that enable
  large behavior changes with small code additions
- **Locality**: How contained the effects of a change are. Changes with high locality
  are safer to make.

## Target Stack

```
User & Tooling Surface
        ↓
Gateway Adapter          ← Adapter layer: translates Claude Code HTTP ↔ internal runtime
        ↓
Request Lifecycle        ← Explicit stage tracking via RequestLifecycle dataclass
        ↓
Request Shaping          ← BridgePreparedPayload carries shaping decisions
        ↓
Deterministic Core       ← RuntimeSession, compression, evidence safety
        ↓
Provider Adapter / Model API  ← Actual upstream calls to api.anthropic.com
```

### Side Surfaces

```
Diagnostics             ← DiagnosticsSnapshot dataclass, health endpoint, stats, audit
Audit & Protocol       ← Live trace emission, tok audit CLI, fixture validation
```

## Module Shape Goals

### Gateway Adapter (src/tok/gateway/)

**Interface**: `BridgeSession` class with methods `activate_session_for_request()`,
`runtime_session`, `tracker`, `smoothness_tracker`

**Implementation**: `_app_factory.py` (FastAPI handlers), `_bridge_streaming.py` (SSE
buffering/translation), `_bridge_request_handler.py` (fail-open retry logic),
`_bridge_runtime_pipeline.py` (request preparation pipeline)

**Depth**: 2 layers from user

- Layer 1: FastAPI route handlers (`_app_factory.py`)
- Layer 2: Pipeline helpers and session management

**Seams**:

- Between `_app_factory.py` and `_bridge_runtime_pipeline.py`: `BridgePreparedPayload`
  is the seam
- Between `_app_factory.py` and `_bridge_streaming.py`: `BridgePreparedPayload` + SSE
  byte stream
- Between streaming and non-streaming paths: both terminate at SSE re-emission

### Request Lifecycle (src/tok/runtime/\_request_lifecycle.py)

**Interface**: `RequestLifecycle` frozen dataclass with stage flags

**Implementation**: Immutable record of which stages were entered during request
processing

**Depth**: 1 layer (pure data, no behavior)

**Seams**: Passed as optional field in `BridgePreparedPayload`; consumers handle `None`

### Request Shaping (src/tok/gateway/\_types.py)

**Interface**: `BridgePreparedPayload` dataclass

**Implementation**: Carries `body`, `behavior_signals`, `request_policy`, `compressed`,
`saved_toks`, `prompt_metrics`, etc.

**Depth**: 1 layer (data carrier only)

**Seams**: Shared between pipeline, streaming, and non-streaming paths

### Deterministic Core (src/tok/runtime/core.py)

**Interface**: `RuntimeSession` class with `prepare_request()`, `process_response()`,
grouped sub-objects

**Implementation**: ~90 fields, ~400 methods. Contains `evidence_safety`,
`streaming_recovery`, `request_policy`, `answer_phase`, `file_delivery` grouped
sub-objects

**Depth**: Core — no layers above, only seams below

**Seams**:

- Grouped sub-objects isolate related state: `evidence_safety`, `streaming_recovery`,
  `request_policy`, etc.
- Each sub-object has its own `reset()` method for session cleanup

### Provider Adapter / Model API (src/tok/gateway/\_bridge_request_handler.py)

**Interface**: `send_with_tok_fail_open_retry()` function

**Implementation**: Rate-limited upstream dispatch with fail-open retry, provider-safe
recanonicalization

**Depth**: 1 layer below deterministic core

**Seams**: Fail-open behavior visible via behavior signals and trace events

## Side Surface: Diagnostics

**Module**: `src/tok/runtime/_diagnostics.py` (new)

**Interface**: `DiagnosticsSnapshot` dataclass with `from_session()` factory and
`to_health_response()` method

**Implementation**: Canonical data structure for health endpoint, stats, audit

**Depth**: 1 layer (internal data structure)

**Seams**: Consumed by health endpoint, stats, audit — all three populate the same
dataclass

**Note**: Rendering unification across health, stats, audit is deferred to 0.2.x. Only
the data model unification is in scope for 0.1.9.

## Side Surface: Audit & Protocol

**Modules**: `src/tok/spec/live_trace.py`, `src/tok/spec/trace.py`,
`src/tok/cli/_audit_commands.py`

**Interface**: `emit_live_trace()` for runtime trace emission; `tok audit` CLI for
fixture validation

**Implementation**: JSONL sidecar files with `request_prepared`, `response_processed`,
`fallback` events. L0-L2 fixture validation.

**Depth**: 1 layer (sidecar, not inline with request path)

**Seams**: Trace emission failures silently logged (non-fatal); artifact capture
requires opt-in

## Architectural Principles for 0.1.9

1. **Extract, don't restructure**: Extract helpers for readability; don't rearchitect
   existing logic
1. **Preserve seams**: New abstractions must not eliminate the ability to substitute
   behavior at existing seams
1. **Favor locality**: Each packet should affect a small number of files; avoid
   cross-cutting changes
1. **Immutable trace context**: `RequestLifecycle` is frozen; it observes without
   modifying behavior
1. **Data model before rendering**: `DiagnosticsSnapshot` unifies data sourcing before
   any rendering changes
1. **No new public surface**: All new types are internal or optional fields on existing
   types

## What 0.1.9 Does NOT Change

- The streaming buffering architecture remains unchanged
- The preflight quarantine logic remains unchanged
- The fail-open retry conditions remain unchanged
- The compression codec registry remains unchanged
- `RuntimeSession` field layout remains unchanged (grouped objects are completed, not
  restructured)
