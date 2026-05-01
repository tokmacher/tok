# Tok 0.2 Architecture Roadmap

This document is a roadmap, not the current runtime contract. Tok 0.1.7 remains a
bridge-first release; the live contract is still described in
[architecture.md](./architecture.md). The purpose of this page is to make the intended
0.2 direction visible to future agents before they edit gateway, runtime, compression,
or release-surface code.

## Current 0.1.7 Posture

- The Claude bridge is the supported product surface.
- `src/tok/gateway/` owns HTTP transport, auth/header shaping, upstream forwarding,
  streaming cleanup, fail-open retry, and bridge session telemetry.
- `src/tok/runtime/` owns request preparation, response processing, memory projection,
  policy decisions, validation, and runtime session state.
- `src/tok/compression/` owns history and tool-result compression, but currently shares
  request-shaping concepts with runtime repeat-target and evidence logic.
- `src/tok/protocol/` owns the canonical Tok schema, parser, encoder, and format bridge.
- `src/tok/cli/`, `src/tok/testing/`, and `src/tok/analysis/` are outer surfaces and
  must not become runtime dependencies.

## Intended 0.2 Target Layers

The 0.2 goal is not a broader public product surface. It is a more agent-legible
implementation of the same bridge-first promise:

1. **Thin gateway adapter**: route requests, select sessions, invoke lifecycle steps,
   and return responses.
1. **Request lifecycle contract**: a typed internal model that names raw request,
   provider-safe request, Tok-prepared request, retry eligibility, response accounting,
   metrics, and live trace evidence.
1. **Request-shaping layer**: the explicit home for the current runtime/compression
   coupling around history cutting, evidence identity, repeat reads, and tool-result
   compression.
1. **Small deterministic core**: pure request/response transforms only. No environment
   reads, transport, persistence, CLI imports, or hidden I/O.
1. **Outer policy and adapters**: config, transport, persistence, telemetry, release
   gates, CLI, benchmark, and analysis concerns.

## Dependency Direction

The allowed direction for 0.2 work should be:

```text
cli / testing / analysis
        ↓
gateway adapter
        ↓
request lifecycle + request shaping
        ↓
runtime policies / compression / validation
        ↓
protocol models + deterministic core
```

Rules to preserve:

- `runtime` and `compression` must not import `gateway` or `cli`.
- `protocol` must not import adapters, gateway, CLI, testing, or analysis code.
- `gateway` may depend on runtime, compression through runtime-facing APIs, protocol,
  and stats/trace helpers.
- `testing`, `analysis`, and benchmark internals may depend inward, but runtime code
  must not depend on them.

## Not Implemented Yet

These are 0.2 contracts to design before coding. They are intentionally not live
abstractions in 0.1.7.

### Request Lifecycle Contract

The future lifecycle model should explicitly name:

- raw inbound request bytes and headers;
- canonical provider-safe request;
- Tok-prepared request;
- retry eligibility and whether raw retry is forbidden;
- upstream call count and retry reason;
- response processing mode;
- metrics and behavior signals;
- live trace evidence emitted for audit.

The gateway route should eventually call named lifecycle steps rather than carrying
large sets of local variables through one function.

### Behavior-Signal Registry

Behavior signals remain accepted as free-form strings for experimental work. For 0.2,
new release-critical signals should be registered before use so runtime behavior,
doctor/status diagnostics, stats, trace audit, and release gates do not drift apart.

The registry should cover at least:

- fail-open and retry signals;
- stream recovery signals;
- request-policy escalation/de-escalation signals;
- validation and provider-safety signals;
- evidence exactness, skeleton, and compression-safety signals;
- release-gate and trace-audit signals.

The first 0.2 implementation step should be additive: introduce the registry and tests,
then migrate call sites gradually. Unknown signals may continue to exist as internal or
experimental details, but anything used for a release decision, user-visible diagnostic,
or evidence-safety proof should have category, severity, display label, and health-gate
metadata in the registry.

### Bridge-Local Observability

0.2 trace work should strengthen Tok's bridge promise without turning Tok into a hosted
agent observability product. The target is a local, privacy-preserving summary for each
turn that can explain:

- compression decision and request policy;
- exact, non-exact, reacquisition, and safety-block state;
- fallback/degradation reason, if any;
- input-token delta and estimated savings;
- whether the trace is metadata-only or artifact-backed.

OpenTelemetry GenAI compatibility may be useful for bridge health summaries later, but
Tok should not export full prompts, tool outputs, hidden reasoning, or general agent
workflow spans as part of the 0.2 bridge architecture.

### Config Strictness Table

0.1.7 may warn and keep safe defaults for some invalid environment values. 0.2 should
centralize config parsing and classify each setting:

- **startup-fatal**: port, bind host, API base, auth/transport mode, release gates;
- **warn-and-default**: tuning thresholds, prompt budgets, benchmark/dev toggles;
- **test-only/dev-only**: experimental analysis and benchmark flags.

Runtime request processing should receive parsed config explicitly instead of reading
environment variables in deep modules.

### Runtime State Grouping

`RuntimeSession` is stateful orchestration state, not the eventual small core. 0.2
should group state into named areas before large refactors:

- caches;
- memory and persistence;
- evidence and exactness;
- recovery and fail-open state;
- smoothness;
- request policy;
- telemetry.

Backwards-compatible properties can remain while new code moves through named groups.

## Evidence-Safety Contract

The bridge may reduce redundant evidence, but must not hide evidence needed for correct
edits or conclusions.

- First file observation must be exact.
- Summarized or skeleton evidence must not authorize edit-like tools by itself.
- Repeated reads may compress only after exact evidence exists for the same identity.
- If a file was delivered as skeleton/summary and then becomes an edit target, the agent
  must see exact content again first.
- Novel failures, provider-sensitive tool pairing failures, and fresh traceback details
  must not be silently compressed away.
- Evidence safety is Tok's bridge-fidelity proof layer, not graph memory, OpenCode
  command integration, or context-pack product expansion.
- Live trace and audit summaries should distinguish exact observations from non-exact
  model-facing references so compression can be defended without claiming summaries are
  original bytes.

0.1.7 should document this contract and preserve existing tests. 0.2 should enforce it
mechanically with a narrow evidence-safety test suite.

## 0.2 Prep Rule

Before adding a live abstraction, add a doc or test that states the boundary. The goal
is to prevent half-built architecture from entering the 0.1.x bridge path.
