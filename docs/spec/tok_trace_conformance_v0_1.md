# Tok Trace Conformance v0.1 Draft

Status: draft for 0.1.8 preparation

This page defines the minimum reader behavior for `tok-trace/v0.1-draft`. It is for
fixture and audit interoperability only. It does not define Resolver networking,
Capability handshakes, Tok Session exchange, signed provenance, a stable SDK, or
agent-to-agent behavior.

## Scope

A conforming L0-L2 reader can validate documented trace fixtures without importing Tok
gateway, runtime, compression, CLI, benchmark, or analysis internals. It may use any
implementation language and any JSON parser, but it must preserve the same draft
semantics. In other words, a reader should be able to validate L0-L2 fixture outcomes
without importing Tok gateway, runtime, compression, CLI, benchmark, or analysis
internals.

The conformance boundary is deliberately narrow: an independent reader must validate
documented trace fixtures without importing Tok gateway, runtime, compression, CLI,
benchmark, or analysis internals.

JSON is the first fixture encoding, not the protocol identity. Future encodings must
preserve the same envelope, observation, content, audit, exactness, resolver-state, and
fallback meanings.

## Levels

| Level | Reader requirement                                                                                                                                                             |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| L0    | Parse fixture JSON arrays and JSONL traces, reject malformed structure, and preserve block order.                                                                              |
| L1    | Validate required fields, enum values, extension namespace rules, canonical payload digests, and pass/warn/fail audit outcomes.                                                |
| L2    | Verify local artifacts, exact content hashes and sizes, exact versus non-exact rules, fallback/degradation reasons, sequence consistency, and supported `unified_diff` deltas. |

For 0.1.8, Tok validates exactly this L0-L2 set:

- L0: parse JSON fixture arrays and JSONL traces, reject malformed JSON or malformed
  fixture structure, and preserve block order.
- L1: validate required fields, enum values, extension namespace rules, canonical
  payload digests, and pass/warn/fail audit outcomes.
- L2: verify local artifact hashes and sizes, exact versus non-exact content claims,
  fallback/degradation reasons, sequence consistency, and supported unified_diff deltas.

Everything above L2 is documentation-only in 0.1.8.

L3, L4, and L5 remain out of scope for this conformance draft. A reader must not claim
cross-cache resolution, capability negotiation, or agent-to-agent compact-state exchange
from L0-L2 fixture success.

L0-L2 fixture success must not claim cross-cache resolution, capability negotiation, or
agent-to-agent compact-state exchange.

L0-L2 success also does not claim resolver networking, remote fetchability, or future
exact recovery. `accept_exact` means this reader verified the exact bytes locally. A
remote resolver reference can be identifiable and still warn because this reader has not
seen the bytes.

## L0-L2 Invariants

- `accept_exact` requires digest identity, size identity, and local byte availability.
- `available_local` always requires `resolver_uri`, `hash`, and `size_bytes`; otherwise
  there is nothing concrete for the L2 reader to verify.
- `accept_reference` is reserved for exact `reference` actions. Missing or remote exact
  references warn; they do not pass as local proof.
- Healthy pass-through metadata uses `accept_pass_through`, not `accept_reference` or
  `accept_fallback`.
- `summary_reference` and `skeleton_reference` are non-exact unless future layers add a
  separate exact recovery path; they cannot authorize edit-like exactness by themselves.
- `result: "ok"` cannot pair with `unresolvable_fallback_required`.
- Fallback, degraded, error, and rejected outcomes require an audit reason.
- A hash proves identity, not availability, authorization, freshness, or permission to
  export raw content.
- Extensions may add namespaced data, but they must not override core `envelope`,
  `observation`, `content`, or `audit` semantics.
- Metadata-only live traces are warnings unless local artifacts allow the reader to
  verify the audited metadata hash and size.

## Standalone Reader Boundary

The fixture corpus, expected outcomes, and this conformance page must be sufficient for
an independent reader to implement L0-L2 checks. Reader implementations should not need
bridge sigils, Claude Code grammar, runtime memory projection, gateway routing, release
gates, or provider-specific request shaping.

The current reference implementation lives inside Tok and is exercised through
`tok audit`, but conformance belongs to the trace records and fixtures, not to the
bridge runtime.
