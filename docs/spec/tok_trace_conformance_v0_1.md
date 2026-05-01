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

L3, L4, and L5 remain out of scope for this conformance draft. A reader must not claim
cross-cache resolution, capability negotiation, or agent-to-agent compact-state exchange
from L0-L2 fixture success.

L0-L2 fixture success must not claim cross-cache resolution, capability negotiation, or
agent-to-agent compact-state exchange.

## L0-L2 Invariants

- Exact content requires digest identity and either local availability or a resolver
  state that honestly reports missing/unavailable bytes.
- `summary_reference` and `skeleton_reference` are non-exact unless future layers add a
  separate exact recovery path; they cannot authorize edit-like exactness by themselves.
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
