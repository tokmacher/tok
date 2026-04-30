# Tok Trace Roadmap

Status: draft for 0.1.7

Tok Trace should grow from audit evidence, not from protocol ambition. The first useful
version is a small verifier-friendly trace format for bridge sessions.

Tok should become a layered protocol family rather than one overloaded file format. In
that family, Tok Trace is the audit-evidence layer. Tok Resolver, Tok Capability, and
Tok Session are future layers described in `tok_protocol_layers_v0_1.md`.

## Ladder

1. Invisible local bridge
   - Tok saves tokens for one user talking to one model API.
1. Auditable trace format
   - Tok can show what was compressed, referenced, restored, degraded, or passed
     through.
1. Stable fixture corpus
   - Other implementations can test their readers against documented examples.
1. Experimental audit command
   - Visible `tok audit` validates draft fixtures and trace files while keeping the
     trace format draft-scoped.
1. Resolver/cache layer
   - Agents and tools can share exact content by hash when bytes are available.
1. Capability handshake
   - Tools can declare support for references, deltas, fallbacks, fixture packs, and
     audit levels.
1. Agent-to-agent protocol
   - Agents exchange compact verified state instead of repeated prose blobs.

The early milestones are deliberately local and boring. The later milestones only become
credible if audit and fixtures stay strict.

## Conformance Levels

| Level | Meaning                                                           | 0.1.7 status                                 |
| ----- | ----------------------------------------------------------------- | -------------------------------------------- |
| L0    | Read documented fixture files.                                    | Covered by fixture tests.                    |
| L1    | Audit live bridge traces.                                         | Covered by `tok audit` and live JSONL tests. |
| L2    | Verify local artifacts, digests, exactness, and supported deltas. | Covered for local files and `unified_diff`.  |
| L3    | Resolve exact content by hash across cache boundaries.            | Deferred.                                    |
| L4    | Negotiate capabilities with another runtime.                      | Deferred.                                    |
| L5    | Exchange compact verified state agent-to-agent.                   | Deferred.                                    |

0.1.7 should only claim L1/L2 draft trace audit. Stable protocol or agent-to-agent
claims require L3+ design and independent conformance testing.

## Adversarial Fixture Roadmap

The protocol needs fixtures that try to lie or confuse readers before it can become a
stable interoperability surface:

- forged payload digest
- resolver URI path escape
- exactness lie
- resolver-state lie
- unsupported delta algorithm
- malformed JSONL line
- duplicate block IDs
- out-of-order turns
- unknown required field or version
- extension attempting to override core semantics

0.1.7 includes local tests for the subset the draft verifier can defend now. Later
releases should promote this into a named adversarial fixture pack with expected audit
outcomes.

## Compatibility Policy

`tok-trace/v0.1-draft` is allowed to change while it is draft, but changes should be
classified clearly.

| Change                                    | Policy                                     |
| ----------------------------------------- | ------------------------------------------ |
| Editorial clarification                   | Keep `tok-trace/v0.1-draft`.               |
| New fixture or expected audit result      | Keep `tok-trace/v0.1-draft`.               |
| New optional extension namespace          | Keep core version unchanged.               |
| New optional core field                   | Keep draft version, document the addition. |
| New required field                        | Bump draft revision before relying on it.  |
| Changed action/result meaning             | Bump to a new draft version.               |
| Changed digest/canonicalization algorithm | Bump to a new draft version.               |

The draft should not be called stable until:

- fixture validation covers malformed and degraded traces
- audit behavior has explicit pass/warn/fail semantics
- exactness and resolver availability are distinct in docs and tests
- at least one reader can validate fixtures without runtime bridge internals

## Runtime Adoption Gate

Bridge/runtime emission should wait until the docs and verifier agree on:

- exact vs non-exact content semantics
- resolver states
- delta requirements
- fallback requirements
- visible draft audit CLI behavior
- release-surface boundaries

0.1.7 may emit metadata-only live traces behind `TOK_TRACE=1`. Artifact-backed runtime
emission should remain opt-in and file-based before any streaming or provider-neutral
transport work begins.
