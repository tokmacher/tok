# Tok Protocol Layers v0.1 Draft

Status: roadmap for 0.1.7 and later

Tok should grow as a layered protocol family, not as one overloaded trace format. The
0.1.7 release ships only the first visible layer: draft trace audit for bridge behavior.
Later layers should be added only after fixtures, audit behavior, and independent
readers prove the lower layers.

## Layers

| Layer          | Purpose                                                                                                 | 0.1.7 status                                               |
| -------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Tok Trace      | Retrospective audit records for what Tok observed, compressed, referenced, passed through, or degraded. | Draft, visible through `tok audit`.                        |
| Tok Resolver   | Content-addressed availability and retrieval semantics for exact bytes by hash.                         | Deferred. Local fixture and metadata artifact checks only. |
| Tok Capability | Runtime declarations for supported references, deltas, fallbacks, fixture packs, and encodings.         | Deferred. No handshake in 0.1.7.                           |
| Tok Session    | Ordered compact-state exchange between agents/tools.                                                    | Deferred. No agent-to-agent protocol in 0.1.7.             |

Tok Trace is evidence. Tok Resolver is availability. Tok Capability is negotiation. Tok
Session is exchange. Keeping those roles separate prevents v0.1 trace blocks from
becoming a dumping ground for every future protocol concern.

## Conformance Levels

| Level | Meaning                                                                                   |
| ----- | ----------------------------------------------------------------------------------------- |
| L0    | Read documented fixture files and reject malformed fixture structure.                     |
| L1    | Audit live bridge JSONL traces with pass/warn/fail outcomes.                              |
| L2    | Verify local artifacts, canonical payload digests, exactness rules, and supported deltas. |
| L3    | Resolve exact content by hash across cache boundaries.                                    |
| L4    | Negotiate capabilities with another runtime.                                              |
| L5    | Exchange compact verified state agent-to-agent.                                           |

Tok 0.1.7 targets L1/L2 only. L3 and above require new design work, adversarial
fixtures, and at least one reader that validates fixtures without importing Tok runtime
internals.

## Adversarial Fixture Roadmap

The protocol should be hardened by fixture packs that try to lie about identity,
availability, order, exactness, and semantics. The first roadmap set is:

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

0.1.7 includes tests for the cases the current draft verifier can defend locally. Future
releases should promote this list into a named adversarial fixture pack before making
stable protocol or interoperability claims.

## Stability Bar

Tok Trace should not be called stable until:

- the fixture corpus includes adversarial examples and expected audit outcomes
- a second implementation or standalone reader can validate the corpus
- resolver trust semantics distinguish identity, availability, authorization, and
  freshness
- capability negotiation has explicit fallback behavior
- signed provenance and replay protection have a design, even if optional
