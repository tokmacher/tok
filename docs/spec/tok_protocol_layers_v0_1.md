# Tok Protocol Layers v0.1 Draft

Status: roadmap for 0.1.x, updated for 0.2.0 local resolver beta

Tok should grow as a layered protocol family, not as one overloaded trace format. The
0.1.x releases ship the first visible layer (draft trace audit). Tok 0.2.0 adds L3a
local resolver beta for content-addressed local storage. Remote resolver routing,
capability handshakes, session exchange, and agent-to-agent behavior remain deferred.
Later layers should be added only after fixtures, audit behavior, and independent
readers prove the lower layers.

## Layers

| Layer          | Purpose                                                                                                 | 0.2.0 status                                   |
| -------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Tok Trace      | Retrospective audit records for what Tok observed, compressed, referenced, passed through, or degraded. | Draft, visible through `tok audit`.            |
| Tok Resolver   | Content-addressed availability and retrieval semantics for exact bytes by hash.                         | L3a local resolver beta. No remote routing.    |
| Tok Capability | Runtime declarations for supported references, deltas, fallbacks, fixture packs, and encodings.         | Deferred. No handshake in 0.2.0.               |
| Tok Session    | Ordered compact-state exchange between agents/tools.                                                    | Deferred. No agent-to-agent protocol in 0.2.0. |

Tok Trace is evidence. Tok Resolver is availability. Tok Capability is negotiation. Tok
Session is exchange. Keeping those roles separate prevents v0.1 trace blocks from
becoming a dumping ground for every future protocol concern.

## Routing Design Axis

Tok Routing is a cross-cutting concern, not a 0.1.x protocol layer. It should answer
where a runtime is allowed to ask for missing content, trace blocks, session roots, or
capability documents. It should not adopt TCP/IP-style packet routing, global routing
tables, a DHT, or ambient public discovery.

Routing decisions belong at the boundary between Resolver, Capability, and Session:

- resolver routing: where to ask for missing content
- capability routing: whether the requester is allowed to ask
- session routing: which peer or session owns a state chain
- privacy routing: whether the lookup itself leaks sensitive metadata

The default route is local-only. Remote routing requires explicit configuration and,
once capabilities exist, capability checks.

Routing asks four questions:

| Question                 | Examples                                                               |
| ------------------------ | ---------------------------------------------------------------------- |
| What is being requested? | content hash, trace block, session root, capability document           |
| Who is asking?           | local user, local runtime, configured peer, remote agent               |
| Who may know?            | local cache only, same project/session, trusted peer, explicit gateway |
| What may be returned?    | full bytes, metadata only, referral, denial, fallback required         |

## Conformance Levels

| Level | Meaning                                                                                   |
| ----- | ----------------------------------------------------------------------------------------- |
| L0    | Read documented fixture files and reject malformed fixture structure.                     |
| L1    | Audit live bridge JSONL traces with pass/warn/fail outcomes.                              |
| L2    | Verify local artifacts, canonical payload digests, exactness rules, and supported deltas. |
| L3    | Resolve exact content by hash across cache boundaries.                                    |
| L3a   | Resolve exact content from a local resolver only.                                         |
| L3b   | Resolve from explicitly configured remote resolvers.                                      |
| L3c   | Follow resolver referrals with loop detection.                                            |
| L4    | Negotiate capabilities with another runtime.                                              |
| L5    | Exchange compact verified state agent-to-agent.                                           |

Tok 0.2.0 targets L1/L2 audit plus L3a local resolver beta. Remote resolver routing
(L3b), referral following (L3c), capability negotiation (L4), and session exchange (L5)
require new design work, adversarial fixtures, and at least one independent reader.

## Named Adversarial Packs

The protocol should be hardened by named fixture packs that try to lie about identity,
availability, order, exactness, and semantics. The current machine-readable pack
manifest is `fixtures/adversarial_packs.json`.

The first locally defensible pack is `trace-l1-l2-core-adversarial`. It covers cases the
0.1.x verifier can judge without Resolver, Capability, or Session implementations:

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

Future Resolver and Routing fixture packs should add:

- resolver referral loop
- unauthorized resolver request
- hash exists but capability missing
- resolver returns wrong bytes
- resolver leaks path or session metadata in an error
- remote unavailable but trace remains valid
- conflicting resolver manifests

Tok includes tests for the cases the current draft verifier can defend locally. Future
releases should promote this list into a named adversarial fixture pack before making
stable protocol or interoperability claims.

L3a local resolver cases are now testable. L3b/L3c/L4/L5 cases must remain documented as
future packs until Tok implements remote routing, capability, or session semantics. They
should not be fake-passed by the L1/L2 audit verifier.

## Stability Bar

Tok Trace should not be called stable until:

- the fixture corpus includes adversarial examples and expected audit outcomes
- a second implementation or standalone reader can validate the corpus
- resolver trust semantics distinguish identity, availability, authorization, and
  freshness
- capability negotiation has explicit fallback behavior
- signed provenance and replay protection have a design, even if optional
