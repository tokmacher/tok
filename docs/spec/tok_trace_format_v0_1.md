# Tok Trace Format v0.1 Draft

Status: draft for 0.1.7

Tok Trace Format v0.1 is an audit format for Tok bridge sessions. It records what the
bridge observed, stored, referenced, delta-compressed, summarized, or passed through
without changing the supported Claude Code bridge workflow.

This is not a universal agent protocol. It is not a stable public SDK contract. It does
not promote the experimental macro system, pointer system, Python runtime APIs, or
existing `tok.protocol` parser tools into the supported 0.1.x surface.

## Scope

Tok 0.1.x remains bridge-first. The supported path is Claude Code routed through the
local Tok bridge. The trace format sits beside the live bridge contract in
`docs/bridge-standard.md` and describes audit records for bridge behavior.

The first milestone is fixture validation:

- Can a trace block be parsed as a structured record?
- Can required audit fields be checked?
- Can exact content be distinguished from compact model-facing references?
- Can missing resolver/cache states be reported without pretending content is
  recoverable?
- Can malformed blocks be rejected before any runtime integration exists?

Runtime trace emission and `tok audit` CLI behavior are intentionally future work.

## Core Concepts

Each trace block has four top-level sections:

- `envelope`: versioning, identity, turn order, direction, and payload digest.
- `observation`: what kind of bridge event this is and which target it concerns.
- `content`: hashes, sizes, resolver information, and optional delta metadata.
- `audit`: verifier expectations, resolver state, and fallback or degradation reason
  when applicable.

An optional `extensions` object may carry namespaced experimental data. Extensions must
not change the meaning of core fields in v0.1.

## Actions

The draft action vocabulary is intentionally small:

| Action               | Meaning                                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| `pass_through`       | Raw content was used because compression was unnecessary or unsafe.      |
| `store`              | Exact content was first observed and made available to a cache/resolver. |
| `reference`          | Previously stored exact content was referenced instead of repeated.      |
| `delta`              | Content was represented relative to a known base.                        |
| `fallback`           | Tok used raw content because compact representation was unsafe.          |
| `skeleton_reference` | A non-exact structural representation points to recoverable content.     |
| `summary_reference`  | A non-exact summary points to recoverable content.                       |

`skeleton_reference` and `summary_reference` are draft-only and must be marked non-exact
unless backed by exact stored or resolvable content.

## Exactness

Tok Trace must distinguish exact recoverability from model-facing compactness.

Exact recoverability is allowed only when original content is stored locally or
resolvable and its digest can be verified. Summaries and skeletons are compact
representations, not original content. When exactness is required and the compact form
is unsafe, Tok must fall back to raw content.

## Resolver States

Resolver state is explicit:

| State                            | Meaning                                                      |
| -------------------------------- | ------------------------------------------------------------ |
| `available_local`                | The referenced bytes are present in the local fixture/cache. |
| `resolvable_remote`              | A URI is present, but local bytes are not included.          |
| `missing_identifiable`           | The content hash identifies bytes that are not available.    |
| `unresolvable_fallback_required` | Exact content cannot be recovered; raw fallback is required. |

A hash proves identity, not availability. A resolver URI is a recovery attempt, not a
guarantee that the receiver has the bytes.

## Proposed `tok audit` Behavior

A future `tok audit` command should:

1. Parse trace fixture files.
1. Validate required fields and enum values.
1. Recompute payload digests once canonicalization is implemented.
1. Resolve local fixture/cache artifacts when available.
1. Verify content hashes and delta relationships.
1. Report missing resolver/cache states separately from hash mismatches.
1. Treat fallback as a valid safety outcome when the trace explains why.

The first 0.1.7 implementation only validates fixture structure. Runtime protocol
compliance should not be claimed until audit behavior exists and passes the fixture
corpus.
