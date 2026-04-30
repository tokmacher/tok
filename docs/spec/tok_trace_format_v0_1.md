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

Tok Trace is the first layer in a larger protocol roadmap. It is not the resolver,
capability-handshake, or session-exchange layer. See `tok_protocol_layers_v0_1.md` for
the future layering model.

The first milestone is fixture validation:

- Can a trace block be parsed as a structured record?
- Can required audit fields be checked?
- Can exact content be distinguished from compact model-facing references?
- Can missing resolver/cache states be reported without pretending content is
  recoverable?
- Can malformed blocks be rejected before any runtime integration exists?

Raw artifact-capturing runtime trace emission is intentionally future work. The visible
`tok audit` command validates draft fixtures and live JSONL traces, but it does not
certify universal protocol compliance.

## Core Concepts

Each trace block has four top-level sections:

- `envelope`: versioning, identity, turn order, direction, and payload digest.
- `observation`: what kind of bridge event this is and which target it concerns.
- `content`: hashes, sizes, resolver information, and optional delta metadata.
- `audit`: verifier expectations, resolver state, and fallback or degradation reason
  when applicable.

An optional `extensions` object may carry namespaced experimental data. Extensions must
not change the meaning of core fields in v0.1.

Extension namespaces must not reuse the core section names `envelope`, `observation`,
`content`, or `audit`.

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

For live 0.1.7 traces, `available_local` can refer to a sanitized metadata artifact
written next to the trace file. That proves the audit record's metadata hash and byte
size, not the original prompt, response, or tool-result bytes unless the block also
marks `content.exact: true`.

## `tok audit` Behavior

The visible `tok audit` command validates draft fixture files and live trace sidecars.
It should:

1. Parse trace fixture files.
1. Validate required fields and enum values.
1. Recompute payload digests for the stable semantic payload.
1. Warn on `draft-uncomputed` payload digests.
1. Resolve local `tok-fixture://` artifacts when available.
1. Verify content hashes and byte sizes.
1. Replay `unified_diff` deltas in fixture-local audit.
1. Report missing resolver/cache states separately from hash mismatches.
1. Treat fallback as a valid safety outcome when the trace explains why.

The first 0.1.7 implementation validates fixture structure and fixture-local artifacts.
It can also emit opt-in metadata-only live traces with `TOK_TRACE=1`. Runtime protocol
compliance should not be claimed until artifact-backed bridge emission exists and passes
the fixture corpus.

## Opt-in Live Trace Emission

Set `TOK_TRACE=1` before starting the bridge to write metadata-only JSONL trace blocks
under `.tok/traces/`. These records are sidecar audit data only. They are never sent to
the model or provider, and they do not change compression decisions.

The live trace surface is intentionally small in 0.1.7:

- `request_prepared`
- `fallback`
- `response_processed`
- `audit_warning` for trace-emission issues

Non-streaming bridge requests emit request and response trace blocks. Streaming bridge
trace coverage is intentionally limited in 0.1.7; streaming behavior remains governed by
the bridge fallback/recovery code and should not be treated as full trace replay proof.

Live traces do not store raw prompts, responses, or tool outputs by default. Since exact
bytes are not captured, `tok audit` reports metadata-only live traces as warnings rather
than exact replay proof.

For a stronger local audit, set `TOK_TRACE_CAPTURE_ARTIFACTS=1` together with
`TOK_TRACE=1`. This writes sanitized metadata artifacts next to the trace file and lets
`tok audit` verify the trace block hash and byte size. This still does not capture raw
prompt, response, or tool-result payloads.
