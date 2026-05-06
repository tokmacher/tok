# Tok Trace Schema v0.1 Draft

Status: draft for 0.1.8

This document describes the compact Tok data model used by the draft fixtures. JSON is
the first fixture encoding, not the protocol itself. Future encodings may reuse the same
data model.

## Block Shape

```tok
tok_trace_block_v0_1 := {
  envelope: envelope,
  observation: observation,
  content: content,
  audit: audit,
  extensions?: map
}
```

## Envelope

```tok
envelope := {
  trace_version: "tok-trace/v0.1-draft",
  block_id: string,
  session_id: string,
  turn: integer,
  step: integer,
  direction: "request" | "response",
  payload_digest: "sha256:<64 lowercase hex>" | "draft-uncomputed"
}
```

`payload_digest` is computed over the stable semantic payload: `observation`, `content`,
`audit`, and optional `extensions`, serialized as sorted-key compact JSON.
`draft-uncomputed` is accepted only as a draft placeholder and produces an audit
warning.

## Observation

```tok
observation := {
  class: "file" | "search" | "tool" | "message" | "system" | "response",
  key: string,
  action:
    "pass_through"
  | "store"
  | "reference"
  | "delta"
  | "fallback"
  | "skeleton_reference"
  | "summary_reference",
  result: "ok" | "degraded" | "error" | "rejected"
}
```

## Content

```tok
content := {
  exact: boolean,
  hash?: "sha256:<64 lowercase hex>",
  size_bytes?: integer,
  resolver_uri?: string,
  base_hash?: "sha256:<64 lowercase hex>",
  base_uri?: string,
  delta_hash?: "sha256:<64 lowercase hex>",
  delta_uri?: string,
  delta_algorithm?: "line" | "unified_diff" | "json_patch" | "binary"
}
```

Exact content requires `hash` and `size_bytes`. `accept_exact` additionally requires
`resolver_state: "available_local"` and `resolver_uri`, so the L0-L2 reader can verify
the bytes locally. Delta actions additionally require `base_hash`, `base_uri`,
`delta_hash`, `delta_uri`, and `delta_algorithm`.

The draft fixture auditor resolves `tok-fixture://...` URIs relative to the fixture
directory. For 0.1.8, only `unified_diff` delta replay is audited; other algorithms are
schema-reserved.

## Audit

```tok
audit := {
  resolver_state:
    "available_local"
  | "resolvable_remote"
  | "missing_identifiable"
  | "unresolvable_fallback_required",
  expectation:
    "accept_exact"
  | "accept_reference"
  | "accept_pass_through"
  | "accept_delta"
  | "accept_fallback"
  | "reject_malformed"
  | "accept_non_exact_reference",
  reason?: string
}
```

Fallback, degraded, error, rejected, missing, and unresolvable states require `reason`.
`draft-uncomputed` payload digests are warnings only; they are never proof.

## L0-L2 Invariant Matrix

Matrix version: `tok-trace-invariant-matrix/v0.1-l0-l2`.

| expectation                  | required core meaning                                                                                        | local audit result                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `accept_exact`               | `content.exact: true`, `resolver_state: "available_local"`, `resolver_uri`, `hash`, and `size_bytes`         | pass only when local bytes resolve and match hash and size                               |
| `accept_reference`           | `action: "reference"`, `content.exact: true`, `hash`, and `size_bytes`                                       | local bytes pass; `missing_identifiable` or `resolvable_remote` warns                    |
| `accept_non_exact_reference` | `action: "summary_reference"` or `"skeleton_reference"`, `content.exact: false`                              | local artifact verifies only the summary or skeleton bytes; missing or remote bytes warn |
| `accept_pass_through`        | `action: "pass_through"`, `result: "ok"`, `content.exact: false`                                             | healthy metadata pass-through only; fallback/degraded records fail                       |
| `accept_delta`               | `action: "delta"`, exact final content, local final/base/delta artifacts, `delta_algorithm: "unified_diff"`  | pass only when replayed delta bytes match final bytes                                    |
| `accept_fallback`            | `action: "fallback"`, degraded/error/rejected result, or `unresolvable_fallback_required`; `reason` required | accepted as fallback or degradation, not as proof of exact recovery                      |
| `reject_malformed`           | `result: "rejected"`                                                                                         | malformed structure still fails schema validation                                        |

Cross-field rules:

- `summary_reference` and `skeleton_reference` are always non-exact.
- `result: "ok"` cannot pair with `resolver_state: "unresolvable_fallback_required"`.
- `resolver_state: "available_local"` must include `resolver_uri`, `hash`, and
  `size_bytes`; the reader must resolve the bytes locally and verify that identity.
- Extension namespaces cannot alter any core action, result, resolver, exactness, hash,
  size, or expectation meaning.
- Any future core enum or required-field semantic change must update this matrix or bump
  `trace_version`.

## Extensions

```tok
extensions := {
  "<namespace>": object
}
```

Extensions are optional, namespaced, and non-normative in v0.1. They must not override
core action, result, resolver, exactness, or hash semantics. Extension namespaces must
not be `envelope`, `observation`, `content`, or `audit`.
