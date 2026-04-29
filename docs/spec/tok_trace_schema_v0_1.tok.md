# Tok Trace Schema v0.1 Draft

Status: draft for 0.1.7

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

Exact content requires `hash`, `size_bytes`, and either `resolver_uri` or an audit state
that explains why the content is unavailable. Delta actions additionally require
`base_hash`, `base_uri`, `delta_hash`, `delta_uri`, and `delta_algorithm`.

The draft fixture auditor resolves `tok-fixture://...` URIs relative to the fixture
directory. For 0.1.7, only `unified_diff` delta replay is audited; other algorithms are
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
  | "accept_delta"
  | "accept_fallback"
  | "reject_malformed"
  | "accept_non_exact_reference",
  reason?: string
}
```

Fallback, degraded, error, rejected, missing, and unresolvable states require `reason`.

## Extensions

```tok
extensions := {
  "<namespace>": object
}
```

Extensions are optional, namespaced, and non-normative in v0.1. They must not override
core action, result, resolver, exactness, or hash semantics.
