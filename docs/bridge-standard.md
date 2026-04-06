# Tok Bridge Standard

This document is the canonical contract for the live Tok bridge. If another document
disagrees with this one, this document wins.

It defines the current bridge contract, not the future universal communication
contract. That broader ambition remains later-stage work.

## Scope

Tok is standardized around the bridge-first runtime in `src/tok/gateway/__init__.py`.

- Primary runtime: the Claude Code bridge
- Secondary runtime: `adapters/orchestrator.py` and related archived/custom-agent paths
- Acceptance target: invisible operation inside the bridge, not protocol purity in isolation
- Current release target: a trustworthy Claude-first public release, not cross-surface standardization

The release-surface manifest in `src/tok/release_surface.py` defines which exports
and commands are supported versus experimental for the 0.1 release story.

## Wire Memory Contract

The bridge working-memory line is a sparse `>>>` record. Canonical fields, in canonical order:

`turns`, `goal`, `files`, `cmds`, `tests`, `errs`, `constraints`, `next`

Rules:

- Fields are emitted in that order only.
- Fields are optional except `turns` when memory exists.
- `turns`, `goal`, and `next` are single-value fields.
- `files`, `cmds`, `tests`, `errs`, and `constraints` are comma-separated bounded lists.
- Additional `key:value` pairs after the canonical fields are non-canonical facts. They may be stored, but they are not part of the standard bridge contract.

## Projection Rules

The bridge projects structured memory back onto the wire from hot memory first.

Rules:

- Structured bridge memory is authoritative when available.
- Raw `memory.tok` is a compatibility fallback only.
- Projection remains sparse and bounded.
- The projected wire state must preserve canonical field order.

## Tok-Native Success

A response counts as Tok-native success only when all of the following hold:

- the response uses Tok markers
- the bridge can project it into at least one user-visible Anthropic content block
- the result contains readable assistant text and/or valid tool-use blocks without needing markdown fallback

This is the preferred success path.

## Fail-Open Compatibility

The bridge may still return a usable response when Tok-native success does not happen.

Compatible degraded behavior:

- non-Tok response with readable text
- malformed Tok response where readable text can still be recovered
- cold start from raw `memory.tok` when structured memory is unavailable

Rules:

- compatibility mode must not block the request
- compatibility mode counts as degraded behavior, not full Tok success
- degraded behavior must be surfaced in telemetry and `tok doctor`

## Inversion Requirements

The bridge must keep inversion semantics stable:

- old history compresses into a bounded working-memory line
- cold starts prefer structured projected memory
- malformed or non-inverted output must not be mistaken for Tok-native success
- file readability must not be sacrificed for local compression ratio

## Conformance Targets

Conformance tests should lock down:

- canonical field order
- structured-memory cold-start precedence
- Tok-native success detection
- fail-open compatibility detection
- malformed/non-inverted response handling
