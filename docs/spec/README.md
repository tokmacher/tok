# Tok Specs

This directory contains draft specification work for Tok 0.1.8 and later.

The current draft is `tok-trace/v0.1-draft`, an audit format for Tok bridge sessions. It
is bridge-first protocol groundwork: fixture and live-audit evidence only, not a
universal agent protocol:

- opt-in metadata-only runtime trace emission via `TOK_TRACE=1`
- optional sanitized metadata artifacts via `TOK_TRACE_CAPTURE_ARTIFACTS=1`
- visible `tok audit` command for draft trace validation
- no change to the supported Claude Code bridge path
- no promotion of experimental protocol/parser, macro, pointer, or SDK surfaces

Start with:

- `tok_trace_format_v0_1.md` for the format narrative and scope
- `tok_trace_schema_v0_1.tok.md` for the compact data model
- `tok_trace_roadmap_v0_1.md` for compatibility and adoption gates
- `tok_protocol_layers_v0_1.md` for the layered protocol-family roadmap
- `tok_trace_conformance_v0_1.md` for L0-L2 reader requirements
- `fixtures/trace_fixtures.json` for draft fixture blocks
- `fixtures/expected_audit_results.json` for expected audit outcomes
- `fixtures/adversarial_packs.json` for named adversarial packs

The first implementation milestone is fixture validation plus metadata-only live trace
validation. Sanitized metadata artifact capture can make live trace audits pass locally;
raw prompt/response/tool artifact capture remains deferred.

Tok 0.1.8 targets Trace L0-L2 audit conformance only. Resolver networking, capability
handshakes, session exchange, signed provenance, and agent-to-agent behavior are future
layers.

The current bridge grammar is adapter-local. `>>>` working-memory lines, Tok markers,
Markdown recovery, and Claude Code bridge shaping are not Tok Session core semantics.
