# Tok Specs

This directory contains draft specification work for Tok 0.1.7 and later.

The current draft is `tok-trace/v0.1-draft`, an audit format for Tok bridge sessions. It
is fixture and documentation groundwork only:

- no runtime trace emission
- no supported `tok audit` command yet
- no change to the supported Claude Code bridge path
- no promotion of experimental protocol/parser, macro, pointer, or SDK surfaces

Start with:

- `tok_trace_format_v0_1.md` for the format narrative and scope
- `tok_trace_schema_v0_1.tok.md` for the compact data model
- `tok_trace_roadmap_v0_1.md` for compatibility and adoption gates
- `fixtures/tok_trace_v0_1_fixtures.json` for draft fixture blocks
- `fixtures/tok_trace_v0_1_expected.json` for expected audit outcomes

The first implementation milestone is fixture validation. Runtime integration comes
after the draft survives schema and audit-shape review.
