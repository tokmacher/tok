# Tok Diagnostics (0.1.5)

This document explains the key health and recovery signals you may see in:

- `tok bridge status`
- `tok doctor --report`
- `tok bridge logs`

The goal is to help you distinguish normal, self-contained recovery (expected under
stress) from a real degradation that should be reported as a bug.

For comparisons with native Claude Code context management and adjacent context tools,
see [`docs/claude-compaction-comparison.md`](./claude-compaction-comparison.md) and
[`docs/positioning-context-tools.md`](./positioning-context-tools.md).

## What Is "Supported" For 0.1.x

The supported product path for `0.1.x` is Claude Code routed through the local Tok
bridge:

```bash
tok bridge start
ANTHROPIC_BASE_URL=http://localhost:9090 claude
```

All guidance below assumes this workflow.

## Quick Triage: GO vs "Something Is Wrong"

These are the top-level signals that matter most.

### Green

- `Fallbacks: 0`
- `Degraded to baseline: no`
- `Tok active: yes`

If these stay true, Tok is operating as an invisible bridge and applying compression.

### Yellow (Expected Under Stress)

- `Session quality: watch`
- `Degradation reason: heavy tool-mode recovery`
- `Session signals` includes `reacq=...` (intentional repeated reads/searches will
  increase this)

These are common during heavy tool usage, repeated file reads, repeated searches, and
parallel tool bursts. They do not imply Tok is bypassing itself.

### Red (Bug Worth Reporting)

- `Fallbacks > 0` during the supported workflow
- `Degraded to baseline: yes` during the supported workflow (outside very short sessions
  where Tok may choose baseline)
- Reproducible user-visible tool failures: malformed tool outputs, missing tool results,
  broken ordering that Claude Code can't handle

If you hit any of these, capture:

- `tok bridge status`
- `tok doctor --report`
- `tok bridge logs 600` (filtered to the relevant signal bundle)

## Interpreting Session Signals

### `compat-fallback`

You may see `compat-fallback=N` in the `Session signals` line of `tok bridge status`.

Meaning:

- This is an internal compatibility response mode, not an upstream bypass.
- It typically appears during turns with heavy parallel tool activity or complicated
  tool-result streams.

What to check:

- `Fallbacks` should remain `0`.
- `Degraded to baseline` should remain `no`.

### `reacq`

`reacq` is a count derived from repeated search and file-read targets. A deliberate
stress test that repeats reads/searches will increase it quickly.

High `reacq` is usually a sign that Tok has useful dedup/delta opportunities, not a
problem by itself.

### Low Or Zero Savings

`tok stats`, `tok doctor`, and `tok bridge status` may show a `Savings note` when Tok is
active but savings are low or absent.

Common explanations:

- the session is too short for bridge compression to amortize overhead;
- `TOK_MODE=baseline` is intentionally disabling compression;
- provider caching or a low-repetition task limits incremental savings;
- MCP servers, file reads, searches, or tools are producing fresh large payloads rather
  than repeated payloads;
- `safe-block` means Tok chose exactness over savings;
- fallback or degraded baseline means Tok protected fidelity and needs log inspection.

Low savings is not automatically a Tok failure. Treat it as a prompt to inspect
`Session signals`, fallback counts, exactness labels, and `tok audit --latest`.

### Evidence-safety labels

You may see compact labels such as `exact=N`, `nonexact=N`, `reacq-safe=X/Y`, or
`safe-block=N` in `tok bridge status`, `tok doctor`, or `tok stats` session panels.

Meaning:

- `exact` means Tok observed exact evidence before treating that evidence identity as
  compressible.
- `nonexact` means Tok emitted a summary, skeleton, or reference for already observed
  evidence.
- `reacq-safe` means an exact reacquisition requirement existed and was satisfied before
  edit-critical use.
- `safe-block` means Tok preserved fidelity by blocking compression for that evidence.

These labels are evidence that Tok preserved the bridge contract. They are not failures
unless they appear alongside actual degradation such as `Fallbacks > 0`,
`Degraded to baseline: yes`, or user-visible tool breakage.

## Interpreting "Response signals" Bundles in Logs

`tok bridge logs` may include a line like:

`Response signals: {...}`

This is a per-turn bundle of internal recovery and compression signals. A few are
especially important in stress tests.

### Tool result repair signals

- `tok_bridge_tool_result_order_repaired`
- `tok_bridge_tool_result_pairing_repaired`
- `tok_bridge_tool_history_pairing_repaired`

Meaning:

- Tok detected a protocol-level mismatch in tool result ordering/pairing and repaired it
  before returning the result to Claude Code.

These are expected to be non-zero in aggressive parallel-tool stress tests. They are
usually good news: the repair machinery is actively protecting the session.

### Answer-ready repair signals

- `answer_ready_turn`
- `answer_ready_repair_active`
- `answer_ready_reacquisition_attempt`
- `answer_phase_fallback_failed_no_anchor`
- `answer_ready_failed_to_answer`
- `answer_ready_repair_failed`

Meaning (plain English):

- Under heavy tool pressure, Tok may try to stabilize the response by forcing a "full
  state resend" and reacquiring the answer anchor ("answer-ready" mode).
- `answer_ready_repair_active=1` means Tok attempted this repair path on that turn.
- `answer_ready_repair_failed=1` means that specific repair attempt could not establish
  a clean answer-ready anchor for that turn.

How to interpret it for `0.1.x`:

- If `Fallbacks` stays `0` and `Degraded to baseline` stays `no`, this is a
  self-contained recovery path. It's acceptable for `0.1.x` under stress, but is useful
  evidence for hardening future versions.
- If you see repeated `answer_ready_repair_failed` on normal usage (not a synthetic
  parallel-tool stress test), file an issue with the filtered log bundle.

## Logging Notes

### "Fail-open check" lines

Tok includes a "fail-open" safety mechanism for handling provider errors in a controlled
way. You may see health assertions about it in logs.

Treat these as diagnostics unless they are accompanied by:

- `Fallbacks > 0`, or
- an explicit "retrying without Tok" / "degraded to baseline" message.
