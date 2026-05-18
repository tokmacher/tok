# Tok Diagnostics (0.2.x)

This document explains the key health and recovery signals you may see in:

- `tok bridge status`
- `tok bridge status --json`
- `tok doctor --report`
- `tok doctor --json`
- `tok bridge logs`

The goal is to help you distinguish normal, self-contained recovery (expected under
stress) from a real degradation that should be reported as a bug.

## What Is "Supported" For 0.2.x

The supported product path for `0.2.x` is Claude Code routed through the local Tok
bridge, with local trace audit and local resolver beta commands available for inspection
and recovery:

```bash
tok claude
```

`tok claude` starts the bridge if needed and routes only that Claude Code process
through Tok. All guidance below assumes this workflow.

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

### Bridge Log Size

`tok bridge start` bounds `~/.tok/bridge.log` before appending to it. By default Tok
trims the log when it exceeds 50 MiB and keeps the newest 10 MiB, with a `log_trimmed`
marker at the top of the retained file.

For local diagnosis you can override the thresholds:

```bash
TOK_BRIDGE_LOG_MAX_BYTES=104857600 TOK_BRIDGE_LOG_KEEP_BYTES=20971520 tok bridge start
```

Set either value to `0` to disable automatic trimming for that start.

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
problem by itself. It is still overhead: `tok stats` shows `Net Tokens Saved` when
recorded reacquisition token cost exists, so compare that line with gross `Tokens Saved`
for high-reacquisition sessions.

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

### JSON Diagnostics Shape

`tok bridge status --json` and `tok doctor --json` use the shared `tok-cli-result/v0.1`
envelope. Important session fields include `session_quality`, `degradation_reason`,
`fallback_count`, `baseline_only`, `tokens_saved`, `savings_pct`, and `goal`.

The `goal` field is a compact orientation hint, not an exact transcript. It is capped at
40 characters by the live bridge health endpoint, so it may end mid-sentence. Tok strips
internal system-reminder tags before exposing it.

`tok audit --json` is different: it returns a bare list of audit results because a trace
file can contain many independently passing, warning, or failing records. See
[`docs/cli-reference.md`](./cli-reference.md) for the item schema.

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

How to interpret it for `0.2.x`:

- If `Fallbacks` stays `0` and `Degraded to baseline` stays `no`, this is a
  self-contained recovery path. It's acceptable for `0.2.x` under stress, but is useful
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
