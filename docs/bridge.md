# Bridge Tutorial

This is the full bridge-first walkthrough for Tok.

If you are new to Tok, start with the quickstart in [`README.md`](../README.md), then
use this page when you want the complete operating flow.

Tok's first open-source release is intentionally narrow:

- install the Python package
- add the `claude()` shell wrapper
- start the bridge
- use Claude normally
- diagnose with `status`, `doctor`, `stats`, and logs

The bridge is the supported product path. Broader platform and SDK work come later.
The default CLI help intentionally centers that bridge-first path for `0.1.0`.

## What The Bridge Does

The Tok bridge sits between Claude Code and the upstream model API:

```mermaid
flowchart LR
    C["Claude Code"] --> B["Tok Bridge"]
    B --> R["Universal Tok Runtime"]
    R --> U["Anthropic / OpenRouter"]
```

The bridge is responsible for transport and process lifecycle. The shared runtime owns:

- request shaping and history compression
- response classification and translation
- memory projection and update
- telemetry and invisible-pressure signals

## Prerequisites

- Python `3.10+`
- macOS or Linux
- Claude Code installed and available as `claude`
- provider/API configuration that already works with Claude Code

## Quickstart

```bash
pip install tok-protocol
tok install
source ~/.zshrc  # or source ~/.bashrc
tok bridge start
claude
tok bridge status
tok doctor
tok bridge stop
tok stats
```

`tok install` adds a `claude()` shell wrapper. It does **not** replace the real `tok` CLI.
If `claude` is still missing after install, reload your shell before digging into
bridge logs or runtime health.

## What Success Looks Like

In a healthy session:

- `tok bridge status` shows the bridge running and Tok active
- `tok doctor` ends with `Recommendation: keep Tok on`
- `tok stats` shows saved dollars, saved percent, and `With Tok vs without Tok`

Typical output shape:

```bash
tok bridge status
# Bridge running on :9090 (PID 12345)
# Bridge Status
# Saved $0.0123 • 48.1% saved
# Verdict                Tok active and helping
# Session quality        clean
# Tok active             yes
# Degraded to baseline   no
# Fallbacks              0

tok doctor
# Current Session
# Saved $0.0123 • 48.1% saved
# Verdict                Tok active and helping
# Tok verdict: compression is active and saving tokens on the current session.
# Recommendation: keep Tok on

tok bridge stop
# Last Session
# Saved $0.0123 • 48.1% saved

tok stats
# Current Session
# Saved $0.0123 • 48.1% saved
# With Tok vs without Tok  45,000 / 86,700 tokens
```

Representative `tok stats --last-session` capture:

```text
╭──────────────── Last Completed Session ─────────────────╮
│ Saved $0.0001 • 30.4% saved                             │
│ Solid savings • 28 tokens avoided                       │
│ Date                               2026-03-20T11:24:26Z │
│ Turns                                                 1 │
│ With Tok vs without Tok                317 / 345 tokens │
│ Cost                                  $0.0003 / $0.0004 │
╰─────────────────────────────────────────────────────────╯
```

The key fields to watch are:

- `Verdict`
- `Mode`
- `Session quality`
- `Degraded to baseline`
- `Fallbacks`
- `Saved $` / `% saved`

## When To Use Baseline

If you want to compare Tok against no compression:

```bash
TOK_MODE=baseline tok bridge start
claude
tok stats
```

That gives you a clean control path for the same workflow.

## Bridge Commands

### Start

```bash
tok bridge start
tok bridge start --foreground
tok bridge start --debug
tok bridge start --capture
tok bridge start --port 8080
tok bridge start --no-fail-open
```

Use `--foreground` for the fastest debugging loop when setup is not behaving the way
you expect.

### Status

`tok bridge status` answers:

- is the bridge running?
- which mode is it in?
- is the session `clean`, `watch`, or `degraded`?
- has the session degraded to baseline?
- are savings visible yet?

### Doctor

`tok doctor` is the fastest “is Tok helping right now?” command.
It now ends with a concrete recommendation:

- `Recommendation: keep Tok on`
- `Recommendation: keep Tok on, but watch this session`
- `Recommendation: investigate degradation before trusting this session`

### Stop

`tok bridge stop` prints a compact session summary, which makes it the easiest end-of-session checkpoint.

### Logs

```bash
tok bridge logs
tok bridge logs 100
```

Use logs when the bridge process exists but `status` or `doctor` suggest fallback or a
non-responsive session.

## Runtime Defaults

- default compressed path: `tok-tool-compatible`
- conservative fallback: `baseline`
- non-default: `tok-minimal`
- non-default: `tok-native`

To force baseline:

```bash
TOK_MODE=baseline tok bridge start
```

## Troubleshooting Basics

### `Degraded to baseline: yes`

The current session degraded to baseline for safety.

Check:

- `tok doctor`
- `tok stats`
- bridge logs for `tok_fallback_activated`

### Fallback count is rising

Tok is protecting the session by serving some requests without compression.

Search the bridge logs for:

- `tok_fallback_activated`
- `processing_error`
- `tok_fail_open_retry`

### Savings are not obvious

Run:

```bash
tok bridge stop
tok stats --last-session
tok stats --recent 5
tok capture-summary ~/.tok/sessions/<capture>.jsonl
tok capture-review ~/.tok/sessions --candidates
tok evidence-gap ~/.tok/sessions --stress-dir tmp/stress_language/<timestamp>
```

This usually gives a cleaner picture than lifetime totals alone.

### `Session quality: watch`

Tok is still saving tokens, but the session shows some friction such as fallback,
reacquisition, or response-contract drift.

Keep Tok on, but inspect the degradation reason before deciding the bridge is at fault.

## Next Docs

- [`docs/cli-reference.md`](./cli-reference.md) for the command surface
- [`docs/troubleshooting.md`](./troubleshooting.md) for fallback and degraded-session diagnosis
- [`docs/production-readiness.md`](./production-readiness.md) for advanced runtime defaults and release posture
- [`docs/architecture.md`](./architecture.md) for deep runtime details
