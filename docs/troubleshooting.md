# Troubleshooting

Start here whenever the install or bridge workflow does not look healthy.

The fastest first check is:

```bash
tok doctor
```

## Quick Checks

### `tok: command not found`

- Make sure the package is installed in the active Python environment.
- Re-activate the environment, then run `pip install tok-protocol` or `pip install .`.
- Verify with `tok --help`.

### `claude: command not found` after `tok install --wrap-claude`

- `tok install --wrap-claude` updated your shell rc file, but the current shell has not
  reloaded it.
- Run `source ~/.zshrc` or `source ~/.bashrc`, or open a new shell.
- Then run `tok doctor` again before assuming the bridge is at fault.

### `Bridge not running`

- Start it again with `tok bridge start`.
- Re-run `tok bridge status` or `tok doctor` immediately after startup.
- If it still fails, use `tok bridge start --foreground` so errors stay in the current
  terminal.
- Inspect recent bridge logs with `tok bridge logs 100`.

### `tok bridge stop` refused with a self-bridged warning

- Tok now protects against in-band self-cutoff.
- Exit Claude first, then run `tok bridge stop` from your shell.
- If you intentionally need in-band shutdown, run `tok bridge stop --force`.

## Runtime Diagnosis

### `Degraded to baseline: yes`

Tok degraded the current session for safety. That means requests are being passed
through without compression until the session is healthy again.

Check:

- `tok doctor`
- `tok bridge status`
- `tok bridge logs 100`

Search the logs for:

- `tok_fallback_activated`
- `processing_error`
- `tok_fail_open_retry`

### `Session quality: watch`

Tok is still active, but the session is showing early signs of friction such as
fallback, reacquisition, or response drift.

Recommended next steps:

- keep Tok on for now
- inspect `Degradation reason` in `tok doctor` or `tok bridge status`
- compare `tok stats --last-session` after the session ends

### Savings are not obvious yet

Very short sessions (under 10-15 turns) may not show clear savings. Tok's compression
benefits accumulate over longer conversations where repeated file reads, tool outputs,
and context build up.

For release verification, use the maintained benchmark and claims matrix flow rather
than a single ad-hoc run. Savings vary significantly by session: long sessions with
heavy tool usage (file reads, search, repeated operations) tend to show the strongest
results. Shorter sessions may show lower savings due to:

- Less opportunity for semantic deduplication
- Fewer repeated tool calls to cache
- Overhead from initial memory setup

Check:

```bash
tok stats --last-session
tok stats --recent 5
tok bridge stop
tok stats
```

When repeated file reads or searches are high, `tok stats` may show `Net Tokens Saved`.
That line subtracts recorded reacquisition token overhead from gross saved tokens. Cost
savings remain an estimate against the baseline model pricing and should be read
alongside the net-token line when `reacq` is high.

### Streaming feels delayed

Tok's normal 0.2.x bridge path buffers the upstream streaming response before it
re-emits server-sent events to Claude Code. That keeps response rewriting deterministic,
but long responses can have a higher first-token delay than direct Claude Code. Some
correctness-first paths, including smoothness/lossless handling and extended-thinking
requests, may send the upstream request as non-streaming.

If you are doing deeper diagnosis with the standard supported commands, use:

```bash
tok doctor
tok stats
tok bridge status
tok audit <trace-file>
```

> **Note:** `tok capture-summary`, `tok capture-review`, and `tok evidence-gap` are
> hidden maintainer-only experimental commands. They are not part of the supported
> diagnostic surface and should not be used in normal agent or user workflows.

For reproducible release claims, refer to:

- [`docs/claims_matrix.md`](./claims_matrix.md)
- [`docs/live_smoke_matrix.md`](./live_smoke_matrix.md)

## Clean-Room Install Check

When validating setup from scratch:

```bash
python -m venv .venv
source .venv/bin/activate
pip install tok-protocol
tok --help
tok install
tok bridge start --help
```

If this sequence fails, fix install and bridge startup before debugging the runtime.

## Repo Checkout Smoke

When validating a release candidate from a repo checkout, run:

```bash
python scripts/run_release_smoke.py
```

This bounded sweep checks the public CLI help surfaces, public import shims, one focused
bridge/runtime/compression path, and a packaging build smoke.

## When To Use Baseline

To compare behavior with compression disabled:

```bash
TOK_MODE=baseline tok bridge start
ANTHROPIC_BASE_URL=http://localhost:9090 claude
tok stats
```

This gives you a clean control path for the same workflow.

## Related Docs

- [`docs/bridge.md`](./bridge.md)
- [`docs/cli-reference.md`](./cli-reference.md)
- [`docs/production-readiness.md`](./production-readiness.md)
