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

### `claude: command not found` after `tok install`

- `tok install` updated your shell rc file, but the current shell has not reloaded it.
- Run `source ~/.zshrc` or `source ~/.bashrc`, or open a new shell.
- Then run `tok doctor` again before assuming the bridge is at fault.

### `Bridge not running`

- Start it again with `tok bridge start`.
- Re-run `tok bridge status` or `tok doctor` immediately after startup.
- If it still fails, use `tok bridge start --foreground` so errors stay in the current
  terminal.
- Inspect recent bridge logs with `tok bridge logs 100`.

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
than a single ad-hoc run. Sessions of 50+ turns are more likely to land in the 45-55%
reference band; shorter sessions may show lower or inconsistent savings due to:

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

If you are doing deeper diagnosis, use:

```bash
tok capture-summary ~/.tok/sessions/<capture>.jsonl
tok capture-review ~/.tok/sessions --candidates
tok evidence-gap ~/.tok/sessions --stress-dir tmp/stress_language/<timestamp>
```

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

If this sequence fails, fix install and shell integration before debugging the runtime.

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
claude
tok stats
```

This gives you a clean control path for the same workflow.

## Related Docs

- [`docs/bridge.md`](./bridge.md)
- [`docs/cli-reference.md`](./cli-reference.md)
- [`docs/production-readiness.md`](./production-readiness.md)
