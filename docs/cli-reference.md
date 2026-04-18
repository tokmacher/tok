# CLI Reference

This page summarizes the supported Tok CLI surface for the first public release. The
default `tok --help` output intentionally highlights only the bridge-first onboarding
path plus the `tok savings` compatibility alias.

If you are new to Tok, start with [`README.md`](../README.md) or the full workflow in
[`docs/bridge.md`](./bridge.md).

## Core Workflow

```bash
tok install
tok init
tok bridge start
ANTHROPIC_BASE_URL=http://localhost:9090 claude
tok bridge status
tok doctor [--report]
tok bridge stop
tok stats
```

`tok install` is a setup/migration helper. To opt into legacy auto-routing, use
`tok install --wrap-claude`.

`tok init` creates a project-local `.tok/` workspace and optional `.env` / `.gitignore`
entries. Run it once per project before starting the bridge.

## Bridge Commands

```bash
tok bridge start [--port 9090] [--keep-turns 2] [--debug] [--foreground]
tok bridge start [--api-base URL] [--capture] [--fail-open/--no-fail-open]
tok bridge status
tok bridge logs [40]
tok bridge stop [--force]
```

Use:

- `start` to launch the bridge
- `start --api-base URL` to point the bridge at a non-Anthropic upstream (e.g.
  `https://openrouter.ai/api/v1`)
- `start --keep-turns N` to control how many recent turns are kept verbatim (default 2)
- `status` to confirm the bridge is live and Tok is helping
- `logs` to inspect the bridge log file
- `stop` to end the session and print a compact summary
- `stop --force` to intentionally stop from an active bridged Claude turn

## Health And Savings

```bash
tok doctor
tok stats [--session | --total | --last-session | --recent N | --since DATE]
tok stats [--breakdown] [--trends] [--window N]
tok savings [same options as tok stats]
```

Use:

- `tok doctor` as the first troubleshooting command after install
- `tok stats` for current, last-session, recent, and lifetime savings views
- `tok savings` only as a compatibility alias for `tok stats`

Advanced maintainer utilities remain available, but they are intentionally hidden from
the default help surface in `0.1.0` so new users land on one clear workflow.

## Capture And Evidence Review

```bash
tok capture-summary PATH.jsonl
tok capture-review DIR [--verdict clean|watch|investigate] [--reason TEXT]
tok capture-review DIR [--candidates] [--coverage] [--stress-dir DIR]
tok evidence-gap DIR [--stress-dir DIR]
```

These commands are for diagnosing repeated degradation classes and ranking replay
coverage opportunities. They are useful for advanced operators and maintainers, not for
the default first-run flow.

## Release Gate

```bash
tok gate-check tests/fixtures/replay \
  --fixtures fixtures.json \
  --config gate-config.json \
  --stability-dir tests/fixtures/stability \
  --frontier-report tmp/compression_frontier/compression_frontier_report.json \
  --required-benchmarks coding-loop-5,research-loop-5 \
  --export results.json
```

Use this for maintainer release checks and CI parity. See
[`docs/release-checklist.md`](./release-checklist.md) and
[`docs/CICD_INTEGRATION.md`](./CICD_INTEGRATION.md).

## CI/CD Integration

For details on running Tok's test and gate tooling in CI, see
[`docs/CICD_INTEGRATION.md`](./CICD_INTEGRATION.md).

## Format Tools

```bash
tok convert INPUT [--to tok|json|md] [--file]
tok parse INPUT [--file]
```

These commands are secondary utilities. The bridge-first workflow is the supported
product path.

## Related Docs

- [`docs/bridge.md`](./bridge.md)
- [`docs/troubleshooting.md`](./troubleshooting.md)
- [`docs/production-readiness.md`](./production-readiness.md)
