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
the default help surface in `0.1.x` so new users land on one clear workflow. Hidden
commands such as capture review, release gating, conversion helpers, and developer tools
are maintainer-only for this release and may change without compatibility guarantees.

For maintainer workflows, see [`docs/release-checklist.md`](./release-checklist.md) and
[`docs/CICD_INTEGRATION.md`](./CICD_INTEGRATION.md) instead of treating those commands
as public onboarding material.

## Related Docs

- [`docs/bridge.md`](./bridge.md)
- [`docs/troubleshooting.md`](./troubleshooting.md)
- [`docs/production-readiness.md`](./production-readiness.md)
