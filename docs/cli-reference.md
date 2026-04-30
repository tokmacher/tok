# CLI Reference

This page summarizes the supported Tok CLI surface for the first public release. The
default `tok --help` output intentionally highlights only the bridge-first onboarding
path plus the `tok stats` command.

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
tok audit --latest
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
tok audit [TRACE_FILE | --latest] [--json]
```

Use:

- `tok doctor` as the first troubleshooting command after install
- `tok stats` for current, last-session, recent, and lifetime savings views
- `tok audit` to validate Tok Trace v0.1 draft fixture files or live bridge sidecars

## Trace Audit

```bash
TOK_TRACE=1 tok bridge start
TOK_TRACE=1 TOK_TRACE_CAPTURE_ARTIFACTS=1 tok bridge start
tok audit --latest
tok audit ~/.tok/traces/20260430_061100_live_example.jsonl
tok audit --latest --json
```

`TOK_TRACE=1` enables opt-in sidecar traces under `~/.tok/traces/`.
`TOK_TRACE_CAPTURE_ARTIFACTS=1` writes sanitized metadata artifacts next to the trace so
audit can verify local hashes and byte sizes. This does not capture raw prompts,
responses, or tool outputs.

`tok audit` is a draft trace-audit feature for inspecting what Tok did. It is not a
universal protocol compliance certificate.

Audit output uses three statuses:

- `PASS`: the trace block shape, digest, and available local artifact checks passed
- `WARN`: the trace is structurally valid, but exact bytes are missing or unavailable
- `FAIL`: the trace is malformed, has a digest mismatch, or failed a local artifact
  check

Metadata-only live traces commonly produce `WARN` because Tok can identify what happened
without storing original prompt, response, or tool-result bytes. Artifact-backed
metadata mode can produce `PASS`, but it still verifies sanitized trace metadata rather
than raw session content.

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
