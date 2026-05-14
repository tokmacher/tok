# Agent Quickstart

This page is for autonomous agents inspecting Tok from a fresh clone.

Tok is a bridge-first, protocol-aimed system for compact, auditable, model-facing
context. The supported 0.2.x product surface is the local Claude Code bridge,
diagnostics, trace audit, and local resolver beta.

Do not treat Tok as a hosted service, general agent framework, repo indexer, universal
protocol implementation, or stable Python SDK.

## Start Here

Read these files first:

```bash
sed -n '1,220p' AGENTS.md
sed -n '1,260p' README.md
sed -n '1,220p' docs/repository_map.md
sed -n '1,220p' docs/cli-reference.md
sed -n '1,220p' docs/bridge.md
sed -n '1,220p' docs/troubleshooting.md
```

Use `docs/agent-contract.json` when a machine-readable support contract is more useful
than prose.

## Install From A Clone

Use the locked development environment:

```bash
uv sync --frozen --extra dev
```

Then inspect the installed CLI:

```bash
uv run tok --version
uv run tok --help
uv run tok claude --help
uv run tok bridge status --help
uv run tok doctor --help
uv run tok stats --help
uv run tok audit --help
uv run tok resolver --help
```

The package name is `tok-protocol`. The CLI command is `tok`. The import package is
`tok`.

## Fastest Safe Smoke Test

Run the agent smoke script:

```bash
uv run python scripts/run_agent_smoke.py
```

This checks the CLI entry point, first-help commands, and the agent docs contract. It
does not prove live bridge behavior or measured savings.

For a focused release-surface check:

```bash
uv run pytest tests/unit/test_agent_docs_contract.py tests/unit/test_release_surface.py -q
```

For broader source validation:

```bash
uv run pytest tests/unit tests/integration -v
uv run ruff check src/tok tests
uv run mypy src/tok
```

Use the full release smoke only when the environment is ready for a longer check:

```bash
uv run python scripts/run_release_smoke.py
```

Do not report any command as passing unless you ran it in this checkout.

## Live Bridge Check

Only run this when Claude Code is installed and available as `claude`:

```bash
uv run tok claude
uv run tok bridge status
uv run tok doctor
uv run tok stats
uv run tok bridge stop
```

Report whether Claude Code was available, whether the bridge was running, whether Tok
degraded to baseline, and whether savings were measured.

Do not claim "bridge works" without bridge health output. Do not claim "savings
achieved" without `tok stats` output from a real session.

Tok `0.2.x` streaming is not fully timing-invisible. The normal bridge path buffers the
upstream stream before re-emitting it to Claude Code, and correctness-first modes may
use a non-streaming upstream request. Report this as first-token latency trade-off, not
as proof that the bridge failed.

## Supported Surface

Supported 0.2.x path:

- local bridge
- Claude Code
- `tok claude`
- diagnostics through `tok bridge status`, `tok doctor`, `tok stats`, and `tok audit`
- local resolver beta commands through `tok resolver`

Experimental or unsupported:

- hosted service behavior
- general agent framework behavior
- repo indexing
- universal protocol compliance
- stable Python SDK compatibility
- Tok Capability, Tok Session, remote resolver, and agent-to-agent exchange

## Safe Editing

Keep changes small and backed by tests.

Do not edit generated artifacts, caches, traces, local state, build output, or planning
notes unless explicitly asked. Do not use `CLAUDE.md`, `docs/plans/`,
`docs/maintainers/`, or `ops/` as public product contracts.

Exactness rules:

- Exact source material must stay exact or be reacquired before use.
- Summaries, skeletons, and references are not exact source material.
- Compression must not hide fallback, degradation, or uncertainty.
- When fidelity is uncertain, Tok should fail open to baseline behavior or report a
  clear failure.

Docs rules:

- Update README and CLI docs when the first-run workflow changes.
- Keep public docs free of hidden prompt instructions.
- Use exact commands and conservative claims.

## Useful Reports

Use `docs/agent-report-template.md` for verification reports.

Include:

- OS, Python version, Tok version, and install mode
- exact commands run
- pass/fail status for each command
- failing traceback or log excerpt
- whether Claude Code was available
- whether the bridge was running
- whether the session degraded to baseline
- whether savings were measured, expected, or not applicable

## Extension Work

For a new runtime such as OpenCode, Codex, or another agent bridge, start by finding the
runtime transport boundary. Do not fork the Claude Code path blindly.

A conservative extension plan:

1. Read `docs/bridge-standard.md`, `docs/spec/README.md`, and
   `src/tok/release_surface.py`.
1. Add fixture tests for the runtime's request and response shape.
1. Label evidence as exact, summary, skeleton, or reference before adding compression.
1. Preserve trace/audit metadata.
1. Keep live behavior behind an explicit command or configuration.
1. Fail open or fail safe explicitly.
1. Do not claim support until tests and docs name the supported surface.
