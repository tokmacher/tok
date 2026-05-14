# AGENTS.md

This repository is agent-operable. Follow this file before making claims about Tok. For
a cold-clone workflow, use `docs/agent_quickstart.md`.

## Project identity

Tok is a bridge-first, protocol-aimed system for compact, auditable, model-facing
context. The supported 0.2.x product is a local Claude Code bridge that performs
deterministic context compression and fails open to baseline behavior when fidelity is
at risk.

Supported 0.2.x path:

- local bridge
- Claude Code
- `tok claude`
- diagnostics through `tok bridge status`, `tok doctor`, `tok stats`, and `tok audit`
- local resolver beta commands through `tok resolver`

Do not describe Tok 0.2.x as:

- a hosted service
- a general agent framework
- a repo indexer
- a universal protocol implementation
- a stable Python SDK

Tok Resolver is implemented as a local-only beta. Tok Capability, Tok Session, and
agent-to-agent exchange are not yet implemented. Do not claim they are. `tok audit`
validates trace structure, not general protocol compliance. Local resolver beta does not
imply a stable protocol.

## Golden path

From a cold clone:

```bash
uv sync --frozen --extra dev
uv run tok --version
uv run tok --help
uv run tok claude --help
uv run tok bridge status --help
uv run tok doctor --help
uv run tok stats --help
uv run tok audit --help
uv run tok resolver --help
uv run python scripts/run_agent_smoke.py
```

For source validation:

```bash
uv run pre-commit run --all-files
uv run ruff check src/tok tests
uv run mypy src/tok
uv run pytest tests/unit tests/integration -v
uv build
```

## Bridge verification

Use this when Claude Code is available:

```bash
tok claude
tok bridge status
tok doctor
tok stats
tok bridge stop
```

A healthy session should show:

- bridge running
- Tok active
- no degraded baseline state
- Fallbacks reported (expected: 0)
- savings visible if the session is long/repetitive enough

Do not claim compression success from a short non-repetitive session.

## First files to inspect

Read these first:

- `README.md`: public product story and human quickstart
- `docs/agent_quickstart.md`: exact cold-clone agent workflow
- `docs/repository_map.md`: directory map and public/internal boundaries
- `docs/cli-reference.md`: supported CLI surface
- `docs/bridge.md`: supported Claude Code bridge workflow
- `docs/troubleshooting.md`: degraded sessions, logs, and fallback handling
- `docs/diagnostics.md`: health signals
- `docs/agent-contract.json`: machine-readable agent contract
- `src/tok/release_surface.py`: tested release surface lists
- `tests/unit/test_release_surface.py`: release-surface regression tests

Treat these as non-default context:

- `CLAUDE.md`: local maintainer/Claude Code notes, not public product contract
- `docs/plans/`: planning artifacts, not current support promises
- `docs/maintainers/`: maintainer notes, not user onboarding
- `ops/`: internal tracking ledgers, not public product docs
- `examples/`: experimental examples outside the default bridge-first path

## Reporting rules

When reporting results, include:

- exact commands run
- pass/fail status
- failing traceback or log excerpt
- whether Claude Code was available
- whether the bridge was actually running
- whether the session degraded to baseline
- whether savings were measured or merely expected

Never claim:

- "protocol compliance" from `tok audit`
- "production ready" from unit tests only
- "savings achieved" without `tok stats`
- "bridge works" without bridge health output

Useful bug reports should include:

- OS, Python version, install mode, and `uv run tok --version`
- exact command sequence
- exact failing traceback or log excerpt
- `uv run tok bridge status --json` when bridge state matters
- `uv run tok doctor --json` when diagnostics matter
- `uv run tok stats --json` only when a bridge session actually ran
- whether Claude Code was installed and available as `claude`
- whether Tok degraded to baseline
- whether savings were measured, expected, or not applicable

## Safe editing rules

Prefer small changes. Do not widen the public API casually. Do not add new protocol
claims in 0.2.x docs. Do not change compression behavior without targeted regression
tests. Do not edit generated artifacts unless explicitly instructed. Update docs when
changing CLI behavior.

Exactness rules:

- Do not treat compressed, summarized, skeletonized, or referenced context as exact
  source material.
- Preserve exact text or require explicit reacquisition before code, security, legal, or
  release claims depend on it.
- Compression may reduce repeated payloads, but diagnostics must make fallback and
  degradation visible.
- Never silently degrade into misleading compressed context. Fall open to baseline
  behavior or report a clear failure.

Coding rules:

- Keep the Claude Code bridge path working while changing shared runtime code.
- Keep transport concerns, trace/audit concerns, resolver storage, and CLI UX separate
  unless a test proves they need to meet.
- Do not add hosted-service, repo-indexer, general-framework, or stable-SDK claims.
- Do not add hidden prompt instructions for downstream agents.
- Do not edit generated files, caches, traces, build outputs, or local state.

Testing rules:

- For docs-only changes, run `uv run pytest tests/unit/test_agent_docs_contract.py -q`
  and `uv run python scripts/run_agent_smoke.py`.
- For CLI help or release-surface changes, also run
  `uv run pytest tests/unit/test_cli.py tests/unit/test_release_surface.py -q`.
- For runtime compression, bridge, trace, or resolver changes, add targeted tests near
  the changed behavior before running broader unit/integration tests.
- Do not claim live bridge success unless Claude Code was available and bridge health
  output was captured.

Docs rules:

- Keep `README.md`, `docs/cli-reference.md`, `docs/bridge.md`,
  `docs/troubleshooting.md`, `docs/diagnostics.md`, and this file aligned when the
  user-visible workflow changes.
- Keep planning notes out of the default onboarding path.
- Use exact commands. Say what was run and what passed or failed.

Release-surface rules:

- Check `src/tok/release_surface.py` before naming a command as supported.
- Update release-surface tests when intentionally changing the supported CLI surface.
- `tok audit` validates trace structure and local sidecars; it is not a general
  protocol-compliance certificate.

## Extension rules

Future runtimes such as OpenCode, Codex, or other agent bridges should start at the
transport boundary, not by copying the Claude Code path blindly.

- Identify the runtime's request/response transport boundary first.
- Identify which data can be exact, summarized, skeletonized, or referenced.
- Preserve trace/audit metadata before adding compression behavior.
- Add fixture and regression tests before wiring live behavior.
- Keep live runtime adapters behind explicit commands or configuration.
- Fail open to the original runtime behavior when fidelity is uncertain.
- Do not imply Tok Capability, Tok Session, remote resolver, or agent-to-agent exchange
  support until those layers are implemented and tested.

## Useful files

- `README.md`: public product story and quickstart
- `docs/agent_quickstart.md`: cold-clone workflow for autonomous agents
- `docs/repository_map.md`: public map of repository areas and support boundaries
- `docs/bridge.md`: bridge tutorial
- `docs/cli-reference.md`: supported CLI surface
- `docs/troubleshooting.md`: degraded sessions and fallbacks
- `docs/diagnostics.md`: health signals
- `docs/spec/README.md`: draft trace/protocol spec map
- `pyproject.toml`: package metadata and tool configuration
- `tests/`: unit, integration, replay, smoke, and stability coverage
