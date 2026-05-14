# Repository Map

This map distinguishes supported runtime code from tests, docs, examples, and
maintainer-only material.

## Root

- `AGENTS.md`: authoritative instructions for autonomous agents working in this
  repository.
- `README.md`: public product overview and human quickstart.
- `CONTRIBUTING.md`: contribution and pull-request expectations.
- `CHANGELOG.md`: release notes.
- `pyproject.toml`: package metadata, CLI entry point, dependency ranges, and tool
  configuration.
- `uv.lock`: locked dependency graph for deterministic development installs.
- `CLAUDE.md`: local maintainer/Claude Code notes. Do not treat it as public product
  documentation.

## Supported Runtime Code

- `src/tok/cli/`: Typer CLI implementation for `tok`.
- `src/tok/gateway/`: local bridge server, request handling, streaming, health, and
  bridge preflight behavior.
- `src/tok/runtime/`: compression runtime state, diagnostics, fallback state, request
  preparation, and exactness-related safety logic.
- `src/tok/compression/`: deterministic compression pipelines and tool-result codecs.
- `src/tok/resolver/`: local resolver beta storage and manifest support.
- `src/tok/spec/`: trace and live-trace helpers.
- `src/tok/protocol/`: draft protocol models and parser/encoder helpers.
- `src/tok/release_surface.py`: tested lists of supported, experimental, and internal
  CLI/public surface.

## Other Source Areas

- `src/tok/adapters/`: adapter experiments and conformance support.
- `src/tok/analysis/`: analysis and audit helper code.
- `src/tok/macros/`: macro and compression research code.
- `src/tok/memory/`: pointer and memory helpers.
- `src/tok/monitoring/`: profiling support.
- `src/tok/testing/`: benchmark and fixture generation helpers.
- `src/tok/utils/`: shared utility modules.

Do not infer public API stability from a module existing under `src/tok/`. Check
`src/tok/release_surface.py`, docs, and tests before making support claims.

## Tests

- `tests/unit/`: fast unit and contract tests. Start here for focused changes.
- `tests/integration/`: integration coverage for bridge/runtime behavior.
- `tests/smoke/`: smoke tests, including live bridge and exactness checks.
- `tests/spec/`: trace, resolver, and fixture conformance tests.
- `tests/runtime/`: runtime safety and policy regression tests.
- `tests/fixtures/`: fixture workspaces. Do not treat fixture code as Tok runtime code.
- `tests/benchmarks/`: benchmark tests and performance helpers.

Useful first checks:

```bash
uv run python scripts/run_agent_smoke.py
uv run pytest tests/unit/test_agent_docs_contract.py tests/unit/test_release_surface.py -q
uv run pytest tests/unit tests/integration -v
```

## Scripts

- `scripts/run_agent_smoke.py`: fastest agent-facing repo smoke check.
- `scripts/agent_smoke.sh`: shell wrapper for the agent smoke check.
- `scripts/run_release_smoke.py`: release smoke runner.
- `scripts/check_repo_hygiene.py`: repository hygiene check used in CI.
- `scripts/run_security_audit.py`: security audit helper.
- `scripts/verify_release_claims.py`: release claim verification helper.
- `scripts/tok_trace_reader.py`: standalone trace fixture reader.
- Benchmark scripts under `scripts/run_*benchmark*`: benchmark and release evidence
  helpers.

Prefer documented scripts over ad hoc command sequences when a script already exists for
the check.

## Public Docs

- `docs/agent_quickstart.md`: autonomous-agent cold-clone workflow.
- `docs/repository_map.md`: this file.
- `docs/bridge.md`: supported bridge tutorial.
- `docs/cli-reference.md`: supported CLI commands.
- `docs/troubleshooting.md`: fallback, degraded sessions, logs, and savings
  interpretation.
- `docs/diagnostics.md`: bridge health signals.
- `docs/agent-report-template.md`: report format for agent verification.
- `docs/claims_matrix.md`: evidence trail for public claims.
- `docs/pricing_verification.md`: pricing and savings verification notes.
- `docs/public-release-decision.md`: supported workflows and release bar.
- `docs/release-checklist.md`: release process checklist.

## Draft Specs And Architecture

- `docs/spec/`: draft trace/protocol specification work and fixtures.
- `docs/bridge-standard.md`: bridge exactness and attribution standard.
- `docs/architecture.md`: current architecture overview.
- `docs/architecture-0.2.md`: roadmap/target context, not the runtime contract.
- `docs/architecture-diagrams.md`: architecture diagrams.
- `docs/savings-accounting.md`: savings accounting rules.

Draft spec files are useful for design context. They do not by themselves prove
universal protocol support.

## Maintainer Or Internal Material

- `docs/maintainers/`: maintainer notes and roadmap material.
- `docs/plans/`: planning artifacts. Keep out of the default onboarding path.
- `docs/audits/`: audit notes and evidence reviews.
- `ops/`: internal tracking ledgers. Do not present these as product docs.
- `archive/`, `tmp/`, `dist/`, cache directories, and trace output: generated or local
  material.

Recommended hygiene: keep `docs/plans/`, `docs/maintainers/`, and `ops/` outside public
navigation unless a public doc explicitly links to a stable, reviewed item.

## Examples

- `examples/`: experimental wrapper/API examples outside the default bridge-first path.

Examples are helpful for orientation, but they are not the support contract for 0.2.x.

## GitHub Metadata

- `.github/ISSUE_TEMPLATE/`: issue templates.
- `.github/PULL_REQUEST_TEMPLATE.md`: pull-request checklist.
- `.github/workflows/`: CI, release, benchmark, SBOM, and security workflows.
- `.github/dependabot.yml`: dependency update policy.

Use the bug template plus `docs/agent-report-template.md` when reporting
agent-discovered failures.
