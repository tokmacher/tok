# RELEASE v0.2.2: Tok Substrate Release

STATUS: pass

## Workstreams

WORKSTREAM A (Codex Adapter): pass

- `docs/bridge_contract.md` documents the adapter/runtime contract.
- Codex fixture coverage includes exactness labels, fallback state, unsupported
  capabilities, and diagnostic signals.
- `docs/plans/codex_adapter_transport_decision.md` records the decision to keep Codex
  traffic out of the Claude-specific gateway.
- `src/tok/gateway/_adapter_proxy.py` provides a gated local Codex adapter proxy proof:
  adapter parse, runtime prepare, provider forwarder, runtime response processing, and
  adapter response build.
- The local adapter remains gated by `TOK_UNSTABLE_ADAPTERS=1`.
- Cross-adapter and adapter-proxy tests pass.

WORKSTREAM A2 (Adapter Consolidation): pass

- Shared private adapter helpers live in `src/tok/adapters/_utils.py`.
- Codex and OpenCode probe adapters live in `src/tok/adapters/probes.py`.
- The deleted `src/tok/adapters/codex_probe.py` and `src/tok/adapters/opencode_probe.py`
  modules are no longer import targets.
- Consolidation tests assert object identity for the shared helpers used by
  `adapters.py` and `probes.py`.

WORKSTREAM B (Session Receipt): pass

- Session receipt schema and examples exist under
  `docs/spec/tok-session-receipt/v0.1-draft/`.
- `src/tok/protocol/session_receipt.py` defines local receipt generation and
  verification.
- `tok session-receipt` is hidden and experimental.
- `tok audit --session-receipt` validates local receipts without claiming protocol
  compliance.
- Adversarial receipt tests pass.

WORKSTREAM C (Savings Proof): pass

- `docs/plans/savings_benchmark_gap_report.md` records the savings and benchmark
  inventory.
- Benchmark family labels map to the maintained families: `execution_patch`,
  `repo_grounding`, and `real_session`.
- `tok stats --json` exposes durable savings evidence including net saved tokens,
  reacquisition cost, fallback/degraded state, cache tokens, and bypass count.
- `docs/claims_matrix.md` is updated for v0.2.2 and the claims gate rejects verified
  savings bands outside the configured evidence band.

WORKSTREAM D (Handoff): pass

- Handoff schema and examples exist under `docs/spec/tok-handoff/v0.1-draft/`.
- `src/tok/protocol/handoff.py` defines export and inspect behavior.
- `tok handoff export` and `tok handoff inspect` are hidden and experimental.
- The two-agent fixture proves Agent B must reacquire exact evidence before edit-like
  work.

WORKSTREAM E (Surface Promotion): pass

- Package version is `0.2.2`.
- Public docs describe the supported Claude Code bridge path and keep session receipt,
  handoff, and adapter work experimental/local.
- Release surface was updated deliberately for hidden experimental commands only.
- No public universal protocol, hosted service, repo-indexer, stable SDK, or broad
  agent-to-agent exchange claim was added.

## Files Created

Key created files and directories:

- `docs/bridge_contract.md`
- `src/tok/adapters/_utils.py`
- `src/tok/adapters/probes.py`
- `docs/spec/tok-session-receipt/v0.1-draft/README.md`
- `docs/spec/tok-session-receipt/v0.1-draft/examples/good_receipt.json`
- `docs/spec/tok-session-receipt/v0.1-draft/examples/degraded_receipt.json`
- `docs/spec/fixtures/session_receipt_good.json`
- `docs/spec/fixtures/session_receipt_adversarial.json`
- `src/tok/protocol/session_receipt.py`
- `src/tok/cli/_session_receipt_commands.py`
- `tests/unit/test_session_receipt.py`
- `tests/spec/test_session_receipt_adversarial.py`
- `docs/spec/tok-handoff/v0.1-draft/README.md`
- `docs/spec/tok-handoff/v0.1-draft/examples/good_handoff.json`
- `docs/spec/tok-handoff/v0.1-draft/examples/good_handoff.md`
- `src/tok/gateway/_adapter_proxy.py`
- `src/tok/protocol/handoff.py`
- `src/tok/cli/_handoff_commands.py`
- `tests/unit/test_handoff.py`
- `tests/integration/test_local_agent_handoff.py`
- `tests/integration/test_codex_adapter_transport.py`
- `tests/unit/test_adapter_consolidation.py`
- `tests/unit/test_utils_edge_cases.py`
- `tests/unit/test_probe_edge_cases.py`
- `tests/unit/test_adapter_proxy_unit.py`
- `tests/unit/test_savings_coverage.py`
- `tests/unit/test_savings_persistence.py`
- `tests/fixtures/codex/diagnostic_signals.json`
- `tests/fixtures/codex/exactness_labels.json`
- `tests/fixtures/codex/fallback_state.json`
- `tests/fixtures/codex/unsupported_capabilities.json`

## Files Deleted

- `src/tok/adapters/codex_probe.py`
- `src/tok/adapters/opencode_probe.py`

## Files Modified

Key modified files:

- `AGENTS.md`
- `README.md`
- `pyproject.toml`
- `uv.lock`
- `sbom.spdx`
- `docs/agent-contract.json`
- `docs/agent_quickstart.md`
- `docs/claims_matrix.md`
- `docs/cli-reference.md`
- `docs/diagnostics.md`
- `docs/plans/claude_code_execution_prompt.md`
- `docs/plans/codex_adapter_transport_decision.md`
- `docs/plans/savings_benchmark_gap_report.md`
- `docs/plans/substrate_execution_prompt.md`
- `docs/plans/tok_0_2_x_development_roadmap.md`
- `docs/repository_map.md`
- `scripts/verify_release_claims.py`
- `src/tok/__init__.py`
- `src/tok/adapters/adapter_config.py`
- `src/tok/adapters/adapters.py`
- `src/tok/cli/__init__.py`
- `src/tok/cli/_audit_commands.py`
- `src/tok/cli/_bridge.py`
- `src/tok/cli/_release.py`
- `src/tok/release_surface.py`
- `src/tok/testing/benchmark_suite.py`
- `src/tok/testing/live_benchmark/_runner.py`
- `src/tok/utils/savings_tracker.py`
- `tests/fixtures/codex/response_expected.json`
- `tests/spec/test_codex_fixtures.py`
- `tests/spec/test_opencode_fixtures.py`
- `tests/unit/test_adapter_conformance.py`
- `tests/unit/test_cli.py`
- `tests/unit/test_release_claims_inputs.py`
- `tests/unit/test_release_surface.py`
- `tests/unit/test_verify_release_claims_script.py`

## Tests Added

Primary new test files or packs:

- `tests/unit/test_adapter_consolidation.py`
- `tests/unit/test_utils_edge_cases.py`
- `tests/unit/test_probe_edge_cases.py`
- `tests/unit/test_adapter_proxy_unit.py`
- `tests/unit/test_savings_coverage.py`
- `tests/unit/test_savings_persistence.py`
- `tests/unit/test_session_receipt.py`
- `tests/spec/test_session_receipt_adversarial.py`
- `tests/unit/test_handoff.py`
- `tests/integration/test_local_agent_handoff.py`
- `tests/integration/test_codex_adapter_transport.py`

Existing adapter, CLI, claims, and release-surface tests were extended.

## Commands Run

- `python -m py_compile src/tok/adapters/probes.py src/tok/adapters/_utils.py src/tok/gateway/_adapter_proxy.py tests/integration/test_codex_adapter_transport.py tests/spec/test_codex_fixtures.py tests/unit/test_adapter_conformance.py`
  - PASS
- `TOK_UNSTABLE_ADAPTERS=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/test_codex_adapter_transport.py tests/spec/test_codex_fixtures.py tests/unit/test_adapter_conformance.py -q`
  - PASS: 18 passed
- `TOK_UNSTABLE_ADAPTERS=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration -k "codex or opencode or adapter" -q`
  - PASS: 4 passed, 41 deselected
- `TOK_UNSTABLE_ADAPTERS=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/spec -k "codex or opencode" -q`
  - PASS: 11 passed, 140 deselected
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit -k "adapter or runtime or fallback or stats" -q`
  - PASS: 544 passed, 1 skipped, 2674 deselected
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/test_release_surface.py -q`
  - PASS: 8 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/test_agent_docs_contract.py -q`
  - PASS: 46 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_agent_smoke.py`
  - PASS: CLI version/help, Claude help, bridge status help, doctor help, stats help,
    audit help, and agent contract tests passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_release_smoke.py`
  - PASS after escalated rerun for local socket binding
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/test_agent_docs_contract.py tests/unit/test_release_surface.py -q`
  - PASS: 54 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit tests/integration -v`
  - PASS: 3261 passed, 3 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/tok tests`
  - PASS
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/tok`
  - PASS: no issues in 257 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv build`
  - PASS: built `tok_protocol-0.2.2.tar.gz` and `tok_protocol-0.2.2-py3-none-any.whl`
- `uv run pytest tests/unit/ tests/spec/ tests/integration/ -q`
  - PASS: 3485 passed, 3 skipped
- `uv run mypy src/tok/`
  - PASS: no issues in 257 source files
- `uv run ruff check src/tok/ tests/`
  - PASS after removing unused imports, sorting new test imports, and refreshing this
    report
- `uv run pytest tests/unit/test_adapter_conformance.py tests/unit/test_adapter_proxy_unit.py tests/unit/test_utils_edge_cases.py -q`
  - PASS: 41 passed
- `uv run pytest tests/unit/test_agent_docs_contract.py -q`
  - PASS: 46 passed
- `uv run python scripts/run_agent_smoke.py`
  - PASS: CLI version/help, Claude help, bridge status help, doctor help, stats help,
    audit help, and agent contract tests passed

## Pass/Fail

All final gates passed.

The first un-escalated `scripts/run_release_smoke.py` attempt failed because the sandbox
denied binding `127.0.0.1:0` in `test_primary_streaming_smoke.py`. The escalated rerun
passed.

## Known Gaps

- A real Codex CLI version and transport override were not verified in a live external
  Codex session. The release includes a gated local proxy proof, not a supported Codex
  CLI user workflow.
- Claude Code availability in the user's environment was not checked.
- No live bridge session health output was captured in this report.
- No live `tok stats` savings were measured in this report; savings evidence is from
  test suites and local accounting artifacts.

## Scope Cuts

- No supported public Codex CLI command was added.
- No universal protocol, stable Tok Session, Tok Capability, remote resolver, hosted
  service, repo-indexer, or broad agent-to-agent exchange claim was added.
