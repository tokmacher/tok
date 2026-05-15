# Changelog

## 0.1.9 (2026-05-12)

### Added

- **Agent operation runbook**: root `AGENTS.md` with golden path, verification commands,
  reporting rules, safe editing rules, and explicit claim boundaries.
- **Machine-readable agent contract**: `docs/agent-contract.json` with supported paths,
  unsupported claims, required verification commands, success signals, and reporting
  fields.
- **Agent smoke check**: `scripts/run_agent_smoke.py` and `scripts/agent_smoke.sh` for
  fast agent verification from a cold clone.
- **Agent report template**: `docs/agent-report-template.md` for structured verification
  reports.
- **Agent docs contract test**: `tests/unit/test_agent_docs_contract.py` prevents
  agent-facing docs from rotting against the real CLI surface.
- **Agent smoke script test**: `tests/unit/test_agent_smoke_script.py` validates the
  smoke script structure and fail-fast behavior.
- **README agent pointer**: brief "Agent Operation" section pointing to `AGENTS.md`.
- **Bridge Capability Manifest**: runtime declaration of supported bridge protocol
  groundwork at `/health` and `tok bridge status`.
- **Evidence exactness bridge standard**: taxonomy for `exact`, `summary`, `skeleton`,
  and `reference` evidence forms.
- **Protocol groundwork status**: planning document for trace receipts, evidence
  exactness, compression decisions, behavior signals, and the capability manifest.
- **Macro expansion engine**: tool mapping and macro expansion for enhanced
  functionality.
- **Hot summary handling**: enhanced hot summary handling and predictive hints for
  eligible repeat targets.
- **Request lifecycle scaffolding**: request lifecycle decomposition for improved
  pipeline clarity.

### Improved

- **Shell integration**: improved shell integration coverage and performance metrics.
- **Protocol detection**: simplified Tok protocol detection logic.
- **Runtime and compression**: refactored compression and runtime state management for
  reduced cross-module drift.
- **Dependency age gate**: relaxed threshold to 5 days; restored urllib3 to 2.7.0 (fixes
  CVE-2026-44431, CVE-2026-44432).

### Fixed

- **Doctor JSON diagnostics**: `tok doctor --json` now correctly sets `ok: false` and
  exits non-zero when the session has degraded to baseline, matching the non-JSON path.
- **Hot-hint state reset**: `reset_session()` now clears `_hot_hints_loaded_from_disk`,
  preventing stale warm-session eligibility after a session reset.
- **Spec test alignment**: roadmap doc test updated to match actual 0.1.9 wording.

## 0.2.0 (2026-05-15)

- **Local resolver beta**: added `tok resolver` commands plus a local content-addressed
  store for `tok-resolver://` URIs.
- **Audit improvements**: `tok audit` can resolve `tok-resolver://` URIs from the local
  resolver store when available.
- **Standalone reader**: added a stdlib-only fixture reader
  (`scripts/tok_trace_reader.py`) that matches `tok audit` results on the fixture packs.
- **Live trace gate**: live trace blocks now assert they never emit `draft-uncomputed`
  payload digests.
- **Diagnostics consistency**: `tok doctor --json` now reports live bridge session
  savings from the same normalized health payload used by `tok stats`, avoiding mixed
  stale/local token counts with live percentages and cost totals.
- **Release docs alignment**: 0.2.0 release docs now describe the `natural-first`
  default request policy and keep `tool-compatible` as an explicit compatibility policy,
  not the default.

### Notes

- This release focuses on the local resolver beta and audit/reader proof. It does not
  add any network routing or referral following.

## 0.1.8 (2026-05-06)

### Improved

- **Compression pipeline**: expanded tool-result detection and caching heuristics, with
  additional unit coverage for result optimizations and common Claude-Code-like
  patterns.
- **Runtime state clarity**: refactored runtime and compression state into grouped
  sub-objects and feature flags to reduce cross-module drift and tighten invariants.

### Notes

- `sbom.spdx` regenerated for this release so `pyproject.toml`, `tok.__version__`, and
  SBOM metadata all report `0.1.8`.

## 0.1.7 (2026-05-04)

### Added

- **Tok Trace v0.1 draft groundwork**: added draft trace docs, fixture corpus, expected
  audit outcomes, fixture-local artifacts, and an internal validator for bridge-session
  trace records.
- **`tok audit` draft trace validation**: validates draft trace fixtures with
  pass/warn/fail outcomes, canonical payload digest checks, local artifact hash/size
  verification, unified-diff delta replay, `--latest`, and JSON output.
- **Opt-in live trace sidecar**: `TOK_TRACE=1` writes metadata-only JSONL trace blocks
  under `.tok/traces/` for `request_prepared`, `fallback`, and `response_processed`
  events. These traces are never sent to the model/provider and audit as warnings unless
  exact artifacts are captured in a future release.
- **Sanitized metadata artifact capture**: `TOK_TRACE_CAPTURE_ARTIFACTS=1` writes
  sidecar metadata artifacts for live trace blocks so `tok audit` can verify hash/size
  locally without storing raw prompts, responses, or tool outputs.
- **Protocol hardening roadmap**: documented Tok Trace as the first layer in a broader
  protocol family, added L0-L5 conformance levels, and started adversarial verifier
  coverage for forged/escaped/out-of-order trace cases.
- **Routing-aware protocol roadmap**: documented routing as a future Resolver/Capability
  design axis with local-first defaults, explicit configuration, and no 0.1.x routing
  implementation.
- **Synthetic adversarial bridge-pressure coverage**: added offline scenario tests for
  Claude-Code-like usage-spike and overcompression risk patterns, including large
  parallel reads, broad audit turns, repeat-read evidence safety, final-answer repair
  guards, skeleton/edit protection, recent-result pressure, and provider-sensitive tool
  pairing repair signals.

### Notes

- This release does not add artifact-backed runtime trace replay, capability handshakes,
  binary trace encodings, or agent-to-agent protocol behavior. The supported workflow
  remains the Claude Code bridge path. The release claim is draft bridge trace/audit
  groundwork, not universal protocol stability.

## 0.1.6 (2026-04-29)

### Fixes

- **Stats labels**: "est. no caching" corrected to "est. no Tok" across all stats
  panels.
- **Current-session stats fallback**: when no active session is running, stats now falls
  back to the live bridge health endpoint instead of showing stale or empty data.
- **Host-stub replay guard regression**: replay logic restored for cases where both
  `stub_text` and `cached_raw_text` are absent; regression tests added.

### Improved

- **Multi-session bucket isolation**: `BridgeSession` now supports per-client session
  isolation via `x-tok-session-id` header, preventing cross-conversation state bleed
  when multiple clients connect to the same bridge.
- **Env var restoration on exit**: `TOK_CAPTURE` and `TOK_RESET_SESSION` are now
  properly restored after `run_bridge` exits, fixing state pollution into parent shell.
- **Release surface validation**: the validator now catches unregistered visible CLI
  commands that bypass the known-command allowlist, preventing accidental exposure of
  experimental commands.
- **Bridge request handler refactor**: handler and response processing paths
  consolidated for clarity and reduced duplication.

### Added

- **Fresh-session pointer notice**: on a new Claude Code session, a one-time runtime
  hint is injected: "see ~/.tok/bridge_memory.tok @pointers for recent file references".
  This surfaces the existence of Tok's pointer registry to the agent without dumping
  full memory state. Fires once per fresh session; not injected on resumption or short
  sessions (under 8 turns).
- `test_bug_audit_regressions.py`: regression tests covering host-stub replay, resend
  strategy, and session state isolation.
- `test_session_state_persistence.py`: tests for `reset_session()` state clearing,
  bucket lifecycle, and fresh-session pointer injection.
- `test_release_surface.py`: CLI surface validation tests.
- `test_stats.py`: coverage for current-session fallback and label accuracy.

### Removed

- `tok _legacy-commands` hidden command and `src/tok/cli/_legacy_commands.py` (293
  lines): functionality superseded by the current bridge-first CLI structure.

## 0.1.5 (2026-04-28)

### Fixes

- **Compression path tracking**: verbatim file observations are now recorded through a
  single `_mark_verbatim_file_observation` helper, closing gaps where path-tracking was
  missed at two of three call sites.
- **Cache fidelity tagging**: results served from a content-hash match are now annotated
  `fidelity:summary|lossy:true` so downstream consumers know the response is a stub, not
  the original content.
- **`first_read_complete` propagation**: cache entries were stored with
  `first_read_complete=False` regardless of content type; now initialised to
  `is_file_like` on store and forced `True` on hash-match serve, preventing spurious
  re-reads.
- **Host-stub replay null guard**: `_should_replay_host_stub` was gating on
  `if not stub_text` alone, which suppressed replay when `cached_raw_text` was also
  absent. Guard now requires both to be falsy.
- **`IS_TOK` false positives**: identifier pattern tightened from `@[A-Za-z_]` to
  `@[A-Za-z_][A-Za-z0-9_]*`, preventing single-character `@x` tokens in user content
  from being misidentified as Tok grammar.
- **`TokMemory` key validation**: regex expanded to permit `/`, `.`, `:`, `[`, `]`
  characters, fixing rejections of valid path-style memory keys.
- **Session persistence**: `load_bridge_memory` failures now increment
  `_persistence_failures` on `RuntimeSession` (previously silent).
- **Episode ledger**: `initialize_session_storage` now loads the episode ledger on
  startup; previously it was skipped until the first explicit load call.
- **`compress_history` return type**: function now returns a three-tuple
  `(messages, tok_state, suppressed_failure_markers)`; call sites in
  `prompt_analyzer.py` and `_release.py` updated accordingly.
- **Cache path normalisation**: path arguments are normalised via
  `normalize_path_target` before result-cache lookup, preventing misses caused by path
  representation differences.
- **mtime check logging**: `OSError` during mtime validation on real-looking paths now
  emits a `debug` log instead of silently passing.
- **Bridge reset URL**: `bridge_reset_session` now uses `bridge_url()` helper instead of
  a hardcoded `localhost:9090`, respecting custom port configuration.
- **Retry fallback recording**: retried requests now call `_record_fallback_once` so
  retry-driven fallbacks are visible in session stats.
- **Turn task IDs**: `smoothness_tracker.start_turn()` now receives a `task_id` derived
  from the current turn counter, improving turn-level observability in diagnostics.

### Added

- `tok bridge reset-session` hidden CLI command: resets the running bridge's first-read
  and first-exact tracking state without restarting the process.
- `_KNOWN_TOK_BLOCKS` frozen set and `_line_start_tok` pattern for more structured Tok
  grammar detection.
- Structured `compression_decision` debug log lines throughout
  `compress_recent_window_impl`, covering preserved/bypassed decisions with reason codes
  (`exact_search_observation`, `first_exact_guard`, `first_session`, `zero_heat`,
  `detection_type_raw`, `pytest_failed`, `size_below_threshold`, `no_compressor`).

## 0.1.4 (2026-04-26)

### Fixes

- **Stats accuracy**: baseline cost model now reflects Tok's compression contribution
  only. Previously, cache token discounts (an API feature) were attributed to Tok,
  inflating cost savings figures to ~90%. Cost savings now tracks token savings (~36% in
  typical sessions).
- **Stats display**: `cost_savings_pct` was missing from completed-session summaries,
  causing the Last Completed Session panel to show token % as the cost headline. Fixed.
- **Compression pipeline**: large `git diff` outputs were being truncated by the generic
  truncation layer after `_compress_git_diff` had already stripped context lines,
  removing actual changed lines. Diffs are now marked already-compressed and exempt from
  further truncation.
- **Stats command**: duplicate rendering branch for Last Completed Session panel
  removed.
- Session persistence errors now increment a `_persistence_failures` counter on
  `RuntimeSession`, making previously silent disk failures observable.
- Result cache entries truncated on save now emit a warning log instead of silently
  dropping characters.
- Memory command `typer.Exit()` handling corrected.

### Improved

- Token and cost row labels across all stats panels now read "Tokens (with Tok / est. no
  Tok)" and "Cost (with Tok / est. no caching)" to make the baseline assumption
  explicit.
- `reset_session()` added to `RuntimeSession` for clean transient state resets without
  touching persisted data.
- Compression level validation introduced for `tok_tool_result`.
- Thread safety improved in result cache operations.

### Removed

- `TokMemory` Pydantic model retained in `protocol/models.py`; call-sites migrated to
  `MacroMemory`.
- Deprecated runtime configuration hints removed.
- Unused `_install.py` CLI module removed.
- Unused `typings/streamlit` stub removed.

## 0.1.3 (2026-04-25)

### Fixes

- Rate limit thundering herd: concurrent 429 retries are now serialized via a shared
  lock and cooldown flag, preventing burst requests from exhausting the rate limit
  allocation on window reset.
- Semantic drift false positives prevented in tool use/result pairing.
- IS_TOK gate now correctly detects line-start `|>` to prevent Tok grammar leaking to
  the user.

### Added

- Harness injection stripping removes injected content from cached results.
- Parallel read deduplication within a single turn for repeated file reads.
- Verbosity tracking and session context in request preparation.

## 0.1.0 (2026-03-31)

### Added

- First-class CLI: `tok bridge start/stop/status/logs`, `tok stats`, `tok install`,
  `tok doctor`
- Bridge decomposed into focused modules: `gateway.py`, `compression.py`,
  `translator.py`, `pricing.py`, `stats.py`
- `BridgeSession` class encapsulating all mutable state (no global mutable state)
- Fail-open resilience: bridge errors pass requests through transparently
- `tok install` auto-configures shell integration for zsh/bash/fish
- `tok doctor` health check command
- Health endpoint at `/health`
- 1478 test functions across 100+ files covering compression, translation, stats,
  pricing, CLI, smoke, and release surface validation
- Mock Anthropic server for integration testing without API keys
- GitHub Actions CI (Python 3.10, 3.11, 3.12 on ubuntu-latest and macos-latest)
- `py.typed` marker for PEP 561
- GitHub issue and PR templates
- Development setup guide (`DEVELOPMENT.md`)
- Maintainer docs (`docs/maintainers/README.md`)
- Production readiness, release checklist, and public release decision docs
- CI/CD integration guide
- Committed `uv.lock` for deterministic dependency resolution

### Changed

- Renamed `bridge.py` (format converter) to `format_bridge.py` to free `bridge` name for
  gateway
- Rebranded "MITM proxy" to "bridge" throughout codebase
- Version synced to 0.1.0 in both `pyproject.toml` and `__init__.py`
- Moved `ruff` and `mypy` from main dependencies to dev
- Research artifacts moved to `archive/`
- Docs consolidated from 30+ files to 4 focused docs
- Tests reorganized into `tests/unit/`, `tests/integration/`, `tests/benchmarks/`
- Release posture documented more explicitly: live Claude validation is the final
  pre-release gate, CLI decomposition beyond the current split is deferred, and
  dependency upper bounds remain a post-`0.1.0` policy follow-up

### Removed

- `[MEMORY-DEBUG]` print statements replaced with proper `logging` module
- Hardcoded memory path replaced with `TOK_PROJECT_DIR` env var
- Global mutable state (`_stats_lock`, module-level config) replaced with
  `BridgeSession`
