# Changelog

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

- Unused `TokMemory` class replaced by `MacroMemory` throughout.
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

- First-class CLI: `tok bridge start/stop/status/logs`, `tok savings`, `tok install`,
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
