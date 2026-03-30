# Changelog

## 0.1.0 (Unreleased)

### Added
- First-class CLI: `tok bridge start/stop/status/logs`, `tok savings`, `tok install`, `tok doctor`
- Bridge decomposed into focused modules: `gateway.py`, `compression.py`, `translator.py`, `pricing.py`, `stats.py`
- `BridgeSession` class encapsulating all mutable state (no global mutable state)
- Fail-open resilience: bridge errors pass requests through transparently
- `tok install` auto-configures shell integration for zsh/bash/fish
- `tok doctor` health check command
- Health endpoint at `/health`
- 903 test functions across 63 files covering compression, translation, stats, pricing, and CLI
- Mock Anthropic server for integration testing without API keys
- GitHub Actions CI (Python 3.9, 3.11, 3.12)
- `py.typed` marker for PEP 561

### Changed
- Renamed `bridge.py` (format converter) to `format_bridge.py` to free `bridge` name for gateway
- Rebranded "MITM proxy" to "bridge" throughout codebase
- Version synced to 0.1.0 in both `pyproject.toml` and `__init__.py`
- Moved `ruff` and `mypy` from main dependencies to dev
- Research artifacts moved to `archive/`
- Docs consolidated from 30+ files to 4 focused docs
- Tests reorganized into `tests/unit/`, `tests/integration/`, `tests/benchmarks/`

### Removed
- `[MEMORY-DEBUG]` print statements replaced with proper `logging` module
- Hardcoded memory path replaced with `TOK_PROJECT_DIR` env var
- Global mutable state (`_stats_lock`, module-level config) replaced with `BridgeSession`
