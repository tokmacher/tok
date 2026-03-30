# Contributing

Thanks for helping improve Tok.

## Before You Start

- read [`README.md`](./README.md) for the public workflow
- read [`DEVELOPMENT.md`](./DEVELOPMENT.md) for local setup
- keep the bridge-first product scope intact unless there is strong evidence to widen it

## Pull Request Expectations

- keep changes focused and reviewable
- preserve existing behavior unless the change intentionally updates docs and tests
- add or update tests when behavior changes
- run the high-signal local checks before opening a PR

Suggested checks:

```bash
pre-commit run --all-files
python scripts/check_repo_hygiene.py
uv run pytest tests/unit/test_cli.py tests/unit/test_gateway.py tests/unit/test_compression.py tests/unit/test_universal_runtime.py tests/unit/test_bridge_memory.py
python -m build
```

## Branching & Commits

- Base every branch on `main` (there is no long-lived release branch yet) and protect `main` with the required status checks.
- Use `[type]: description` (Conventional Commits) for commit messages so release automation and changelog tooling can recognize intent.
- Prefix branch names with `feature/`, `fix/`, or `chore/` depending on the work being done.

## Testing Expectations

- Run the full pytest suite before merging: `pytest tests/unit/ tests/integration/ -v`.
- The release-surface coverage gate is 80%. Use the checked-in `.coveragerc` when running `pytest --cov=src/tok`.
- The `uv` lockfile governs precise dependency versions. `uv sync` keeps you aligned with the lockfile, while `pip install -e "[.dev]"` may resolve newer versions; prefer `uv sync` if you need deterministic results.

## Docs Expectations

- update `README.md` when the first-run workflow changes
- update [`docs/bridge.md`](./docs/bridge.md), [`docs/cli-reference.md`](./docs/cli-reference.md), and [`docs/troubleshooting.md`](./docs/troubleshooting.md) when the user-visible CLI changes
- keep maintainer planning docs out of the default onboarding path

## Community

Please follow [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md).
