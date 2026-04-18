# CI/CD Integration

Using Tok with continuous integration and delivery pipelines.

## Running Tok in CI

Tok is designed for local development with Claude Code. It is not currently designed to
run as a CI service. However, you can use Tok's test and release gate tooling in CI:

### Release Gate Check

```bash
tok gate-check tests/fixtures/replay \
  --fixtures fixtures.json \
  --config gate-config.json \
  --stability-dir tests/fixtures/stability \
  --required-benchmarks coding-loop-5,research-loop-5 \
  --export results.json
```

This validates replay fixtures and stability artifacts against configurable thresholds.

### Lint and Test

The CI pipeline runs these checks on every push and PR:

```bash
uv sync --frozen --extra dev
uv run pre-commit run --all-files
uv run ruff check src/tok tests
uv run mypy src/tok/
uv run pytest tests/unit tests/integration -v --cov=src/tok --cov-fail-under=80
uv build
```

For maintainers and CI parity, `uv` is the canonical workflow. For end-user
installation/onboarding, keep `pip install tok-protocol` as the default path.

Tag pushes (`v*`) trigger the release workflow, which builds the package, publishes to
PyPI using GitHub trusted publishing, and creates a GitHub Release with the built
artifacts attached.

The security workflow runs dependency review on pull requests and CodeQL on `main` plus
a weekly schedule.

## CI Status Badges

- **CI**: ![CI](https://github.com/tokmacher/tok/actions/workflows/ci.yml/badge.svg)
- **PyPI**: ![PyPI version](https://img.shields.io/pypi/v/tok-protocol.svg)

## Local Development Checks

Before opening a PR, run:

```bash
uv sync --frozen --extra dev
uv run pre-commit run --all-files
uv run ruff check src/tok tests
uv run mypy src/tok/
uv run pytest tests/unit tests/integration -v --cov=src/tok --cov-fail-under=80
uv build
```

For the release candidate itself, use the same sequence and then verify the built wheel
includes the bundled shell integration script and other package data.
