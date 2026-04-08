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
pre-commit run --all-files
ruff check src/tok tests
mypy src/tok/
pytest tests/unit tests/integration -v --cov=src/tok --cov-fail-under=80
python -m build
```

`python -m build` is still the canonical release check. If you are validating in an
offline or sandboxed environment that already has build requirements installed,
`python -m build --no-isolation` is an acceptable fallback for local verification.

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
pre-commit run --all-files
ruff check src/tok tests
mypy src/tok/
pytest tests/unit tests/integration -v --cov=src/tok --cov-fail-under=80
python -m build
```

For the release candidate itself, use the same sequence and then verify the built wheel
includes the bundled shell integration script and other package data.
