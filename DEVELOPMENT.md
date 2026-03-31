# Development Setup

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Install

```bash
git clone https://github.com/tokmacher/tok.git
cd tok
uv sync
uv run pre-commit install
```

If not using uv:

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/unit/ tests/integration/ -v
```

To include live or benchmark tests:

```bash
pytest tests/ -v -m "live or benchmark"
```

## Linting and Type Checking

```bash
ruff check src/tok/ tests/
mypy src/tok/
pre-commit run --all-files
python -m build
```

## Code Style

- Line length: 79 characters
- Linter: ruff (E, F, I, N, W, UP, B, C901 rules)
- Type checker: mypy (strict on release surface)

## Project Layout

- `src/tok/` - runtime, bridge, CLI, and library code
- `src/tok/runtime/` - universal runtime core
- `src/tok/gateway/` - bridge gateway (primary production adapter)
- `src/tok/cli/` - CLI surface
- `src/tok/protocol/` - Tok language/protocol layer
- `src/tok/compression/` - compression engine
- `tests/unit/` - unit tests
- `tests/integration/` - integration tests
- `tests/benchmarks/` - performance benchmarks
- `docs/` - public documentation
- `docs/maintainers/` - internal planning docs
