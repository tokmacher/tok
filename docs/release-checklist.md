# Release Checklist

Steps for cutting a Tok release.

## Before Release

- [ ] Worktree is clean or intentionally quarantined; do not tag from a churn-heavy bridge/runtime branch
- [ ] All CI checks pass on `main`
- [ ] Run full test suite locally: `pytest tests/unit tests/integration -v`
- [ ] Run lint and hygiene: `pre-commit run --all-files && ruff check src/tok tests`
- [ ] Run type check: `mypy src/tok/`
- [ ] Build package: `python -m build`
- [ ] Verify wheel installs cleanly in a fresh venv
- [ ] Run the clean-room install verification from the README
- [ ] Confirm `tok --help` only emphasizes the bridge-first public workflow for `0.1.0`
- [ ] Run live Claude bridge validation on the supported path: `tok install`, `tok bridge start`, `claude`, `tok bridge status`, `tok doctor`, `tok stats`, `tok bridge stop`
- [ ] Confirm only supported examples remain in `examples/`
- [ ] Update `CHANGELOG.md` with release date
- [ ] Update `__version__` in `src/tok/__init__.py`
- [ ] Update version in `pyproject.toml`
- [ ] Confirm README badges and repository URLs resolve publicly
- [ ] Confirm the deferred `0.1.0` follow-ups are documented: CLI decomposition and dependency upper-bound policy

Recommended local gate sequence for the exact release candidate:

```bash
pre-commit run --all-files
ruff check src/tok tests
mypy src/tok
pytest tests/unit tests/integration -v --cov=src/tok --cov-fail-under=80
python -m build
```

If the release check is running in an offline or sandboxed environment that
already has build requirements installed, `python -m build --no-isolation` is an
acceptable local fallback. The canonical release command remains `python -m build`.

## Release

- [ ] Reconfirm the exact release candidate still passes the supported bridge-first live validation path
- [ ] Tag the commit: `git tag v0.x.x`
- [ ] Push the tag: `git push origin v0.x.x`
- [ ] Watch the `Release` GitHub Actions workflow build the artifacts
- [ ] Confirm the workflow publishes to PyPI via trusted publishing
- [ ] Confirm the workflow creates the GitHub Release and uploads `dist/*`

## After Release

- [ ] Verify the package on PyPI
- [ ] Verify the README renders correctly on PyPI
- [ ] Verify the GitHub Release notes and assets are correct
- [ ] Update any internal references to the new version
- [ ] Announce if appropriate
