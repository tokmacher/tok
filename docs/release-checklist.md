# Release Checklist

Steps for cutting a Tok release.

## Before Release

- [ ] Worktree is clean or intentionally quarantined; do not tag from a churn-heavy
  bridge/runtime branch
- [ ] All CI checks pass on `main`
- [ ] Run full test suite locally: `uv run pytest tests/unit tests/integration -v`
- [ ] Run lint and hygiene:
  `uv run pre-commit run --all-files && uv run ruff check src/tok tests`
- [ ] Run type check: `uv run mypy src/tok/`
- [ ] Run maintainer release smoke: `uv run python scripts/run_release_smoke.py`
- [ ] Regenerate release metadata artifacts:
  `uv run python scripts/generate_spdx_sbom.py`
- [ ] Confirm the release-surface gate passes and no experimental root exports are
  advertised as canonical
- [ ] Reconcile pricing claims via `docs/pricing_verification.md`
- [ ] Reconcile benchmark/savings claims via `docs/claims_matrix.md`
- [ ] Confirm automated and manual live-smoke coverage via `docs/live_smoke_matrix.md`
- [ ] Build package: `uv build`
- [ ] Validate built artifact metadata and README rendering locally:
  `uv run --with twine python -m twine check dist/*.whl dist/*.tar.gz`
- [ ] Verify wheel installs cleanly in a fresh venv
- [ ] Verify the sdist is present in `dist/` and includes release-critical metadata
  files
- [ ] Verify `tok --version` reports the tagged release version from the installed
  artifact
- [ ] Run the clean-room install verification from the README
- [ ] Confirm `tok --help` only emphasizes the bridge-first public workflow for `0.1.x`
- [ ] For `0.1.7`, confirm Tok Trace remains draft-scoped: `tok audit` is visible, live
  trace emission is opt-in via `TOK_TRACE=1`, and `uv run pytest tests/spec -q` passes
- [ ] Run live Claude bridge validation on the supported path: `tok claude`,
  `tok bridge status`, `tok doctor`, `tok stats`, then exit Claude and run
  `tok bridge stop`
- [ ] Treat any frontier/OpenRouter report as advisory validation only; do not let it
  redefine the Claude bridge default
- [ ] Confirm only supported examples remain in `examples/`
- [ ] Update `CHANGELOG.md` with release date
- [ ] Update `__version__` in `src/tok/__init__.py`
- [ ] Update version in `pyproject.toml`
- [ ] Confirm `pyproject.toml`, `src/tok/__init__.py`, `sbom.spdx`, and security utility
  User-Agents all report the same release version
- [ ] Confirm README badges and repository URLs resolve publicly
- [ ] Confirm the deferred `0.1.x` follow-ups are documented: CLI decomposition and
  dependency upper-bound policy

Recommended local gate sequence for the exact release candidate:

```bash
uv sync --frozen --extra dev
uv run python scripts/generate_spdx_sbom.py
uv run python scripts/run_release_smoke.py
uv run pre-commit run --all-files
uv run ruff check src/tok tests
uv run mypy src/tok
uv run pytest tests/unit tests/integration -v --cov=src/tok --cov-fail-under=80
uv build
```

For maintainers and CI parity, `uv` is canonical. End-user install/onboarding commands
in user-facing docs remain `pip`-first.

## Release

- [ ] Reconfirm the exact release candidate still passes the supported bridge-first live
  validation path
- [ ] Tag the commit: `git tag v0.x.x`
- [ ] Push the tag: `git push origin v0.x.x`
- [ ] Watch the `Release` GitHub Actions workflow build the artifacts
- [ ] Confirm publish only ran from a `v*` tag path and not from any manual dispatch
  path
- [ ] Confirm the workflow publishes to PyPI via trusted publishing
- [ ] Confirm the workflow creates the GitHub Release and uploads `dist/*`

## After Release

- [ ] Verify the package on PyPI
- [ ] Verify the README renders correctly on PyPI
- [ ] Verify the GitHub Release notes and assets are correct
- [ ] Update any internal references to the new version
- [ ] Announce if appropriate
