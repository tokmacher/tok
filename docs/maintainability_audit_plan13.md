# Plan 13 Maintainability Audit (Post-Freeze)

Date: 2026-04-08

## Baseline Verification (Frozen)

- `uv run python scripts/run_release_smoke.py` -> pass on 2026-04-08.
- `uv run pytest tests/smoke/test_api_base_smoke.py -q` -> `2 passed` on 2026-04-08.
- `uv run pytest tests/unit/test_streaming_cleanup_exact_once.py -q` -> `5 passed` on
  2026-04-08.
- RC-ready defended-surface status is recorded in `hardening_ledger.md`.

## File Map (Python Scope for this Audit)

- `src/tok/`: 150 Python files.
- `scripts/`: 17 Python files.
- `tests/`: 106 Python files.

## Buckets

### Likely Maintainability Targets

- `scripts/run_release_smoke.py`
- `src/tok/runtime/pipeline/request_validation.py`
- `src/tok/runtime/_request_preparation.py`
- `src/tok/runtime/pipeline/request_preparation.py`
- `src/tok/runtime/pipeline/response_processing.py`

### Protected Files (Do Not Touch Unless Evidence Forces)

- `src/tok/gateway/_bridge_streaming.py`
- `src/tok/gateway/_app_factory.py`
- `src/tok/runtime/core.py`

### Docs / Claim / Pricing / Benchmark Files

- `docs/claims_matrix.md`
- `docs/live_smoke_matrix.md`
- `docs/pricing_verification.md`
- `docs/benchmark_findings_0.1.0.md`
- `src/tok/gateway/pricing.py`
- `src/tok/utils/pricing.py`
- `tests/unit/test_pricing.py`

### Audit-Only (No Edits in Plan 13)

- `src/tok/compression/_history_pipeline.py`
- `src/tok/testing/stress/runner.py`
- large `tests/unit/` monoliths (including `test_gateway.py`,
  `test_universal_runtime.py`, `test_stress_harness.py`)

## Maintainability Scoring and Disposition

Scale: 1 (low concern) to 5 (high concern)

| file                                              | size/density | responsibilities | naming clarity | magic literals | comment drift | testability | defended relevance | disposition                  |
| ------------------------------------------------- | ------------ | ---------------- | -------------- | -------------- | ------------- | ----------- | ------------------ | ---------------------------- |
| `src/tok/runtime/pipeline/request_validation.py`  | 5            | 5                | 3              | 4              | 2             | 3           | 5                  | candidate for targeted split |
| `src/tok/runtime/_request_preparation.py`         | 4            | 5                | 3              | 4              | 2             | 3           | 5                  | candidate for targeted split |
| `scripts/run_release_smoke.py`                    | 3            | 3                | 4              | 3              | 2             | 4           | 5                  | safe clarity refactor        |
| `src/tok/runtime/pipeline/request_preparation.py` | 3            | 3                | 4              | 2              | 2             | 4           | 4                  | safe clarity refactor        |
| `src/tok/runtime/pipeline/response_processing.py` | 3            | 4                | 3              | 3              | 2             | 3           | 4                  | safe clarity refactor        |
| `src/tok/compression/_history_pipeline.py`        | 4            | 4                | 3              | 3              | 2             | 3           | 3                  | audit only                   |
| `src/tok/testing/stress/runner.py`                | 5            | 5                | 2              | 3              | 3             | 2           | 2                  | audit only                   |
| `src/tok/gateway/_bridge_streaming.py`            | 4            | 4                | 3              | 3              | 2             | 3           | 5                  | do not touch now             |
| `src/tok/gateway/_app_factory.py`                 | 4            | 4                | 3              | 3              | 2             | 3           | 5                  | do not touch now             |
| `src/tok/runtime/core.py`                         | 4            | 4                | 3              | 3              | 2             | 3           | 5                  | do not touch now             |

## Frozen Maintainability Rules

- Prefer explicit over implicit.
- Prefer one obvious responsibility per helper.
- Prefer descriptive names over short clever names.
- Delete dead code rather than commenting around it.
- Comments must explain intent or constraints, not restate code.
- Comments must not mention stale plans/phases unless they are still current.
- Literals that encode policy should become named constants.
- Splitting is allowed only when it improves ownership and reasoning.
- No rule in this audit implies architecture redesign.

## Frozen Naming Rules and Canonical Terms

- Names must reflect actual responsibility.
- Avoid vague names such as `data`, `thing`, `result`, `info` unless tightly scoped.
- Avoid stale names tied to old implementation phases.
- Reuse defended terminology from ledger/docs.

Canonical terms for this pass:

- defended surface
- candidate path
- internal compatibility seam
- smoke gate
- boundary
- cleanup ownership
- request validation
- request preparation

## Dead-Code Candidate Ledger (Initial)

No removals performed yet. Candidates requiring proof before deletion:

| symbol / block                                                     | file                                             | why candidate                                   | evidence status                        | classification                                          |
| ------------------------------------------------------------------ | ------------------------------------------------ | ----------------------------------------------- | -------------------------------------- | ------------------------------------------------------- |
| Local validation/check constants and duplicated threshold literals | `src/tok/runtime/pipeline/request_validation.py` | policy literals currently embedded in functions | used at runtime                        | keep, refactor for clarity only                         |
| Inline smoke harness literals in validation-failure script         | `scripts/run_release_smoke.py`                   | dense literals reduce readability               | script behavior intentionally explicit | keep, extract constants only when zero-drift is obvious |

Removal policy in effect:

- remove only with explicit non-use evidence;
- do not remove compatibility seams or defended-surface-linked behavior without
  dedicated proof.

## Plan 13 Execution Summary (This Pass)

### Implemented Changes

- `src/tok/runtime/pipeline/request_validation.py`
  - Extracted policy literals into named constants:
    - `_DEFAULT_PROMPT_BLOAT_THRESHOLD_CHARS`
    - `_DEFAULT_PROMPT_OPTIMIZE_LIMIT_CHARS`
    - `_USER_PROMPT_LEAK_MIN_CHARS`
    - `_USER_PROMPT_LEAK_SNIPPET_CHARS`
  - Replaced inline threshold literals with the constants above (behavior-preserving).
- `src/tok/runtime/_request_preparation.py`
  - Extracted env-int parsing helper: `_env_int_or_default(name, default)`.
  - Replaced duplicated threshold parsing blocks for JIT/speculative macro thresholds.
  - Renamed local `_spec_threshold` to `speculative_hit_threshold` for clarity.
- `scripts/run_release_smoke.py`
  - Comment hygiene: removed stale plan-phase phrasing in one comment block.

### Verification Outcomes (2026-04-08)

- Compile checks:
  - `python -m py_compile src/tok/runtime/pipeline/request_validation.py` -> pass
  - `python -m py_compile src/tok/runtime/_request_preparation.py` -> pass
  - `python -m py_compile scripts/run_release_smoke.py` -> pass
- Focused tests:
  - `uv run pytest tests/unit/test_prompt_optimization.py -q` -> `10 passed`
  - `uv run pytest tests/unit/test_jit_execution.py -q` -> `4 passed`
  - `uv run pytest tests/unit/test_request_preparation_resend.py -q` -> `9 passed`
- Freeze evidence tests:
  - `uv run pytest tests/smoke/test_api_base_smoke.py -q` -> `2 passed`
  - `uv run pytest tests/unit/test_streaming_cleanup_exact_once.py -q` -> `5 passed`
- Release smoke:
  - `uv run python scripts/run_release_smoke.py` -> pass

### Known Existing Test Failure In Current Tree

- `uv run pytest tests/unit/test_request_validation.py -q` reports:
  - `1 failed, 38 passed`
  - failing test: `test_prepare_request_discards_history_rewrite_that_breaks_pairing`
  - failure type: monkeypatched fake `compress_recent_window` does not accept the
    currently passed keyword argument `first_exact_evidence_seen`
  - this failure is outside the changes made in this Plan 13 pass and indicates a
    pre-existing test seam mismatch in current repo state.
