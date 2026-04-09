# Production Benchmark Catalog

This directory defines the checked-in pilot benchmark catalog for production Tok.

- Headline comparison is always `baseline` vs production Tok (`tok-universal`).
- `production_claude_lane` is the only release-relevant headline lane.
- Any `adapter_*` lanes are compatibility-only and must stay separate from headline
  claims.
- Legacy replay probes remain in `tests/fixtures/replay` and `tests/fixtures/stability`;
  they are not part of this catalog.

## Kickoff Workflow

Use `anthropic/claude-sonnet-4.6` as the initial benchmark model for the first
reportable production runs.

### 1. Preflight assets

Verify the checked-in asset pack before any reportable run:

```bash
uv run python scripts/prepare_benchmark_assets.py --root benchmarks verify
```

Reportable `execution_patch` tasks also require a private evaluator overlay. Use a
pinned tag or commit for hosted runs, not a moving branch ref such as `main` or `HEAD`.

### 2. Local shakedown smoke

Run the bounded side-by-side smoke slice locally first. The dated output layout below is
the expected kickoff shape for April 9, 2026:

```bash
uv run python scripts/run_release_smoke.py \
  --benchmark-mode smoke \
  --benchmark-output tmp/benchmark-smoke/20260409-local-smoke \
  --model anthropic/claude-sonnet-4.6 \
  --private-evaluator-root <abs-overlay-path>
```

This writes `legacy/`, `catalog/`, and `summary.md` under the benchmark output directory
and then runs `tok gate-check --benchmark-report` against the catalog report.

### 3. Hosted smoke and public sweep

Use the manual `benchmark-smoke.yml` workflow for the first official artifacts.

- Smoke inputs: `mode=smoke` `model=anthropic/claude-sonnet-4.6`
  `private-evaluator-ref=<pinned-tag-or-commit>`
- Public full inputs: `mode=public_full` `model=anthropic/claude-sonnet-4.6`
  `private-evaluator-ref=<same pinned ref>`

Treat `benchmark-smoke-smoke` as the first official kickoff artifact. After it passes,
run `public_full` as the first release-candidate baseline artifact.

For local model loops that mirror the older benchmark shell scripts, use:

```bash
uv run python scripts/run_benchmark_smoke_matrix.py \
  --mode smoke \
  --model deepseek/deepseek-v3.2
```

The matrix runner wraps `scripts/run_release_smoke.py`, auto-discovers a usable private
evaluator overlay when possible, and writes one dated output directory per model under
`tmp/benchmark_smoke_multimodel_<YYYYMMDD>/`.

### 4. Report review

Release gating comes from the `Public Production Lane` section in `catalog/report.json`
and `catalog/report.md` only.

- The headline lane must remain `production_claude_lane`.
- `consistency_gate_passed` and `public_claim_allowed` must both be `true`.
- Ignore `Supplemental Internal/Advisory Tasks` for release gating and public claims.

### 5. Supplemental diagnostics

Run internal execution and advisory real-session tasks separately from the public
benchmark bundle:

```bash
uv run tok dev live-benchmark \
  --program catalog \
  --mode compare \
  --model anthropic/claude-sonnet-4.6 \
  --output tmp/benchmark-smoke/20260409-local-supplemental \
  --include-advisory \
  --private-evaluator-root <abs-overlay-path> \
  --task exec.tok.bridge-canonicalization \
  --task exec.tok.first-exact-search \
  --task session.context-reacquisition.answer \
  --task session.response-contract.patch \
  --task session.fail-open.answer \
  --task session.answer-anchor.patch
```

Do not use supplemental artifacts for `tok gate-check --benchmark-report` or for
headline/public claims.
