# Savings Metrics Audit (Tok 0.2.x)

## Goal

Audit Tok savings metrics for:

- consistency across CLI and health surfaces
- clarity on what is measured vs estimated
- reconstructability from durable records

This audit is about the current code on this branch, not a product promise.

## Baseline (2026-05-13)

### Exact commands run

```
git branch --show-current
git status
git log --oneline -5
git diff --stat develop...HEAD

UV_CACHE_DIR=/private/tmp/uv-cache uv sync --frozen --extra dev

UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/unit/test_stats.py tests/unit/test_json_diagnostics.py tests/unit/test_diagnostics_snapshot.py -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/unit/test_agent_operable_regression.py -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff check src/tok tests
UV_CACHE_DIR=/private/tmp/uv-cache uv run mypy src/tok
UV_CACHE_DIR=/private/tmp/uv-cache uv build

UV_CACHE_DIR=/private/tmp/uv-cache uv run tok stats --json 2>/dev/null | python -m json.tool
UV_CACHE_DIR=/private/tmp/uv-cache uv run tok doctor --json 2>/dev/null | python -m json.tool
```

### Outputs (high signal)

- `git branch --show-current` -> `0.2.0-implementation-branch`
- `git status` -> working tree clean
- `git log --oneline -5` (top two)
  - `721b6aa feat: enhance stats command with detailed output and add tests for new functionality`
  - `d94b700 feat: introduce local resolver beta commands and update documentation for Tok 0.2.x`
- `uv sync --frozen --extra dev` -> pass
- `pytest (stats/json_diagnostics/diagnostics_snapshot)` -> `73 passed`
- `pytest (agent_operable_regression)` -> `46 passed`
- `ruff check` -> pass
- `mypy` -> `Success: no issues found in 194 source files`
- `uv build` -> pass (required network access to fetch build requirements)

### Baseline JSON surfaces

`tok stats --json` returned `ok: true` and reported `bridge_running: true` with a
populated session + lifetime section.

`tok doctor --json` returned `ok: false` while still reporting `bridge_running: true`,
with `health_reachable: false` and warning
`Unable to reach health endpoint: ConnectError`.

This confirms a real disagreement axis: “PID alive” vs “health reachable”.

## Current savings surfaces

- `/health` (bridge HTTP) -> health payload based on a diagnostics snapshot
- `tok bridge status` / `tok bridge status --json` -> reads health payload only
- `tok stats` / `tok stats --json` -> reads local stats on disk, then may overlay with
  health payload when bridge is running
- `tok doctor` / `tok doctor --json` -> reads local stats + health payload and merges
  them

## Metric field inventory (key mapping)

Field names differ between surfaces, even when they refer to the same idea.

| Concept      | `session_summary` key | `/health` key          | `tok stats --json` key        | `tok bridge status --json` key |
| ------------ | --------------------- | ---------------------- | ----------------------------- | ------------------------------ |
| tokens saved | `tokens_saved`        | `session_tokens_saved` | `data.session.tokens_saved`   | `data.tokens_saved`            |
| savings pct  | `savings_pct`         | `session_savings_pct`  | `data.session.savings_pct`    | `data.savings_pct`             |
| cost saved   | `cost_saved_usd`      | `cost_saved_usd`       | `data.session.cost_saved_usd` | `data.cost_saved_usd`          |

## Savings computation findings (headline meaning)

These are the main “shape” findings about the numbers:

- `saved_tokens` headline includes estimate terms (reacquisition avoidance and hot-hint
  offsets), not just measured deltas.
- `baseline_input_tokens` is derived from `actual_input_tokens + input_saved_tokens`,
  not independently measured.
- Many type-level savings counters use a character-based approximation (`chars // 4`)
  rather than token counting.
- Provider cache is correctly excluded from headline savings (cost uses cache rates in
  both actual and baseline, so discount does not show up as “Tok savings”).

## Consistency findings (risks)

- RISK-001 (MEDIUM): `tok stats` and `tok bridge status` can disagree while bridge is
  running because stats may overlay from health data and bridge status uses health data
  only.
- RISK-002 (LOW): `tok doctor` merges local + health dictionaries, which can produce a
  hybrid view if one source is missing fields.
- RISK-003 (MEDIUM): `reacquisition_tokens_avoided_estimate` is an estimate but included
  in headline `tokens_saved`.
- RISK-004 (LOW): type breakdown savings use `chars // 4` approximation.
- RISK-005 (HIGH): savings are not reconstructable per request from durable records
  (per-request events are not persisted).

## Persistence and reconstructability

- Durable today:
  - session aggregates (stats file)
  - lifetime ledger aggregates + per-session log rows
- Not durable today:
  - per-request savings events

Result:

- Session totals can be reconstructed from the session stats file.
- Lifetime totals can be reconstructed from the lifetime ledger + per-session log.
- Individual request savings cannot be reconstructed after the fact.

## Recommended next packet

Start with:

1. A canonical per-request savings event schema.
1. An append-only per-request ledger (JSONL).

See `docs/plans/0.2.0/savings-ledger-hardening-plan.md` for a staged plan.

## Final report (current state)

- Runtime code changes in this audit: **No** (one test-only hardening fix so the
  baseline suite passes reliably when a bridge PID exists but health is unreachable).
- Provider cache included in Tok headline savings: **No** (cache is costed at cache
  rates in both actual and baseline).
- Savings reconstructable from durable records: **Partially** (session + lifetime yes;
  per-request no).
