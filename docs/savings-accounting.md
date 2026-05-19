# Tok Savings Accounting

## Goal

Define what “savings” means in Tok, which parts are measured vs estimated, and which
numbers are allowed in headline metrics.

## Core rule (headline meaning)

Headline Tok savings should mean Tok-attributable reduction in input tokens:

```
input_tokens_saved = max(baseline_input_tokens - actual_input_tokens, 0)
```

Today, `baseline_input_tokens` is not measured independently. It is derived as:

```
baseline_input_tokens = actual_input_tokens + input_tokens_saved_by_tok
```

That makes headline “baseline” a derived counter, not a ground truth replay.

## Core terms

| Term                        | Meaning                                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| `actual_input_tokens`       | Provider-reported input tokens for the request.                                                           |
| `input_tokens_saved_by_tok` | Tok’s measured (or estimated) reduction in the request input.                                             |
| `baseline_input_tokens`     | Derived as `actual_input_tokens + input_tokens_saved_by_tok`.                                             |
| `actual_cost_usd`           | Cost computed from actual tokens, using per-token rates (including cache rates).                          |
| `baseline_cost_usd`         | Cost computed from baseline tokens, using the same cache rates so cache discounts do not inflate savings. |
| `cost_saved_usd`            | `baseline_cost_usd - actual_cost_usd`.                                                                    |

## Headline vs attribution

Tok currently reports a session-level `tokens_saved` headline number that mixes:

- measured input token savings (often derived from character deltas)
- measured output token savings (when applicable)
- some estimates (for reacquisition avoidance and hot-hint cost offsets)

This file treats that as the “current model”, and also sets a “desired model”.

### Headline (current model)

Included in `tokens_saved` today:

- `input_saved_tokens`
- `output_saved_tokens`
- `reacquisition_tokens_avoided_estimate` (estimate)
- minus `hot_hint_tokens_added` (estimate offset)

### Attribution-only

Tracked as signals but not included in headline `tokens_saved`:

- `macro_savings_attributed` (estimate)

### Not Tok savings

Not allowed in Tok headline savings:

- Provider prompt cache discounts (they use their own cache rates in both actual and
  baseline cost).

## Required invariants (what should always hold)

- `actual_*_tokens >= 0`
- `baseline_*_tokens >= 0`
- `tokens_saved >= 0`
- Cache discount does not inflate Tok headline savings.
- A fail-open / fallback request should report zero Tok headline savings for that
  request.

## Savings path classification (current implementation shape)

### Direct measurement (character delta -> estimated tokens)

Many per-type savings counters are derived from character deltas using an approximation
like `chars // 4`, not true token counting.

### Direct measurement (token counting)

Some counters (like prompt token accounting) can be based on token counting depending on
where the measurement is taken.

### Estimates (need to be called out)

- `reacquisition_tokens_avoided_estimate`
- `hot_hint_tokens_added`
- `macro_savings_attributed`

## Desired future ledger event shape (for reconstructability)

Tok does not currently persist per-request durable records. The target is an append-only
per-request event stream (for example JSONL) with a schema like:

```json
{
  "schema": "tok-savings-event/v1",
  "event_id": "uuid4",
  "session_id": "hash8",
  "request_id": "turn-N-model",
  "timestamp": "2026-05-13T12:00:00Z",
  "model": "claude-sonnet-4",
  "mode": "tool-compatible",
  "request_policy": "natural_first",
  "baseline_input_tokens": 5000,
  "actual_input_tokens": 3500,
  "input_tokens_saved": 1500,
  "baseline_output_tokens": 500,
  "actual_output_tokens": 450,
  "output_tokens_saved": 50,
  "cache_read_tokens": 2000,
  "cache_write_tokens": 100,
  "baseline_cost_usd": 0.025,
  "actual_cost_usd": 0.020,
  "cost_saved_usd": 0.005,
  "fallback": false,
  "degraded_to_baseline": false,
  "compression_paths": {
    "file": 800,
    "semantic_dedup": 400,
    "saved_prompt_tokens": 300
  },
  "non_headline_estimates": {
    "reacquisition_tokens_avoided_estimate": 200,
    "hot_hint_tokens_added": 50,
    "macro_savings_attributed": 100
  }
}
```
