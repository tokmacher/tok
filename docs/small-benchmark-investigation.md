# Small Patch Benchmark Investigation Prompt

## Pre-Run Investigation Checklist

### 1. Task Structure Analysis

- **Tasks**: invoice.totals, text.pipeline, auth.rules
- **step_budget**: 30 (vs tiny's 20)
- **Expected turns**: 15-25 (vs tiny's ~10)

### 2. Known Issues from Tiny Benchmark (RCA)

| Issue                         | Root Cause                          | Impact                |
| ----------------------------- | ----------------------------------- | --------------------- |
| Premature finals              | Hint cap (1 hint when constrained)  | 6 early exits in tiny |
| Behavior inflation            | More turns/tool calls than baseline | +14% token cost       |
| File codec missing (deepseek) | Different model behavior vs OpenAI  | Lost 170k savings     |

### 3. Questions to Answer

#### For each task, examine:

1. **Tool usage pattern**: Does task use restricted tools (`view_file`, `grep_search`,
   `edit_file`, `run_tests`) or more complex tool chains?
1. **File read density**: How many file reads per turn? (File codec needs density to
   work)
1. **Session length**: Actual turns taken vs step_budget (30)
1. **Compression triggers**: Which codecs activate?
   - `file`: File read compression
   - `pytest`: Test failure compression
   - `semantic_dedup`: Repeated content dedup
   - `search_repeat_cached`: Search result caching

#### Expected vs Actual:

- Baseline: ~6-10 tool calls (from old run)
- Tok: currently +5 tool calls (11 vs 6)
- Is the extra tool call pattern continuing?

### 4. Key Metrics to Watch

```
fairness_diagnostics:
  - premature_final_counts: should be 0 (fixed from tiny's 6)
  - constrained_tool_profile_active: expected > 0 (benchmark restricts tools)
  - tool_required_latch_active: expected 0

comparators[tok-universal]:
  - token_delta_total_median: negative = saving, positive = cost
  - compression_recovery_estimate_total: codec savings
  - behavior_inflation_estimate_total: extra cost from behavior changes
  - codec_family_saved_tokens: breakdown by codec type
```

### 5. Hypotheses to Validate

1. **H1**: Small should show better savings than tiny (longer sessions = more
   compression opportunity)
1. **H2**: Deepseek still won't trigger file codec (model-specific behavior)
1. **H3**: Premature finals should be 0 (hint cap fix applied)
1. **H4**: Behavior inflation persists but smaller magnitude than tiny

## Post-Run Analysis

### If Token Delta is Positive (Cost):

1. Check `behavior_inflation_estimate_total` vs `compression_recovery_estimate_total`
1. Which specific behavior counters are elevated?
1. Is `constrained_tool_profile_active` high?
1. Compare turn counts: baseline vs tok

### If Token Delta is Negative (Savings):

1. Which codecs are providing savings?
1. Is it sustainable or driven by outliers?
1. What's the p90 delta (stability)?

### Red Flags:

- `premature_final_counts` > 0 (regression)
- `completion_success_rate` dropped vs baseline
- `tool_call_delta` > 5 extra calls
- `codec_family_saved_tokens` shows 0 for major codecs

## Command to Run

```bash
python -m tok dev patch-benchmark --size small --model deepseek/deepseek-v3.2 --repeats 5
```
