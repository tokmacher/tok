# Tok Compression Benchmark (Experimental)

> Last run: 2026-05-24 | Python 3.11.5 | Status: experimental Fixtures ship with the
> package. Reproduce from any install: `tok benchmark run && tok benchmark report` Note:
> fixtures are designed by the Tok maintainers — treat results as internal regression
> data, not independent proof.

## Aggregate Results

| Metric                            | Value |
| --------------------------------- | ----- |
| Median token savings              | 23.7% |
| P90 token savings                 | 73.2% |
| Total tokens saved (all fixtures) | 5,150 |
| Average fallback rate             | 37.3% |
| Fixtures run                      | 41    |

## Per-Fixture Results

| Fixture                      | Turns | Saved % | Fallback Rate | Top Strategy                |
| ---------------------------- | ----- | ------- | ------------- | --------------------------- |
| alternating_adapters         | 1     | 26.9%   | 0.0%          | —                           |
| branching_tests              | 1     | 0.0%    | 0.0%          | —                           |
| bridge_vs_orchestrator       | 1     | 48.2%   | 0.0%          | —                           |
| burst_retries                | 1     | 19.5%   | 0.0%          | —                           |
| cache_sensitivity            | 1     | 13.9%   | 0.0%          | —                           |
| cache_stable_research_turns  | 1     | 43.9%   | 0.0%          | —                           |
| claude_coding_loop           | 1     | 0.0%    | 0.0%          | —                           |
| comprehensive_test           | 5     | 10.4%   | 0.0%          | —                           |
| compression_hypothesis_churn | 1     | 40.4%   | 0.0%          | —                           |
| context_pinned_file          | 1     | 93.1%   | 0.0%          | —                           |
| dedup_opportunity_corpus     | 4     | 0.0%    | 0.0%          | command_cached              |
| episodes_multi_phase         | 1     | 20.4%   | 100.0%        | —                           |
| file_heavy_operations        | 2     | 44.7%   | 0.0%          | cache_stored                |
| gemini_coding_loop           | 1     | 45.3%   | 0.0%          | —                           |
| gpt_coding_loop              | 1     | 37.2%   | 0.0%          | —                           |
| grammar_drift                | 4     | 2.9%    | 100.0%        | output_minimalist           |
| healing_drift                | 2     | 0.0%    | 100.0%        | —                           |
| heavy_tool_event             | 1     | 40.9%   | 0.0%          | —                           |
| high_pressure_scenario       | 2     | 13.4%   | 0.0%          | —                           |
| jit_loop                     | 1     | 2.6%    | 0.0%          | cache_stored                |
| long_coding_session          | 3     | 23.7%   | 66.7%         | cache_stored                |
| markdown_fallback            | 1     | 0.0%    | 100.0%        | —                           |
| metric_long_debug            | 1     | 75.1%   | 0.0%          | pytest_diff                 |
| multi_model_session          | 2     | 59.8%   | 100.0%        | —                           |
| neuro_loop                   | 1     | 6.9%    | 0.0%          | cache_stored                |
| orchestrator_parity          | 1     | 30.5%   | 0.0%          | —                           |
| pressure_session             | 1     | 31.8%   | 0.0%          | cache_hit                   |
| refined_search_recovery      | 1     | 5.1%    | 0.0%          | —                           |
| release_reacquisition        | 1     | 60.0%   | 0.0%          | search_overlap_delta        |
| repeat_search_pressure       | 1     | 24.8%   | 0.0%          | raw_cached                  |
| research_loop                | 1     | 20.8%   | 0.0%          | —                           |
| research_loop_extended       | 1     | 76.8%   | 0.0%          | cache_stored                |
| runtime_conformance          | 1     | 37.4%   | 0.0%          | —                           |
| search_intensive_workflow    | 2     | 28.8%   | 0.0%          | cache_stored                |
| straddling_boundary          | 1     | 7.3%    | 0.0%          | —                           |
| subtle_drift                 | 2     | 0.0%    | 100.0%        | —                           |
| test_cli_fixture             | 2     | 19.4%   | 0.0%          | —                           |
| test_coding_fixture          | 3     | 14.2%   | 0.0%          | —                           |
| test_search_fixture          | 6     | 15.2%   | 0.0%          | cache_stored                |
| tool_density_micro           | 1     | 73.2%   | 100.0%        | command_cache_reached_apply |
| verbose_payload              | 1     | 73.5%   | 0.0%          | —                           |

## Methodology

- Fixtures: deterministic JSONL replay captures bundled at `tok/bench_fixtures/replay/`
- Token counting: tiktoken cl100k_base (fallback: chars/4)
- Compression pipeline: `tok.utils.replay_metrics.analyze_replay_fixture()`
- Fallback rate: `min(1, (non_tok_response + fail_open_compat_response) / turns)` —
  clamped to 1.0
- Note: some fixtures are specifically designed to test fallback behavior, so high
  fallback rates are expected for those
- Default results file: user cache path `tok/benchmark_latest.json`
