# Claude Live-Smoke Matrix

Last updated: **2026-04-08**.

## Automated Smokes

| Category                       | Automated Check                                                                            | Evidence Captured                                     | One-Line Pass Criterion                                              | Owning Boundary                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------ | ----------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------ |
| Basic non-streaming happy path | `tests/smoke/test_primary_non_streaming_smoke.py`                                          | endpoint, non-stream mode, upstream call count        | One request yields one upstream call and valid message response      | `src/tok/gateway/_bridge_request_handler.py`     |
| Streaming happy path           | `tests/smoke/test_primary_streaming_smoke.py`                                              | endpoint, stream mode, SSE events, call count         | Stream contains start/stop events and exactly one upstream execution | `src/tok/gateway/_bridge_streaming.py`           |
| Malformed request rejection    | Validation-failure step in `scripts/run_release_smoke.py`                                  | normalized error body, blocked upstream count         | malformed request fails locally with no upstream execution           | `src/tok/runtime/pipeline/request_validation.py` |
| Long-context path              | `tests/smoke/test_live_claude_smoke_matrix.py::test_live_smoke_long_context_path`          | endpoint, payload size, call count                    | long input reaches upstream once and returns success                 | `src/tok/runtime/_request_preparation.py`        |
| Tool-use path                  | `tests/smoke/test_live_claude_smoke_matrix.py::test_live_smoke_tool_use_path`              | endpoint, tool-use block shape, call count            | tool_use response survives bridge and executes once                  | `src/tok/runtime/pipeline/request_validation.py` |
| Repeated-call guard            | `tests/smoke/test_live_claude_smoke_matrix.py::test_live_smoke_repeated_call_guard`        | endpoint, per-request call-count delta                | two identical requests yield exactly two upstream calls              | `src/tok/gateway/_bridge_request_handler.py`     |
| Endpoint override path         | `tests/smoke/test_api_base_smoke.py`                                                       | selected endpoint marker, default endpoint call count | explicit `api_base` wins with zero fallback to default env endpoint  | `src/tok/gateway/__init__.py`                    |
| Cleanup / early-close path     | `tests/smoke/test_live_claude_smoke_matrix.py::test_live_smoke_stream_cleanup_early_close` | stream mode, early-close behavior, call count         | early client close does not create duplicate upstream execution      | `src/tok/gateway/_bridge_streaming.py`           |
| Import/install sanity          | Clean install/import step in `scripts/run_release_smoke.py`                                | fresh venv import contract check                      | installed wheel imports and release surface matches declaration      | `src/tok/__init__.py`                            |
| Release-surface drift sanity   | Release-surface drift step in `scripts/run_release_smoke.py`                               | exported names versus manifest                        | effective exports equal defended manifest with no experimental leaks | `src/tok/release_surface.py`                     |

## Manual Live Runs

Manual operator checks remain required for a real Claude session:

1. `tok install`
1. `tok bridge start`
1. run `claude` on a real task
1. `tok bridge status`
1. `tok doctor`
1. `tok stats`
1. `tok bridge stop`

Manual pass criterion: bridge-first workflow completes without hidden fallback or
wrong-target behavior.
