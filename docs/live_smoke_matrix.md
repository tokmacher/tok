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

1. `tok claude` on a real task
1. `tok bridge status`
1. `tok doctor`
1. `tok stats`
1. exit Claude, then `tok bridge stop` (or `tok bridge stop --force` if stopping
   in-band)

Manual pass criterion: `tok claude` workflow completes without hidden fallback or
wrong-target behavior.

## Manual Claude Code Prompt Suite

Run these prompts in one fresh `tok claude` session before cutting a release candidate.
Use a disposable branch or scratch copy when a prompt asks Claude Code to inspect or
plan changes. Do not ask Claude Code to commit, push, tag, publish, or edit release
artifacts.

Before starting:

```bash
tok bridge stop --force || true
tok claude
```

After each prompt, run in a separate shell:

```bash
tok bridge status
tok doctor
tok stats
```

Record whether the bridge is running, whether the session degraded to baseline, fallback
count, and whether savings are measured or only expected. A short early session may show
no savings; do not claim savings until `tok stats` reports them.

| Order | Prompt                                                                                                                                                                                                                                                                                                                      | What It Exercises                                             | Pass Criterion                                                                                                               |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 1     | `Read AGENTS.md, README.md, and docs/cli-reference.md. Summarize the supported 0.2.0 user path in 8 bullets. Do not edit files.`                                                                                                                                                                                            | broad reads, bridge-first claims, docs exactness              | Mentions local Claude Code bridge, `tok claude`, diagnostics, and local resolver beta without stable/global protocol claims. |
| 2     | `Inspect src/tok/resolver/manifest.py, src/tok/resolver/store.py, and docs/spec/tok_resolver_manifest_v0_2.md. Tell me the local resolver invariants and any release risks. Do not edit files.`                                                                                                                             | exact resolver reads, content-addressed store reasoning       | Calls out local-only, SHA-256, no remote routing/referral following, and manifest/store consistency.                         |
| 3     | `Use the CLI help only: run tok resolver --help, tok resolver status --help, tok resolver put --help, and tok resolver get --help. Report the visible resolver surface. Do not edit files.`                                                                                                                                 | CLI-facing usability, tool execution, help output compression | Reports only `init`, `status`, `store`, `put`, `get`; does not invent network commands.                                      |
| 4     | `Create a temporary file under tmp/live-resolver-smoke.txt, store it with tok resolver put, fetch it with tok resolver get --out tmp/live-resolver-smoke.out, compare bytes, then delete only those two temp files. Report exact commands and results.`                                                                     | resolver put/get, exact recovery, safe reversible writes      | Bytes match; only temp files are touched; no remote/network resolver behavior appears.                                       |
| 5     | `Audit docs/spec/fixtures/trace_fixtures.json with tok audit. Then explain what tok audit proves and what it does not prove for 0.2.0. Do not edit files.`                                                                                                                                                                  | audit semantics, forbidden compliance claims                  | Describes fixture/live trace validation only; does not claim protocol compliance or production readiness.                    |
| 6     | `Find all docs that mention remote resolver routing, referral following, capability negotiation, or agent-to-agent exchange. Report whether each is marked deferred for 0.2.0. Do not edit files.`                                                                                                                          | forbidden-claim search, release claim safety                  | Every 0.2.0 mention is deferred/non-goal; no stable/global protocol claim is introduced.                                     |
| 7     | `Review src/tok/runtime/_request_preparation.py and src/tok/compression/_feature_flags.py for risky irreversible-action behavior. Tell me whether exact reads or resolver-backed content are required before risky actions. Do not edit files.`                                                                             | safety philosophy, lossy-summary guardrails                   | Does not recommend acting on lossy summaries; flags any unclear exactness boundary.                                          |
| 8     | `Run uv run pytest tests/unit/test_resolver_manifest.py tests/unit/test_resolver_store.py tests/unit/test_resolver_cli.py tests/spec/test_resolver_backed_audit.py -q and summarize pass/fail with any traceback. Do not edit files.`                                                                                       | focused regression gate under Claude Code                     | Tests pass, or failures are reported with exact traceback and no “works” claim.                                              |
| 9     | `Pretend you are preparing release notes. Draft only the 0.2.0 claims that are supported by this repo. Do not mention stable protocol, hosted service, global resolver, remote routing, referral following, agent-to-agent exchange, capability negotiation, session state roots, or stable Python SDK. Do not edit files.` | claim discipline, user-facing language                        | Draft stays narrow: local resolver beta, content-addressed local store, audit/reader proof, bridge-first workflow.           |
| 10    | `Final check: run tok bridge status, tok doctor, and tok stats. Report bridge running state, fallback count, degraded/baseline state, and whether savings were measured. Do not edit files.`                                                                                                                                | end-of-session diagnostics and reporting discipline           | Report includes exact commands, pass/fail status, baseline/degraded state, and savings evidence status.                      |

End the session with:

```bash
tok bridge stop
```

The prompt suite passes only if the session stays on the local bridge path, avoids
forbidden 0.2.0 claims, handles resolver content by exact bytes, and reports diagnostics
with the required caveats.
