# Architecture

For visual flow diagrams, see [architecture-diagrams.md](./architecture-diagrams.md). For the active roadmap and tranche sequencing, see `roadmap.md` in the repository root.

Current architecture posture:

- near-term product: invisible bridge-first runtime for user-to-agent workflows
- next release target: a narrow public release of that bridge-first path
- later-stage ambition: reliable agent-to-agent handoff and a broader runtime contract

## Runtime Topology

```
┌──────────────────┐
│ Universal Runtime│
│ request shaping  │
│ response shaping │
│ memory projection│
│ tool ontology    │
│ replay + gating  │
│ telemetry        │
└────────┬─────────┘
         │
 ┌───────┼───────────────┬───────────────────┬─────────────────┐
 │       │               │                   │                 │
 ▼       ▼               ▼                   ▼                 ▼
Claude bridge      OpenAI chat        Text loop         Orchestrator
adapter            adapter            adapter           adapter boundary
```

The runtime is the only canonical execution path. Adapters own transport mapping and
session mechanics only.

That does not mean every adapter is equally productized today. The Claude bridge is the
current product surface; broader cross-surface runtime expansion remains later-stage
work until parity and trust are stronger.

## Primary Bridge Flow

The Claude bridge remains the primary acceptance surface.

### Request Path

1. Claude Code sends a `/v1/messages` POST to `localhost:9090`
2. The bridge adapter forwards the normalized request into `UniversalTokRuntime`
3. If history exceeds `keep_turns` human turns:
   - Older messages are compressed into a `>>>` rolling state line using the live bridge memory schema
   - Only the last `keep_turns` human turns are kept verbatim
   - Tool-use/tool-result pairs are never split
4. The runtime injects the Tok output directive and projected memory
5. Compressed request is forwarded to `api.anthropic.com`

### Response Path

1. Anthropic returns a response (streaming SSE or non-streaming JSON)
2. For streaming: the full SSE stream is buffered, then the bridge adapter passes the accumulated text to the runtime
3. The runtime detects if the response is in Tok grammar:
   - **Tok mode**: `@thought` blocks are stripped, `|>` content from `@msg role:assistant` is extracted
   - **Markdown fallback**: Headers, bold, italic, horizontal rules are stripped
4. The bridge re-emits the processed response to Claude Code
5. Memory, family-mode state, savings, and invisible-pressure signals are recorded through shared runtime logic

### Fail-Open Behavior

If any error occurs during compression or translation, the bridge transparently passes the raw request/response through. The user never sees a Tok-related error; worst case is higher token cost for that request.

## Runtime Contract

### Canonical IDL Ownership

For `0.1.0`, Tok should be described as having one canonical protocol IDL and two
derived translation layers:

- Canonical protocol IDL: `src/tok/protocol/schema.py` and `src/tok/protocol/models.py`
- Derived runtime tool-input contract: `TOOL_SCHEMAS` in `src/tok/protocol/models.py`, enforced by `RuntimeToolExecutor._compiler_guard()` in `src/tok/runtime/tools.py`
- Bridge / wire adaptation: request shaping in `src/tok/runtime/pipeline/request_preparation.py` and transport canonicalization in `src/tok/runtime/pipeline/request_validation.py`

Top-level exports in `src/tok/__init__.py` are convenience re-exports, not an
independent source of truth. Release audits should score schema drift against the
protocol layer first, then classify any runtime-tool or bridge-wire divergence as
derived-contract drift rather than a competing second IDL.

### Wire Memory Schema

- The working-memory state is a sparse `>>>` line emitted in canonical order: `turns`, `goal`, `files`, `cmds`, `tests`, `errs`, `constraints`, `next`.
- Structured bridge memory is authoritative when present; `memory.tok` is a compatibility fallback only.
- Projection remains bounded and deterministic—fields are emitted only when populated, but ordering never changes.

### Request Preparation Rules

1. Keep the last `keep_turns` human turns verbatim; everything older compresses into the state line via `BridgeMemoryState`.
2. Tool-use/tool-result pairs must remain intact (never split across compression boundaries).
3. Tool density determines whether history rewrite is skipped to preserve fidelity on heavy tool sessions.
4. The runtime keeps semantic deduplication, tool-result compression, and history winnowing active by default, then escalates into tool-compatible shaping only when the session is recovery-sensitive.

### Response Classification

- **Tok-native success:** response uses Tok markers, has a visible `@msg role:assistant` block, and yields readable assistant/tool content without markdown fallback.
- **Model-Agnostic Leniency:** The parser supports both `:` and `=` as attribute separators. It also recognizes hybrid `@Tool name {json}` blocks with unquoted keys and TitleCase normalization (e.g., `@Tool ReadFile` maps to `read`).
- **Tool-compatible mode:** plain text is accepted when the upstream request declared tool compatibility. Telemetry records `tool_compatible_response` instead of `non_tok_response`.
- **Fail-open compatibility:** malformed or markdown-only responses pass through but increment `fail_open_compat_response` and `non_tok_response` signals.
- **Malformed enforcement:** hybrid tool JSON, non-inverted assistant blocks, or bad headers increment the specific `malformed_tok_*` signals and still fail to count as Tok-native success.

### Inversion & Memory Guarantees

- Cold starts prefer structured memory; wire fallback is only used when structured memory is empty.
- Memory ingestion always writes the latest `>>>` line to both structured state and fallback files, keeping bridge replays consistent.
- File/search snapshots are recorded through runtime helpers and surfaced to telemetry as `file_snapshot_recorded` / `search_snapshot_recorded` signals.

### Telemetry & Conformance Signals

- Behavior signals cover `tok_native_response`, `non_tok_response`, `fail_open_compat_response`, `malformed_tok_*`, cold-start metrics, invisible pressure, and mutation detection.
- `tok doctor` and CI gates consume these signals; any regression of the contract is treated as a release blocker per the roadmap.

## Module Layout

```
src/tok/
├── universal_runtime.py          # Canonical runtime facade
├── protocol/
│   ├── schema.py                 # Canonical protocol schema registry
│   ├── models.py                 # Canonical protocol AST/data model
│   ├── parser.py                 # TokParser
│   ├── encoder.py                # TokEncoder
│   └── format_bridge.py          # Explicit Tok language/conversion layer
├── runtime/
│   ├── core.py                   # Shared runtime implementation
│   ├── tools.py                  # Derived runtime tool-input validation/execution
│   └── pipeline/request_validation.py  # Bridge request canonicalization
├── gateway/__init__.py           # Primary Claude bridge adapter
├── cli/__init__.py               # CLI and replay entry points
├── stats.py                      # Shared telemetry and savings ledger
└── compression/__init__.py       # Request compression primitives
```

## Repo Classification

- Canonical runtime: `universal_runtime.py`, `runtime/core.py`
- Primary adapter: `gateway/__init__.py`
- Secondary adapters: `adapters.py`, `live_runner.py`, `agent.py`, partial `tok_orchestrator.py`
- Canonical protocol IDL: `protocol/schema.py`, `protocol/models.py`
- Derived runtime contract: `runtime/tools.py`
- Bridge transport adaptation: `runtime/pipeline/request_preparation.py`, `runtime/pipeline/request_validation.py`
- Experimental / legacy: deeper orchestrator internals and archive material

This means Tok should currently be described as bridge-first infrastructure, not as a
fully generalized protocol product.

## Governing Metric

The governing metric is minimum token spend per successful reasoning step.

Success requires:

- lower token drag
- less repeated context acquisition
- high-fidelity working memory
- low invisible pressure

### Dual-Axis Evaluation

Runtime health is evaluated on two axes that must move together:

| Axis | Metric | Direction |
|------|--------|-----------|
| Efficiency | tokens per successful step | ↓ lower is better |
| Depth | `reasoning_depth_per_token` = (steps × tool_diversity) / tokens | ↑ higher is better |

If token savings increase while `reasoning_depth_per_token` decreases, the compression is over-aggressive and the change should be treated as a regression.

`semantic_regression_score` (sum of non-Tok responses, fail-open churn, repeat reads, blocker rediscovery) is the third axis that guards against protocol drift silently growing behind improving efficiency numbers.

### Episode Ledger

`EpisodeLedger` (in `universal_runtime.py`) stores a rolling window of completed reasoning episodes. Each episode captures:

- `goal` — the active objective when the episode started
- `outcome` — success / failure / partial / open
- `learnings` — one-line causal summary (e.g. "test X failed because missing import Y; fixed by adding Z")
- `artifacts` — key files or commands touched

The ledger is projected into the working-memory state line to prevent the model from re-opening solved subproblems across session boundaries.

## Compression Algorithm

The input compression (`compress_history`) works by:

1. Walking backwards through the message list
2. Counting human turns at safe-cut boundaries (messages without `tool_result` blocks)
3. After `keep_turns` human turns, everything before that point becomes "old"
4. Old messages are summarized into bounded working-memory fields:
   - `turns`: compressed user-turn count
   - `goal`: active objective
   - `files`: hottest referenced code/doc files
   - `cmds`: recent high-signal commands
   - `tests`: failing/passing test evidence
   - `errs`: recent error signals
   - `constraints`: user instructions that must persist
   - `next`: near-term intended action when available
5. The summary is encoded as a `>>>` state line appended to the injected runtime directive

This produces O(1) context growth — the state line is a fixed size regardless of conversation length.
