# Discovery-Safe Deduplication Benchmark

This document defines the benchmark specification and "first exact observation" contract for evaluating discovery-safe deduplication improvements.

## Purpose

Establish a fixed before/after benchmark so that discovery improvements are measurable and the "first exact observation" contract is clearly defined.

______________________________________________________________________

## Benchmark Task A: Streaming & API Infrastructure

### Task A.1: Find streaming client lifetime ownership

Locate where streaming client lifetime is managed, including:

- Client initialization and cleanup
- Connection pooling boundaries
- Resource ownership transfer

### Task A.2: Find `--api-base` plumbing

Trace the `--api-base` parameter through the system:

- CLI argument parsing
- Configuration propagation
- Adapter/transport layer consumption

### Task A.3: Find real `aiter_bytes()` coverage

Identify actual test coverage for `aiter_bytes()`:

- Unit test locations
- Integration test scenarios
- Mock vs. real implementation testing

______________________________________________________________________

## Benchmark Task B: Assistant & Validation Logic

### Task B.1: Find latest-assistant thinking preservation

Locate the logic that preserves assistant thinking/reasoning across:

- Message boundary handling
- Context window management
- State serialization/deserialization

### Task B.2: Find system-block validation boundary

Identify where system-block validation occurs:

- Input validation layer
- Security boundary checks
- Content policy enforcement points

### Task B.3: Find release-surface enforcement

Locate release surface controls:

- Feature flag evaluation
- Version-gated functionality
- Deployment stage restrictions

______________________________________________________________________

## Metrics

The following metrics must be collected for each benchmark run:

| Metric                       | Description                                                         |
| ---------------------------- | ------------------------------------------------------------------- |
| `time-to-first-correct-file` | Time (or turns) until the first relevant file is identified         |
| `total-turns`                | Total number of tool calls / interactions to complete task          |
| `repeated-searches`          | Number of times the same search query is issued                     |
| `repeated-file-rereads`      | Number of times a file is read again with same parameters           |
| `repeated-offset-reads`      | Number of times the same file offset range is re-read               |
| `manual-bypass-count`        | Number of times the user had to manually correct or guide discovery |

______________________________________________________________________

## First Exact Observation Contract

### Definition

A **first exact observation** is the initial retrieval of specific content from the codebase using exact (non-summarized) retrieval methods.

### Observation Types

#### File Read

- First `read_file` call for a specific absolute path
- Content retrieved with exact line numbers and full file contents
- Not a summary, not a cached reference, not a user-provided excerpt

#### Symbol Body Retrieval

- First time a function/method/class body is retrieved in full
- Includes complete implementation, not just signature
- May be via `read_file` with offset or via semantic symbol retrieval

#### Grep/Search Result Set

- First `grep_search` or `code_search` returning matches for a query
- Complete result set (up to tool limits), not a pre-filtered summary
- Original file paths and line numbers preserved

#### Directory Slice

- First `list_dir` or directory enumeration for a path
- Complete listing (within reasonable limits), not a curated subset
- Original file names and structure preserved

### Contract Rules

#### Rule 1: First Exact Observation Must Not Be Replaced by Summary-Only Substitution

When a first exact observation occurs:

- The full, exact content must be preserved in context
- No summarization or abstraction may replace the actual content
- The model must have direct access to the literal text/structure discovered

This ensures:

- Ground truth is established for subsequent reasoning
- The model can verify claims against original source
- No information loss occurs at discovery boundary

#### Rule 2: First-Pass Search Must Return Evidence Sufficient for Grounding

When a discovery search is issued on first pass:

- First-pass discovery search on the live discovery surface should return line-level evidence with surrounding context by default
- Path-only search output is valid only for explicitly navigational searches and is not sufficient first-pass exact evidence for content discovery
- Repeat compression is only allowed after line-level or equivalent exact evidence has been seen in-session

This ensures:

- The model can reason directly from returned evidence without requiring immediate re-queries
- File paths alone do not constitute "exact observation" for code content discovery
- Content discovery requires actual content (lines, context, matches) not just location hints

#### Rule 3: Unified First-Observation Rule with Evidence-Specific Identity

No cache or stabilization layer may replace the first exact observation in-session. This applies globally across:

- **Stable-result paths**: Result stabilization must not summarize on first observation
- **Hot-recent search paths**: Hot-recent cache must not substitute first exact observation with stubs or summaries
- **Shared compressor paths**: Compression layers must not affect first exact observations
- **Discovery-oriented wrappers**: Any wrapper over grep/search/listing outputs (e.g., exploration mode, discovery helpers) must preserve first exact observation

**Evidence-Specific Identity Keys:**

| Evidence Type      | Identity Key                                      |
| ------------------ | ------------------------------------------------- |
| File reads         | Canonical absolute path                           |
| Directory listings | Canonical path + listing mode (recursive vs flat) |
| Grep/Search        | Normalized query identity + scope/flags           |

#### Rule 4: Repeat Deduplication After Exact Evidence Established

Once first exact evidence has been established in-session, repeated identical observations may still compress:

- Subsequent identical retrievals may deduplicate to stable summaries
- The same content accessed via the same evidence identity may use compressed representation
- Cross-session observations follow normal caching policies

______________________________________________________________________

## Running the Benchmark

1. Reset context window (fresh session)
1. Present benchmark task description
1. Record all tool invocations and their results
1. Compute metrics at task completion
1. Verify contract compliance via audit log

______________________________________________________________________

## Phase 3: Contradictory Live Smoke Test

### Observed Failure

During latest benchmark run, the first-pass search behavior conflicted with previous smoke test expectations:

| Failure Mode                    | Description                                      |
| ------------------------------- | ------------------------------------------------ |
| Path-only first search          | First search returned only file paths            |
| Missing matching lines          | No matching lines were visible in initial result |
| Insufficient grounding evidence | Content required for reasoning was absent        |

### Later Improvement

Subsequent search behavior showed correct patterns:

| Improvement Mode          | Description                                         |
| ------------------------- | --------------------------------------------------- |
| Content+context on repeat | Subsequent search returned exact lines with context |
| Targeted regex worked     | Regex-like targeted search operated correctly       |
| Clean offset reads        | Offset-based file reads functioned properly         |

### Classification

The audit classifies first-pass path-only `Search(pattern: ...)` results on discovery-oriented surfaces as a **behavioral defect**, not the intended default.

- **First pass**: discovery search should surface line-level evidence with surrounding context first.
- **Upgrade trigger**: later repeats may compress once exact evidence is already in-session.
- **Root cause**: wrapper / tool-selection behavior and search-result shaping, not cache warming or repeat dedup alone.
- **Semantics split**: treat path-only output as navigational-only; treat content+context as the discovery default.

### Phase 3 Metrics

| Metric                                             | Description                                                                              |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `first-pass-path-only-results`                     | Number of first-pass searches returning only file paths (no line content)                |
| `later-searches-upgraded-to-content`               | Number of later searches that upgraded from path-only to content+context                 |
| `search-surfaces-modes-with-differential-behavior` | Number of search surfaces/modes observed with inconsistent first-pass vs repeat behavior |

______________________________________________________________________

## Step 0: Release Transport Hardening Benchmark

### 0.1 Extended Benchmark Scope

The benchmark has shifted from discovery behavior to release transport correctness. This section captures the release-critical infrastructure issues and their resolution criteria.

#### 0.1.1 New Section: Release Transport Hardening Benchmark

This extension adds release-blocking infrastructure validation to the existing discovery-safe deduplication benchmark.

#### 0.1.2 Confirmed Wins (Out of Blocker Scope)

| Win                               | Status   | Description                                                                                     |
| --------------------------------- | -------- | ----------------------------------------------------------------------------------------------- |
| First-pass discovery search       | Verified | Working correctly - returns line-level evidence with surrounding context                        |
| Repeat compression on live search | Verified | Working correctly - subsequent identical searches compress after first exact observation        |
| Hot-search advisory behavior      | Verified | Working correctly - hot-search hints remain advisory and never suppress first exact observation |
| Targeted search/read              | Verified | Working correctly - targeted regex searches and file reads operate with content+context         |

#### 0.1.3 Confirmed Release Blockers

| Blocker                       | Severity | Description                                                                                  |
| ----------------------------- | -------- | -------------------------------------------------------------------------------------------- |
| Streaming client lifetime bug | Critical | Upstream client can close before generator consumption, causing premature stream termination |
| `--api-base` plumbing break   | Critical | Configured API base fails to reach request URL construction in subprocess bridge mode        |
| Mock-only streaming coverage  | Critical | No real streaming verification path exercises actual `aiter_bytes()` lifecycle               |

#### 0.1.4 Live Operational Symptoms

| Symptom            | Current State                                 |
| ------------------ | --------------------------------------------- |
| Session quality    | WATCH - degradation observed under load       |
| Degradation reason | Stream transport instability                  |
| Read-error count   | Elevated - requires monitoring from tok stats |
| Recovery holdovers | Present - automatic retry logic engaged       |

______________________________________________________________________

### 0.2 Release Gate Lock

Precise definition of "ready to release" for the 0.2.0 milestone:

#### 0.2.1 Streaming Response Lifetime Rule

**Rule:** Tok must not return a streaming response whose upstream client can close before generator consumption.

- **Validation:** All streaming paths must ensure client lifetime extends through full generator exhaustion
- **Test:** Verify `aiter_bytes()` completes without `ConnectionClosed` or `GeneratorExit` errors

#### 0.2.2 API Base Plumbing Rule

**Rule:** Configured API base must reach request URL construction in both foreground and subprocess bridge modes.

- **Validation:** `--api-base` parameter propagates through CLI → config → adapter → transport
- **Test:** Custom API base is used in actual HTTP requests in both execution modes

#### 0.2.3 Real Streaming Coverage Rule

**Rule:** At least one real streaming verification path must exercise the actual `aiter_bytes()` lifecycle.

- **Validation:** Integration test with real (non-mock) streaming backend
- **Test:** Full bytes→chunks→generator→consumption flow verified end-to-end

#### 0.2.4 Session Quality Stability Rule

**Rule:** Live session quality must no longer degrade due to stream transport instability under normal benchmark use.

- **Validation:** 10 consecutive benchmark runs without transport-related errors
- **Test:** Tok stats show zero read-errors and no recovery holdovers during standard workload

______________________________________________________________________

## Version

**Version:** 0.2.0-draft
**Last Updated:** 2026-04-07
**Status:** Release gate locked - awaiting blocker resolution

______________________________________________________________________

## Phase 2: Post First-Exact Guard Benchmark

### Observed Wins

The following behaviors were verified as working correctly after implementing the first-exact guard:

| Win                                 | Description                                                                                  |
| ----------------------------------- | -------------------------------------------------------------------------------------------- |
| Fresh file reads returned raw       | First-time file reads in a session return exact, uncompressed content with full line numbers |
| Repeated reads compressed correctly | Subsequent identical file reads are properly deduplicated to stable summaries                |
| Bypass was not required             | No manual user intervention needed to force exact content retrieval                          |

### Remaining Pain Points

The following issues were observed during Phase 2 testing and require further attention:

| Issue                                             | Description                                                                                            | Severity |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------- |
| `@hot_recent_search` first-pass stub substitution | Hot-recent cache path incorrectly substitutes first exact observation with stub instead of raw results | High     |
| File-targeted grep error stubs                    | Grep searches targeting specific files return error stubs on first pass rather than exact matches      | High     |
| Truncation at answer lines                        | Content truncation occurs at critical answer/decision lines, hiding the decisive evidence              | Medium   |
| Shallow stable-summary anchoring                  | Stable summaries fail to expose the specific line or region containing the answer                      | Medium   |

### Phase 2 Rerun Metrics

Additional metrics collected during Phase 2 to quantify remaining issues:

| Metric                                   | Description                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| `hot-cache-first-pass-substitutions`     | Number of times hot-recent cache replaced first exact observation with a stub        |
| `file-targeted-grep-error-stubs`         | Number of times file-targeted grep returned error stub on first pass                 |
| `repeated-offset-reads-from-truncation`  | Number of repeated offset reads caused by truncation at answer lines                 |
| `stable-summaries-missing-decisive-line` | Number of stable summaries that did not expose the line containing the actual answer |

______________________________________________________________________

## Phase 4: Live Search Surface Repeat-Compression Audit

### Scope

This audit focuses specifically on the Claude-facing live `Search(pattern: ...)` surface, including wrapper/cache/compression layers that can affect first-pass output mode. The issue is no longer broad discovery safety—it is a single surface-specific repeat-compression question.

### Confirmed Working Behaviors

| Behavior                       | Status  | Description                                                        |
| ------------------------------ | ------- | ------------------------------------------------------------------ |
| First-pass search raw          | Working | Initial search calls return exact, uncompressed content            |
| First-pass read exact          | Working | File reads on first access return full exact content               |
| Hot-search first-pass behavior | Working | Hot-recent search path correctly preserves first exact observation |
| Targeted discovery usable      | Working | Targeted regex searches operate correctly with content+context     |

### Remaining Question

| Question                                           | Status      |
| -------------------------------------------------- | ----------- |
| Repeated identical Search(pattern: ...) stayed raw | Under audit |

**Observation:** During testing, repeated identical `Search(pattern: ...)` calls remained in raw/full form rather than compressing to stable summaries after the first exact observation.

### Acceptance Question

**Is this live surface intended to compress on repeat?**

The answer to this question determines the classification of the observed behavior:

- **If YES**: The current behavior (remaining raw on repeat) is a gap/bug that should be fixed.
- **If NO**: The current behavior is intentional and must be treated as an explicit surface behavior, not an accidental gap.

______________________________________________________________________

## Audit Findings: Live Search Surface Implementation Trace

### Implementation Path

| Component                        | Location                                             | Purpose                                                                            |
| -------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `SEARCH_LIKE_TOOLS`              | `src/tok/runtime/repeat_targets.py:16`               | Defines search-like tools: `frozenset({"grep", "grep_search", "search", "rg"})`    |
| `evidence_identity_key()`        | `src/tok/runtime/repeat_targets.py:531-606`          | Generates unique key for search: `search\|{"flags":..., "query":..., "scope":...}` |
| `search_result_evidence_level()` | `src/tok/runtime/repeat_targets.py:310-332`          | Classifies output as "navigation" or "exact_content"                               |
| `_first_exact_guard()`           | `src/tok/compression/_history_pipeline.py:1194-1217` | Prevents compression of first exact observation                                    |
| `compress_recent_window_impl()`  | `src/tok/compression/_history_pipeline.py:1152-1354` | Compression for recent window with evidence level check                            |
| `compress_tool_results_impl()`   | `src/tok/compression/_history_pipeline.py`           | Compression for tool results - **GAP: missing evidence level check**               |

### Payload Shape

- Search tool context includes: `name`, `args`, `path`, `query`
- Evidence identity key format: `search|{"flags":{...},"query":"...","scope":"..."}`
- Evidence levels: `"navigation"` (path-only) vs `"exact_content"` (line-level)

### Gap Identified

**`compress_recent_window_impl()`** (lines 1319-1323) checks evidence level before tracking:

```python
if (
    tool_name in SEARCH_LIKE_TOOLS
    and search_result_evidence_level(raw) == "navigation"
):
    continue  # Skip tracking, stay raw
```

**`compress_tool_results_impl()`** does NOT have this check - it tracks evidence key regardless of level.

**Impact:**

- Path-only results incorrectly count as "first exact evidence" in `compress_tool_results_impl`
- Later content-level results may be treated as "repeat" and compress incorrectly

______________________________________________________________________

## Decision: YES, This Surface Should Compress on Repeat

### Rationale

1. `SEARCH_LIKE_TOOLS` includes `"search"` - the surface IS covered by repeat compression logic
1. The first-exact guard mechanism exists and works for `exact_content` results
1. Rule 4 in this benchmark explicitly allows repeat dedup after exact evidence established
1. Existing tests confirm intended behavior: `test_second_identical_hot_search_result_may_compress_after_exact_seen`

### Locked No-Regression Rules

| Rule                    | Description                                                              |
| ----------------------- | ------------------------------------------------------------------------ |
| First exact stays raw   | First `exact_content` search MUST stay raw                               |
| Path-only stays raw     | `navigation` results MUST stay raw on repeat (no evidence tracking)      |
| Repeat may compress     | Repeat `exact_content` search MAY compress after first exact observation |
| Identity includes scope | Evidence identity key must include query + scope + flags                 |

### Current Behavior Classification

| Scenario                      | `compress_recent_window_impl` | `compress_tool_results_impl` |
| ----------------------------- | ----------------------------- | ---------------------------- |
| First `exact_content` search  | Raw (CORRECT)                 | Raw (CORRECT)                |
| Repeat `exact_content` search | Can compress (CORRECT)        | Can compress (CORRECT)       |
| Path-only results             | Stay raw (CORRECT)            | Stay raw (CORRECT)           |

### Status: FIXED

The gap has been resolved by adding a search-specific repeat compression path in `compress_tool_results_impl()` that:

1. Checks if tool is in `SEARCH_LIKE_TOOLS`
1. Generates evidence identity key from query + scope + flags
1. Only compresses if key is already in `first_exact_evidence_seen` (repeat)
1. Skips navigation-only results (path-only stays raw)
1. Applies result cache or semantic hash compression for repeat exact-content searches

**Commit:** Search-specific repeat compression path added to `src/tok/compression/_history_pipeline.py`

______________________________________________________________________

## Policy Boundary: Live Search Surface Compression Rules

### Rule A: Compressor-Managed Surfaces

If a live search surface is compressor-managed, repeated identical exact-content results **may compress after first exact observation**.

- The first exact observation establishes the evidence identity
- Subsequent identical observations via the same evidence key may use compressed representation
- Compression must not occur until after first exact evidence is established in-session

### Rule B: Intentionally Raw Surfaces

If a live search surface is intentionally raw, that must be treated as **an explicit surface behavior, not an accidental gap**.

- Raw-on-repeat behavior must be documented and intentional
- The surface contract must clearly state that compression is not applied
- This is a valid design choice for critical discovery paths

### Rule C: First-Pass Raw Grounding Mandatory

Regardless of repeat behavior:

- **First-pass raw grounding remains mandatory**
- First discovery search must return line-level evidence with surrounding context
- Path-only output is not sufficient first-pass exact evidence for content discovery
- No cache, stabilization, or compression layer may replace the first exact observation
