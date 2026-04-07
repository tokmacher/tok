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

## Version

**Version:** 0.1.0
**Last Updated:** 2026-04-06

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
