# Tok And Claude Code Compaction

Tok and Claude Code compaction are complementary. Use Claude Code compaction for native
conversation management; use Tok when you want a local bridge to reduce repeated
machine-facing context before requests reach the model.

## Practical Difference

| Tool                        | Where it acts                                  | Best for                                                              | What to watch                                                                              |
| --------------------------- | ---------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Claude Code `/compact`      | Inside Claude Code conversation state          | Manually shrinking a long visible conversation                        | The model receives a compacted conversation summary rather than the full prior transcript. |
| Claude Code auto-compaction | Inside Claude Code when context pressure rises | Native long-session continuity                                        | Timing and summary shape are controlled by Claude Code.                                    |
| Tok bridge                  | Between Claude Code and the model API          | Repeated file reads, searches, and tool outputs in sustained sessions | Tok must preserve Claude Code's expected request/response shape or fall back.              |
| `TOK_MODE=baseline`         | Tok bridge with compression disabled           | Measuring impact or debugging                                         | No Tok compression savings should appear.                                                  |

## When Tok Helps

Tok is most useful when the same machine-readable evidence appears repeatedly: file
contents, search results, command output, diffs, and status text. It can replace
repeated payloads with deterministic references or deltas while preserving the normal
Claude Code workflow.

Claude Code compaction is still useful for long visible conversations. Tok does not
replace it, and Tok's `0.1.x` release does not claim control over Claude Code's native
conversation-management behavior.

## Measuring Tok Against Baseline

Run the same kind of work with Tok active, then compare with baseline mode:

```bash
tok bridge start
ANTHROPIC_BASE_URL=http://localhost:9090 claude
tok stats
```

```bash
TOK_MODE=baseline tok bridge start
ANTHROPIC_BASE_URL=http://localhost:9090 claude
tok stats
```

Use `tok doctor` and `tok bridge status` to check whether Tok stayed active or degraded
to baseline for safety. Short sessions and non-repetitive sessions may show little
difference.

## 0.1.x Boundary

For `0.1.x`, Tok supports the explicit Claude Code bridge workflow. It does not replace
Claude Code `/compact`, does not manage Claude Code's local conversation files, and does
not promise broad multi-provider behavior beyond the bridge validation paths documented
elsewhere.
