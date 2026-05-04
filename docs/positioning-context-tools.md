# Tok Among Context Tools

Tok is a local bridge-layer compression tool. It is intentionally narrower than memory
systems, code indexers, MCP servers, hosted observability tools, or prompt-compression
APIs.

## What Tok Is

- A deterministic compression layer for repeated Claude Code context.
- A local bridge that preserves the normal Claude Code workflow.
- A safety-first runtime that falls back to baseline when compact representation may
  lose important evidence.
- A diagnostic surface: `tok bridge status`, `tok doctor`, `tok stats`, logs, and
  optional `tok audit` traces.

## What Tok Is Not

| Category               | Tok's position                                                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| Memory product         | Tok keeps compact bridge state, but it is not a user-facing long-term memory system.                            |
| Code indexer           | Tok may compress repeated file/search evidence, but it does not build a semantic repo index for users to query. |
| MCP server marketplace | Tok does not provide or broker tools. It sits under Claude Code's existing workflow.                            |
| Prompt-compression API | Tok is not a hosted API for arbitrary prompts. The supported path is the local Claude Code bridge.              |
| Observability platform | Tok exposes local diagnostics and trace sidecars, but it is not a hosted dashboard or monitoring service.       |
| Agent framework        | Tok does not orchestrate agents or define broad agent-to-agent handoff in `0.1.x`.                              |

## Why The Boundary Matters

Tok's first release keeps a small public surface so claims stay testable:

- Claude Code is the supported default client.
- The local bridge is the supported integration path.
- Compression is deterministic rather than LLM-summarized.
- Fallback is preferred over risky compression.
- `tok audit` in `0.1.7` is draft bridge trace/audit groundwork, not universal protocol
  conformance.

This makes Tok useful for developers who already have Claude Code working and want to
reduce repeated model-facing context in sustained sessions without adopting a new agent
framework.

## Adjacent Tools Can Still Help

Tok can sit alongside native Claude Code compaction, repo search, MCP tools, and
external observability. The clean mental model is:

- Use Claude Code and your existing tools as usual.
- Route Claude Code through Tok when you want bridge-layer compression.
- Use `tok doctor`, `tok stats`, and optional `tok audit` to understand Tok's behavior.
