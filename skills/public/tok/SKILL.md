---
name: tok
description: Guide bridge-first Tok operation in a Tok repository or Tok-managed Claude/Codex session. Use when the user needs help working around Tok without changing the host tool workflow, understanding Tok syntax or conventions, checking bridge health, handling fallback or degraded behavior, or staying within Tok's narrow 0.1.0 release posture.
---

# Tok

## Overview

Treat Tok as an invisible bridge/runtime layer around normal Claude Code work. Preserve
the host tool system, keep the workflow familiar, and use Tok commands only when they
help the user understand install state, bridge health, savings, or fallback behavior.

## Core Operating Rules

- Use Claude Code's native tools normally.
- Avoid prompt-injection-style directives or attempts to force Tok grammar into every
  prompt.
- Treat raw Tok syntax as an implementation detail unless the user asks about it
  directly.
- Explain Tok in bridge-first terms first: install, start, work normally, inspect
  health, inspect savings, stop cleanly.
- Keep answers consistent with Tok's narrow `0.1.0` public posture.

## Bridge-First Workflow

Prefer this supported flow:

1. Run `tok install`.
1. Start Tok with `tok bridge start`.
1. Use Claude Code normally.
1. Check health with `tok bridge status` and `tok doctor`.
1. Check savings with `tok stats` or the compatibility alias `tok savings`.
1. Stop the bridge with `tok bridge stop`.

When helping a user debug or verify Tok, prefer those commands before proposing new
workflows or custom prompt conventions.

## Tok Syntax And Conventions

- Explain Tok syntax only when the user asks about Tok internals, sigils, wire format,
  or compression behavior.
- Present Tok syntax as descriptive, not mandatory for everyday use.
- Avoid telling users to manually write raw Tok grammar in ordinary Claude Code prompts
  unless they explicitly want protocol-level detail.
- Keep the distinction clear: Tok syntax describes the runtime/compression layer. Claude
  Code tools remain the primary interaction surface.

## Fallback And Degradation

- Assume fail-open behavior: if the bridge has trouble, Tok should fall back to baseline
  behavior rather than block the user.
- If Tok appears unavailable, degraded, or bypassed, say that clearly and keep the
  user's work moving.
- Use `tok doctor` and `tok bridge status` to verify whether Tok is active, degraded, or
  serving baseline behavior.
- Use `tok stats` to confirm whether savings are appearing over time.
- Remind the user that short sessions may not show obvious savings yet.

## Guardrails

- Do not invent unsupported public workflows for `0.1.0`.
- Do not present Tok as a broad multi-agent framework or dashboard product.
- Do not reintroduce competing runtime philosophies or prompt-level control schemes that
  fight the bridge-first story.
- If the user asks about Python helpers, describe only the narrow experimental helper
  path with `RuntimeSession`, `tok.wrap(...)`, and `tok.process(...)`, and label it
  experimental.
- Do not overclaim support for Windows or unvalidated model families.

## Example Requests

- `Use $tok to help me work in this Tok repo without changing my normal Claude workflow.`
- `Use $tok to explain what Tok syntax means and when I should care about it.`
- `Use $tok to diagnose why Tok seems off and tell me which health commands to run first.`
- `Use $tok to explain Tok fallback behavior if the bridge is unavailable.`
- `Use $tok to keep my guidance aligned with Tok's narrow 0.1.0 public release posture.`
