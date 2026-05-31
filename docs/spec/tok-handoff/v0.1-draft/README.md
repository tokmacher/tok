# Tok Handoff Packet v0.1 Draft

> **Status:** Draft local schema for v0.2.2 Workstream D. **Created:** 2026-05-26
> **Scope:** Local agent-to-agent state handoff built from a `TokSessionReceipt`.

## Purpose

A handoff packet is a compact local artifact that lets one agent stop work and another
agent resume with clear evidence boundaries. It is not a remote protocol, not signed
provenance, and not a claim that summary, skeleton, or reference evidence is exact.

## JSON Structure

```json
{
  "schema": "tok-handoff/v0.1-draft",
  "handoff_id": "handoff_...",
  "created_at": "2026-05-26T00:00:00Z",
  "source_agent": "agent-a",
  "target_agent": "agent-b",
  "session_receipt": {
    "receipt_id": "tsr_...",
    "path": "out/session_receipt.json",
    "digest": "sha256:..."
  },
  "task_state": {
    "goal": "Continue bounded task",
    "completed": [],
    "next_steps": []
  },
  "evidence": [],
  "required_reacquisitions": [],
  "warnings": []
}
```

## Field Rules

| Field                     | Rule                                                                                   |
| ------------------------- | -------------------------------------------------------------------------------------- |
| `schema`                  | Must be `tok-handoff/v0.1-draft`.                                                      |
| `handoff_id`              | Local unique identifier. No remote trust semantics.                                    |
| `session_receipt`         | Must reference a local `TokSessionReceipt`; digest is verified locally when available. |
| `task_state.goal`         | Plain English current goal.                                                            |
| `task_state.completed`    | Completed work summaries. These are not exact source evidence.                         |
| `task_state.next_steps`   | Bounded continuation steps for the receiving agent.                                    |
| `evidence`                | Evidence entries labeled `exact`, `summary`, `skeleton`, or `reference`.               |
| `required_reacquisitions` | Exact files/artifacts the receiver must reacquire before edit-like work.               |
| `warnings`                | Degradation, fallback, or missing verification notes.                                  |

## Markdown Rendering Rules

The human rendering must include:

- Session receipt id and digest status.
- Current goal.
- Completed work.
- Next steps.
- Evidence table with exactness labels.
- Required exact reacquisitions before edit-like work.
- Warnings about fallback or degraded baseline state.

The rendering must not hide non-exact evidence behind exact-sounding language.

## Exactness Rules

| Exactness   | Receiver May Use For Edits? | Rule                                                     |
| ----------- | --------------------------: | -------------------------------------------------------- |
| `exact`     |                         Yes | Digest or local path must be recoverable before use.     |
| `summary`   |                          No | May guide search or review only.                         |
| `skeleton`  |                          No | May guide structure discovery only.                      |
| `reference` |                          No | Must be resolved and reacquired as exact evidence first. |

## Reacquisition Rules

Before code, security, legal, or release claims depend on evidence, the receiver must
reacquire exact source material. A handoff may list summaries and references, but those
entries only guide reacquisition.

## Validation Levels

| Level | Meaning                                         |
| ----- | ----------------------------------------------- |
| L0    | Packet schema parses.                           |
| L1    | Internal references are consistent.             |
| L2    | Linked session receipt digest verifies locally. |
| L3    | Required local exact artifacts are recoverable. |
| L4    | Signed provenance, deferred.                    |
| L5    | Remote verification, deferred.                  |

## Relation to TokSessionReceipt

`TokSessionReceipt` is the audit anchor. The handoff packet is the operational summary
for the next agent. The handoff must reference the receipt instead of duplicating full
trace or savings state.

## Non-Claims

This schema does not define Tok Capability, Tok Session as a stable protocol, remote
resolver routing, signed provenance, or universal agent-to-agent exchange.
