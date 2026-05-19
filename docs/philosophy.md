# Tok Philosophy: The Invisible Bridge

**Version:** 0.2.0 **Audience:** contributors, reviewers, and anyone proposing a change
to Tok behaviour

______________________________________________________________________

## The Contract

A session with Tok active must be indistinguishable from a session without it — except
cheaper.

The model completes its task at the same quality, in the same number of turns, using the
same tools it would have chosen. Tok's job is to reduce the bandwidth of that journey,
not to influence it. If Tok causes a model to make a different decision — use fewer
tools, stop earlier, phrase an answer differently — that is a bug, not a feature.

This is the **invisible bridge** principle. Everything else in this document follows
from it.

______________________________________________________________________

## What Tok Is Allowed To Do

### Compress redundant history

Once a tool result has been seen and acted upon, sending it again in full is wasteful.
Tok may replace a previously-seen result with a compact stub (`@stable_result`) when the
content is byte-for-byte identical to a prior turn. The model is told what the stub
means and can trust it.

### Inject compact session state

A long conversation contains a lot of context the model needs but has already processed.
Tok maintains a rolling memory block (`>>>`) that captures the distilled state — facts,
current goal, key findings — so the model doesn't have to re-read a 40-turn history to
know where it is. This is compression of context, not instruction to the model.

### Explain its own signals

When Tok introduces a non-standard token into the conversation (a `@stable_result` stub,
a `@hot_recent_file` summary), it appends a brief explanation to the system prompt so
the model understands what it is seeing. This is documentation, not steering.

### Suppress exact duplicates

If a model calls the same tool with the same arguments and gets the same result it
already has, Tok may suppress the repeated delivery. The model already has this
information; sending it again costs tokens with no benefit.

______________________________________________________________________

## What Tok Is Not Allowed To Do

### Tell the model when to use tools, or which tool to use

The model understands its task. It chooses tools based on what it needs. Tok must not
inject hints like "make concrete progress with one tool action before finalising" or
"use the read-only tools first." These instructions second-guess the model and produce
unpredictable results.

### Tell the model when to produce a final answer

"Answer-ready" pressure — injecting hints that the model should now stop using tools and
deliver its answer — is steering. If the model has work left to do, it will not
finalise. If it has no work left, it will finalise on its own. Tok adding urgency does
not help and frequently causes premature finalisation on incomplete tasks.

### Remove or truncate novel failure information

If a test run produces a new failure — a failure that has not appeared in the session
before — that output must reach the model in full. The model needs to read the actual
error to fix it. Compressing a novel failure into `>>> tool:pytest|failed:2|passed:47`
removes the information the model needs and forces it to guess, producing extra
iterations that cost more than the compression saved.

### Inject correction, urgency, or protocol-enforcement language

Phrases like "protocol drift detected", "you should finalise now", or "required before
finalising" are attempts to override the model's judgement. They add tokens, add noise,
and create divergence from baseline behaviour that is difficult to predict or test.

______________________________________________________________________

## The Compression Test

A proposed Tok feature passes if and only if all three conditions hold:

1. **Same outcome.** The model would reach the same conclusion — same answer, same
   edits, same number of effective tool calls — with or without this feature.

1. **Net savings.** It saves tokens averaged across a realistic session, not just in the
   turn where it fires. A feature that saves 200 tokens now but causes one extra
   iteration (≈1,500 tokens) is a net loss.

1. **Invisible.** It does not require the model to "understand", "follow", or "comply
   with" the feature. If the feature would break silently if the model ignored it, it is
   steering.

When unsure, apply the inversion: *if we removed this feature tomorrow, would the model
produce worse results?* If yes, the feature is filling an information gap and is
legitimate. If no — or if the model would actually perform better — the feature is
steering and should be removed.

______________________________________________________________________

## The Steering Anti-Pattern

Steering is always a losing bet. Here is why:

**It costs tokens at injection.** Every hint added to the system prompt is paid for
immediately, before knowing whether it helps.

**The model partially follows it unpredictably.** A well-calibrated model sometimes
ignores hints, sometimes follows them literally, and sometimes over-indexes on them.
Premature finalisation of an incomplete patch is a direct consequence of answer-ready
pressure.

**It creates benchmark divergence.** When Tok-assisted sessions diverge from baseline in
turn count or tool usage, the benchmark correctly flags this as a fairness concern.
Steering is the primary cause of this divergence.

**The correct response to steering temptation is an information audit.** When we find
ourselves wanting to tell the model what to do, the right question is: *what information
is the model missing that would lead it to the right decision naturally?* Usually the
answer is: we compressed something we should not have.

______________________________________________________________________

## The Decision Framework for New Features

Before adding anything to Tok, answer these questions:

| Question                                                                    | If yes          | If no           |
| --------------------------------------------------------------------------- | --------------- | --------------- |
| Does it replace redundant content with a lossless or summary-faithful stub? | Likely valid    | Needs scrutiny  |
| Does it inject state the model has already derived and acted on?            | Likely valid    | Likely steering |
| Does it require the model to change its behaviour?                          | Steering — stop | Fine            |
| Does it fire even when the model is making correct decisions?               | Steering — stop | Fine            |
| Would removing it improve benchmark fairness?                               | Remove it       | Fine            |

______________________________________________________________________

## Stability and Fail-Open

Tok's fail-open mechanism — falling back to uncompressed baseline after 3 consecutive
failures — is the right safety valve. When in doubt, doing nothing is always correct. A
session that passes through Tok unchanged is a success. A session where Tok tried to
help and made things worse is a failure, regardless of the token savings on paper.

The goal is for Tok to be so unobtrusive that removing it from a session produces no
observable change in model behaviour, only a change in cost.

______________________________________________________________________

*This document is the design constitution for Tok. Proposals that conflict with it
require explicit justification and a benchmark demonstrating no behavioural divergence
from baseline.*
