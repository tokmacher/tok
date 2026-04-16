# Tok Metaprompt: Self-Diagnostic Exploration Audit

Use this prompt in a fresh Claude session (with Tok active) to evaluate whether Tok's
compression behavior is helping or hindering discovery workflows.

______________________________________________________________________

## Prompt for Claude

```
You are auditing the tiny `src/tok/memory/` subfolder (just 2 files).

Your task: Find and explain what "memory pointers" are in this codebase. Specifically:
1. What does pointers.py define?
2. How is it used (check __init__.py exports)?
3. Search for usages elsewhere in the repo

IMPORTANT: As you perform each tool call, narrate:

A) What you asked for (exact query/path)
B) What you got back (full content? summary? paths only?)
C) Was it immediately useful, or did you need to re-query?

For each call, note:
- **First-time vs repeat**: First time asking this, or repeat?
- **Content richness**: Line-level evidence or just paths/summaries?
- **Friction**: Any workarounds needed?

At the end, list:
- Where Tok HELPED (good compression, useful summaries)
- Where Tok HINDERED (over-compression, missing content, re-queries needed)
```

______________________________________________________________________

## Fixes Applied (2026-04-15)

### 1. Small File Truncation Bypass

**Location:** `src/tok/compression/_tool_result_codecs.py:truncate_large_result()`

Files with < 100 lines and reasonable line lengths (< 100 chars avg) are no longer
truncated at default limits. This prevents mid-content truncation of small discovery
targets like the 77-line `pointers.py`.

### 2. `__other__` Snippet Visibility

**Location:** `src/tok/compression/_tool_result_codecs.py:_compress_grep()`

Lines that don't match standard grep format now show first 3 snippets explicitly before
collapsing. Previously all `__other__` content was hidden behind a count.

### 3. Multi-File Grep Evidence

**Location:** `src/tok/compression/_tool_result_codecs.py:_compress_grep()`

Multi-file grep now shows up to 3 snippets per file (not just 1), providing line-level
evidence for discovery without requiring re-queries.

______________________________________________________________________

## Expected Good Outcomes

- First file read of `pointers.py` returns full content (no truncation)
- First search for "pointers" usages returns line-level matches (3 per file)
- Repeat reads show compressed summary (only after first exact)
- No "got paths but needed content" moments

## Red Flags

- First read returns summary instead of full content
- First search returns only file paths (no line content)
- Immediate re-queries needed due to insufficient results

## Scoring (1-5 each)

| Dimension                 | 1 = Poor        | 5 = Excellent          |
| ------------------------- | --------------- | ---------------------- |
| First-pass grounding      | Path-only       | Full line evidence     |
| Repeat compression timing | Too early       | Only after first exact |
| Discovery friction        | Many re-queries | One-shot works         |

______________________________________________________________________

## Prompt B: Large Search + Repeat Read Test

```
You are testing Tok's behavior on larger searches and repeat reads.

Task: Find all uses of "compress" functions in the codebase.

Step 1: Search for "def _compress" across src/tok/
Step 2: Read the first file that appears (full file)
Step 3: Search again for the same pattern (to test repeat behavior)
Step 4: Re-read the same file you read in step 2

After each step, note:
- What you got back (full content? compressed? paths only?)
- Any truncation or suppression
- Whether repeat queries behaved differently than first-time

At the end, report:
- Did the first search show line-level evidence?
- Did the repeat search compress/summarize?
- Did the repeat file read compress?
- Any `__other__` content that was hidden?
```
