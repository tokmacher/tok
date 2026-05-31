# Tok Handoff

Session receipt: `tsr_good_001` Digest status: local digest required

## Goal

Continue the bounded docs validation task.

## Completed

- Defined session receipt schema and local verifier.

## Next Steps

- Inspect required exact files before editing.
- Run the targeted docs contract test.

## Evidence

| Label                 | Exactness | Path                                       |
| --------------------- | --------- | ------------------------------------------ |
| session receipt model | exact     | `src/tok/protocol/session_receipt.py`      |
| plan summary          | summary   | `docs/plans/substrate_execution_prompt.md` |

## Required Reacquisitions

- `src/tok/protocol/session_receipt.py`: Source edits require exact local file content.
