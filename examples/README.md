# Examples

Tok ships two supported examples for the v0.1.0 public release.

## Supported Examples

- [`tok_wrap_example.py`](./tok_wrap_example.py): minimal experimental Python wrapper flow using `RuntimeSession`, `tok.wrap(...)`, and `tok.process(...)`
- [`natural_first_openrouter.py`](./natural_first_openrouter.py): exercises the same wrapper flow against OpenRouter with the natural-first request policy for cheap iteration.

Older playground/demo scripts are intentionally not shipped as part of the public example set because they are broader than the supported bridge-first and wrapper flows.

If you are new to Tok, start with the bridge flow in [`README.md`](../README.md) first.

## Verbatim Reads (Claude Code)

When you need exact quoting from a large file, prefer a small precision read
(`offset`/`limit`, or a tight line slice). Tok will replay cached bytes for
precision reads even if the host UI reports "Unchanged since last read".
