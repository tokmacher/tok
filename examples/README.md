# Examples

Tok ships examples for the v0.1.0 public release. All examples use the **experimental**
SDK path (`tok.wrap`/`tok.process`), not the supported bridge-first CLI.

## Examples

- [`tok_wrap_example.py`](./tok_wrap_example.py): minimal experimental Python wrapper
  flow using `RuntimeSession`, `tok.wrap(...)`, and `tok.process(...)`
- [`natural_first_openrouter.py`](./natural_first_openrouter.py): exercises the same
  wrapper flow against OpenRouter with the natural-first request policy for cheap
  iteration.
- [`tok_universal_demo.py`](./tok_universal_demo.py): demonstrates the universal runtime
  preparation step (no API call — shows compression setup only).

> **Note**: These examples exercise the experimental SDK path. The supported 0.1.0
> workflow is the bridge-first CLI documented in [`README.md`](../README.md).

If you are new to Tok, start with the bridge flow in [`README.md`](../README.md) first.
