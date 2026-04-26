# Examples

Tok ships examples alongside the `0.1.x` release, but the examples are for the
**experimental** Python path, not the supported bridge-first CLI.

## Examples

- [`tok_wrap_example.py`](./tok_wrap_example.py): minimal experimental Python flow using
  explicit submodule imports such as `RuntimeSession` and `RuntimeRequest`
- [`natural_first_openrouter.py`](./natural_first_openrouter.py): exercises the same
  wrapper flow against OpenRouter with the natural-first request policy for cheap
  iteration.
- [`tok_universal_demo.py`](./tok_universal_demo.py): demonstrates the universal runtime
  preparation step (no API call — shows compression setup only).

> **Note**: These examples exercise experimental submodule APIs. The supported `0.1.x`
> workflow is the bridge-first CLI documented in [`README.md`](../README.md).

If you are new to Tok, start with the bridge flow in [`README.md`](../README.md) first.
