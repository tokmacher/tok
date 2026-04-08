# Pricing Verification Ledger

Canonical pricing source for Tok is
[`src/tok/utils/pricing.py`](../src/tok/utils/pricing.py).

Last reviewed: **2026-04-08**.

## Verification Status

| Model Prefix                    | Current Tok Rates (input/output/cache_read/cache_write) | Status                              | Source                                                                             |
| ------------------------------- | ------------------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------- |
| `claude-opus-4`                 | `15.00 / 75.00 / 1.50 / 18.75`                          | Verified                            | [Anthropic pricing](https://docs.anthropic.com/en/docs/about-claude/pricing)       |
| `claude-sonnet-4`               | `3.00 / 15.00 / 0.30 / 3.75`                            | Verified                            | [Anthropic pricing](https://docs.anthropic.com/en/docs/about-claude/pricing)       |
| `claude-haiku-4`                | `0.80 / 4.00 / 0.08 / 1.00`                             | Provisional mapping                 | [Anthropic pricing](https://docs.anthropic.com/en/docs/about-claude/pricing)       |
| `openai/gpt-5.4-pro`            | `30.00 / 180.00 / 0.00 / 0.00`                          | Verified (cache fields unpublished) | [OpenAI GPT-5.4 model docs](https://developers.openai.com/api/docs/models/gpt-5.4) |
| `google/gemini-3-flash-preview` | `0.50 / 3.00 / 0.00 / 0.00`                             | Aggregator-derived                  | [Gemini API pricing](https://ai.google.dev/pricing)                                |
| `z-ai/glm-5`                    | `0.72 / 2.30 / 0.00 / 0.00`                             | Aggregator-derived                  | [OpenRouter model pricing](https://openrouter.ai/models)                           |
| `x-ai/grok-4.20-beta`           | `2.00 / 6.00 / 0.00 / 0.00`                             | Aggregator-derived                  | [OpenRouter model pricing](https://openrouter.ai/models)                           |
| `moonshotai/kimi-k2.5`          | `0.38 / 1.72 / 0.00 / 0.00`                             | Aggregator-derived                  | [OpenRouter model pricing](https://openrouter.ai/models)                           |
| `minimax/minimax-m2.7`          | `0.30 / 1.20 / 0.06 / 0.38`                             | Aggregator-derived                  | [MiniMax model page](https://www.minimax.io/models/text/m27)                       |

## Policy

- Release-defining claims should use values marked **Verified**.
- Aggregator-derived rows are allowed for runtime estimates, but should be treated as
  non-release-defining.
- If a provider does not publish cache read/write rates, Tok stores `0.00` for those
  fields until verified.
