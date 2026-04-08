"""
Demonstrate tok-universal profile behavior across model families.

This example shows how the tok-universal profile:
1. Preserves assistant thinking blocks (Claude models)
2. Maintains robust compression across all model families
3. Handles tool use consistently
4. Achieves 30-50% token savings while prioritizing task completion

Env knobs (defaults shown):
  ANTHROPIC_API_KEY    required for Claude
  OPENAI_API_KEY       required for GPT
  OPENROUTER_API_KEY   required for other models via OpenRouter
  TOK_MODEL            claude-sonnet-4-20250514 (or gpt-4o, openai/gpt-4o, etc.)
  TOK_UNIVERSAL_MODE   1 (forces tok-universal profile)

Example runs:
  # Claude with thinking preservation
  ANTHROPIC_API_KEY=... python examples/tok_universal_demo.py

  # GPT via OpenRouter
  OPENROUTER_API_KEY=... TOK_MODEL=openai/gpt-4o python examples/tok_universal_demo.py

  # DeepSeek via OpenRouter
  OPENROUTER_API_KEY=... TOK_MODEL=deepseek/deepseek-chat python examples/tok_universal_demo.py
"""

from __future__ import annotations

import os

import tok
from tok.runtime.policy.smart_policy import UNIVERSAL_MODE
from tok.utils.token_utils import count_tokens


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        msg = f"Set {name}=... before running."
        raise SystemExit(msg)
    return val


def create_universal_demo_prompt() -> list[dict]:
    """
    Create a multi-turn conversation that exercises tok-universal behavior.

    This prompt is designed to:
    - Trigger assistant reasoning/thinking blocks
    - Include tool-use patterns
    - Require context retention across turns
    - Show compression effectiveness
    """
    return [
        {
            "role": "user",
            "content": (
                "Analyze this Python function and explain what it does, "
                "then suggest two specific improvements with reasoning.\n\n"
                "```python\n"
                "def process_data(items):\n"
                "    results = []\n"
                "    for i in range(len(items)):\n"
                "        item = items[i]\n"
                "        if item.get('active'):\n"
                "            results.append(transform(item))\n"
                "    return results\n"
                "```"
            ),
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": (
                        "Let me analyze this function step by step:\n"
                        "1. It iterates over items using index-based loop\n"
                        "2. Checks if item has 'active' key set to truthy\n"
                        "3. Transforms active items and collects results\n"
                        "Potential issues: O(n) complexity with Python iteration, "
                        "could use list comprehension for clarity"
                    ),
                },
                {
                    "type": "text",
                    "text": (
                        "This function filters active items and transforms them.\n\n"
                        "**Issues I see:**\n"
                        "1. Uses index-based iteration (range(len())) instead of direct iteration\n"
                        "2. Could be simplified with list comprehension\n\n"
                        "**Suggested improvements:**\n"
                        "1. Use direct iteration: `for item in items:`\n"
                        "2. Use list comprehension with filter: "
                        "`[transform(i) for i in items if i.get('active')]`"
                    ),
                },
            ],
        },
        {
            "role": "user",
            "content": (
                "Now write a test case for the improved version that covers "
                "edge cases: empty list, all inactive items, and missing 'active' key."
            ),
        },
    ]


def run_universal_demo() -> None:
    """Run the tok-universal demonstration."""
    # Force universal mode
    os.environ["TOK_MODE"] = "tok-universal"
    os.environ["TOK_REQUEST_POLICY"] = "tool_compatible"

    model = os.getenv("TOK_MODEL", "claude-sonnet-4-20250514")
    session = tok.RuntimeSession()

    # Verify universal mode is active
    policy = tok.policy_for_model(model)
    assert policy.default_mode == UNIVERSAL_MODE, "Universal mode not active!"

    # Create demonstration messages
    messages = create_universal_demo_prompt()

    # Wrap with tok-universal
    prepared = tok.wrap(messages, model=model, session=session)

    # Show savings percentage if we had baseline info
    baseline_tokens = sum(count_tokens(str(m)) for m in messages)
    savings_pct = (prepared.input_saved_tokens / baseline_tokens * 100) if baseline_tokens > 0 else 0
    print(f"  Savings: {savings_pct:.1f}%")

    # Demonstrate that the output is ready for API use
    # Summary


if __name__ == "__main__":
    run_universal_demo()
