"""
Demonstrate the Tok SDK preparation step (no API call).

This is an experimental example showing how tok.wrap() prepares a request
for compression. It does not send any API calls.

Requires: TOK_MODE and TOK_REQUEST_POLICY can be set via environment.

Usage:
  python examples/tok_universal_demo.py
"""

from __future__ import annotations

import logging
import os

import tok

logging.disable(logging.WARNING)


def create_demo_messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": "Analyze this Python function and suggest improvements.",
        },
    ]


def main() -> None:
    model = os.getenv("TOK_MODEL", "claude-sonnet-4-20250514")
    session = tok.RuntimeSession()
    messages = create_demo_messages()
    prepared = tok.wrap(messages, model=model, session=session)
    print(f"Prepared request for {model}")
    print(f"Input tokens saved: {prepared.input_saved_tokens}")


if __name__ == "__main__":
    main()
