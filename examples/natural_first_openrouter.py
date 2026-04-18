"""
Run Tok wrap/process with OpenRouter using the natural-first policy.

Env knobs (defaults shown):
  OPENROUTER_API_KEY   required
  OPENROUTER_API_BASE  https://openrouter.ai/api/v1
  TOK_MODEL            gpt-4o-mini
  TOK_TURNS            10
  TOK_DELAY_SECONDS    0.2
  TOK_PROMPT           "Give me a one-line repo summary."
  TOK_MODE             tool-compatible
  TOK_REQUEST_POLICY   natural_first
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import openai

import tok


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        msg = f"Set {name}=... in your shell or .env before running."
        raise SystemExit(msg)
    return val


def _load_config() -> dict[str, str | int | float]:
    """Load and return all configuration from environment variables."""
    return {
        "api_key": _require_env("OPENROUTER_API_KEY"),
        "base_url": os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        "model": os.getenv("TOK_MODEL", "gpt-4o-mini"),
        "turns": int(os.getenv("TOK_TURNS", "10")),
        "delay": float(os.getenv("TOK_DELAY_SECONDS", "0.2")),
        "prompt": os.getenv("TOK_PROMPT", "Give me a one-line repo summary."),
        "artifact_path": os.getenv("TOK_FRONTIER_ARTIFACT", ""),
    }


def _run_turn(
    i: int,
    prompt: str,
    model: str,
    client: openai.OpenAI,
    session: tok.RuntimeSession,
) -> dict[str, object]:
    """Run a single turn and return the result row."""
    messages = [{"role": "user", "content": f"{prompt} (turn {i})"}]
    prepared = tok.wrap(messages, model=model, session=session)

    body_messages = (
        [{"role": "system", "content": prepared.body["system"]}] if prepared.body.get("system") else []
    ) + prepared.body["messages"]

    response = client.chat.completions.create(
        model=model,
        messages=body_messages,
        temperature=0.2,
        max_tokens=200,
    )
    text = response.choices[0].message.content or ""

    processed = tok.process(text, model=model, session=session)
    combined_signals = dict(prepared.behavior_signals)
    for key, value in processed.behavior_signals.items():
        combined_signals[key] = combined_signals.get(key, 0) + int(value)

    return {
        "turn": i,
        "input_saved_tokens": int(prepared.input_saved_tokens),
        "output_saved_tokens": int(processed.output_saved_tokens),
        "provider_total_tokens": int(getattr(response.usage, "total_tokens", 0)),
        "behavior_signals": combined_signals,
        "text_preview": text.strip()[:120],
    }


def main() -> None:
    """Run Tok wrap/process with OpenRouter."""
    config = _load_config()

    os.environ.setdefault("TOK_MODE", os.getenv("TOK_MODE", "tool-compatible"))
    os.environ.setdefault("TOK_REQUEST_POLICY", os.getenv("TOK_REQUEST_POLICY", "natural_first"))

    client = openai.OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    session = tok.RuntimeSession()
    rows: list[dict[str, object]] = []

    for i in range(config["turns"]):
        row = _run_turn(
            i,
            config["prompt"],
            config["model"],
            client,
            session,
        )
        rows.append(row)

        if config["delay"] > 0:
            time.sleep(config["delay"])

    if config["artifact_path"]:
        path = Path(config["artifact_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
