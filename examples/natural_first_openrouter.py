"""Run Tok wrap/process with OpenRouter using the natural-first policy.

Env knobs (defaults shown):
  OPENROUTER_API_KEY   required
  OPENROUTER_API_BASE  https://openrouter.ai/api/v1
  TOK_MODEL            gpt-4o-mini
  TOK_TURNS            10
  TOK_DELAY_SECONDS    0.2
  TOK_PROMPT           "Give me a one-line repo summary."
  TOK_MODE             tool-compatible   # or TOK_REQUEST_POLICY=natural_first
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import openai

import tok
from tok.testing.frontier import DEFAULT_FRONTIER_PROFILES, apply_frontier_env


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise SystemExit(
            f"Set {name}=... in your shell or .env before running."
        )
    return val


def main() -> None:
    api_key = _require_env("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    model = os.getenv("TOK_MODEL", "gpt-4o-mini")
    turns = int(os.getenv("TOK_TURNS", "10"))
    delay = float(os.getenv("TOK_DELAY_SECONDS", "0.2"))
    prompt = os.getenv("TOK_PROMPT", "Give me a one-line repo summary.")
    profile_name = os.getenv("TOK_FRONTIER_PROFILE", "balanced")
    artifact_path = os.getenv("TOK_FRONTIER_ARTIFACT", "")

    profile = next(
        (
            candidate
            for candidate in DEFAULT_FRONTIER_PROFILES
            if candidate.name == profile_name
        ),
        None,
    )
    if profile is None:
        raise SystemExit(f"Unknown TOK_FRONTIER_PROFILE={profile_name!r}")

    # Keep the existing natural-first behavior as the default baseline.
    os.environ.setdefault(
        "TOK_MODE", profile.env.get("TOK_MODE", "tool-compatible")
    )
    os.environ.setdefault(
        "TOK_REQUEST_POLICY",
        profile.env.get("TOK_REQUEST_POLICY", "natural_first"),
    )

    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    session = tok.RuntimeSession()
    rows: list[dict[str, object]] = []

    with apply_frontier_env(profile.env):
        for i in range(turns):
            messages = [{"role": "user", "content": f"{prompt} (turn {i})"}]
            prepared = tok.wrap(messages, model=model, session=session)

            # Assemble OpenAI-format body (OpenRouter is OpenAI-compatible).
            body_messages = (
                [{"role": "system", "content": prepared.body["system"]}]
                if prepared.body.get("system")
                else []
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
                combined_signals[key] = combined_signals.get(key, 0) + int(
                    value
                )
            row = {
                "turn": i,
                "profile": profile.name,
                "mode": profile.mode,
                "input_saved_tokens": int(prepared.input_saved_tokens),
                "output_saved_tokens": int(processed.output_saved_tokens),
                "provider_total_tokens": int(
                    getattr(response.usage, "total_tokens", 0)
                ),
                "behavior_signals": combined_signals,
                "text_preview": text.strip()[:120],
            }
            rows.append(row)

            print(
                f"turn {i}\tprofile={profile.name}\tmode={profile.mode}\t"
                f"in_saved={prepared.input_saved_tokens}\t"
                f"out_saved={processed.output_saved_tokens}\t"
                f"text={text.strip()[:120]}"
            )

            if delay > 0:
                time.sleep(delay)

    if artifact_path:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2))
        print(f"artifact={path}")


if __name__ == "__main__":
    main()
