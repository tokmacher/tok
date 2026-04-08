"""
Minimal recipe for the experimental tok.wrap / tok.process path.

Caller owns the RuntimeSession, sends ``prepared.body`` to an OpenAI-compatible
SDK call, then reuses the same session on the next turn.
"""

from __future__ import annotations

import os

import openai

import tok

# 1. Create one RuntimeSession and keep it for the whole conversation.
session = tok.RuntimeSession()

# 2. Build messages as usual.
messages = [
    {
        "role": "user",
        "content": "What file in this repo owns core runtime semantics?",
    },
]
model = os.getenv("TOK_MODEL", "deepseek/deepseek-v3.2")

# 3. Prepare the request. For OpenAI-style chat APIs, prepend the system string
#    as a system message, then append prepared.body["messages"].
prepared = tok.wrap(messages, model=model, session=session)


# 4. Send the prepared request through any OpenAI-compatible client.
client = openai.OpenAI(
    base_url=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
)

request_body = {
    "model": model,
    "messages": (
        ([{"role": "system", "content": prepared.body["system"]}] if prepared.body.get("system") else [])
        + prepared.body["messages"]
    ),
    "temperature": 0.0,
    "max_tokens": 200,
}

response = client.chat.completions.create(**request_body)
response_text = response.choices[0].message.content or ""

# 5. Process the model text. Inspect visible text, updated_memory, and saved tokens.
result = tok.process(response_text, model=model, session=session)
visible_blocks = [b for b in result.content_blocks if b.get("type") == "text"]

# 6. Reuse the same RuntimeSession on the next turn.
next_messages = [
    *messages,
    {"role": "assistant", "content": response_text},
    {
        "role": "user",
        "content": "Answer in one line and keep the same context.",
    },
]
next_prepared = tok.wrap(next_messages, model=model, session=session)
