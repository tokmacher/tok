"""
Minimal recipe for the experimental explicit-submodule runtime path.

Caller owns the RuntimeSession, sends ``prepared.body`` to an OpenAI-compatible
SDK call, then reuses the same session on the next turn.
"""

from __future__ import annotations

import os

import openai

from tok.runtime.core import RuntimeSession
from tok.runtime.types import RuntimeRequest

session = RuntimeSession()

messages = [
    {
        "role": "user",
        "content": "What file in this repo owns core runtime semantics?",
    },
]
model = os.getenv("TOK_MODEL", "deepseek/deepseek-v3.2")


runtime = __import__("tok.universal_runtime", fromlist=["UniversalTokRuntime"]).UniversalTokRuntime()
request = RuntimeRequest(model=model, messages=messages, system=None, adapter_kind="wrap", tool_compatible=True)
prepared = runtime.prepare_request(request, session)

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

result = runtime.process_response(response_text, model=model, session=session, tool_compatible=True)
visible_blocks = [b for b in result.content_blocks if b.get("type") == "text"]

next_messages = [
    *messages,
    {"role": "assistant", "content": response_text},
    {
        "role": "user",
        "content": "Answer in one line and keep the same context.",
    },
]
next_request = RuntimeRequest(
    model=model, messages=next_messages, system=None, adapter_kind="wrap", tool_compatible=True
)
next_prepared = runtime.prepare_request(next_request, session)
