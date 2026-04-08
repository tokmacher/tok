"""
LLM client interfaces and implementations.

This module provides protocol definitions and concrete implementations
for interacting with various Large Language Model providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI


class ChatLLM(Protocol):
    """
    Protocol for chat-based LLM interactions.

    Defines the interface for chat-based language models that can
    process system and user messages to generate responses.
    """

    def chat(self, *, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "deepseek/deepseek-v3.2"
    temperature: float = 0.2
    max_tokens: int = 700
    timeout_s: float = 120.0


class OpenRouterClient:
    def __init__(self, cfg: OpenRouterConfig) -> None:
        # Convenient local setup: if python-dotenv is installed, load `.env` from CWD.
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

        api_key = os.getenv(cfg.api_key_env, "")
        if not api_key:
            msg = f"{cfg.api_key_env} is not set"
            raise ValueError(msg)
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=api_key,
            timeout=cfg.timeout_s,
            max_retries=0,
        )

    def chat(self, *, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class StubClient:
    """Offline stub: always returns a placeholder function."""

    def chat(self, *, system: str, user: str) -> str:
        _ = system, user
        if "extract a concise, reusable rule" in system:
            return "Rule: define knowledge as power"
        if "Multiple Choice" in system or "choices" in user.lower():
            return "A"
        return "```python\ndef solve(x):\n    raise NotImplementedError\n```"
