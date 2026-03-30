"""
Tok Live Runner - Industrial Validation Module
==============================================
Real LLM execution via OpenRouter with usage tracking.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, cast

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from ..utils import config
from ..adapters import OpenAIChatAdapter

load_dotenv()

MODEL = "deepseek/deepseek-v3.2"

# Token pricing - DeepSeek V3.2 ($/M tokens)
PRICING = {
    "prompt": 0.26,
    "completion": 0.38,
}

enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Usage:
    """Token usage from API call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float


class LiveAgent:
    """Live LLM agent with OpenRouter and usage tracking."""

    def __init__(
        self,
        model: str = MODEL,
        temperature: float = 0.7,
        max_tokens: int = 500,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        api_key = config.API_KEY or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in config or environment"
            )

        self.client = OpenAI(
            base_url=config.API_BASE,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
        self.adapter = OpenAIChatAdapter()

        self.last_prompt = ""
        self.last_response = ""
        self.last_usage: Usage | None = None
        self.call_count = 0

    def __call__(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[str, Usage]:
        """
        Execute a single call to the LLM.

        Returns:
            (response_text, Usage)
        """
        self.last_prompt = prompt
        self.call_count += 1

        messages, prepared = self.adapter.build_chat_messages(
            model=self.model,
            user_text=prompt,
            system_prompt=system_prompt,
        )

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=cast(Any, messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            cost = (
                prompt_tokens * PRICING["prompt"] / 1_000_000
                + completion_tokens * PRICING["completion"] / 1_000_000
            )

            self.last_usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

            raw_response = response.choices[0].message.content or ""
            processed = self.adapter.finalize(
                text=raw_response,
                model=self.model,
                behavior_signals=prepared.behavior_signals,
            )
            self.last_response = (
                self.adapter.visible_text(processed) or raw_response
            )

            return self.last_response, self.last_usage

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.last_usage = Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=latency_ms,
                cost_usd=0,
            )
            self.last_response = f"ERROR: {str(e)}"
            return self.last_response, self.last_usage

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using cl100k_base."""
        return len(enc.encode(text))

    def get_stats(self) -> dict[str, Any]:
        """Get cumulative statistics."""
        return {
            "calls": self.call_count,
            "last_prompt_tokens": self.count_tokens(self.last_prompt),
            "last_response_tokens": self.count_tokens(
                self.last_response or ""
            ),
            "last_usage": self.last_usage,
        }

    def stream_call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """True SSE streaming from OpenRouter. Yields response chunks."""
        self.last_prompt = prompt
        self.call_count += 1

        messages, prepared = self.adapter.build_chat_messages(
            model=self.model,
            user_text=prompt,
            system_prompt=system_prompt,
        )

        start_time = time.time()
        buffer = ""
        tokens_to_use = (
            max_tokens if max_tokens is not None else self.max_tokens
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=cast(Any, messages),
                temperature=self.temperature,
                max_tokens=tokens_to_use,
                stream=True,
            )

            for chunk in response:
                if (
                    not hasattr(chunk, "choices")
                    or not cast(Any, chunk).choices
                ):
                    continue
                delta = cast(Any, chunk).choices[0].delta
                if delta and delta.content:
                    buffer += delta.content
                    yield buffer

            latency_ms = (time.time() - start_time) * 1000

            usage = self.client.chat.completions.create(
                model=self.model,
                messages=cast(Any, messages),
                max_tokens=1,
            ).usage

            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            cost = (
                prompt_tokens * PRICING["prompt"] / 1_000_000
                + completion_tokens * PRICING["completion"] / 1_000_000
            )

            self.last_usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
            )
            processed = self.adapter.finalize(
                text=buffer,
                model=self.model,
                behavior_signals=prepared.behavior_signals,
            )
            self.last_response = self.adapter.visible_text(processed) or buffer

        except Exception as e:
            self.last_response = f"ERROR: {str(e)}"
            yield self.last_response


def test_connection() -> bool:
    """Test OpenRouter connection."""
    agent = LiveAgent()
    try:
        response, usage = agent("Say 'hello' in one word.")
        print(f"Connection OK: {response[:50]}...")
        print(f"Usage: {usage}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
