"""Model pricing table for cost calculations."""

from __future__ import annotations

# USD per million tokens: (input, output, cache_read, cache_write)
PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-4": (15.00, 75.00, 1.50, 18.75),
    "claude-sonnet-4": (3.00, 15.00, 0.30, 3.75),
    "claude-haiku-4": (0.80, 4.00, 0.08, 1.00),
    "claude-3-5-sonnet": (3.00, 15.00, 0.30, 3.75),
    "claude-3-5-haiku": (0.80, 4.00, 0.08, 1.00),
    "claude-3-opus": (15.00, 75.00, 1.50, 18.75),
    "claude-3-haiku": (0.25, 1.25, 0.03, 0.30),
    "claude-3-sonnet": (3.00, 15.00, 0.30, 3.75),
    "xiaomi/mimo-v2-pro": (1.00, 3.00, 0.10, 1.25),
    "z-ai/glm-5": (0.80, 3.00, 0.08, 1.00),
    "openai/gpt-5.4-pro": (30.00, 180.00, 3.00, 37.50),
    "x-ai/grok-4.20-beta": (2.00, 6.00, 0.20, 2.50),
    "google/gemini-3-flash-preview": (0.50, 3.00, 0.05, 0.60),
    "minimax/minimax-m2.7": (0.30, 1.20, 0.03, 0.35),
    "moonshotai/kimi-k2.5": (0.50, 2.80, 0.05, 0.70),
    "x-ai/grok-4.1-fast": (0.20, 0.50, 0.02, 0.15),
}

PRICING_DEFAULT = (3.00, 15.00, 0.30, 3.75)  # Sonnet rates fallback


def get_pricing(model: str) -> tuple[float, float, float, float]:
    """Look up pricing for a model by prefix match."""
    for prefix, rates in PRICING.items():
        if model.startswith(prefix):
            return rates
    return PRICING_DEFAULT
