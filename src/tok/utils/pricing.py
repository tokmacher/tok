"""Model pricing table for cost calculations."""

from __future__ import annotations

# Last reviewed against external provider pages on 2026-04-08.
# Canonical source file: src/tok/utils/pricing.py
# See docs/pricing_verification.md for verification status and source links.

# USD per million tokens: (input, output, cache_read, cache_write)
PRICING: dict[str, tuple[float, float, float, float]] = {
    # Anthropic (verified from docs.anthropic.com pricing page).
    "claude-opus-4": (15.00, 75.00, 1.50, 18.75),
    "claude-sonnet-4": (3.00, 15.00, 0.30, 3.75),
    "claude-haiku-4": (0.80, 4.00, 0.08, 1.00),
    "claude-3-5-sonnet": (3.00, 15.00, 0.30, 3.75),
    "claude-3-5-haiku": (0.80, 4.00, 0.08, 1.00),
    "claude-3-opus": (15.00, 75.00, 1.50, 18.75),
    "claude-3-haiku": (0.25, 1.25, 0.03, 0.30),
    "claude-3-sonnet": (3.00, 15.00, 0.30, 3.75),
    # OpenAI (verified from developers.openai.com model docs).
    # Cache rates not published for this model -> track as zero until verified.
    "openai/gpt-5.4-pro": (30.00, 180.00, 0.00, 0.00),
    # OpenRouter/aggregator-derived references (reviewed, not release-defining).
    "xiaomi/mimo-v2-pro": (1.00, 3.00, 0.10, 1.25),
    "z-ai/glm-5": (0.72, 2.30, 0.00, 0.00),
    "x-ai/grok-4.20-beta": (2.00, 6.00, 0.00, 0.00),
    "google/gemini-3-flash-preview": (0.50, 3.00, 0.00, 0.00),
    "minimax/minimax-m2.7": (0.30, 1.20, 0.06, 0.38),
    "moonshotai/kimi-k2.5": (0.38, 1.72, 0.00, 0.00),
    "x-ai/grok-4.1-fast": (0.20, 0.50, 0.00, 0.00),
}

PRICING_DEFAULT = (3.00, 15.00, 0.30, 3.75)  # Sonnet rates fallback


def get_pricing(model: str) -> tuple[float, float, float, float]:
    """Look up pricing for a model by prefix match."""
    for prefix, rates in PRICING.items():
        if model.startswith(prefix):
            return rates
    return PRICING_DEFAULT
