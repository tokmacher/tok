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

# Sonnet rates, kept as a labelled reference only. It is deliberately NOT the
# fallback for unknown models: pricing an unknown model at a fabricated rate
# produces misleading dollar cost/savings (an overclaim or underclaim that looks
# authoritative). Callers that genuinely want a reference rate can import this.
PRICING_DEFAULT = (3.00, 15.00, 0.30, 3.75)

# Conservative pricing for a model whose rates we do not actually know. Dollar
# cost/savings collapse to zero so the bridge never fabricates a dollar figure;
# token savings (which are rate-independent) are unaffected.
PRICING_UNKNOWN = (0.0, 0.0, 0.0, 0.0)


def _lookup_pricing(model: str) -> tuple[float, float, float, float] | None:
    """Return the registered rates for *model* by prefix match, or ``None``."""
    for prefix, rates in PRICING.items():
        if model.startswith(prefix):
            return rates
    return None


def has_known_pricing(model: str) -> bool:
    """Return ``True`` when *model* has registered (non-fabricated) pricing.

    Lets cost/savings surfaces distinguish a real dollar figure from one that
    would be guessed, so they can suppress or annotate dollar claims for models
    Tok does not have verified rates for.
    """
    return _lookup_pricing(model) is not None


def get_pricing(model: str) -> tuple[float, float, float, float]:
    """Look up pricing for a model by prefix match.

    Unknown models return conservative zero rates (``PRICING_UNKNOWN``) rather
    than a fabricated default, so the bridge never reports misleading dollar
    savings for a model whose price it does not actually know. Use
    :func:`has_known_pricing` to detect the unknown case explicitly.
    """
    rates = _lookup_pricing(model)
    return rates if rates is not None else PRICING_UNKNOWN
