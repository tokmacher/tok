"""Backward-compatible facade for tok.utils.pricing."""

from __future__ import annotations

from .utils.pricing import *  # noqa: F403
from .utils.pricing import PRICING_DEFAULT, get_pricing  # noqa: F401
