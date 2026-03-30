"""Backward-compatible alias for tok.testing.live_runner.

Tests patch tok.live_runner.OpenAI / tok.live_runner.config directly, so this
module must share the same module object as the canonical implementation.
"""

from __future__ import annotations

import sys
import tok.testing.live_runner as _live_mod

sys.modules[__name__] = _live_mod
