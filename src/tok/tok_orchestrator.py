"""Backward-compatible facade for tok.adapters.orchestrator.

This module registers itself as an alias of tok.adapters.orchestrator so that
monkeypatching tok.tok_orchestrator.OpenAI (etc.) affects TokOrchestrator.__init__.
"""

from __future__ import annotations

import sys

import tok.adapters.orchestrator as _orch_mod

# Make tok.tok_orchestrator and tok.adapters.orchestrator share the same module
# object, so monkeypatching either namespace affects the other.
sys.modules[__name__] = _orch_mod
