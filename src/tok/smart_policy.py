"""Backward-compatible facade for tok.runtime.policy.smart_policy."""

from __future__ import annotations

from .runtime.policy.smart_policy import *  # noqa: F403
from .runtime.policy.smart_policy import (  # noqa: F401
    MemoryProjectionProfile,
    FamilyAdaptiveState,
    advance_state,
    identify_model_family,
    initial_state,
    policy_for_model,
    pressure_score,
    CANONICAL_WIRE_FIELD_ORDER,
)
