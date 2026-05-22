"""Internal registry for behavior signals that matter to diagnostics and release gates.

Signal definitions live in _signal_definitions.py to keep this module navigable.
All public names from the previous flat implementation remain importable from here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from tok.runtime._signal_definitions import (  # noqa: F401 — re-export for backward compat
    _SIGNALS,
    _VALID_CATEGORIES,
    _VALID_SEVERITIES,
    SignalDefinition,
    _signal,
)

logger = logging.getLogger("tok.signals")


class SignalRegistry:
    """Typed registry for SignalDefinition instances.

    Wraps the _SIGNALS tuple with a dict-backed lookup and a clean query API.
    The module-level SIGNAL_REGISTRY dict is the same object as _registry._dict,
    so all existing dict-based reads continue to work without copying.
    """

    def __init__(self, signals: tuple[SignalDefinition, ...]) -> None:
        self._signals = signals
        self._dict: dict[str, SignalDefinition] = {s.name: s for s in signals}

    def get(self, name: str) -> SignalDefinition | None:
        return self._dict.get(name)

    def all(self) -> tuple[SignalDefinition, ...]:
        return self._signals

    def by_category(self, category: str) -> tuple[SignalDefinition, ...]:
        return tuple(s for s in self._signals if s.category == category)

    def names(self) -> frozenset[str]:
        return frozenset(self._dict)

    def __contains__(self, name: object) -> bool:
        return name in self._dict

    def __len__(self) -> int:
        return len(self._signals)


_REGISTRY = SignalRegistry(_SIGNALS)

# Preserve the existing dict interface — same object as _REGISTRY._dict.
SIGNAL_REGISTRY: dict[str, SignalDefinition] = _REGISTRY._dict

EVIDENCE_SAFETY_SIGNAL_NAMES: tuple[str, ...] = tuple(
    signal.name for signal in _SIGNALS if signal.category == "evidence_safety"
)


def signal_definition(name: str) -> SignalDefinition | None:
    """Return registered metadata for a signal, or None for accepted internal signals."""
    return SIGNAL_REGISTRY.get(name)


def is_registered_signal(name: str) -> bool:
    return name in SIGNAL_REGISTRY


def signals_by_category(category: str) -> tuple[SignalDefinition, ...]:
    return tuple(signal for signal in _SIGNALS if signal.category == category)


def aggregate_signal_category(signals: Mapping[str, int], category: str) -> int:
    names = {signal.name for signal in signals_by_category(category)}
    return sum(int(value) for name, value in signals.items() if name in names)


def aggregate_registered_categories(signals: Mapping[str, int]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for name, value in signals.items():
        definition = signal_definition(name)
        if definition is None:
            continue
        totals[definition.category] = totals.get(definition.category, 0) + int(value)
    return totals


def unregistered_signals(signals: Mapping[str, int]) -> dict[str, int]:
    return {name: int(value) for name, value in signals.items() if name not in SIGNAL_REGISTRY}


def debug_unregistered_signals(signals: Mapping[str, int]) -> None:
    """Log unregistered signal keys at session end when TOK_DEBUG=1.

    Intended to be called once per session at teardown. Only emits output
    when the ``TOK_DEBUG`` environment variable is set to ``1``.
    """
    import os

    if os.environ.get("TOK_DEBUG") != "1":
        return
    unknown = unregistered_signals(signals)
    if not unknown:
        return
    names = ", ".join(sorted(unknown))
    logger.debug(
        "Session ended with %d unregistered behavior signal(s): %s",
        len(unknown),
        names,
    )


def warn_unregistered_signals(signals: Mapping[str, int]) -> None:
    """Emit a WARNING for each unregistered signal when TOK_ENFORCE_SIGNAL_REGISTRY=1.

    Intended to be called at session end. No-op when the env var is absent or
    not ``"1"``. This is the enforcement mode that can be made the default in a
    future release once all signals are registered.
    """
    import os

    if os.environ.get("TOK_ENFORCE_SIGNAL_REGISTRY") != "1":
        return
    unknown = unregistered_signals(signals)
    if not unknown:
        return
    names = ", ".join(sorted(unknown))
    logger.warning(
        "TOK_ENFORCE_SIGNAL_REGISTRY: %d unregistered behavior signal(s) emitted: %s",
        len(unknown),
        names,
    )


__all__ = [
    "EVIDENCE_SAFETY_SIGNAL_NAMES",
    "SIGNAL_REGISTRY",
    "SignalDefinition",
    "SignalRegistry",
    "aggregate_registered_categories",
    "aggregate_signal_category",
    "debug_unregistered_signals",
    "is_registered_signal",
    "signal_definition",
    "signals_by_category",
    "unregistered_signals",
    "warn_unregistered_signals",
]
