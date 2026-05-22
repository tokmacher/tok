"""RED tests for the SignalRegistry class interface.

Written against the planned interface before it exists in signals.py.
These tests verify the new typed class without breaking the existing module-level API.
"""

from __future__ import annotations


class TestSignalRegistryClass:
    def test_registry_class_get_known_signal(self) -> None:
        from tok.runtime.signals import _SIGNALS, SignalRegistry

        registry = SignalRegistry(_SIGNALS)
        result = registry.get("tok_fallback_activated")
        assert result is not None
        assert result.name == "tok_fallback_activated"

    def test_registry_class_get_unknown_returns_none(self) -> None:
        from tok.runtime.signals import _SIGNALS, SignalRegistry

        registry = SignalRegistry(_SIGNALS)
        assert registry.get("not_a_real_signal_xyz") is None

    def test_registry_class_all_returns_same_as_tuple(self) -> None:
        from tok.runtime.signals import _SIGNALS, SignalRegistry

        registry = SignalRegistry(_SIGNALS)
        assert registry.all() == _SIGNALS

    def test_registry_class_by_category_matches_module_function(self) -> None:
        from tok.runtime.signals import _SIGNALS, SignalRegistry, signals_by_category

        registry = SignalRegistry(_SIGNALS)
        assert registry.by_category("fallback") == signals_by_category("fallback")

    def test_registry_class_contains_known_signal(self) -> None:
        from tok.runtime.signals import _SIGNALS, SignalRegistry

        registry = SignalRegistry(_SIGNALS)
        assert "tok_fallback_activated" in registry

    def test_registry_class_not_contains_unknown(self) -> None:
        from tok.runtime.signals import _SIGNALS, SignalRegistry

        registry = SignalRegistry(_SIGNALS)
        assert "not_a_real_signal_xyz" not in registry

    def test_registry_class_len_matches_signals_tuple(self) -> None:
        from tok.runtime.signals import _SIGNALS, SignalRegistry

        registry = SignalRegistry(_SIGNALS)
        assert len(registry) == len(_SIGNALS)

    def test_registry_class_names_is_frozenset(self) -> None:
        from tok.runtime.signals import _SIGNALS, SignalRegistry

        registry = SignalRegistry(_SIGNALS)
        names = registry.names()
        assert isinstance(names, frozenset)
        assert "tok_fallback_activated" in names


class TestModuleLevelBackwardCompat:
    def test_signal_registry_dict_still_subscriptable(self) -> None:
        from tok.runtime.signals import SIGNAL_REGISTRY

        result = SIGNAL_REGISTRY["tok_fallback_activated"]
        assert result.name == "tok_fallback_activated"

    def test_module_level_registry_instance_accessible(self) -> None:
        """_REGISTRY should be the module-level SignalRegistry instance."""
        from tok.runtime.signals import _REGISTRY, SignalRegistry

        assert isinstance(_REGISTRY, SignalRegistry)

    def test_signal_definitions_importable_from_new_module(self) -> None:
        """_SIGNALS must be importable from _signal_definitions after the refactor."""
        from tok.runtime._signal_definitions import _SIGNALS

        assert len(_SIGNALS) > 100  # at minimum 174 signals defined

    def test_signal_factory_importable_from_new_module(self) -> None:
        from tok.runtime._signal_definitions import _signal

        result = _signal(
            "test_signal_xyz",
            category="trace",
            severity="info",
            label="test only",
            release_critical=False,
        )
        assert result.name == "test_signal_xyz"
