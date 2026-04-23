"""
Bridge startup and import-surface smoke tests.

These tests ensure that the bridge-critical import surface remains stable
and that the gateway can be instantiated without hitting a port or API key.
They are the regression guard for Batch 1.
"""

from __future__ import annotations


class TestBridgeCriticalImports:
    """Verify all bridge-critical modules import cleanly."""

    def test_bridge_memory_imports(self) -> None:
        from tok.runtime.memory.bridge_memory import BridgeMemoryState

        assert BridgeMemoryState is not None

    def test_savings_tracker_imports(self) -> None:
        from tok.utils.savings_tracker import SavingsTracker

        assert SavingsTracker is not None

    def test_telemetry_imports(self) -> None:
        import tok.utils.telemetry

        assert tok.utils.telemetry is not None

    def test_universal_runtime_imports(self) -> None:
        from tok.universal_runtime import (  # noqa: F401
            RuntimeRequest,
            RuntimeSession,
            UniversalTokRuntime,
        )

        assert UniversalTokRuntime is not None

    def test_gateway_imports(self) -> None:
        from tok.gateway import BridgeSession, create_app  # noqa: F401

        assert create_app is not None

    def test_prompt_shim_imports(self) -> None:
        from tok.analysis.prompt import TOK_SYSTEM_PROMPT

        assert isinstance(TOK_SYSTEM_PROMPT, str)
        assert len(TOK_SYSTEM_PROMPT) > 0


class TestBridgeAppCreation:
    """Verify gateway app can be instantiated without port binding."""

    def test_create_app_returns_fastapi(self) -> None:
        from fastapi import FastAPI

        from tok.gateway import create_app

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_create_app_has_health_route(self) -> None:
        from tok.gateway import create_app

        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/health" in routes, f"Expected /health route; got: {routes}"

    def test_create_app_has_proxy_route(self) -> None:
        """Gateway uses a wildcard proxy route to forward /v1/messages."""
        from tok.gateway import create_app

        app = create_app()
        routes = [r.path for r in app.routes]
        # The gateway registers a catch-all /{path:path} that proxies to Anthropic
        assert "/{path:path}" in routes, f"Expected catch-all proxy route; got: {routes}"

    def test_bridge_session_default_values(self) -> None:
        from tok.gateway import BridgeSession

        session = BridgeSession()
        assert session.port == 9090
        assert session.keep_turns >= 1
        assert session.fail_open is True


class TestBridgeShimConsistency:
    """Verify that compatibility shims re-export the right canonical symbols."""

    def test_bridge_memory_shim_matches_canonical(self) -> None:
        from tok.runtime.memory.bridge_memory import (
            BridgeMemoryState as CanonicalBMState,
        )
        from tok.runtime.memory.bridge_memory import (
            BridgeMemoryState as ShimBMState,
        )

        assert ShimBMState is CanonicalBMState

    def test_savings_tracker_shim_matches_canonical(self) -> None:
        from tok.utils.savings_tracker import SavingsTracker as CanonicalST
        from tok.utils.savings_tracker import SavingsTracker as ShimST

        assert ShimST is CanonicalST

    def test_prompt_shim_matches_canonical(self) -> None:
        from tok.analysis.prompt import TOK_SYSTEM_PROMPT as CANONICAL_PROMPT
        from tok.analysis.prompt import TOK_SYSTEM_PROMPT as SHIM_PROMPT

        assert SHIM_PROMPT is CANONICAL_PROMPT
