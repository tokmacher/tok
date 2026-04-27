"""Tests for Bug 0.1.5-3: session state persistence across conversation restarts."""

from __future__ import annotations

import pytest

from tok.runtime.core import RuntimeSession


class TestResetSessionClearsState:
    """reset_session() must clear all first-exact / first-read tracking state."""

    def test_clears_files_read_this_session(self) -> None:
        rs = RuntimeSession()
        rs._files_read_this_session.add("/repo/src/foo.py")
        rs._files_read_this_session.add("/repo/src/bar.py")
        assert len(rs._files_read_this_session) == 2
        rs.reset_session()
        assert len(rs._files_read_this_session) == 0

    def test_clears_first_exact_evidence_seen(self) -> None:
        rs = RuntimeSession()
        rs._first_exact_evidence_seen.add("/repo/src/foo.py")
        assert len(rs._first_exact_evidence_seen) == 1
        rs.reset_session()
        assert len(rs._first_exact_evidence_seen) == 0

    def test_clears_pending_exact_evidence_keys(self) -> None:
        rs = RuntimeSession()
        rs._pending_exact_evidence_keys.add("/repo/src/baz.py")
        assert len(rs._pending_exact_evidence_keys) == 1
        rs.reset_session()
        assert len(rs._pending_exact_evidence_keys) == 0

    def test_clears_skeleton_delivered_paths(self) -> None:
        rs = RuntimeSession()
        rs._skeleton_delivered_paths.add("/repo/src/README.md")
        rs.reset_session()
        assert len(rs._skeleton_delivered_paths) == 0

    def test_clears_result_cache(self) -> None:
        rs = RuntimeSession()
        rs.result_cache["key1"] = {"raw": "data"}
        assert len(rs.result_cache) == 1
        rs.reset_session()
        assert len(rs.result_cache) == 0


class TestResetSessionEndpoint:
    """POST /reset-session must be reachable and reset transient state."""

    @pytest.fixture()
    def test_client(self):  # type: ignore[no-untyped-def]
        from httpx import ASGITransport, AsyncClient

        from tok.gateway import BridgeSession
        from tok.gateway._app_factory import create_app_impl

        session = BridgeSession()
        app = create_app_impl(session=session)

        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test"), session

    @pytest.mark.asyncio
    async def test_reset_session_endpoint_returns_ok(self, test_client: tuple) -> None:  # type: ignore[no-untyped-def]
        client, session = test_client
        async with client:
            resp = await client.post("/reset-session")
            assert resp.status_code == 200
            body = resp.json()
            assert body.get("status") == "ok"
            assert "reset" in body.get("action", "").lower()

    @pytest.mark.asyncio
    async def test_reset_session_endpoint_clears_first_exact(self, test_client: tuple) -> None:  # type: ignore[no-untyped-def]
        client, session = test_client
        session.runtime_session._first_exact_evidence_seen.add("/foo.py")
        session.runtime_session._files_read_this_session.add("/foo.py")
        assert len(session.runtime_session._first_exact_evidence_seen) == 1

        async with client:
            resp = await client.post("/reset-session")
            assert resp.status_code == 200

        assert len(session.runtime_session._first_exact_evidence_seen) == 0
        assert len(session.runtime_session._files_read_this_session) == 0

    @pytest.mark.asyncio
    async def test_health_endpoint_still_works(self, test_client: tuple) -> None:  # type: ignore[no-untyped-def]
        client, session = test_client
        async with client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
