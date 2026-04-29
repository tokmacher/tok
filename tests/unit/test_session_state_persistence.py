"""Tests for Bug 0.1.5-3: session state persistence across conversation restarts."""

from __future__ import annotations

import pytest

from tok.runtime.core import RuntimeSession
from tok.runtime.memory.bridge_memory import MemoryEntry


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

    def test_clears_hot_bridge_memory_but_preserves_durable(self) -> None:
        rs = RuntimeSession()
        rs.bridge_memory.hot["facts"] = [MemoryEntry(value="answer_file:src/old.py")]
        rs.bridge_memory.durable["facts"] = [MemoryEntry(value="project_fact")]
        rs.bridge_memory.rolling_cmds = [MemoryEntry(value="pytest")]

        rs.reset_session()

        assert rs.bridge_memory.hot == {}
        assert rs.bridge_memory.rolling_cmds == []
        assert rs.bridge_memory.durable["facts"][0].value == "project_fact"


class TestBridgeSessionBuckets:
    def test_explicit_session_headers_select_distinct_buckets(self, tmp_path) -> None:
        from tok.gateway import BridgeSession

        session = BridgeSession(memory_dir=tmp_path / ".tok")
        session.activate_session_for_request(
            {"x-tok-session-id": "alpha"}, {"messages": [{"role": "user", "content": "a"}]}
        )
        alpha_runtime = session.runtime_session
        alpha_runtime._files_read_this_session.add("alpha.py")
        alpha_runtime.bridge_memory.hot["facts"] = [MemoryEntry(value="answer_file:alpha.py")]

        session.activate_session_for_request(
            {"x-tok-session-id": "beta"}, {"messages": [{"role": "user", "content": "b"}]}
        )
        beta_runtime = session.runtime_session
        beta_runtime._files_read_this_session.add("beta.py")

        session.activate_session_for_request(
            {"x-tok-session-id": "alpha"}, {"messages": [{"role": "user", "content": "a"}]}
        )

        assert session.runtime_session is alpha_runtime
        assert session.runtime_session is not beta_runtime
        assert session.runtime_session._files_read_this_session == {"alpha.py"}
        assert session.runtime_session.bridge_memory.hot["facts"][0].value == "answer_file:alpha.py"

    def test_client_session_header_fallback_is_accepted(self, tmp_path) -> None:
        from tok.gateway import BridgeSession

        session = BridgeSession(memory_dir=tmp_path / ".tok")
        key = session.resolve_session_key({"x-claude-session-id": "claude-session"}, None)

        assert key.startswith("hdr:")

    def test_ephemeral_keys_separate_unknown_initial_conversations(self, tmp_path) -> None:
        from tok.gateway import BridgeSession

        session = BridgeSession(memory_dir=tmp_path / ".tok")
        headers = {"authorization": "Bearer secret", "user-agent": "client"}
        first = session.resolve_session_key(headers, {"messages": [{"role": "user", "content": "first"}]})
        second = session.resolve_session_key(headers, {"messages": [{"role": "user", "content": "second"}]})

        assert first.startswith("auto:")
        assert second.startswith("auto:")
        assert first != second

    def test_bucket_lru_eviction_keeps_active_bucket(self, tmp_path) -> None:
        from tok.gateway import BridgeSession

        session = BridgeSession(memory_dir=tmp_path / ".tok", max_sessions=2)
        session.activate_session_for_request({"x-tok-session-id": "one"}, None)
        session.activate_session_for_request({"x-tok-session-id": "two"}, None)
        active_key = session.activate_session_for_request({"x-tok-session-id": "three"}, None)

        assert active_key in session._session_buckets
        assert len(session._session_buckets) == 2

    def test_bound_session_view_survives_parent_activation_swap(self, tmp_path) -> None:
        from tok.gateway import BridgeSession

        session = BridgeSession(memory_dir=tmp_path / ".tok")
        alpha_key = session.activate_session_for_request({"x-tok-session-id": "alpha"}, None)
        alpha_runtime = session.runtime_session
        alpha_bound = session.bound_session_for_key(alpha_key)

        session.activate_session_for_request({"x-tok-session-id": "beta"}, None)

        assert session.runtime_session is not alpha_runtime
        assert alpha_bound.runtime_session is alpha_runtime

    def test_reset_all_sessions_tolerates_stale_active_key(self, tmp_path) -> None:
        from tok.gateway import BridgeSession

        session = BridgeSession(memory_dir=tmp_path / ".tok")
        session.activate_session_for_request({"x-tok-session-id": "alpha"}, None)
        session._active_session_key = "missing"

        session.reset_all_sessions()

        assert session.runtime_session is not None


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
    async def test_reset_session_endpoint_resets_only_current_bucket(self, test_client: tuple) -> None:  # type: ignore[no-untyped-def]
        client, session = test_client
        session.activate_session_for_request({"x-tok-session-id": "alpha"}, None)
        alpha_runtime = session.runtime_session
        alpha_runtime._files_read_this_session.add("alpha.py")
        session.activate_session_for_request({"x-tok-session-id": "beta"}, None)
        beta_runtime = session.runtime_session
        beta_runtime._files_read_this_session.add("beta.py")

        async with client:
            resp = await client.post("/reset-session", headers={"x-tok-session-id": "alpha"})
            assert resp.status_code == 200

        assert alpha_runtime._files_read_this_session == set()
        assert beta_runtime._files_read_this_session == {"beta.py"}

    @pytest.mark.asyncio
    async def test_reset_session_without_header_resolves_request_bucket(self, test_client: tuple) -> None:  # type: ignore[no-untyped-def]
        client, session = test_client
        headers = {"user-agent": "claude-a", "x-api-key": "test"}
        body = {"messages": [{"role": "user", "content": "first"}]}
        session.activate_session_for_request(headers, body)
        target_runtime = session.runtime_session
        target_runtime._files_read_this_session.add("target.py")
        session.tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=10,
            cache_read=0,
            cache_write=0,
            input_saved=25,
            output_saved=0,
        )
        session.activate_session_for_request({"x-tok-session-id": "other"}, None)
        other_runtime = session.runtime_session
        other_runtime._files_read_this_session.add("other.py")

        async with client:
            resp = await client.post("/reset-session", headers=headers)
            assert resp.status_code == 200

        assert target_runtime._files_read_this_session == set()
        assert other_runtime._files_read_this_session == {"other.py"}

    @pytest.mark.asyncio
    async def test_reset_session_scope_all_resets_all_buckets(self, test_client: tuple) -> None:  # type: ignore[no-untyped-def]
        client, session = test_client
        session.activate_session_for_request({"x-tok-session-id": "alpha"}, None)
        alpha_runtime = session.runtime_session
        alpha_runtime._files_read_this_session.add("alpha.py")
        session.activate_session_for_request({"x-tok-session-id": "beta"}, None)
        beta_runtime = session.runtime_session
        beta_runtime._files_read_this_session.add("beta.py")

        async with client:
            resp = await client.post("/reset-session?scope=all")
            assert resp.status_code == 200

        assert alpha_runtime._files_read_this_session == set()
        assert beta_runtime._files_read_this_session == set()

    @pytest.mark.asyncio
    async def test_session_headers_are_not_forwarded_upstream(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import httpx
        from fastapi.testclient import TestClient

        from tok.gateway import BridgeSession
        from tok.gateway._app_factory import create_app_impl

        client = TestClient(create_app_impl(session=BridgeSession()))
        captured_headers = {}

        async def fake_send(self, request, stream=False):  # type: ignore[no-untyped-def]
            del self, stream
            captured_headers.update(dict(request.headers))
            return httpx.Response(
                200,
                json={
                    "model": "claude-sonnet-4",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "content": [{"type": "text", "text": "ok"}],
                },
            )

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

        resp = client.post(
            "/v1/messages",
            headers={"x-api-key": "test", "x-tok-session-id": "alpha"},
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

        assert resp.status_code == 200
        assert "x-tok-session-id" not in captured_headers
        assert captured_headers.get("x-api-key") == "test"

    @pytest.mark.asyncio
    async def test_health_endpoint_still_works(self, test_client: tuple) -> None:  # type: ignore[no-untyped-def]
        client, session = test_client
        async with client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_reports_non_empty_bucket_when_active_bucket_is_empty(self, test_client: tuple) -> None:  # type: ignore[no-untyped-def]
        client, session = test_client
        session.activate_session_for_request({"x-tok-session-id": "alpha"}, None)
        session.tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=10,
            cache_read=0,
            cache_write=0,
            input_saved=25,
            output_saved=0,
        )
        session.activate_session_for_request({"x-tok-session-id": "empty"}, None)

        async with client:
            resp = await client.get("/health")
            assert resp.status_code == 200

        payload = resp.json()
        assert payload["session_tokens_saved"] > 0
        assert payload["session_savings_pct"] > 0
