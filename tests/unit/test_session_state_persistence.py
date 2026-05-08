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
    def test_auth_bucket_token_cache_is_bounded(self, monkeypatch) -> None:
        import tok.gateway as gateway

        monkeypatch.setattr(gateway, "_AUTH_TOKEN_CACHE_MAX", 2)
        gateway._AUTH_TOKEN_CACHE.clear()

        gateway._auth_bucket_token("Bearer a")
        gateway._auth_bucket_token("Bearer b")
        gateway._auth_bucket_token("Bearer c")

        assert len(gateway._AUTH_TOKEN_CACHE) == 2
        assert "Bearer a" not in gateway._AUTH_TOKEN_CACHE

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

    def test_bucket_eviction_persists_tracker_to_ledger(self, tmp_path) -> None:
        from tok.gateway import BridgeSession

        merged: list[str] = []

        class _FakeTracker:
            def __init__(self, key: str) -> None:
                self._key = key
                self.savings_file = "fake.tok"

            def merge_session_to_ledger(self) -> None:
                merged.append(self._key)

            def reset_session_stats(self) -> None:
                return None

            def session_summary(self) -> dict[str, int]:
                return {"calls": 1}

        session = BridgeSession(memory_dir=tmp_path / ".tok", max_sessions=1)
        session._new_savings_tracker = lambda key: _FakeTracker(key)  # type: ignore[method-assign]

        session.activate_session_for_request({"x-tok-session-id": "one"}, None)
        session.activate_session_for_request({"x-tok-session-id": "two"}, None)

        assert any(key.startswith("hdr:") for key in merged)
        assert len(merged) == 1

    def test_auto_fingerprint_map_pruned_when_bucket_evicted(self, tmp_path) -> None:
        from tok.gateway import BridgeSession

        session = BridgeSession(memory_dir=tmp_path / ".tok", max_sessions=1)
        h1 = {"authorization": "Bearer one", "user-agent": "ua"}
        h2 = {"authorization": "Bearer two", "user-agent": "ua"}

        key1 = session.activate_session_for_request(h1, {"messages": [{"role": "user", "content": "one"}]})
        assert key1.startswith("auto:")
        fp1 = session._auto_fingerprint({k.lower(): v for k, v in h1.items()})
        assert session._auto_fingerprint_to_key.get(fp1) == key1

        key2 = session.activate_session_for_request(h2, {"messages": [{"role": "user", "content": "two"}]})
        assert key2.startswith("auto:")
        assert key1 not in session._session_buckets
        assert fp1 not in session._auto_fingerprint_to_key

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

    @pytest.mark.asyncio
    async def test_health_explicit_session_header_reports_that_bucket(self, test_client: tuple) -> None:  # type: ignore[no-untyped-def]
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
        session.activate_session_for_request({"x-tok-session-id": "beta"}, None)
        session.runtime_session._baseline_only = True
        session.runtime_session._bump_signals({"tok_fallback_activated": 1})

        async with client:
            resp = await client.get("/health", headers={"x-tok-session-id": "beta"})
            assert resp.status_code == 200

        payload = resp.json()
        assert payload["baseline_only"] is True
        assert payload["fallback_count"] == 1
        assert payload["session_tokens_saved"] == 0


class TestIsFirstRequestFlag:
    """RuntimeSession._is_first_request tracks fresh session start vs resumption."""

    def test_starts_true_on_new_session(self) -> None:
        rs = RuntimeSession()
        assert rs._is_first_request is True

    def test_stays_false_after_being_set(self) -> None:
        rs = RuntimeSession()
        rs._is_first_request = False
        assert rs._is_first_request is False

    def test_reset_session_resets_to_true(self) -> None:
        rs = RuntimeSession()
        rs._is_first_request = False
        rs.reset_session()
        assert rs._is_first_request is True


class TestFreshSessionPointerInjection:
    """On a fresh session start, a pointer existence notice is injected once."""

    def test_pointer_notice_injected_on_first_request(self, tmp_path) -> None:
        from unittest.mock import patch

        from tok.compression import inject_system_additions
        from tok.runtime._request_preparation import prepare_request_impl
        from tok.runtime.core import RuntimeSession, UniversalTokRuntime
        from tok.runtime.memory.bridge_memory import MemoryEntry
        from tok.runtime.types import RuntimeRequest

        rs = RuntimeSession(memory_dir=tmp_path / ".tok")
        rs.bridge_memory.turn = 100
        rs.bridge_memory.pointers.get_pointer("/repo/src/foo.py")
        rs.bridge_memory.durable["facts"] = [MemoryEntry(value="answer_file:/repo/src/foo.py")]
        (tmp_path / ".tok").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".tok" / "bridge_memory.tok").write_text("@mem v:b1 t:100\n")
        assert rs._is_first_request is True

        runtime = UniversalTokRuntime()

        captured_hints: list[str] = []

        def capture_inject(body, **kwargs):
            captured_hints.extend(kwargs.get("runtime_hints", []) or [])
            return inject_system_additions(body, **kwargs)

        req = RuntimeRequest(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "hello"}],
            adapter_kind="claude-bridge",
            tool_compatible=False,
        )

        with patch("tok.runtime._request_preparation.inject_system_additions", side_effect=capture_inject):
            with patch("tok.runtime._request_preparation._should_skip_history_rewrite", return_value=(False, "")):
                prepare_request_impl(runtime, req, rs)

        assert any("[tok] Session memory:" in h and "bridge_memory.tok" in h for h in captured_hints), (
            f"Memory hint not found in runtime_hints: {captured_hints}"
        )
        assert rs._is_first_request is False

    def test_pointer_notice_not_injected_on_second_request(self, tmp_path) -> None:
        from unittest.mock import patch

        from tok.compression import inject_system_additions
        from tok.runtime._request_preparation import prepare_request_impl
        from tok.runtime.core import RuntimeSession, UniversalTokRuntime
        from tok.runtime.memory.bridge_memory import MemoryEntry
        from tok.runtime.types import RuntimeRequest

        rs = RuntimeSession(memory_dir=tmp_path / ".tok")
        rs.bridge_memory.turn = 100
        rs.bridge_memory.pointers.get_pointer("/repo/src/bar.py")
        rs.bridge_memory.durable["facts"] = [MemoryEntry(value="answer_file:/repo/src/bar.py")]
        rs._is_first_request = False

        runtime = UniversalTokRuntime()

        captured_hints: list[str] = []

        def capture_inject(body, **kwargs):
            captured_hints.extend(kwargs.get("runtime_hints", []) or [])
            return inject_system_additions(body, **kwargs)

        req = RuntimeRequest(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "hello"}],
            adapter_kind="claude-bridge",
            tool_compatible=False,
        )

        with patch("tok.runtime._request_preparation.inject_system_additions", side_effect=capture_inject):
            with patch("tok.runtime._request_preparation._should_skip_history_rewrite", return_value=(False, "")):
                prepare_request_impl(runtime, req, rs)

        assert not any("[tok] Session memory:" in h for h in captured_hints), (
            f"Memory hint unexpectedly found in runtime_hints on second request: {captured_hints}"
        )

    def test_short_session_no_injection(self, tmp_path) -> None:
        from unittest.mock import patch

        from tok.compression import inject_system_additions
        from tok.runtime._request_preparation import prepare_request_impl
        from tok.runtime.core import RuntimeSession, UniversalTokRuntime
        from tok.runtime.types import RuntimeRequest

        rs = RuntimeSession(memory_dir=tmp_path / ".tok")
        rs.bridge_memory.turn = 3
        rs.bridge_memory.pointers.get_pointer("/repo/src/baz.py")
        rs._is_first_request = True

        runtime = UniversalTokRuntime()

        captured_hints: list[str] = []

        def capture_inject(body, **kwargs):
            captured_hints.extend(kwargs.get("runtime_hints", []) or [])
            return inject_system_additions(body, **kwargs)

        req = RuntimeRequest(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "hello"}],
            adapter_kind="claude-bridge",
            tool_compatible=False,
        )

        with patch("tok.runtime._request_preparation.inject_system_additions", side_effect=capture_inject):
            with patch("tok.runtime._request_preparation._should_skip_history_rewrite", return_value=(False, "")):
                prepare_request_impl(runtime, req, rs)

        assert not any("[tok] Session memory:" in h for h in captured_hints), (
            f"Memory hint unexpectedly found for short session: {captured_hints}"
        )
        assert rs._is_first_request is False

    def test_first_request_flag_clears_even_without_pointers(self, tmp_path) -> None:
        from tok.runtime._request_preparation import prepare_request_impl
        from tok.runtime.core import RuntimeSession, UniversalTokRuntime
        from tok.runtime.types import RuntimeRequest

        rs = RuntimeSession(memory_dir=tmp_path / ".tok")
        rs.bridge_memory.turn = 100
        rs._is_first_request = True
        assert not rs.bridge_memory.pointers.map

        runtime = UniversalTokRuntime()
        req = RuntimeRequest(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "hello"}],
            adapter_kind="claude-bridge",
            tool_compatible=False,
        )
        prepare_request_impl(runtime, req, rs)
        assert rs._is_first_request is False


class TestCrossSessionMacroPersistence:
    """distill_bridge_history must be called at session eviction to persist macro patterns."""

    def test_distill_called_on_bucket_eviction(self, tmp_path) -> None:
        """When a bucket is evicted, distill_bridge_history must run on its bridge_memory."""
        from unittest.mock import patch

        from tok.gateway import BridgeSession

        distilled: list[str] = []

        with patch(
            "tok.gateway.distill_bridge_history",
            side_effect=lambda _memory_state, **_kw: distilled.append("called") or [],
        ):
            session = BridgeSession(memory_dir=tmp_path / ".tok", max_sessions=1)
            session.activate_session_for_request({"x-tok-session-id": "alpha"}, None)
            session.activate_session_for_request({"x-tok-session-id": "beta"}, None)

        assert distilled, "distill_bridge_history was not called during eviction"

    def test_distill_called_on_flush_all(self, tmp_path) -> None:
        """merge_all_trackers_to_ledger must also trigger distill_bridge_history on each bucket."""
        from unittest.mock import patch

        from tok.gateway import BridgeSession

        distilled: list[str] = []

        with patch(
            "tok.gateway.distill_bridge_history",
            side_effect=lambda _memory_state, **_kw: distilled.append("called") or [],
        ):
            session = BridgeSession(memory_dir=tmp_path / ".tok", max_sessions=5)
            session.activate_session_for_request({"x-tok-session-id": "one"}, None)
            session.activate_session_for_request({"x-tok-session-id": "two"}, None)
            session.merge_all_trackers_to_ledger()

        assert len(distilled) >= 2, f"Expected >=2 distill calls, got {len(distilled)}"

    def test_distill_failure_does_not_block_ledger_merge(self, tmp_path) -> None:
        """Macro mining is best-effort; a mining error must not lose savings stats."""
        from unittest.mock import patch

        from tok.gateway import BridgeSession

        session = BridgeSession(memory_dir=tmp_path / ".tok")
        bucket = session._session_buckets["default"]
        bucket.tracker.record_call(
            model="claude-sonnet-4-6",
            actual_input=1000,
            actual_output=200,
            cache_read=0,
            cache_write=0,
            input_saved=500,
            output_saved=0,
        )

        with patch("tok.gateway.distill_bridge_history", side_effect=RuntimeError("mine failed")):
            session.merge_all_trackers_to_ledger()

        assert bucket.tracker.ledger_path.exists()
        summary = bucket.tracker.lifetime_summary()
        assert summary is not None
        assert summary["sessions"] >= 1
        assert summary["tokens_saved"] >= 500


class TestDefaultTrackerPromotion:
    def test_default_tracker_flushed_before_promotion(self, tmp_path) -> None:
        """Stats recorded on the default tracker must reach the ledger when the
        default bucket is promoted to a keyed session. Previously the old tracker
        was silently replaced without flushing, losing all pre-identification turns."""
        from unittest.mock import patch

        from tok.gateway import BridgeSession
        from tok.stats import SavingsTracker

        ledger_path = tmp_path / "global_savings.tok"
        savings_file = str(tmp_path / "tok_savings.tok")

        session = BridgeSession(memory_dir=tmp_path / ".tok")
        default_tracker = SavingsTracker(savings_file=savings_file, ledger_path=ledger_path)
        session.tracker = default_tracker
        session._session_buckets["default"].tracker = default_tracker

        session.tracker.record_call(
            model="claude-sonnet-4-6",
            actual_input=1000,
            actual_output=200,
            cache_read=0,
            cache_write=0,
            input_saved=500,
            output_saved=0,
        )

        with patch("tok.gateway._default_savings_file", return_value=savings_file):
            session.activate_session_for_request({"x-tok-session-id": "s1"}, None)

        assert ledger_path.exists(), "Ledger not written on default tracker promotion"
        text = ledger_path.read_text()
        assert "sessions: 1" in text, "Pre-promotion session missing from ledger"
        assert "tokens_saved: 500" in text, "Pre-promotion savings missing from ledger"
