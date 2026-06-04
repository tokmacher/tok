"""Tests for delivery-state persistence across bridge process restarts."""

from __future__ import annotations

import json
import time

from tok.runtime._session_persistence import (
    delivery_state_file,
    load_delivery_state,
    save_delivery_state,
)
from tok.runtime.core import RuntimeSession


class TestDeliveryStateRoundTrip:
    def test_save_and_load_restores_all_trackers(self, tmp_path) -> None:
        session = RuntimeSession()
        session.memory_dir = tmp_path / ".tok"
        session.memory_dir.mkdir(parents=True, exist_ok=True)

        session._files_read_this_session.add("/repo/src/foo.py")
        session._files_read_this_session.add("/repo/src/bar.py")
        session._files_fully_delivered["/repo/src/foo.py"] = 5
        session._skeleton_delivered_paths.add("/repo/src/README.md")
        session._files_read_fingerprints["/repo/src/foo.py"] = "abc123"
        session.evidence_safety.first_exact_seen.add("/repo/src/foo.py")
        session.evidence_safety.pending_exact_keys.add("/repo/src/baz.py")
        session.evidence_safety.alias_map["foo"] = "/repo/src/foo.py"
        session.evidence_safety.ledger["/repo/src/foo.py"] = type(
            "Entry", (), {"key": "/repo/src/foo.py", "latest_digest": "d1"}
        )()
        session.evidence_safety.ledger["/repo/src/foo.py"].__dataclass_fields__ = None

        from tok.runtime.evidence_safety import EvidenceLedgerEntry

        session.evidence_safety.ledger["/repo/src/foo.py"] = EvidenceLedgerEntry(
            key="/repo/src/foo.py", latest_digest="d1"
        )

        save_delivery_state(session)

        fresh = RuntimeSession()
        fresh.memory_dir = tmp_path / ".tok"
        load_delivery_state(fresh)

        assert fresh._files_read_this_session == {"/repo/src/foo.py", "/repo/src/bar.py"}
        assert fresh._files_fully_delivered == {"/repo/src/foo.py": 5}
        assert fresh._skeleton_delivered_paths == {"/repo/src/README.md"}
        assert fresh._files_read_fingerprints == {"/repo/src/foo.py": "abc123"}
        assert fresh.evidence_safety.first_exact_seen == {"/repo/src/foo.py"}
        assert fresh.evidence_safety.pending_exact_keys == {"/repo/src/baz.py"}
        assert fresh.evidence_safety.alias_map == {"foo": "/repo/src/foo.py"}
        assert "/repo/src/foo.py" in fresh.evidence_safety.ledger
        assert fresh.evidence_safety.ledger["/repo/src/foo.py"].latest_digest == "d1"

    def test_load_fresh_session_no_file_no_error(self, tmp_path) -> None:
        session = RuntimeSession()
        session.memory_dir = tmp_path / ".tok"
        session.memory_dir.mkdir(parents=True, exist_ok=True)
        load_delivery_state(session)
        assert len(session._files_read_this_session) == 0

    def test_load_corrupted_file_no_error(self, tmp_path) -> None:
        session = RuntimeSession()
        session.memory_dir = tmp_path / ".tok"
        session.memory_dir.mkdir(parents=True, exist_ok=True)
        delivery_state_file(session).write_text("NOT JSON{{{")
        load_delivery_state(session)
        assert len(session._files_read_this_session) == 0

    def test_stale_file_is_ignored(self, tmp_path, monkeypatch) -> None:
        import tok.runtime._session_persistence as _sp

        monkeypatch.setattr(_sp, "RESULT_CACHE_TTL_SECONDS", 10)

        session = RuntimeSession()
        session.memory_dir = tmp_path / ".tok"
        session.memory_dir.mkdir(parents=True, exist_ok=True)
        session._files_read_this_session.add("/repo/src/old.py")
        save_delivery_state(session)

        payload = json.loads(delivery_state_file(session).read_text())
        payload["timestamp"] = time.time() - 100
        delivery_state_file(session).write_text(json.dumps(payload))

        fresh = RuntimeSession()
        fresh.memory_dir = tmp_path / ".tok"
        load_delivery_state(fresh)
        assert len(fresh._files_read_this_session) == 0


class TestResetSessionClearsDeliveryStateFile:
    def test_reset_removes_delivery_state_file(self, tmp_path) -> None:
        from tok.gateway import BridgeSession

        session = BridgeSession(memory_dir=tmp_path / ".tok")
        session.activate_session_for_request(
            {"x-tok-session-id": "reset-test"},
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        rs = session.runtime_session
        rs._files_read_this_session.add("/repo/src/foo.py")
        save_delivery_state(rs)
        assert delivery_state_file(rs).exists()

        rs.reset_session()
        assert not delivery_state_file(rs).exists()
        assert len(rs._files_read_this_session) == 0


class TestBridgeSessionIdleShutdownField:
    def test_default_idle_shutdown_is_7200(self) -> None:
        from tok.gateway import BridgeSession

        session = BridgeSession()
        assert session.idle_shutdown_seconds == 7200

    def test_idle_shutdown_env_override(self, monkeypatch) -> None:
        from tok.gateway import BridgeSession

        monkeypatch.setenv("TOK_BRIDGE_IDLE_SHUTDOWN_SECONDS", "600")
        session = BridgeSession()
        assert session.idle_shutdown_seconds == 600

    def test_idle_shutdown_disabled_when_zero(self, monkeypatch) -> None:
        from tok.gateway import BridgeSession

        monkeypatch.setenv("TOK_BRIDGE_IDLE_SHUTDOWN_SECONDS", "0")
        session = BridgeSession()
        assert session.idle_shutdown_seconds == 0
