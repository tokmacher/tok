from __future__ import annotations

from types import SimpleNamespace

import tok.gateway as gateway
from tok.gateway import BridgeSession
from tok.universal_runtime import RuntimeSession


class _Tracker:
    def merge_session_to_ledger(self) -> None:
        pass


def _fake_session(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        tracker=_Tracker(),
        api_base=kwargs["api_base"],
        request_policy_default="natural_first",
    )


def test_run_bridge_binds_to_loopback_by_default(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(gateway, "TOK_DIR", tmp_path / ".tok")
    monkeypatch.setattr(gateway, "PID_FILE", tmp_path / ".tok" / "bridge.pid")
    monkeypatch.setattr(gateway, "BridgeSession", _fake_session)
    monkeypatch.setattr(gateway, "create_app", lambda session: object())
    monkeypatch.setattr(gateway.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(
        gateway.uvicorn,
        "run",
        lambda app, host, port, log_level: captured.update(
            {"app": app, "host": host, "port": port, "log_level": log_level}
        ),
    )
    monkeypatch.delenv("TOK_BRIDGE_BIND_HOST", raising=False)
    monkeypatch.setenv("TOK_BRIDGE_HOST", "bridge.example.test")

    gateway.run_bridge(port=9090, keep_turns=2, debug=False, fail_open=True, _api_base="https://api.anthropic.com")

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9090


def test_run_bridge_honors_explicit_bind_host_override(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(gateway, "TOK_DIR", tmp_path / ".tok")
    monkeypatch.setattr(gateway, "PID_FILE", tmp_path / ".tok" / "bridge.pid")
    monkeypatch.setattr(gateway, "BridgeSession", _fake_session)
    monkeypatch.setattr(gateway, "create_app", lambda session: object())
    monkeypatch.setattr(gateway.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(
        gateway.uvicorn,
        "run",
        lambda app, host, port, log_level: captured.update(
            {"app": app, "host": host, "port": port, "log_level": log_level}
        ),
    )
    monkeypatch.setenv("TOK_BRIDGE_HOST", "localhost")
    monkeypatch.setenv("TOK_BRIDGE_BIND_HOST", "0.0.0.0")

    gateway.run_bridge(port=9090, keep_turns=2, debug=False, fail_open=True, _api_base="https://api.anthropic.com")

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9090


def test_bridge_session_reads_env_defaults_at_construction_time(monkeypatch) -> None:
    monkeypatch.setenv("TOK_BRIDGE_PORT", "7777")
    monkeypatch.setenv("TOK_KEEP_TURNS", "9")
    monkeypatch.setenv("TOK_FAIL_OPEN", "0")
    monkeypatch.setenv("TOK_CAPTURE", "1")
    monkeypatch.setenv("TOK_RATE_LIMIT_RETRY_MAX_ATTEMPTS", "5")

    session = BridgeSession()

    assert session.port == 7777
    assert session.keep_turns == 9
    assert session.runtime_session.keep_turns == 9
    assert session.fail_open is False
    assert session.capture is True
    assert session.rate_limit_retry_max_attempts == 5


def test_bridge_session_keep_turns_reaches_runtime_without_explicit_memory_dir() -> None:
    session = BridgeSession(keep_turns=7)

    assert session.runtime_session.keep_turns == 7
    assert session.runtime_session.adaptive_keep_turns() == 7


def test_bridge_session_honors_zero_keep_turns_without_young_session_floor() -> None:
    session = BridgeSession(keep_turns=0)

    assert session.runtime_session.keep_turns == 0
    assert session.runtime_session.adaptive_keep_turns() == 0


def test_raw_runtime_session_keeps_young_session_floor() -> None:
    session = RuntimeSession(keep_turns=2)

    assert session.adaptive_keep_turns() == 3
