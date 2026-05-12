"""Tests for --json output on tok bridge status, tok stats, and tok doctor."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from tok.cli import app

runner = CliRunner()


class TestJsonEnvelope:
    def test_json_envelope_helper_shape(self) -> None:
        from tok.cli._cli_support import json_envelope

        env = json_envelope("tok test", ok=True, status="ok")
        assert env["schema"] == "tok-cli-result/v0.1"
        assert env["command"] == "tok test"
        assert env["ok"] is True
        assert env["status"] == "ok"
        assert isinstance(env["data"], dict)
        assert isinstance(env["warnings"], list)
        assert isinstance(env["next_steps"], list)


class TestStatsJson:
    def test_stats_json_produces_valid_json(self) -> None:
        result = runner.invoke(app, ["stats", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["schema"] == "tok-cli-result/v0.1"
        assert data["command"] == "tok stats"
        assert "ok" in data
        assert "status" in data
        assert isinstance(data["data"], dict)
        assert isinstance(data["warnings"], list)
        assert "bridge_running" in data["data"]
        assert "port" in data["data"]

    def test_stats_json_has_session_and_lifetime_keys(self) -> None:
        result = runner.invoke(app, ["stats", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        d = data["data"]
        assert "session" in d or "lifetime" in d or len(data["warnings"]) > 0


class TestDoctorJson:
    def test_doctor_json_produces_valid_json(self) -> None:
        result = runner.invoke(app, ["doctor", "--json"])
        data = json.loads(result.output)
        assert data["schema"] == "tok-cli-result/v0.1"
        assert data["command"] == "tok doctor"
        assert "ok" in data
        assert "status" in data
        assert isinstance(data["data"], dict)
        assert isinstance(data["warnings"], list)
        assert "bridge_running" in data["data"]

    def test_doctor_json_not_running_reports_in_warnings(self, monkeypatch) -> None:
        from tok.cli import _release as _rel

        monkeypatch.setattr(_rel, "get_running_bridge_pid", lambda port: None)
        result = runner.invoke(app, ["doctor", "--json"])
        data = json.loads(result.output)
        assert data["data"]["bridge_running"] is False
        assert any("not running" in w.lower() for w in data["warnings"])

    def test_doctor_json_reports_memory_state(self) -> None:
        result = runner.invoke(app, ["doctor", "--json"])
        data = json.loads(result.output)
        assert "memory_structured" in data["data"] or len(data["warnings"]) > 0


class TestBridgeStatusJson:
    def test_bridge_status_json_not_running_returns_json(self, monkeypatch) -> None:
        from tok.cli import _bridge as _br

        monkeypatch.setattr(_br, "get_running_bridge_pid", lambda port: None)
        result = runner.invoke(app, ["bridge", "status", "--json"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["schema"] == "tok-cli-result/v0.1"
        assert data["command"] == "tok bridge status"
        assert data["ok"] is False
        assert data["data"]["bridge_running"] is False

    def test_bridge_status_json_has_port(self, monkeypatch) -> None:
        from tok.cli import _bridge as _br

        monkeypatch.setattr(_br, "get_running_bridge_pid", lambda port: None)
        result = runner.invoke(app, ["bridge", "status", "--json"])
        data = json.loads(result.output)
        assert "port" in data["data"]
        assert isinstance(data["data"]["port"], int)

    def test_bridge_status_json_running_bridge_returns_ok(self) -> None:
        result = runner.invoke(app, ["bridge", "status", "--json"])
        data = json.loads(result.output)
        assert data["schema"] == "tok-cli-result/v0.1"
        assert data["command"] == "tok bridge status"
        if data["data"]["bridge_running"]:
            assert data["ok"] is True
            assert "tokens_saved" in data["data"]
            assert "savings_pct" in data["data"]
