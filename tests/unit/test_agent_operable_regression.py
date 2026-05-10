"""Regression tests for agent-operable Tok: JSON diagnostics, smoke runner, contract drift.

These tests protect against regressions in:
- Stable JSON output envelope shape and field presence
- --json output under mocked bridge states (running, not running, degraded, unreachable)
- Agent smoke runner report generation and claim level logic
- Agent contract synchronization with AGENTS.md and CLI surface
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tok.cli import app

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()

_REQUIRED_ENVELOPE_KEYS = ("schema", "command", "ok", "status", "data", "warnings", "next_steps")
_SCHEMA_VALUE = "tok-cli-result/v0.1"


def _make_fake_health_response(
    *,
    status_code: int = 200,
    baseline_only: bool = False,
    mode: str = "natural-first",
    fallback_count: int = 0,
    tokens_saved: int = 500,
    savings_pct: float = 10.0,
    cost_saved_usd: float = 0.03,
    session_quality: str = "clean",
    degradation_reason: str = "",
    capability: dict[str, Any] | None = None,
    **extra: Any,
) -> type:
    payload: dict[str, Any] = {
        "status": "ok",
        "bridge": "tok",
        "port": 9090,
        "mode": mode,
        "baseline_only": baseline_only,
        "fallback_count": fallback_count,
        "session_tokens_saved": tokens_saved,
        "session_savings_pct": savings_pct,
        "cost_saved_usd": cost_saved_usd,
        "session_quality": session_quality,
        "last_degradation_reason": degradation_reason,
        "actual_tokens": 10000,
        "baseline_tokens": 10500,
        "actual_cost_usd": 0.10,
        "baseline_cost_usd": 0.13,
        "api_base": "",
        "request_policy": "natural_first",
        "semantic_drift_count": 0,
        "fail_open_count": 0,
        "non_tok_count": 0,
        "answer_anchor_miss_count": 0,
        "repeat_search_count": 0,
        "repeat_file_read_count": 0,
        "state_resend_full_count": 0,
        "state_resend_delta_count": 0,
        "state_resend_suppressed_count": 0,
        "preflight_block_original_payload_count": 0,
        "preflight_block_rewritten_payload_count": 0,
        "stream_recovery_empty_success_count": 0,
        "stream_recovery_read_error_count": 0,
        "request_policy_held_by_recovery_count": 0,
        "evidence_exact_observed_count": 0,
        "evidence_non_exact_reference_count": 0,
        "evidence_non_exact_summary_count": 0,
        "evidence_non_exact_skeleton_count": 0,
        "evidence_exact_reacquisition_required_count": 0,
        "evidence_exact_reacquisition_satisfied_count": 0,
        "evidence_compression_blocked_for_safety_count": 0,
        "calls": 10,
        "smoothness_score": 0,
        "labour_index": 0,
        "current_mode": "",
        "stream_instability_events": 0,
        "thinking_mutation_events": 0,
        "repeated_active_file_reads": 0,
        "task_score": 0,
    }
    if capability is not None:
        payload["capability"] = capability
    payload.update(extra)

    class FakeResponse:
        pass

    FakeResponse.status_code = status_code

    def _json():
        return payload

    FakeResponse.json = staticmethod(_json)
    return FakeResponse


# ---------------------------------------------------------------------------
# 1. JSON envelope contract
# ---------------------------------------------------------------------------


class TestEnvelopeContractRegression:
    def test_envelope_has_all_required_keys(self) -> None:
        from tok.cli._cli_support import json_envelope

        env = json_envelope("tok test", ok=True, status="ok")
        for key in _REQUIRED_ENVELOPE_KEYS:
            assert key in env, f"Envelope missing required key: {key}"

    def test_envelope_ok_false_status_error(self) -> None:
        from tok.cli._cli_support import json_envelope

        env = json_envelope("tok test", ok=False, status="error", warnings=["bad"])
        assert env["ok"] is False
        assert env["status"] == "error"
        assert env["warnings"] == ["bad"]

    def test_envelope_preserves_data(self) -> None:
        from tok.cli._cli_support import json_envelope

        data = {"bridge_running": True, "port": 9090}
        env = json_envelope("tok test", ok=True, status="ok", data=data, next_steps=["step1"])
        assert env["data"] == data
        assert env["next_steps"] == ["step1"]

    def test_envelope_defaults_empty_collections(self) -> None:
        from tok.cli._cli_support import json_envelope

        env = json_envelope("tok test", ok=True, status="ok")
        assert env["data"] == {}
        assert env["warnings"] == []
        assert env["next_steps"] == []

    def test_envelope_schema_is_stable(self) -> None:
        from tok.cli._cli_support import json_envelope

        env = json_envelope("tok test", ok=True, status="ok")
        assert env["schema"] == _SCHEMA_VALUE


# ---------------------------------------------------------------------------
# 2. tok bridge status --json regression
# ---------------------------------------------------------------------------


class TestBridgeStatusJsonRegression:
    def test_bridge_not_running_returns_json_not_prose(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: None)
        result = runner.invoke(app, ["bridge", "status", "--json"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["ok"] is False
        assert data["data"]["bridge_running"] is False
        assert "Bridge not running" not in result.output or data["ok"] is False

    def test_bridge_running_healthy_session(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 12345)
        fake = _make_fake_health_response()
        monkeypatch.setattr("tok.cli._bridge.get_bridge_health_response", lambda *a, **kw: fake())
        result = runner.invoke(app, ["bridge", "status", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert data["status"] == "ok"
        assert data["data"]["bridge_running"] is True
        assert data["data"]["pid"] == 12345
        assert data["data"]["tok_active"] is True
        assert data["data"]["baseline_only"] is False
        assert data["data"]["degraded_to_baseline"] is False
        assert data["data"]["fallback_count"] == 0
        assert data["data"]["tokens_saved"] == 500
        assert data["data"]["savings_pct"] == 10.0

    def test_bridge_running_degraded_session(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 12345)
        fake = _make_fake_health_response(
            baseline_only=True,
            session_quality="degraded",
            degradation_reason="baseline fallback",
            fallback_count=3,
        )
        monkeypatch.setattr("tok.cli._bridge.get_bridge_health_response", lambda *a, **kw: fake())
        result = runner.invoke(app, ["bridge", "status", "--json"])
        data = json.loads(result.output)
        assert data["data"]["baseline_only"] is True
        assert data["data"]["degraded_to_baseline"] is True
        assert data["data"]["fallback_count"] == 3
        assert data["data"]["session_quality"] == "degraded"

    def test_bridge_running_includes_capability_conformance(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 12345)
        fake = _make_fake_health_response(
            capability={"bridge_mode": "tok", "trace_version": "0.1", "max_conformance_level": "L2"},
        )
        monkeypatch.setattr("tok.cli._bridge.get_bridge_health_response", lambda *a, **kw: fake())
        result = runner.invoke(app, ["bridge", "status", "--json"])
        data = json.loads(result.output)
        assert data["data"]["conformance"] == "L2"

    def test_bridge_running_health_unreachable(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 12345)
        monkeypatch.setattr(
            "tok.cli._bridge.get_bridge_health_response",
            lambda *a, **kw: (_ for _ in ()).throw(Exception("Connection refused")),
        )
        result = runner.invoke(app, ["bridge", "status", "--json"])
        data = json.loads(result.output)
        assert data["data"]["bridge_running"] is True
        assert data["data"]["health_reachable"] is False
        assert data["ok"] is False

    def test_no_secrets_in_json_output(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 12345)
        fake = _make_fake_health_response(api_base="https://secret-api-key@host.com")
        monkeypatch.setattr("tok.cli._bridge.get_bridge_health_response", lambda *a, **kw: fake())
        result = runner.invoke(app, ["bridge", "status", "--json"])
        output_lower = result.output.lower()
        for forbidden in ("api_key", "secret", "password", "token=", "bearer "):
            assert forbidden not in output_lower, f"JSON output contains forbidden string: {forbidden}"

    def test_json_output_is_parseable_by_python_json_tool(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 12345)
        fake = _make_fake_health_response()
        monkeypatch.setattr("tok.cli._bridge.get_bridge_health_response", lambda *a, **kw: fake())
        result = runner.invoke(app, ["bridge", "status", "--json"])
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
        roundtripped = json.loads(json.dumps(parsed))
        assert roundtripped == parsed

    def test_bridge_non_200_response_returns_json(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 12345)
        fake = _make_fake_health_response(status_code=502)
        monkeypatch.setattr("tok.cli._bridge.get_bridge_health_response", lambda *a, **kw: fake())
        result = runner.invoke(app, ["bridge", "status", "--json"])
        data = json.loads(result.output)
        assert data["ok"] is False
        assert data["status"] == "error"
        assert data["data"]["bridge_running"] is True
        assert data["data"]["http_status"] == 502

    def test_bridge_malformed_payload_returns_json(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 12345)

        class BadJsonResponse:
            status_code = 200

            def json(self):
                raise ValueError("bad json")

        monkeypatch.setattr(
            "tok.cli._bridge.get_bridge_health_response",
            lambda *a, **kw: BadJsonResponse(),
        )
        result = runner.invoke(app, ["bridge", "status", "--json"])
        data = json.loads(result.output)
        assert data["ok"] is False
        assert data["status"] == "error"
        assert data["data"]["bridge_running"] is True
        assert data["data"]["malformed_payload"] is True
        for tag in ("[bold]", "[green]", "[yellow]", "[red]", "[dim]"):
            assert tag not in result.output


# ---------------------------------------------------------------------------
# 3. tok stats --json regression
# ---------------------------------------------------------------------------


class TestStatsJsonRegression:
    def test_stats_json_no_session_data_still_valid(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "nonexistent.tok"))
        monkeypatch.setenv("TOK_GLOBAL_LEDGER", str(tmp_path / "nonexistent_ledger.tok"))
        result = runner.invoke(app, ["stats", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["schema"] == _SCHEMA_VALUE
        assert data["data"]["bridge_running"] is False

    def test_stats_json_with_tracker_data(self, monkeypatch, tmp_path) -> None:
        from tok.stats import SavingsTracker

        tracker = SavingsTracker(
            savings_file=str(tmp_path / "savings.tok"),
            ledger_path=tmp_path / "ledger.tok",
        )
        tracker.record_call(
            model="claude-sonnet-4-5",
            actual_input=1000,
            actual_output=200,
            cache_read=100,
            cache_write=50,
            input_saved=300,
            output_saved=100,
        )
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)
        monkeypatch.setattr("tok.cli._release.SavingsTracker", lambda: tracker)
        result = runner.invoke(app, ["stats", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "session" in data["data"]
        assert data["data"]["session"]["calls"] == 1
        assert data["data"]["session"]["tokens_saved"] == 400
        assert data["data"]["session"]["savings_pct"] > 0

    def test_stats_json_session_has_required_fields(self, monkeypatch, tmp_path) -> None:
        from tok.stats import SavingsTracker

        tracker = SavingsTracker(
            savings_file=str(tmp_path / "savings.tok"),
            ledger_path=tmp_path / "ledger.tok",
        )
        tracker.record_call(
            model="claude-sonnet-4-5",
            actual_input=1000,
            actual_output=200,
            cache_read=0,
            cache_write=0,
            input_saved=300,
            output_saved=100,
        )
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)
        monkeypatch.setattr("tok.cli._release.SavingsTracker", lambda: tracker)
        result = runner.invoke(app, ["stats", "--json"])
        data = json.loads(result.output)
        session = data["data"]["session"]
        for field in (
            "calls",
            "actual_tokens",
            "baseline_tokens",
            "tokens_saved",
            "savings_pct",
            "actual_cost_usd",
            "baseline_cost_usd",
            "cost_saved_usd",
            "fallback_count",
            "baseline_only",
            "session_quality",
            "degradation_reason",
        ):
            assert field in session, f"Session data missing field: {field}"

    def test_stats_json_with_live_bridge(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 12345)
        fake = _make_fake_health_response(
            tokens_saved=5000,
            savings_pct=20.0,
            actual_tokens=20000,
            baseline_tokens=25000,
            calls=50,
        )
        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *a, **kw: fake())
        from tok.stats import SavingsTracker

        tracker = SavingsTracker(
            savings_file=str(tmp_path / "savings.tok"),
            ledger_path=tmp_path / "ledger.tok",
        )
        monkeypatch.setattr("tok.cli._release.SavingsTracker", lambda: tracker)
        result = runner.invoke(app, ["stats", "--json"])
        data = json.loads(result.output)
        assert data["data"]["bridge_running"] is True
        assert data["data"]["session"]["tokens_saved"] == 5000

    def test_stats_json_does_not_emit_rich_markup(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)
        result = runner.invoke(app, ["stats", "--json"])
        for tag in ("[bold]", "[green]", "[yellow]", "[red]", "[dim]"):
            assert tag not in result.output, f"Stats --json output contains Rich markup: {tag}"


# ---------------------------------------------------------------------------
# 4. tok doctor --json regression
# ---------------------------------------------------------------------------


class TestDoctorJsonRegression:
    def test_doctor_bridge_not_running_exits_nonzero(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)
        result = runner.invoke(app, ["doctor", "--json"])
        data = json.loads(result.output)
        assert result.exit_code == 1
        assert data["ok"] is False
        assert data["data"]["bridge_running"] is False

    def test_doctor_healthy_bridge(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 12345)
        fake_cls = _make_fake_health_response()
        fake_instance = fake_cls()
        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *a, **kw: fake_instance)
        tok_dir = tmp_path / ".tok"
        tok_dir.mkdir()
        (tok_dir / "bridge_memory.tok").write_text("test")
        monkeypatch.setattr("tok.cli._release.memory_root", lambda: tok_dir)
        result = runner.invoke(app, ["doctor", "--json"])
        data = json.loads(result.output)
        assert data["ok"] is True, f"Unexpected output: {result.output[:1000]}"
        assert data["data"]["bridge_running"] is True
        assert data["data"]["health_reachable"] is True
        assert data["data"]["tok_active"] is True
        assert data["data"]["fallback_count"] == 0

    def test_doctor_degraded_baseline(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 12345)
        fake_cls = _make_fake_health_response(
            baseline_only=True,
            session_quality="degraded",
            degradation_reason="baseline fallback",
        )
        fake_instance = fake_cls()
        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *a, **kw: fake_instance)
        tok_dir = tmp_path / ".tok"
        tok_dir.mkdir()
        (tok_dir / "bridge_memory.tok").write_text("test")
        monkeypatch.setattr("tok.cli._release.memory_root", lambda: tok_dir)
        result = runner.invoke(app, ["doctor", "--json"])
        data = json.loads(result.output)
        assert data["data"]["degraded_to_baseline"] is True
        assert data["data"]["baseline_only"] is True
        assert any("degraded" in w.lower() for w in data["warnings"])

    def test_doctor_missing_memory_dir(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 12345)
        fake_cls = _make_fake_health_response()
        fake_instance = fake_cls()
        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *a, **kw: fake_instance)
        monkeypatch.setattr("tok.cli._release.memory_root", lambda: tmp_path / "no_such_dir")
        result = runner.invoke(app, ["doctor", "--json"])
        data = json.loads(result.output)
        assert any("memory" in w.lower() for w in data["warnings"])

    def test_doctor_reports_cold_start_signals(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 12345)
        fake_cls = _make_fake_health_response()
        fake_instance = fake_cls()
        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *a, **kw: fake_instance)
        tok_dir = tmp_path / ".tok"
        tok_dir.mkdir()
        (tok_dir / "bridge_memory.tok").write_text("test")
        monkeypatch.setattr("tok.cli._release.memory_root", lambda: tok_dir)
        result = runner.invoke(app, ["doctor", "--json"])
        data = json.loads(result.output)
        assert "cold_start_structured" in data["data"]
        assert "cold_start_fallback" in data["data"]
        assert isinstance(data["data"]["cold_start_structured"], int)
        assert isinstance(data["data"]["cold_start_fallback"], int)

    def test_doctor_json_does_not_emit_rich_markup(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)
        result = runner.invoke(app, ["doctor", "--json"])
        for tag in ("[bold]", "[green]", "[yellow]", "[red]", "[dim]", "✅", "❌", "⚠️"):
            assert tag not in result.output, f"Doctor --json output contains Rich/emoji: {tag}"


# ---------------------------------------------------------------------------
# 5. Smoke runner regression
# ---------------------------------------------------------------------------


class TestSmokeRunnerRegression:
    def _load_smoke_module(self):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("run_agent_smoke", ROOT / "scripts" / "run_agent_smoke.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_report_json_has_required_fields(self, monkeypatch, tmp_path) -> None:
        module = self._load_smoke_module()
        monkeypatch.setattr(module, "OUT_DIR", tmp_path)
        monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "agent_smoke_report.json")
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""),
        )
        exit_code = module.main([])
        assert exit_code == 0
        report = json.loads((tmp_path / "agent_smoke_report.json").read_text())
        assert report["schema"] == "tok-agent-smoke-report/v0.1"
        assert "overall" in report
        assert "checks" in report
        assert "live_bridge_requested" in report
        assert "live_bridge_result" in report
        assert "claim_level" in report

    def test_claim_level_is_test_suite_on_full_pass(self, monkeypatch, tmp_path) -> None:
        module = self._load_smoke_module()
        monkeypatch.setattr(module, "OUT_DIR", tmp_path)
        monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "agent_smoke_report.json")
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""),
        )
        module.main([])
        report = json.loads((tmp_path / "agent_smoke_report.json").read_text())
        assert report["claim_level"] == "test_suite"

    def test_claim_level_is_source_only_on_failure(self, monkeypatch, tmp_path) -> None:
        module = self._load_smoke_module()
        monkeypatch.setattr(module, "OUT_DIR", tmp_path)
        monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "agent_smoke_report.json")
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(args=a, returncode=1, stdout="", stderr="error"),
        )
        module.main([])
        report = json.loads((tmp_path / "agent_smoke_report.json").read_text())
        assert report["claim_level"] == "source_only"
        assert report["overall"] == "FAIL"

    def test_live_bridge_not_requested_skips(self, monkeypatch, tmp_path) -> None:
        module = self._load_smoke_module()
        monkeypatch.setattr(module, "OUT_DIR", tmp_path)
        monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "agent_smoke_report.json")
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""),
        )
        module.main([])
        report = json.loads((tmp_path / "agent_smoke_report.json").read_text())
        assert report["live_bridge_requested"] is False
        assert report["live_bridge_result"] == "skipped"
        assert "live_bridge_checks" not in report

    def test_live_bridge_requested_adds_checks(self, monkeypatch, tmp_path) -> None:
        module = self._load_smoke_module()
        monkeypatch.setattr(module, "OUT_DIR", tmp_path)
        monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "agent_smoke_report.json")
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="{}", stderr=""),
        )
        module.main(["--live-bridge"])
        report = json.loads((tmp_path / "agent_smoke_report.json").read_text())
        assert report["live_bridge_requested"] is True
        assert report["live_bridge_result"] == "pass"
        assert "live_bridge_checks" in report
        assert len(report["live_bridge_checks"]) == 3
        assert report["claim_level"] == "live_bridge"

    def test_report_checks_all_have_name_and_status(self, monkeypatch, tmp_path) -> None:
        module = self._load_smoke_module()
        monkeypatch.setattr(module, "OUT_DIR", tmp_path)
        monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "agent_smoke_report.json")
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""),
        )
        module.main([])
        report = json.loads((tmp_path / "agent_smoke_report.json").read_text())
        for check in report["checks"]:
            assert "name" in check
            assert "status" in check
            assert check["status"] in ("PASS", "FAIL")

    def test_report_is_valid_json_roundtrip(self, monkeypatch, tmp_path) -> None:
        module = self._load_smoke_module()
        monkeypatch.setattr(module, "OUT_DIR", tmp_path)
        monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "agent_smoke_report.json")
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""),
        )
        module.main([])
        raw = (tmp_path / "agent_smoke_report.json").read_text()
        parsed = json.loads(raw)
        roundtripped = json.loads(json.dumps(parsed))
        assert roundtripped == parsed

    def test_live_bridge_failure_overall_fail(self, monkeypatch, tmp_path) -> None:
        module = self._load_smoke_module()
        monkeypatch.setattr(module, "OUT_DIR", tmp_path)
        monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "agent_smoke_report.json")
        call_count = 0
        base_step_count = len(module.build_steps())

        def _fail_on_live(*a, **_kw):
            nonlocal call_count
            call_count += 1
            if call_count <= base_step_count:
                return subprocess.CompletedProcess(args=a, returncode=0, stdout="{}", stderr="")
            return subprocess.CompletedProcess(args=a, returncode=1, stdout="", stderr="error")

        monkeypatch.setattr(module.subprocess, "run", _fail_on_live)
        exit_code = module.main(["--live-bridge"])
        report = json.loads((tmp_path / "agent_smoke_report.json").read_text())
        assert report["overall"] == "FAIL"
        assert report["live_bridge_result"] == "fail"
        assert exit_code == 1

    def test_live_bridge_measured_savings_from_nested_stats(self, monkeypatch, tmp_path) -> None:
        module = self._load_smoke_module()
        monkeypatch.setattr(module, "OUT_DIR", tmp_path)
        monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "agent_smoke_report.json")
        call_count = 0
        base_step_count = len(module.build_steps())

        def _nested_savings(*a, **_kw):
            nonlocal call_count
            call_count += 1
            if call_count <= base_step_count:
                return subprocess.CompletedProcess(args=a, returncode=0, stdout="{}", stderr="")
            stats_json = json.dumps(
                {
                    "schema": "tok-cli-result/v0.1",
                    "command": "tok stats",
                    "ok": True,
                    "status": "ok",
                    "data": {"session": {"tokens_saved": 100}},
                    "warnings": [],
                    "next_steps": [],
                }
            )
            return subprocess.CompletedProcess(args=a, returncode=0, stdout=stats_json, stderr="")

        monkeypatch.setattr(module.subprocess, "run", _nested_savings)
        module.main(["--live-bridge"])
        report = json.loads((tmp_path / "agent_smoke_report.json").read_text())
        assert report["claim_level"] == "live_bridge_with_measured_savings"


# ---------------------------------------------------------------------------
# 6. Agent contract drift regression
# ---------------------------------------------------------------------------


class TestContractDriftRegression:
    @pytest.fixture(autouse=True)
    def _load_contract(self) -> None:
        self._contract = json.loads((ROOT / "docs" / "agent-contract.json").read_text())

    def test_json_output_commands_in_live_bridge_checks(self) -> None:
        checks = self._contract["live_bridge_checks"]
        assert any("bridge status" in c for c in checks)
        assert any("doctor" in c for c in checks)
        assert any("stats" in c for c in checks)
        for check in checks:
            assert "--json" in check

    def test_forbidden_claims_no_duplicates(self) -> None:
        claims = self._contract["forbidden_claims"]
        assert len(claims) == len(set(claims))

    def test_protocol_layers_are_exhaustive(self) -> None:
        layers = self._contract["implemented_protocol_layers"]
        expected_keys = {"tok_trace", "tok_resolver", "tok_capability", "tok_session", "agent_to_agent_exchange"}
        assert set(layers.keys()) == expected_keys

    def test_only_tok_trace_is_not_deferred(self) -> None:
        layers = self._contract["implemented_protocol_layers"]
        non_deferred = [k for k, v in layers.items() if v != "deferred"]
        assert non_deferred == ["tok_trace"]

    def test_agents_md_mentions_deferred_layers(self) -> None:
        agents_content = (ROOT / "AGENTS.md").read_text().lower()
        for layer in ("resolver", "capability", "session"):
            assert layer in agents_content, f"AGENTS.md missing reference to deferred layer: {layer}"

    def test_contract_required_cli_checks_match_release_surface(self) -> None:
        from tok.release_surface import SUPPORTED_CLI_ROOT_COMMANDS

        supported = set(SUPPORTED_CLI_ROOT_COMMANDS)
        for cmd in self._contract["required_verification_commands"]:
            bare = cmd.removeprefix("tok ").strip()
            parts = bare.split()
            if bare in ("--version", "--help"):
                continue
            if len(parts) >= 2 and parts[0] == "bridge":
                continue
            assert parts[0] in supported, f"Contract references unsupported command: {cmd}"

    def test_forbidden_claims_and_unsupported_claims_both_in_agents_md(self) -> None:
        agents_content = (ROOT / "AGENTS.md").read_text().lower()
        for claim in self._contract["forbidden_claims"]:
            keywords = [w for w in claim.lower().split() if len(w) > 3]
            short_words = [
                w for w in claim.lower().split() if len(w) <= 3 and w not in ("the", "a", "an", "by", "is", "it", "to")
            ]
            matchable = keywords or short_words
            assert any(m in agents_content for m in matchable), (
                f"Forbidden claim '{claim}' has no keyword match in AGENTS.md"
            )
        for claim in self._contract["unsupported_claims"]:
            assert claim.lower() in agents_content, f"Unsupported claim '{claim}' not mentioned in AGENTS.md"

    def test_contract_and_agents_md_forbidden_claims_overlap(self) -> None:
        agents_content = (ROOT / "AGENTS.md").read_text().lower()
        agents_has_do_not = "do not" in agents_content
        agents_has_never = "never" in agents_content
        assert agents_has_do_not or agents_has_never, "AGENTS.md missing forbidding language"

    def test_live_bridge_checks_reference_real_json_commands(self) -> None:
        for check in self._contract["live_bridge_checks"]:
            cmd_parts = check.split()
            assert cmd_parts[0] == "tok"
            assert "--json" in cmd_parts

    def test_reporting_required_fields_comprehensive(self) -> None:
        fields = self._contract["reporting_required_fields"]
        for required in ("commands_run", "pass_fail", "bridge_running", "claude_available"):
            assert required in fields, f"Missing required reporting field: {required}"


# ---------------------------------------------------------------------------
# 7. Cross-cutting: all --json commands return valid envelope
# ---------------------------------------------------------------------------


class TestJsonEnvelopeConsistencyRegression:
    def test_all_json_commands_use_same_schema(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: None)
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)

        results = []
        for cmd in (["bridge", "status", "--json"], ["stats", "--json"], ["doctor", "--json"]):
            r = runner.invoke(app, cmd)
            data = json.loads(r.output)
            results.append((cmd, data))

        schemas = [d["schema"] for _, d in results]
        assert len(set(schemas)) == 1, f"Inconsistent schemas across commands: {schemas}"
        assert schemas[0] == _SCHEMA_VALUE

    def test_all_json_commands_have_command_field_matching_invocation(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: None)
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)

        expected_commands = {
            "bridge status --json": "tok bridge status",
            "stats --json": "tok stats",
            "doctor --json": "tok doctor",
        }
        for args, expected_cmd in expected_commands.items():
            r = runner.invoke(app, args.split())
            data = json.loads(r.output)
            assert data["command"] == expected_cmd, (
                f"Command field mismatch for {args}: got '{data['command']}', expected '{expected_cmd}'"
            )
