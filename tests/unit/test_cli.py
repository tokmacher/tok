"""Tests for tok.cli — CLI commands via typer testing."""

import json
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

from typer.testing import CliRunner

from tok.cli import app
from tok.stats import SavingsTracker

runner = CliRunner()


class TestCLI:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "bridge" in result.output
        assert "stats" in result.output
        assert "savings" in result.output
        assert "install" in result.output
        assert "doctor" in result.output
        assert "metrics" not in result.output
        assert "dev" not in result.output
        assert "memory-snap" not in result.output
        assert "capture-review" not in result.output
        assert "gate-check" not in result.output
        assert "convert" not in result.output
        assert "parse" not in result.output

    def test_bridge_help(self) -> None:
        result = runner.invoke(app, ["bridge", "--help"])
        assert result.exit_code == 0
        assert "start" in result.output
        assert "stop" in result.output
        assert "status" in result.output

    def test_doctor_help(self) -> None:
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "Check bridge health and runtime contract conformance" in (result.output)
        assert "--report" in result.output
        assert "--verbose" in result.output

    def test_stats_help(self) -> None:
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0
        assert "Show token savings and fallback state" in result.output
        assert "--last-session" in result.output
        assert "--recent" in result.output
        assert "--since" in result.output

    def test_metrics_help(self) -> None:
        result = runner.invoke(app, ["metrics", "--help"])
        assert result.exit_code == 0
        assert "pressure" in result.output
        assert "memory" in result.output
        assert "savings-trend" in result.output
        assert "fallback" in result.output
        assert "health" in result.output

    def test_dev_help(self) -> None:
        result = runner.invoke(app, ["dev", "--help"])
        assert result.exit_code == 0
        assert "generate-fixture" in result.output
        assert "live-benchmark" in result.output
        assert "stress-language" in result.output

    def test_hidden_legacy_command_aliases_still_work(self, monkeypatch, tmp_path) -> None:
        calls = {}

        monkeypatch.setattr(
            "tok.utils.metrics.pressure_trends",
            lambda window, export: calls.update({"window": window, "export": export}),
        )

        export_path = tmp_path / "pressure.json"
        result = runner.invoke(app, ["pressure", "--window", "3", "--export", str(export_path)])

        assert result.exit_code == 0
        assert calls == {"window": 3, "export": str(export_path)}

    def test_hidden_legacy_generate_fixture_alias_still_works(self, monkeypatch) -> None:
        calls = {}

        class FakeGenerator:
            def generate_coding_session(self, name, turns, template, complexity):
                calls["generate"] = (name, turns, template, complexity)
                return "fixture-data", '{"name": "demo"}'

            def save_fixture(self, name, fixture, metadata, output) -> None:
                calls["save"] = (name, fixture, metadata, output)

        monkeypatch.setattr("tok.testing.fixture_generator.FixtureGenerator", FakeGenerator)

        result = runner.invoke(app, ["generate-fixture", "coding", "legacy-demo"])

        assert result.exit_code == 0
        assert calls["generate"] == (
            "legacy-demo",
            5,
            "standard_claude",
            "medium",
        )

    def test_install_uses_shell_integration_backend(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("tok.utils.shell_integration.install", lambda: tmp_path / ".zshrc")

        result = runner.invoke(app, ["install"])

        assert result.exit_code == 0
        assert "Tok shell integration installed in" in result.output
        assert ".zshrc" in result.output
        assert "Reload your shell:" in result.output
        assert "tok bridge start" in result.output

    def test_bridge_status_shows_mode_and_session_summary(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-empty-savings-status.tok")
        Path("/tmp/test-empty-savings-status.tok").unlink(missing_ok=True)

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "ok",
                    "bridge": "tok",
                    "port": 9090,
                    "mode": "tool-compatible",
                    "request_policy": "natural_first",
                    "baseline_only": True,
                    "fallback_count": 2,
                    "session_tokens_saved": 140,
                    "session_savings_pct": 48.3,
                    "session_quality": "degraded",
                    "last_degradation_reason": "baseline fallback",
                    "semantic_drift_count": 0,
                    "fail_open_count": 0,
                    "non_tok_count": 0,
                    "answer_anchor_miss_count": 0,
                    "repeat_search_count": 0,
                    "repeat_file_read_count": 0,
                    "state_resend_full_count": 0,
                    "state_resend_delta_count": 0,
                    "state_resend_suppressed_count": 0,
                }

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())

        result = runner.invoke(app, ["bridge", "status"])
        assert result.exit_code == 0
        assert "Bridge running on :9090 (PID 321)" in result.output
        assert "Bridge Status" in result.output
        assert "Saved $0.0000" in result.output
        assert "48.3%" in result.output
        assert "Session degraded to baseline" in result.output
        assert "Session degraded to baseline" in result.output
        assert "Tok active" in result.output
        assert "Degraded to baseline" in result.output
        assert "Mode" in result.output
        assert "Request policy" in result.output
        assert "natural_first" in result.output
        assert "Fallbacks" in result.output
        assert "Session quality" in result.output
        assert "Degradation reason" in result.output
        assert "Session signals" in result.output
        assert "tok doctor" in result.output
        assert "tok bridge logs 100" in result.output

    def test_bridge_status_shows_watch_session_verdict(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-empty-watch-status.tok")
        Path("/tmp/test-empty-watch-status.tok").unlink(missing_ok=True)

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "ok",
                    "bridge": "tok",
                    "port": 9090,
                    "mode": "tool-compatible",
                    "request_policy": "natural_first",
                    "baseline_only": False,
                    "fallback_count": 1,
                    "session_tokens_saved": 140,
                    "session_savings_pct": 48.3,
                    "session_quality": "watch",
                    "last_degradation_reason": "context reacquisition",
                    "semantic_drift_count": 0,
                    "fail_open_count": 0,
                    "non_tok_count": 0,
                    "answer_anchor_miss_count": 0,
                    "repeat_search_count": 1,
                    "repeat_file_read_count": 0,
                    "state_resend_full_count": 0,
                    "state_resend_delta_count": 0,
                    "state_resend_suppressed_count": 0,
                }

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())

        result = runner.invoke(app, ["bridge", "status"])

        assert result.exit_code == 0
        assert "Tok active, watch session" in result.output
        assert "context reacquisition" in result.output

    def test_bridge_status_surfaces_attribution_signals(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 321)

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "ok",
                    "bridge": "tok",
                    "port": 9090,
                    "mode": "natural-first",
                    "baseline_only": False,
                    "fallback_count": 0,
                    "session_tokens_saved": 120,
                    "session_savings_pct": 41.4,
                    "actual_tokens": 100,
                    "baseline_tokens": 200,
                    "actual_cost_usd": 0.01,
                    "baseline_cost_usd": 0.02,
                    "cost_saved_usd": 0.01,
                    "session_quality": "watch",
                    "last_degradation_reason": "request-shape incompatibility",
                    "semantic_drift_count": 0,
                    "fail_open_count": 0,
                    "non_tok_count": 0,
                    "answer_anchor_miss_count": 0,
                    "repeat_search_count": 0,
                    "repeat_file_read_count": 0,
                    "preflight_block_original_payload_count": 1,
                    "preflight_block_rewritten_payload_count": 2,
                    "stream_recovery_empty_success_count": 3,
                    "stream_recovery_read_error_count": 4,
                    "request_policy_held_by_recovery_count": 1,
                }

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())

        result = runner.invoke(app, ["bridge", "status"])

        assert result.exit_code == 0
        assert "request-shape incompatibility" in result.output
        assert "shape-orig=1" in result.output
        assert "shape-rewrite=2" in result.output
        assert "stream-empty=3" in result.output
        assert "stream-read=4" in result.output
        assert "held=1" in result.output

    def test_bridge_status_prefers_live_payload_over_stale_local_summary(self, monkeypatch, tmp_path) -> None:
        tracker = SavingsTracker(
            savings_file=str(tmp_path / "tok_savings.tok"),
            ledger_path=tmp_path / "global_savings.tok",
        )
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=120,
            actual_output=30,
            cache_read=0,
            cache_write=0,
            input_saved=80,
            output_saved=20,
            behavior_signals={
                "tok_fallback_activated": 2,
                "baseline_only_session": 1,
            },
        )
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "ok",
                    "bridge": "tok",
                    "port": 9090,
                    "mode": "tool-compatible",
                    "baseline_only": False,
                    "fallback_count": 0,
                    "session_tokens_saved": 250,
                    "session_savings_pct": 25.0,
                    "actual_cost_usd": 0.75,
                    "baseline_cost_usd": 1.0,
                    "cost_saved_usd": 0.25,
                    "session_quality": "clean",
                    "last_degradation_reason": "",
                    "semantic_drift_count": 0,
                    "fail_open_count": 0,
                    "non_tok_count": 0,
                    "answer_anchor_miss_count": 0,
                    "repeat_search_count": 0,
                    "repeat_file_read_count": 0,
                    "state_resend_full_count": 0,
                    "state_resend_delta_count": 0,
                    "state_resend_suppressed_count": 0,
                }

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())

        result = runner.invoke(app, ["bridge", "status"])

        assert result.exit_code == 0
        assert "Saved $0.2500" in result.output
        assert "25.0% saved" in result.output
        assert "Tok active and helping" in result.output
        assert "Session degraded to baseline" not in result.output

    def test_bridge_status_ignores_unremovable_stale_pid(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _bridge, _cli_support

        pid_file = tmp_path / "bridge.pid"
        pid_file.write_text("not-a-pid")
        monkeypatch.setattr(_cli_support, "PID_FILE", pid_file)
        monkeypatch.setattr(_bridge, "PID_FILE", pid_file)
        monkeypatch.setattr(_cli_support, "find_pids_on_port", lambda port: [])

        from pathlib import Path as _Path

        def deny_unlink(*args, **kwargs) -> NoReturn:
            msg = "sandbox"
            raise PermissionError(msg)

        monkeypatch.setattr(_Path, "unlink", deny_unlink)

        result = runner.invoke(app, ["bridge", "status"])
        assert result.exit_code == 1, f"exit={result.exit_code} output={result.output!r}"
        assert "Bridge not running" in result.output
        assert "tok bridge start" in result.output

    def test_bridge_start_with_capture_prints_capture_directory(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _bridge, _cli_support

        def no_bridge(port) -> None:
            return None

        def no_collector(_debug=False) -> None:
            return None

        def fake_memory_root():
            return tmp_path / ".tok"

        for mod in (_bridge, _cli_support):
            monkeypatch.setattr(mod, "get_running_bridge_pid", no_bridge)
            monkeypatch.setattr(mod, "start_collector", no_collector)
            monkeypatch.setattr(mod, "LOG_FILE", tmp_path / "bridge.log")
            monkeypatch.setattr(mod, "PID_FILE", tmp_path / "bridge.pid")
            monkeypatch.setattr(mod, "memory_root", fake_memory_root)

        class FakeProcess:
            pid = 4321

        class FakeResponse:
            status_code = 200

        monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: FakeProcess())
        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())

        result = runner.invoke(app, ["bridge", "start", "--capture"])

        assert result.exit_code == 0
        assert "Bridge started on :9090 (PID 4321)" in result.output
        assert "Capture directory:" in result.output
        assert ".tok/sessions" in result.output
        assert "run `claude`" in result.output

    def test_bridge_start_enables_session_reset(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _bridge, _cli_support

        def no_bridge(port) -> None:
            return None

        def no_collector(_debug=False) -> None:
            return None

        for mod in (_bridge, _cli_support):
            monkeypatch.setattr(mod, "get_running_bridge_pid", no_bridge)
            monkeypatch.setattr(mod, "start_collector", no_collector)
            monkeypatch.setattr(mod, "LOG_FILE", tmp_path / "bridge.log")
            monkeypatch.setattr(mod, "PID_FILE", tmp_path / "bridge.pid")

        captured = {}

        class FakeProcess:
            pid = 4321

        class FakeResponse:
            status_code = 200

        def _fake_popen(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return FakeProcess()

        monkeypatch.setattr("subprocess.Popen", _fake_popen)
        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())

        result = runner.invoke(app, ["bridge", "start"])

        assert result.exit_code == 0
        assert captured["env"]["TOK_RESET_SESSION"] == "1"

    def test_bridge_start_foreground_forwards_api_base(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _bridge, _cli_support

        def no_bridge(port) -> None:
            return None

        def no_collector(_debug=False) -> None:
            return None

        for mod in (_bridge, _cli_support):
            monkeypatch.setattr(mod, "get_running_bridge_pid", no_bridge)
            monkeypatch.setattr(mod, "start_collector", no_collector)
            monkeypatch.setattr(mod, "LOG_FILE", tmp_path / "bridge.log")
            monkeypatch.setattr(mod, "PID_FILE", tmp_path / "bridge.pid")

        captured = {}

        def _fake_run_bridge(**kwargs) -> None:
            captured.update(kwargs)

        monkeypatch.setattr("tok.gateway.run_bridge", _fake_run_bridge)

        result = runner.invoke(
            app,
            [
                "bridge",
                "start",
                "--foreground",
                "--api-base",
                "https://example.test/custom",
            ],
        )

        assert result.exit_code == 0
        assert captured["_api_base"] == "https://example.test/custom"

    def test_bridge_start_subprocess_exports_custom_api_base(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _bridge, _cli_support

        def no_bridge(port) -> None:
            return None

        def no_collector(_debug=False) -> None:
            return None

        def fake_memory_root():
            return tmp_path / ".tok"

        for mod in (_bridge, _cli_support):
            monkeypatch.setattr(mod, "get_running_bridge_pid", no_bridge)
            monkeypatch.setattr(mod, "start_collector", no_collector)
            monkeypatch.setattr(mod, "LOG_FILE", tmp_path / "bridge.log")
            monkeypatch.setattr(mod, "PID_FILE", tmp_path / "bridge.pid")
            monkeypatch.setattr(mod, "memory_root", fake_memory_root)

        captured = {}

        class FakeProcess:
            pid = 4321

        class FakeResponse:
            status_code = 200

        def _fake_popen(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return FakeProcess()

        monkeypatch.setattr("subprocess.Popen", _fake_popen)
        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())

        result = runner.invoke(
            app,
            [
                "bridge",
                "start",
                "--api-base",
                "https://example.test/custom",
            ],
        )

        assert result.exit_code == 0
        assert captured["env"]["TOK_API_BASE"] == "https://example.test/custom"
        assert captured["env"]["TOK_RESET_SESSION"] == "1"

    def test_savings_no_data(self) -> None:
        result = runner.invoke(app, ["savings"])
        assert result.exit_code == 0

    def test_stats_alias_shows_fallback_and_baseline_state(self, tmp_path, monkeypatch) -> None:
        tracker = SavingsTracker(
            savings_file=str(tmp_path / "tok_savings.tok"),
            ledger_path=tmp_path / "global_savings.tok",
        )
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=120,
            actual_output=30,
            cache_read=0,
            cache_write=0,
            input_saved=80,
            output_saved=20,
            behavior_signals={
                "tok_fallback_activated": 2,
                "baseline_only_session": 1,
            },
        )
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))

        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 0
        assert "Current Session" in result.output
        assert "Saved $0.0005" in result.output
        assert "40.0% saved" in result.output
        assert "Session degraded to baseline" in result.output
        assert "Strong savings" in result.output
        assert "Fallbacks" in result.output
        assert "Degraded to baseline" in result.output
        assert "yes" in result.output
        assert "Session quality" in result.output
        assert "Degradation reason" in result.output

    def test_stats_total_shows_lifetime_fallback_counts(self, tmp_path, monkeypatch) -> None:
        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n  sessions: 2\n  total_turns: 10\n  total_tokens: 2000\n  total_cost_usd: 0.020000\n  estimated_baseline_cost_usd: 0.030000\n  tokens_saved: 1000\n  cost_saved_usd: 0.010000\n  savings_pct: 33.3\n  tok_fallback_activated: 3\n  baseline_only_session: 1\n\n@per_session_log"
            "\n"
        )
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))

        result = runner.invoke(app, ["stats", "--total"])
        assert result.exit_code == 0
        assert "Lifetime" in result.output
        assert "Saved $0.0100" in result.output
        assert "33.3% saved" in result.output
        assert "1,000 tokens avoided" in result.output
        assert "Fallbacks" in result.output
        assert "Baseline-only requests" in result.output

    def test_stats_last_session_reads_latest_completed_session(self, tmp_path, monkeypatch) -> None:
        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n  sessions: 2\n  total_turns: 10\n  total_tokens: 2000\n  total_cost_usd: 0.020000\n  estimated_baseline_cost_usd: 0.045000\n  tokens_saved: 1200\n  cost_saved_usd: 0.025000\n  savings_pct: 55.0\n\n@per_session_log\n  # format: date;session_id;turns;tokens;cost_usd;baseline_cost_usd;saved_usd;tokens_saved;invisible_pressure;non_tok_response\n  2026-03-17T10:00:00Z;aaa11111;5;1000;0.010000;0.020000;0.010000;500;3;1\n  2026-03-18T10:00:00Z;bbb22222;5;1000;0.010000;0.025000;0.015000;700;1;0"
            "\n"
        )
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))

        result = runner.invoke(app, ["stats", "--last-session"])
        assert result.exit_code == 0
        assert "Last Completed Session" in result.output
        assert "Saved $0.0150" in result.output
        assert "60.0% saved" in result.output
        assert "Strong savings" in result.output
        assert "2026-03-18T10:00:00Z" in result.output
        assert "Session quality" in result.output
        assert "Degradation reason" in result.output

    def test_stats_recent_shows_recent_window_summary(self, tmp_path, monkeypatch) -> None:
        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n  sessions: 3\n  total_turns: 12\n  total_tokens: 2400\n  total_cost_usd: 0.024000\n  estimated_baseline_cost_usd: 0.042000\n  tokens_saved: 1200\n  cost_saved_usd: 0.018000\n  savings_pct: 42.9\n\n@per_session_log\n  2026-03-17T10:00:00Z;aaa11111;4;800;0.008000;0.014000;0.006000;400;1;0\n  2026-03-18T10:00:00Z;bbb22222;4;800;0.008000;0.016000;0.008000;500;1;0\n  2026-03-19T10:00:00Z;ccc33333;4;800;0.008000;0.012000;0.004000;300;1;0"
            "\n"
        )
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))

        result = runner.invoke(app, ["stats", "--recent", "2", "--total"])
        assert result.exit_code == 0
        assert "Recent Sessions (2)" in result.output
        assert "Saved $0.0120" in result.output
        assert "42.9% saved" in result.output
        assert "Strong savings" in result.output
        assert "800 tokens" in result.output
        assert "2026-03-18T10:00:00Z" in result.output

    def test_stats_since_shows_filtered_window_summary(self, tmp_path, monkeypatch) -> None:
        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n  sessions: 3\n  total_turns: 12\n  total_tokens: 2400\n  total_cost_usd: 0.024000\n  estimated_baseline_cost_usd: 0.042000\n  tokens_saved: 1200\n  cost_saved_usd: 0.018000\n  savings_pct: 42.9\n\n@per_session_log\n  2026-03-17T10:00:00Z;aaa11111;4;800;0.008000;0.014000;0.006000;400;1;0\n  2026-03-18T10:00:00Z;bbb22222;4;800;0.008000;0.016000;0.008000;500;1;0\n  2026-03-19T10:00:00Z;ccc33333;4;800;0.008000;0.012000;0.004000;300;1;0"
            "\n"
        )
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))

        result = runner.invoke(app, ["stats", "--since", "2026-03-18", "--total"])
        assert result.exit_code == 0
        assert "Since 2026-03-18" in result.output
        assert "Saved $0.0120" in result.output
        assert "42.9% saved" in result.output
        assert "Sessions" in result.output

    def test_bridge_stop_prints_compact_session_summary(self, tmp_path, monkeypatch, capsys) -> None:
        from tok.cli import _bridge

        tracker = SavingsTracker(
            savings_file=str(tmp_path / "tok_savings.tok"),
            ledger_path=tmp_path / "global_savings.tok",
        )
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=120,
            actual_output=30,
            cache_read=0,
            cache_write=0,
            input_saved=80,
            output_saved=20,
            behavior_signals={"tok_fallback_activated": 1},
        )
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setattr(_bridge, "get_running_bridge_pid", lambda port: 123)
        monkeypatch.setattr(_bridge, "read_collector_pid", lambda: None)
        monkeypatch.setattr(_bridge, "PID_FILE", tmp_path / "bridge.pid")
        monkeypatch.setattr(_bridge, "COLLECTOR_PID_FILE", tmp_path / "collector.pid")

        calls = {"checked": False}

        def fake_kill(pid, sig) -> None:
            assert pid == 123
            if sig == signal.SIGTERM:
                return
            if sig == 0 and not calls["checked"]:
                calls["checked"] = True
                raise ProcessLookupError
            return

        monkeypatch.setattr(_bridge.os, "kill", fake_kill)

        _bridge.bridge_stop()
        output = capsys.readouterr().out
        assert "Bridge stopped" in output
        assert "Last Session" in output
        assert "Saved $" in output
        assert "Verdict" in output

    def test_savings_breakdown_shows_behavior_signals(self, tmp_path, monkeypatch) -> None:
        tracker = SavingsTracker(
            savings_file=str(tmp_path / "tok_savings.tok"),
            ledger_path=tmp_path / "global_savings.tok",
        )
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=100,
            actual_output=50,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=5,
            behavior_signals={"repeat_file_read": 2, "python_c_workaround": 1},
        )
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))

        result = runner.invoke(app, ["savings", "--breakdown"])
        assert result.exit_code == 0
        assert "Tok health:" in result.output
        assert "Request-side savings:" in result.output
        assert "input_saved_tokens=10" in result.output
        assert "Behavior signals" in result.output
        assert "repeat_file_read" in result.output

    def test_savings_trends_shows_recent_trend_summary(self, tmp_path, monkeypatch) -> None:
        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n  sessions: 2\n  total_turns: 10\n  total_tokens: 2000\n  tokens_saved: 1200\n  total_cost_usd: 0.020000\n  cost_saved_usd: 0.025000\n  savings_pct: 55.0\n\n@per_session_log\n  # format: date;session_id;turns;tokens;cost_usd;baseline_cost_usd;saved_usd;tokens_saved;invisible_pressure;non_tok_response\n  2026-03-17T10:00:00Z;aaa11111;5;1000;0.010000;0.020000;0.010000;500;3;1\n  2026-03-18T10:00:00Z;bbb22222;5;1000;0.010000;0.025000;0.015000;700;1;0"
            "\n"
        )
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "savings.tok"))

        result = runner.invoke(app, ["savings", "--total", "--trends"])
        assert result.exit_code == 0
        assert "Trend:" in result.output
        assert "direction=improving" in result.output

    def test_convert_json_to_tok(self) -> None:
        result = runner.invoke(app, ["convert", '{"key": "value"}', "--to", "tok"])
        assert result.exit_code == 0

    def test_parse_tok(self) -> None:
        result = runner.invoke(app, ["parse", "@block\n  |> content"])
        assert result.exit_code == 0

    def test_generate_fixture_uses_fixture_generator_backend(self, monkeypatch) -> None:
        calls = {}

        class FakeGenerator:
            def generate_coding_session(self, name, turns, template, complexity):
                calls["generate"] = (name, turns, template, complexity)
                return "fixture-data", '{"name": "demo"}'

            def save_fixture(self, name, fixture, metadata, output) -> None:
                calls["save"] = (name, fixture, metadata, output)

        monkeypatch.setattr("tok.testing.fixture_generator.FixtureGenerator", FakeGenerator)

        result = runner.invoke(
            app,
            [
                "dev",
                "generate-fixture",
                "coding",
                "demo-fixture",
                "--turns",
                "3",
                "--template",
                "standard_claude",
                "--complexity",
                "simple",
                "--output",
                "tmp-fixtures",
            ],
        )

        assert result.exit_code == 0
        assert calls["generate"] == (
            "demo-fixture",
            3,
            "standard_claude",
            "simple",
        )
        assert calls["save"] == (
            "demo-fixture",
            "fixture-data",
            '{"name": "demo"}',
            "tmp-fixtures",
        )
        assert "Generated coding fixture: demo-fixture" in result.output

    def test_pressure_command_invokes_metrics_backend(self, monkeypatch, tmp_path) -> None:
        calls = {}

        monkeypatch.setattr(
            "tok.utils.metrics.pressure_trends",
            lambda window, export: calls.update({"window": window, "export": export}),
        )

        export_path = tmp_path / "pressure.json"
        result = runner.invoke(
            app,
            [
                "metrics",
                "pressure",
                "--window",
                "7",
                "--export",
                str(export_path),
            ],
        )

        assert result.exit_code == 0
        assert calls == {"window": 7, "export": str(export_path)}

    def test_memory_command_invokes_metrics_backend(self, monkeypatch) -> None:
        calls = {}

        monkeypatch.setattr(
            "tok.utils.metrics.memory_trends",
            lambda window: calls.update({"window": window}),
        )

        result = runner.invoke(app, ["metrics", "memory", "--window", "6"])

        assert result.exit_code == 0
        assert calls == {"window": 6}

    def test_savings_trend_command_invokes_metrics_backend(self, monkeypatch) -> None:
        calls = {}

        monkeypatch.setattr(
            "tok.utils.metrics.savings_trends",
            lambda window: calls.update({"window": window}),
        )

        result = runner.invoke(app, ["metrics", "savings-trend", "--window", "8"])

        assert result.exit_code == 0
        assert calls == {"window": 8}

    def test_fallback_command_invokes_metrics_backend(self, monkeypatch) -> None:
        calls = {}

        monkeypatch.setattr(
            "tok.utils.metrics.fallback_trends",
            lambda window: calls.update({"window": window}),
        )

        result = runner.invoke(app, ["metrics", "fallback", "--window", "4"])

        assert result.exit_code == 0
        assert calls == {"window": 4}

    def test_health_command_invokes_metrics_backend(self, monkeypatch, tmp_path) -> None:
        calls = {}

        monkeypatch.setattr(
            "tok.utils.metrics.health_summary",
            lambda window, export: calls.update({"window": window, "export": export}),
        )

        export_path = tmp_path / "health.json"
        result = runner.invoke(
            app,
            [
                "metrics",
                "health",
                "--window",
                "9",
                "--export",
                str(export_path),
            ],
        )

        assert result.exit_code == 0
        assert calls == {"window": 9, "export": str(export_path)}

    def test_replay_shows_behavior_signals(self, tmp_path) -> None:
        session_file = tmp_path / "capture.jsonl"
        record = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "a1",
                            "name": "view_file",
                            "input": {"path": "src/tok/gateway.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "a2",
                            "name": "view_file",
                            "input": {"path": "src/tok/gateway.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "a3",
                            "name": "bash",
                            "input": {"command": "python -c 'print(1)' >&2"},
                        },
                    ],
                }
            ]
        }
        session_file.write_text(json.dumps(record) + "\n")

        result = runner.invoke(app, ["replay", str(session_file)])
        assert result.exit_code == 0
        assert "Behavior replay:" in result.output
        assert "repeat_file_read" in result.output

    def test_replay_gate_fails_on_high_pressure(self, tmp_path) -> None:
        session_file = tmp_path / "capture.jsonl"
        record = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "a1",
                            "name": "view_file",
                            "input": {"path": "src/tok/gateway.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "a2",
                            "name": "view_file",
                            "input": {"path": "src/tok/gateway.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "a3",
                            "name": "grep",
                            "input": {"query": "create_app"},
                        },
                        {
                            "type": "tool_use",
                            "id": "a4",
                            "name": "grep",
                            "input": {"query": "create_app"},
                        },
                        {
                            "type": "tool_use",
                            "id": "a5",
                            "name": "bash",
                            "input": {"command": "python -c 'print(1)' >&2"},
                        },
                    ],
                }
            ]
        }
        session_file.write_text(json.dumps(record) + "\n")
        session_file.with_suffix(".jsonl.meta.json").write_text(
            json.dumps(
                {
                    "model": "claude-sonnet-4",
                    "provider": "anthropic",
                    "family": "claude",
                    "min_savings_pct": 20.0,
                    "max_invisible_pressure": 3,
                    "max_repeat_file_read": 0,
                    "max_repeat_search": 0,
                    "max_non_tok_response": 0,
                    "max_fail_open_compat_response": 0,
                    "max_malformed_tok_response": 0,
                    "max_blocker_rediscovery": 0,
                }
            )
        )

        result = runner.invoke(app, ["replay", str(session_file), "--gate"])
        assert result.exit_code == 1

    def test_replay_gate_requires_metadata(self, tmp_path) -> None:
        session_file = tmp_path / "capture.jsonl"
        session_file.write_text(json.dumps({"messages": []}) + "\n")

        result = runner.invoke(app, ["replay", str(session_file), "--gate"])
        assert result.exit_code == 1

    def test_capture_summary_reports_verdict_and_reacquisition(self, tmp_path) -> None:
        session_file = tmp_path / "capture.jsonl"
        session_file.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "event": "request",
                            "model": "claude-sonnet-4",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "tool_use",
                                            "id": "a1",
                                            "name": "view_file",
                                            "input": {"path": "src/tok/gateway.py"},
                                        },
                                        {
                                            "type": "tool_use",
                                            "id": "a2",
                                            "name": "view_file",
                                            "input": {"path": "src/tok/gateway.py"},
                                        },
                                    ],
                                }
                            ],
                        }
                    ),
                    json.dumps(
                        {
                            "event": "response",
                            "model": "claude-sonnet-4",
                            "baseline_only": False,
                            "fallback_count": 1,
                            "behavior_signals": {"tok_fallback_activated": 1},
                            "last_degradation_reason": "context reacquisition",
                        }
                    ),
                ]
            )
            + "\n"
        )

        result = runner.invoke(app, ["capture-summary", str(session_file)])

        assert result.exit_code == 0
        assert "Capture Summary" in result.output
        assert "Requests" in result.output
        assert "claude-sonnet-4" in result.output
        assert "Repeat file read" in result.output
        assert "watch" in result.output
        assert "context reacquisition" in result.output

    def test_capture_review_aggregates_sessions_and_exports_json(self, tmp_path) -> None:
        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        (capture_dir / "a.jsonl").write_text(
            json.dumps(
                {
                    "event": "response",
                    "model": "claude-sonnet-4",
                    "baseline_only": False,
                    "fallback_count": 1,
                    "session_quality": "watch",
                    "session_savings_pct": 31.0,
                    "last_degradation_reason": "context reacquisition",
                }
            )
            + "\n"
        )
        (capture_dir / "b.jsonl").write_text(
            json.dumps(
                {
                    "event": "response",
                    "model": "gpt-4.1-mini",
                    "baseline_only": True,
                    "fallback_count": 2,
                    "session_quality": "degraded",
                    "session_savings_pct": 12.0,
                    "last_degradation_reason": "baseline fallback",
                }
            )
            + "\n"
        )
        export_path = tmp_path / "review.json"

        result = runner.invoke(
            app,
            [
                "capture-review",
                str(capture_dir),
                "--candidates",
                "--json",
                str(export_path),
            ],
        )

        assert result.exit_code == 0
        assert "Capture Review" in result.output
        assert "Captured Sessions" in result.output
        assert "Promotion Candidates" in result.output
        data = json.loads(export_path.read_text())
        assert "sessions" in data
        assert "aggregate" in data
        assert "candidates" in data
        assert data["aggregate"]["total_sessions"] == 2

    def test_capture_review_supports_coverage_with_stress_evidence(self, tmp_path) -> None:
        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        (capture_dir / "a.jsonl").write_text(
            json.dumps(
                {
                    "event": "response",
                    "model": "claude-sonnet-4",
                    "baseline_only": False,
                    "fallback_count": 0,
                    "session_quality": "watch",
                    "last_degradation_reason": "context reacquisition",
                }
            )
            + "\n"
        )
        stress_dir = tmp_path / "stress"
        stress_dir.mkdir()
        (stress_dir / "breakpoints.json").write_text(json.dumps([{"breakpoint_class": "protocol_drift"}]))
        replay_dir = tmp_path / "replay"
        replay_dir.mkdir()
        (replay_dir / "release_reacquisition.jsonl.meta.json").write_text("{}")
        (replay_dir / "runtime_conformance.jsonl.meta.json").write_text("{}")
        gate_config = tmp_path / "gate-config.json"
        gate_config.write_text(
            json.dumps(
                {
                    "required_fixtures": [
                        "runtime_conformance",
                        "alternating_adapters",
                        "release_reacquisition",
                    ]
                }
            )
        )

        result = runner.invoke(
            app,
            [
                "capture-review",
                str(capture_dir),
                "--coverage",
                "--stress-dir",
                str(stress_dir),
                "--fixtures-dir",
                str(replay_dir),
                "--gate-config",
                str(gate_config),
            ],
        )

        assert result.exit_code == 0
        assert "Evidence Coverage" in result.output
        assert "context reacquisition" in result.output
        assert "response contract drift" in result.output

    def test_evidence_gap_writes_json(self, tmp_path) -> None:
        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        (capture_dir / "a.jsonl").write_text(
            json.dumps(
                {
                    "event": "response",
                    "model": "claude-sonnet-4",
                    "baseline_only": False,
                    "fallback_count": 1,
                    "session_quality": "watch",
                    "last_degradation_reason": "baseline fallback",
                }
            )
            + "\n"
        )
        replay_dir = tmp_path / "replay"
        replay_dir.mkdir()
        gate_config = tmp_path / "gate-config.json"
        gate_config.write_text(json.dumps({"required_fixtures": []}))
        export_path = tmp_path / "gap.json"

        result = runner.invoke(
            app,
            [
                "evidence-gap",
                str(capture_dir),
                "--fixtures-dir",
                str(replay_dir),
                "--gate-config",
                str(gate_config),
                "--json",
                str(export_path),
            ],
        )

        assert result.exit_code == 0
        assert "Evidence Gap" in result.output
        data = json.loads(export_path.read_text())
        assert "coverage" in data
        assert data["coverage"][0]["candidate"] == "baseline fallback"

    def test_gate_check_reports_trend_as_warning_not_fixture_failure(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        fixture = fixtures_dir / "single_fixture.jsonl"
        fixture.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {
                            "role": "assistant",
                            "content": ">>> t:1|usr:hello\n@msg role:assistant\n  |> ok",
                        },
                    ]
                }
            )
            + "\n"
        )
        fixture.with_suffix(".jsonl.meta.json").write_text(
            json.dumps(
                {
                    "model": "claude-sonnet-4",
                    "provider": "anthropic",
                    "family": "claude",
                    "min_savings_pct": 0.0,
                    "max_invisible_pressure": 0,
                    "max_repeat_file_read": 0,
                    "max_repeat_search": 0,
                    "max_non_tok_response": 0,
                    "max_fail_open_compat_response": 0,
                    "max_malformed_tok_response": 0,
                    "max_blocker_rediscovery": 0,
                }
            )
        )

        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n  sessions: 2\n  total_turns: 10\n  total_tokens: 2000\n  tokens_saved: 1000\n  total_cost_usd: 0.020000\n  cost_saved_usd: 0.005000\n  savings_pct: 20.0\n\n@per_session_log\n  # format: date;session_id;turns;tokens;cost_usd;baseline_cost_usd;saved_usd;tokens_saved;invisible_pressure;non_tok_response\n  2026-03-17T10:00:00Z;aaa11111;5;1000;0.010000;0.030000;0.020000;800;1;0\n  2026-03-18T10:00:00Z;bbb22222;5;1000;0.010000;0.015000;0.005000;200;9;0"
            "\n"
        )
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "savings.tok"))

        result = runner.invoke(app, ["gate-check", str(fixtures_dir)])

        assert result.exit_code == 0
        assert "single_fixture" in result.output
        assert "✅ PASS" in result.output
        assert "Set Trend:" in result.output
        assert "status=clean" in result.output

    def test_doctor_shows_runtime_mode_and_session_savings(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/claude")
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-empty-savings-doctor.tok")
        Path("/tmp/test-empty-savings-doctor.tok").unlink(missing_ok=True)

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "ok",
                    "bridge": "tok",
                    "port": 9090,
                    "mode": "baseline",
                    "baseline_only": True,
                    "fallback_count": 3,
                    "session_tokens_saved": 120,
                    "session_savings_pct": 41.4,
                    "session_quality": "degraded",
                    "last_degradation_reason": "baseline fallback",
                    "semantic_drift_count": 0,
                    "fail_open_count": 0,
                    "non_tok_count": 0,
                    "answer_anchor_miss_count": 0,
                    "repeat_search_count": 0,
                    "repeat_file_read_count": 0,
                    "state_resend_full_count": 0,
                    "state_resend_delta_count": 0,
                    "state_resend_suppressed_count": 0,
                }

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())
        monkeypatch.setattr(
            "tok.cli._release.memory_root",
            lambda: Path("/tmp/nonexistent_tok"),
        )

        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
        assert "Current Session" in result.output
        assert "Saved $0.0000" in result.output
        assert "41.4%" in result.output
        assert "Session degraded to baseline" in result.output
        assert "Fallbacks" in result.output
        assert "Tok verdict:" in result.output
        assert "baseline" in result.output
        assert "Recommendation:" in result.output
        assert "investigate degradation before trusting this session" in result.output

    def test_doctor_surfaces_request_shape_and_stream_attribution(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/claude")

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "ok",
                    "bridge": "tok",
                    "port": 9090,
                    "mode": "natural-first",
                    "baseline_only": False,
                    "fallback_count": 0,
                    "session_tokens_saved": 120,
                    "session_savings_pct": 41.4,
                    "actual_tokens": 100,
                    "baseline_tokens": 200,
                    "actual_cost_usd": 0.01,
                    "baseline_cost_usd": 0.02,
                    "cost_saved_usd": 0.01,
                    "session_quality": "watch",
                    "last_degradation_reason": "stream transport instability",
                    "semantic_drift_count": 0,
                    "fail_open_count": 0,
                    "non_tok_count": 0,
                    "answer_anchor_miss_count": 0,
                    "repeat_search_count": 0,
                    "repeat_file_read_count": 0,
                    "preflight_block_original_payload_count": 1,
                    "preflight_block_rewritten_payload_count": 0,
                    "stream_recovery_empty_success_count": 2,
                    "stream_recovery_read_error_count": 3,
                    "request_policy_held_by_recovery_count": 1,
                }

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())
        monkeypatch.setattr(
            "tok.cli._release.memory_root",
            lambda: Path("/tmp/nonexistent_tok"),
        )

        result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 1
        assert "stream transport instability" in result.output
        assert "shape-orig=1" in result.output
        assert "stream-empty=2" in result.output
        assert "stream-read=3" in result.output
        assert "held=1" in result.output

    def test_gate_check_export_includes_behavior_signals_and_fixture_metadata(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        fixture = fixtures_dir / "grammar_drift.jsonl"
        fixture.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": ">>> t:1|usr:hello\n@msg role:assistant\n  | bad",
                                }
                            ],
                        },
                    ]
                }
            )
            + "\n"
        )
        fixture.with_suffix(".jsonl.meta.json").write_text(
            json.dumps(
                {
                    "model": "claude-sonnet-4",
                    "provider": "anthropic",
                    "family": "claude",
                    "min_savings_pct": 0.0,
                    "max_invisible_pressure": 10,
                    "max_repeat_file_read": 0,
                    "max_repeat_search": 0,
                    "max_non_tok_response": 0,
                    "max_fail_open_compat_response": 5,
                    "max_malformed_tok_response": 5,
                    "max_blocker_rediscovery": 0,
                    "fixture_kind": "conformance",
                    "provenance": "synthetic",
                }
            )
        )
        export_path = tmp_path / "results.json"

        result = runner.invoke(
            app,
            ["gate-check", str(fixtures_dir), "--export", str(export_path)],
        )

        assert result.exit_code == 0
        data = json.loads(export_path.read_text())
        exported = data["results"][0]
        assert exported["fixture"] == "grammar_drift"
        assert exported["behavior_signals"]["malformed_tok_response"] == 1
        assert exported["fixture_format"] == "session-records"
        assert exported["fixture_kind"] == "conformance"
        assert exported["provenance"] == "synthetic"

    def test_gate_check_export_includes_common_path_summary(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        fixture = fixtures_dir / "claude_coding_loop.jsonl"
        fixture.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {
                            "role": "assistant",
                            "content": ">>> t:1|usr:hello\n@msg role:assistant\n  |> ok",
                        },
                    ]
                }
            )
            + "\n"
        )
        fixture.with_suffix(".jsonl.meta.json").write_text(
            json.dumps(
                {
                    "model": "claude-sonnet-4",
                    "provider": "anthropic",
                    "family": "claude",
                    "min_savings_pct": 0.0,
                    "max_invisible_pressure": 0,
                    "max_repeat_file_read": 0,
                    "max_repeat_search": 0,
                    "max_non_tok_response": 0,
                    "max_fail_open_compat_response": 0,
                    "max_malformed_tok_response": 0,
                    "max_blocker_rediscovery": 0,
                    "fixture_kind": "compression",
                    "provenance": "synthetic",
                    "common_path": True,
                    "usage_weight": 2.5,
                }
            )
        )
        export_path = tmp_path / "results.json"

        result = runner.invoke(
            app,
            ["gate-check", str(fixtures_dir), "--export", str(export_path)],
        )

        assert result.exit_code == 0
        data = json.loads(export_path.read_text())
        assert data["common_path_summary"]["fixtures"] == 1
        assert data["common_path_summary"]["total_weight"] == 2.5

    def test_gate_check_export_includes_release_summary(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        fixture = fixtures_dir / "grammar_drift.jsonl"
        fixture.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": ">>> t:1|usr:hello\n@msg role:assistant\n  | bad",
                                }
                            ],
                        },
                    ]
                }
            )
            + "\n"
        )
        fixture.with_suffix(".jsonl.meta.json").write_text(
            json.dumps(
                {
                    "model": "claude-sonnet-4",
                    "provider": "anthropic",
                    "family": "claude",
                    "min_savings_pct": 0.0,
                    "max_invisible_pressure": 10,
                    "max_repeat_file_read": 0,
                    "max_repeat_search": 0,
                    "max_non_tok_response": 1,
                    "max_fail_open_compat_response": 5,
                    "max_malformed_tok_response": 5,
                    "max_blocker_rediscovery": 0,
                }
            )
        )
        export_path = tmp_path / "results.json"

        result = runner.invoke(
            app,
            ["gate-check", str(fixtures_dir), "--export", str(export_path)],
        )

        assert result.exit_code == 0
        data = json.loads(export_path.read_text())
        assert "release_summary" in data
        assert "avg_savings_pct" in data["release_summary"]
        assert "avg_invisible_pressure" in data["release_summary"]
        assert "fallback_fixture_rate" in data["release_summary"]
        assert "reacquisition_fixture_rate" in data["release_summary"]
        assert "billing_delta_usd" in data["release_summary"]
        assert isinstance(data["release_summary"]["billing_delta_usd"], float)
        assert data["release_summary"]["billing_delta_usd"] >= 0.0

    def test_stress_language_reports_breakpoint_classes(self, tmp_path, monkeypatch) -> None:
        class FakeHarness:
            def __init__(self, config) -> None:
                self.config = config

            def run(self):
                return SimpleNamespace(
                    tasks_completed=2,
                    required_classes=("protocol_drift", "baseline_fallback"),
                    validated_anchor_count=3,
                    reuse_checks_run=1,
                    checkpoint_checks_run=1,
                    reuse_probe_attempts=1,
                    reuse_probe_successes=1,
                    retention_probe_attempts=1,
                    retention_probe_successes=0,
                    late_retention_probe_attempts=0,
                    late_retention_probe_successes=0,
                    tool_contract_probe_attempts=1,
                    tool_contract_failure_events_seen=1,
                    mixed_answer_tool_events_seen=1,
                    unsupported_tool_events_seen=0,
                    bad_tool_args_events_seen=0,
                    toolless_fresh_answer_events_seen=1,
                    reacquisition_events_seen=1,
                    retention_substitution_events_seen=0,
                    anchors_before_baseline=2,
                    seed_searches=2,
                    seed_direct_reads=2,
                    seed_answer_attempts=1,
                    seed_evidence_sufficient=True,
                    first_anchor_failure_mode="answer_assembly",
                    tool_backed_turns=5,
                    turns=[
                        SimpleNamespace(
                            task_id="retention_probe_early",
                            phase_name="retention-probe",
                        ),
                        SimpleNamespace(task_id="other", phase_name="fresh-grounding"),
                    ],
                    resend_modes_seen=["full", "delta"],
                    payload_pressure_reached=True,
                    compaction_eligible=True,
                    run_diagnosis="memory_surface_reached",
                    weak_run_reasons=[],
                    breakpoints=[
                        SimpleNamespace(breakpoint_class="protocol_drift"),
                        SimpleNamespace(breakpoint_class="baseline_fallback"),
                    ],
                    baseline_only=True,
                )

        monkeypatch.setattr("tok.testing.stress.StressHarness", FakeHarness)
        monkeypatch.setattr(
            "tok.testing.stress.write_stress_artifacts",
            lambda output_dir, result: {
                "stress_run": output_dir / "run.json",
                "breakpoints": output_dir / "breakpoints.json",
                "stress_report": output_dir / "report.md",
                "language_refactor_plan": output_dir / "plan.md",
            },
        )
        monkeypatch.setattr(
            "tok.testing.stress.summarize_implicated_files",
            lambda breakpoints: [{"path": "src/tok/cli.py", "count": 2}],
        )

        result = runner.invoke(app, ["dev", "stress-language", "--output", str(tmp_path)])

        assert result.exit_code == 0
        assert "breakpoints=2" in result.output
        assert "baseline_only=True" in result.output
        assert "Breakpoint classes:" in result.output
        assert "Coverage:" in result.output
        assert "Anchors:" in result.output
        assert "Seed phase:" in result.output
        assert "Memory checks:" in result.output
        assert "Reuse probes:" in result.output
        assert "Retention probes:" in result.output
        assert "Tool contract probes:" in result.output
        assert "Tool contract signals:" in result.output
        assert "Late retention probes:" in result.output
        assert "Early retention probe:" in result.output
        assert "Late retention probe:" in result.output
        assert "Compaction eligibility:" in result.output
        assert "Run diagnosis:" in result.output
        assert "First-anchor failure mode:" in result.output
        assert "baseline_fallback" in result.output
        assert "protocol_drift" in result.output
        assert "report.md" in result.output

    def test_live_benchmark_compare_writes_artifacts(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)

        class FakeRunner:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def run(self, definition, *, mode, turns):
                del definition
                totals = {
                    "baseline": 120,
                    "tok-universal": 85,
                }
                total = totals[mode]
                return SimpleNamespace(
                    benchmark="coding-loop",
                    mode=mode,
                    model="deepseek/deepseek-v3.2",
                    provider="openrouter",
                    fixture_path="tests/fixtures/replay/claude_coding_loop.jsonl",
                    provider_usage=SimpleNamespace(
                        prompt_tokens=total - 20,
                        completion_tokens=20,
                        total_tokens=total,
                        latency_ms=10.0,
                        cost_usd=None,
                    ),
                    compression_metrics={"total_saved_tokens": 10},
                    prompt_metrics={"tok_overhead_tokens": 5},
                    response_metrics={
                        "invisible_pressure": 0,
                        "reacquisition_cost_tokens": 0,
                    },
                    diagnostics={},
                    task_success=True,
                    matched_success_terms=["gateway.py", "passed"],
                    request_messages=2,
                    turn_count=turns,
                    turns=[
                        {
                            "turn": 1,
                            "outbound_payload": {
                                "system": "s",
                                "messages": [{"role": "user", "content": "x"}],
                            },
                        }
                    ],
                    visible_response="File=gateway.py\nVerification=1 passed in 0.05s",
                    raw_response="File=gateway.py\nVerification=1 passed in 0.05s",
                    notes=[],
                    to_dict=lambda self=None, mode=mode, total=total: {
                        "benchmark": "coding-loop",
                        "mode": mode,
                        "model": "deepseek/deepseek-v3.2",
                        "provider": "openrouter",
                        "fixture_path": "tests/fixtures/replay/claude_coding_loop.jsonl",
                        "provider_usage": {
                            "prompt_tokens": total - 20,
                            "completion_tokens": 20,
                            "total_tokens": total,
                            "latency_ms": 10.0,
                            "cost_usd": None,
                        },
                        "compression_metrics": {"total_saved_tokens": 10},
                        "prompt_metrics": {"tok_overhead_tokens": 5},
                        "response_metrics": {
                            "invisible_pressure": 0,
                            "reacquisition_cost_tokens": 0,
                        },
                        "diagnostics": {},
                        "task_success": True,
                        "matched_success_terms": ["gateway.py", "passed"],
                        "request_messages": 2,
                        "turn_count": turns,
                        "turns": [
                            {
                                "turn": 1,
                                "outbound_payload": {
                                    "system": "s",
                                    "messages": [{"role": "user", "content": "x"}],
                                },
                            }
                        ],
                        "visible_response": "File=gateway.py\nVerification=1 passed in 0.05s",
                        "raw_response": "File=gateway.py\nVerification=1 passed in 0.05s",
                        "notes": [],
                    },
                )

        class FakeComparison:
            def __init__(self, mode: str) -> None:
                self.mode = mode

            def to_dict(self) -> dict[str, Any]:
                return {
                    "benchmark": "coding-loop",
                    "candidate_mode": self.mode,
                    "tok_improved": True,
                }

        monkeypatch.setattr("tok.testing.live_benchmark.LiveBenchmarkRunner", FakeRunner)
        monkeypatch.setattr(
            "tok.testing.live_benchmark.load_benchmark_definition",
            lambda name: SimpleNamespace(name=name, default_turns=5),
        )
        monkeypatch.setattr(
            "tok.testing.live_benchmark.compare_results",
            lambda baseline, tok: FakeComparison(tok.mode),
        )
        monkeypatch.setattr(
            "tok.testing.live_benchmark.render_comparison_markdown",
            lambda baseline, comparisons: "# compare\n",
        )
        monkeypatch.setattr(
            "tok.testing.live_benchmark.select_preferred_mode",
            lambda baseline, comparisons: "tok-universal",
        )

        output_dir = tmp_path / "artifacts"
        result = runner.invoke(
            app,
            [
                "dev",
                "live-benchmark",
                "--benchmark",
                "coding-loop",
                "--mode",
                "compare",
                "--repeats",
                "3",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "coding-loop_baseline.json").exists()
        assert (output_dir / "coding-loop_tok-universal.json").exists()
        assert (output_dir / "coding-loop_compare_tok-universal.json").exists()
        for run_index in range(1, 4):
            assert (output_dir / f"coding-loop_run{run_index}_baseline.json").exists()
            assert (output_dir / f"coding-loop_run{run_index}_tok-universal.json").exists()
            assert (output_dir / f"coding-loop_run{run_index}_compare_tok-universal.json").exists()
        assert (output_dir / "coding-loop_compare.md").exists()
        assert (output_dir / "coding-loop_triage.json").exists()
        assert (output_dir / "coding-loop_stability.json").exists()
        assert (output_dir / "coding-loop_stability.md").exists()
        assert "Best mode:" in result.output

    def test_gate_check_expected_failure_fixture_passes_when_signal_detected(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        fixture = fixtures_dir / "grammar_drift.jsonl"
        fixture.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": ">>> t:1|usr:hello\n@msg role:assistant\n  | bad",
                                }
                            ],
                        },
                    ]
                }
            )
            + "\n"
        )
        fixture.with_suffix(".jsonl.meta.json").write_text(
            json.dumps(
                {
                    "model": "claude-sonnet-4",
                    "provider": "anthropic",
                    "family": "claude",
                    "min_savings_pct": 0.0,
                    "max_invisible_pressure": 10,
                    "max_repeat_file_read": 0,
                    "max_repeat_search": 0,
                    "max_non_tok_response": 5,
                    "max_fail_open_compat_response": 0,
                    "max_malformed_tok_response": 1,
                    "max_blocker_rediscovery": 0,
                    "fixture_kind": "conformance",
                    "provenance": "synthetic",
                    "common_path": False,
                    "usage_weight": 0.4,
                    "expected_failure": True,
                    "expected_failures": ["max_fail_open_compat_response"],
                    "expected_signals": ["malformed_tok_response"],
                }
            )
        )

        result = runner.invoke(app, ["gate-check", str(fixtures_dir)])

        assert result.exit_code == 0
        assert "grammar_drift" in result.output
        assert "✅ PASS" in result.output


class TestStabilityGate:
    """Tests for the --stability-dir gate in gate-check."""

    def _make_stability_json(
        self,
        tmp_path: Path,
        benchmark: str,
        success_rate: float,
        preferred_ttc: int,
        runs: int = 5,
    ) -> None:
        """Write a minimal stability JSON file for a benchmark."""
        data = {
            "runs": runs,
            "preferred_mode_counts": {"tok-tool-compatible": preferred_ttc},
            "mode_summaries": {
                "tok-tool-compatible": {
                    "runs": runs,
                    "success_rate": success_rate,
                    "success_count": int(success_rate * runs),
                    "median_total_tokens": 900,
                    "min_total_tokens": 800,
                    "max_total_tokens": 1000,
                    "median_prompt_tokens": 700,
                    "median_completion_tokens": 200,
                    "median_latency_ms": 500.0,
                },
                "baseline": {
                    "runs": runs,
                    "success_rate": 1.0,
                    "success_count": runs,
                    "median_total_tokens": 1700,
                    "min_total_tokens": 1600,
                    "max_total_tokens": 1800,
                    "median_prompt_tokens": 1400,
                    "median_completion_tokens": 300,
                    "median_latency_ms": 600.0,
                },
            },
        }
        f = tmp_path / f"{benchmark}_stability.json"
        f.write_text(json.dumps(data))

    def _make_passing_fixture(self, fixtures_dir: Path) -> None:
        """Write a minimal passing fixture + meta that won't fail the replay gate."""
        f = fixtures_dir / "dummy.jsonl"
        f.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {
                            "role": "assistant",
                            "content": ">>> t:1|usr:hello\n@msg role:assistant\n  |> ok",
                        },
                    ]
                }
            )
            + "\n"
        )
        f.with_suffix(".jsonl.meta.json").write_text(
            json.dumps(
                {
                    "model": "test-model",
                    "provider": "test",
                    "family": "test",
                    "min_savings_pct": 0.0,
                    "max_invisible_pressure": 100,
                    "max_repeat_file_read": 100,
                    "max_repeat_search": 100,
                    "max_non_tok_response": 100,
                    "max_fail_open_compat_response": 100,
                    "max_malformed_tok_response": 100,
                    "max_blocker_rediscovery": 100,
                }
            )
        )

    def _make_frontier_json(
        self,
        tmp_path: Path,
        *,
        release_profile: str = "balanced",
        benchmark_verdict: str = "stable",
        probe_verdict: str | None = None,
    ) -> Path:
        data: dict[str, object] = {
            "checkpoints": [
                {
                    "checkpoint": {
                        "label": "current-head",
                        "ref": "CURRENT",
                    },
                    "benchmark_profiles": [
                        {
                            "profile": {"name": release_profile},
                            "verdict": benchmark_verdict,
                        }
                    ],
                    "openrouter_profiles": [],
                    "default_release_profile": release_profile,
                    "experimental_profiles": ["extreme"],
                }
            ]
        }
        if probe_verdict is not None:
            checkpoint = data["checkpoints"][0]
            checkpoint["openrouter_profiles"] = [
                {
                    "profile": release_profile,
                    "verdict": probe_verdict,
                }
            ]
        path = tmp_path / "frontier.json"
        path.write_text(json.dumps(data))
        return path

    def test_stability_gate_pass(self, tmp_path, monkeypatch) -> None:
        """Both required benchmarks present and passing → exit 0, PASS output."""
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        stab_dir = tmp_path / "stability"
        stab_dir.mkdir()
        self._make_stability_json(stab_dir, "coding-loop-5", 1.0, 5)
        self._make_stability_json(stab_dir, "research-loop-5", 1.0, 5)
        self._make_passing_fixture(fixtures_dir)

        result = runner.invoke(
            app,
            [
                "gate-check",
                str(fixtures_dir),
                "--stability-dir",
                str(stab_dir),
                "--required-benchmarks",
                "coding-loop-5,research-loop-5",
                "--continue-on-error",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "STABILITY PASS" in result.output
        assert "Stability gate: PASS" in result.output

    def test_stability_gate_export_includes_benchmark_results(self, tmp_path, monkeypatch) -> None:
        """Export includes benchmark-level stability status when a stability dir is used."""
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        stab_dir = tmp_path / "stability"
        stab_dir.mkdir()
        self._make_stability_json(stab_dir, "coding-loop-5", 1.0, 5)
        self._make_stability_json(stab_dir, "research-loop-5", 1.0, 5)
        self._make_passing_fixture(fixtures_dir)
        export_path = tmp_path / "results.json"

        result = runner.invoke(
            app,
            [
                "gate-check",
                str(fixtures_dir),
                "--stability-dir",
                str(stab_dir),
                "--required-benchmarks",
                "coding-loop-5,research-loop-5",
                "--export",
                str(export_path),
                "--continue-on-error",
            ],
        )

        assert result.exit_code == 0, result.output
        data = json.loads(export_path.read_text())
        assert data["stability_check"]["coding-loop-5"]["passed"] is True
        assert data["stability_check"]["research-loop-5"]["passed"] is True

    def test_stability_gate_fail_low_success_rate(self, tmp_path, monkeypatch) -> None:
        """A benchmark with success_rate < 1.0 → exit 1, FAIL output."""
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        stab_dir = tmp_path / "stability"
        stab_dir.mkdir()
        self._make_stability_json(stab_dir, "coding-loop-5", 1.0, 5)
        self._make_stability_json(stab_dir, "research-loop-5", 0.8, 5)  # failing
        self._make_passing_fixture(fixtures_dir)

        result = runner.invoke(
            app,
            [
                "gate-check",
                str(fixtures_dir),
                "--stability-dir",
                str(stab_dir),
                "--required-benchmarks",
                "coding-loop-5,research-loop-5",
                "--continue-on-error",
            ],
        )
        assert result.exit_code == 1
        assert "STABILITY FAIL" in result.output
        assert "research-loop-5" in result.output
        assert "Stability gate: FAIL" in result.output

    def test_stability_gate_fail_missing_file(self, tmp_path, monkeypatch) -> None:
        """A required benchmark file missing → exit 1, file-not-found message."""
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        stab_dir = tmp_path / "stability"
        stab_dir.mkdir()
        self._make_stability_json(stab_dir, "coding-loop-5", 1.0, 5)
        # research-loop-5 file intentionally absent
        self._make_passing_fixture(fixtures_dir)

        result = runner.invoke(
            app,
            [
                "gate-check",
                str(fixtures_dir),
                "--stability-dir",
                str(stab_dir),
                "--required-benchmarks",
                "coding-loop-5,research-loop-5",
                "--continue-on-error",
            ],
        )
        assert result.exit_code == 1
        assert "STABILITY FAIL" in result.output
        assert "research-loop-5" in result.output
        assert "file not found" in result.output

    def test_frontier_gate_pass(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        self._make_passing_fixture(fixtures_dir)
        frontier_path = self._make_frontier_json(tmp_path)

        result = runner.invoke(
            app,
            [
                "gate-check",
                str(fixtures_dir),
                "--frontier-report",
                str(frontier_path),
                "--continue-on-error",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "FRONTIER PASS" in result.output
        assert "Compression frontier gate: PASS" in result.output

    def test_frontier_gate_export_includes_frontier_check(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        self._make_passing_fixture(fixtures_dir)
        frontier_path = self._make_frontier_json(tmp_path)
        export_path = tmp_path / "results.json"

        result = runner.invoke(
            app,
            [
                "gate-check",
                str(fixtures_dir),
                "--frontier-report",
                str(frontier_path),
                "--export",
                str(export_path),
                "--continue-on-error",
            ],
        )

        assert result.exit_code == 0, result.output
        data = json.loads(export_path.read_text())
        assert data["frontier_check"]["passed"] is True
        assert data["release_summary"]["frontier_release_profile"] == "balanced"
        assert data["release_summary"]["frontier_status"] == "pass"

    def test_frontier_gate_fails_for_baseline_release_lane(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        self._make_passing_fixture(fixtures_dir)
        frontier_path = self._make_frontier_json(tmp_path, release_profile="baseline")

        result = runner.invoke(
            app,
            [
                "gate-check",
                str(fixtures_dir),
                "--frontier-report",
                str(frontier_path),
                "--continue-on-error",
            ],
        )

        assert result.exit_code == 1
        assert "FRONTIER FAIL" in result.output
        assert "Compression frontier gate: FAIL" in result.output
