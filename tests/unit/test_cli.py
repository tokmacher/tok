"""Tests for tok.cli — CLI commands via typer testing."""

import json
import re
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

from typer.testing import CliRunner

from tok import __version__
from tok.cli import app
from tok.stats import SavingsTracker

runner = CliRunner()


def _normalized_help(text: str) -> str:
    stripped = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return stripped.replace("-", "").lower()


class TestCLI:
    def test_bridge_pid_recovers_from_tok_health_when_pidfile_missing(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _cli_support

        pid_file = tmp_path / "bridge.pid"
        monkeypatch.delenv("TOK_PROJECT_DIR", raising=False)
        monkeypatch.delenv("TOK_DIR", raising=False)
        monkeypatch.setattr(_cli_support, "PID_FILE", pid_file)
        monkeypatch.setattr(_cli_support, "find_pids_on_port", lambda port: [4321])

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json() -> dict[str, Any]:
                return {"status": "ok", "bridge": "tok", "port": 9090}

        monkeypatch.setattr(_cli_support, "get_bridge_health_response", lambda *args, **kwargs: FakeResponse())

        assert _cli_support.get_running_bridge_pid(9090) == 4321
        assert pid_file.read_text() == "4321"

    def test_bridge_pid_does_not_recover_non_tok_listener(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _cli_support

        pid_file = tmp_path / "bridge.pid"
        monkeypatch.delenv("TOK_PROJECT_DIR", raising=False)
        monkeypatch.delenv("TOK_DIR", raising=False)
        monkeypatch.setattr(_cli_support, "PID_FILE", pid_file)
        monkeypatch.setattr(_cli_support, "find_pids_on_port", lambda port: [4321])

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json() -> dict[str, Any]:
                return {"status": "ok", "bridge": "other", "port": 9090}

        monkeypatch.setattr(_cli_support, "get_bridge_health_response", lambda *args, **kwargs: FakeResponse())

        assert _cli_support.get_running_bridge_pid(9090) is None
        assert not pid_file.exists()

    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "bridge" in result.output
        assert "claude" in result.output
        assert "stats" in result.output
        assert "install" in result.output
        assert "doctor" in result.output
        assert "metrics" not in result.output
        assert "dev" not in result.output
        assert "memory-snap" not in result.output
        assert "capture-review" not in result.output
        assert "gate-check" not in result.output
        assert "convert" not in result.output
        assert "parse" not in result.output

    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_shows_program_name(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert result.output.strip().startswith("tok ")

    def test_bridge_help(self) -> None:
        result = runner.invoke(app, ["bridge", "--help"])
        assert result.exit_code == 0
        assert "start" in result.output
        assert "stop" in result.output
        assert "status" in result.output

    def test_bridge_stop_help_shows_force_flag(self) -> None:
        result = runner.invoke(app, ["bridge", "stop", "--help"])
        assert result.exit_code == 0
        assert "force" in _normalized_help(result.output)

    def test_bridge_stop_force_flag_forwards_to_backend(self, monkeypatch) -> None:
        calls: dict[str, bool] = {}

        monkeypatch.setattr("tok.cli._bridge.bridge_stop", lambda force=False: calls.setdefault("force", force))
        result = runner.invoke(app, ["bridge", "stop", "--force"])

        assert result.exit_code == 0
        assert calls["force"] is True

    def test_doctor_help(self) -> None:
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        normalized = _normalized_help(result.output)
        assert "Check bridge health and runtime contract conformance" in result.output
        assert "report" in normalized
        assert "verbose" in normalized

    def test_stats_help(self) -> None:
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0
        normalized = _normalized_help(result.output)
        assert "Show token savings and fallback state" in result.output
        assert ("lastsession" in normalized) or ("session" in normalized)
        assert "recent" in normalized
        assert "since" in normalized

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

    def test_install_default_mode_does_not_write_shell_wrapper(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.utils.shell_integration.uninstall", lambda: [])
        monkeypatch.setattr(
            "tok.utils.shell_integration.install",
            lambda: (_ for _ in ()).throw(AssertionError("install() should not be called in default mode")),
        )

        result = runner.invoke(app, ["install"])

        assert result.exit_code == 0
        assert "Tok install complete." in result.output
        assert "Default mode is explicit and shell-safe" in result.output
        assert "tok claude" in result.output
        assert "tok install --wrap-claude" in result.output

    def test_install_default_mode_removes_legacy_wrapper_when_present(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("tok.utils.shell_integration.uninstall", lambda: [tmp_path / ".zshrc"])
        monkeypatch.setattr(
            "tok.utils.shell_integration.install",
            lambda: (_ for _ in ()).throw(AssertionError("install() should not be called in default mode")),
        )

        result = runner.invoke(app, ["install"])

        assert result.exit_code == 0
        assert "Removed legacy `claude()` Tok wrapper from:" in result.output
        assert ".zshrc" in result.output
        assert "Default mode is explicit and shell-safe" in result.output

    def test_claude_starts_bridge_and_launches_with_bridge_env(self, monkeypatch) -> None:
        from tok.cli import _claude

        calls: dict[str, Any] = {}

        monkeypatch.setattr(_claude._bridge, "get_running_bridge_pid", lambda port: None)

        def fake_bridge_start(**kwargs) -> None:
            calls["bridge_start"] = kwargs

        class FakeCompleted:
            returncode = 0

        def fake_run(argv, *, env, check):
            calls["argv"] = argv
            calls["env"] = env
            calls["check"] = check
            return FakeCompleted()

        monkeypatch.setattr(_claude._bridge, "bridge_start", fake_bridge_start)
        monkeypatch.setattr(_claude.subprocess, "run", fake_run)

        result = runner.invoke(app, ["claude", "--port", "9191", "--api-base", "custom-api-host.internal", "--debug"])

        assert result.exit_code == 0
        assert calls["bridge_start"]["port"] == 9191
        assert calls["bridge_start"]["api_base"] == "custom-api-host.internal"
        assert calls["bridge_start"]["debug"] is True
        assert calls["argv"] == ["claude"]
        assert calls["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:9191"
        assert calls["env"]["TOK_SELF_BRIDGED_SESSION"] == "1"
        assert calls["env"]["TOK_BRIDGE_PORT"] == "9191"
        assert calls["check"] is False

    def test_claude_reuses_existing_bridge_without_starting_new_one(self, monkeypatch) -> None:
        from tok.cli import _claude

        calls: dict[str, Any] = {}

        monkeypatch.setattr(_claude._bridge, "get_running_bridge_pid", lambda port: 4321)
        monkeypatch.setattr(
            _claude._bridge,
            "bridge_start",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("bridge_start should not be called")),
        )

        class FakeCompleted:
            returncode = 0

        def fake_run(argv, *, env, check):
            calls["argv"] = argv
            calls["env"] = env
            return FakeCompleted()

        monkeypatch.setattr(_claude.subprocess, "run", fake_run)

        result = runner.invoke(app, ["claude"])

        assert result.exit_code == 0
        assert "Using existing Tok bridge on :9090 (PID 4321)." in result.output
        assert calls["argv"] == ["claude"]
        assert calls["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:9090"

    def test_claude_passes_trailing_args_unchanged(self, monkeypatch) -> None:
        from tok.cli import _claude

        calls: dict[str, Any] = {}

        monkeypatch.setattr(_claude._bridge, "get_running_bridge_pid", lambda port: 4321)

        class FakeCompleted:
            returncode = 0

        def fake_run(argv, *, env, check):
            calls["argv"] = argv
            return FakeCompleted()

        monkeypatch.setattr(_claude.subprocess, "run", fake_run)

        result = runner.invoke(app, ["claude", "--", "--model", "sonnet", "hello"])

        assert result.exit_code == 0
        assert calls["argv"] == ["claude", "--model", "sonnet", "hello"]

    def test_claude_returns_subprocess_exit_code(self, monkeypatch) -> None:
        from tok.cli import _claude

        monkeypatch.setattr(_claude._bridge, "get_running_bridge_pid", lambda port: 4321)

        class FakeCompleted:
            returncode = 42

        monkeypatch.setattr(_claude.subprocess, "run", lambda *args, **kwargs: FakeCompleted())

        result = runner.invoke(app, ["claude"])

        assert result.exit_code == 42

    def test_claude_reports_missing_claude_executable(self, monkeypatch) -> None:
        from tok.cli import _claude

        monkeypatch.setattr(_claude._bridge, "get_running_bridge_pid", lambda port: 4321)
        monkeypatch.setattr(
            _claude.subprocess,
            "run",
            lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
        )

        result = runner.invoke(app, ["claude"])

        assert result.exit_code == 127
        assert "Claude Code executable not found" in result.output

    def test_install_wrap_claude_uses_shell_integration_backend(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("tok.utils.shell_integration.install", lambda: tmp_path / ".zshrc")

        result = runner.invoke(app, ["install", "--wrap-claude"])

        assert result.exit_code == 0
        assert "Tok shell integration installed in" in result.output
        assert ".zshrc" in result.output
        assert "Reload your shell:" in result.output
        assert "Wrapper mode enabled" in result.output

    def test_install_uninstall_removes_existing_shell_integration(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("tok.utils.shell_integration.uninstall", lambda: [tmp_path / ".zshrc"])

        result = runner.invoke(app, ["install", "--uninstall"])

        assert result.exit_code == 0
        assert "Tok shell integration removed from:" in result.output
        assert ".zshrc" in result.output

    def test_install_rejects_wrap_claude_and_uninstall_combination(self) -> None:
        result = runner.invoke(app, ["install", "--wrap-claude", "--uninstall"])
        assert result.exit_code == 2
        assert "Cannot combine `--wrap-claude` with `--uninstall`." in result.output

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
                    "api_base": "custom-api-host.internal",
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
        assert "Saved 140 tokens" in result.output
        assert "48.3%" in result.output
        assert "% cost savings" in result.output
        assert "Session degraded to baseline" in result.output
        assert "Session degraded to baseline" in result.output
        assert "Tok active" in result.output
        assert "Degraded to baseline" in result.output
        assert "Mode" in result.output
        assert "Request policy" in result.output
        assert "API base" in result.output
        assert "custom-api-host.internal" in result.output
        assert "natural_first" in result.output
        assert "Fallbacks" in result.output
        assert "Session quality" in result.output
        assert "Degradation reason" in result.output
        assert "Session signals" in result.output
        assert "tok doctor" in result.output
        assert "tok bridge logs 100" in result.output

    def test_bridge_status_invalid_port_config_falls_back_without_traceback(self, monkeypatch) -> None:
        monkeypatch.setenv("TOK_BRIDGE_PORT", "bad")
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: None)

        result = runner.invoke(app, ["bridge", "status"])

        assert result.exit_code == 1
        assert "Invalid integer config TOK_BRIDGE_PORT='bad'; using fallback 9090." in result.output
        assert "Bridge not running" in result.output
        assert "Traceback" not in result.output

    def test_bridge_status_reports_malformed_health_payload_honestly(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 321)

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "status": "ok",
                    "session_tokens_saved": "not-an-int",
                    "session_savings_pct": "not-a-float",
                }

        monkeypatch.setattr("tok.cli._bridge.get_bridge_health_response", lambda *args, **kwargs: FakeResponse())

        result = runner.invoke(app, ["bridge", "status"])

        assert result.exit_code == 1
        assert "health payload is malformed" in result.output
        assert "not responding" not in result.output
        assert "Traceback" not in result.output

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

    def test_bridge_status_recovers_when_localhost_probe_fails(self, monkeypatch) -> None:
        import httpx

        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setenv("TOK_BRIDGE_HOST", "localhost")

        attempted_urls: list[str] = []

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
                    "fallback_count": 0,
                    "session_tokens_saved": 20,
                    "session_savings_pct": 10.0,
                    "session_quality": "clean",
                    "last_degradation_reason": "",
                }

        def fake_get(url: str, timeout: float):
            attempted_urls.append(url)
            if "localhost" in url:
                return FakeResponse()
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr("httpx.get", fake_get)

        result = runner.invoke(app, ["bridge", "status"])

        assert result.exit_code == 0
        assert "Bridge running on :9090 (PID 321)" in result.output
        assert any("127.0.0.1" in url for url in attempted_urls)
        assert any("localhost" in url for url in attempted_urls)

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
        assert "Saved 250 tokens" in result.output
        assert "$0.2500 saved" in result.output
        assert "Saved 250 tokens" in result.output
        assert "25.0% token" in result.output
        assert "savings • 25.0% cost savings" in result.output
        assert "25.0% cost savings" in result.output
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
        from tok.cli import _cli_support

        def no_bridge(port) -> None:
            return None

        def fake_memory_root():
            return tmp_path / ".tok"

        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", no_bridge)
        for mod in (_cli_support,):
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
        assert "ANTHROPIC_BASE_URL=http://localhost:9090 claude" in result.output

    def test_bridge_start_enables_session_reset(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _cli_support

        def no_bridge(port) -> None:
            return None

        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", no_bridge)
        for mod in (_cli_support,):
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

    def test_bridge_reset_session_honors_configured_bridge_port(self, monkeypatch) -> None:
        captured = {}

        class FakeResponse:
            status_code = 200

        def fake_post(url, timeout):
            captured["url"] = url
            captured["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setenv("TOK_BRIDGE_PORT", "7777")
        monkeypatch.setattr("httpx.post", fake_post)

        result = runner.invoke(app, ["bridge", "reset-session"])

        assert result.exit_code == 0
        assert captured == {
            "url": "http://localhost:7777/reset-session",
            "timeout": 5,
        }

    def test_bridge_start_foreground_forwards_api_base(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _cli_support

        def no_bridge(port) -> None:
            return None

        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", no_bridge)
        for mod in (_cli_support,):
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

    def test_bridge_start_foreground_preserves_env_api_base_without_explicit_override(
        self, monkeypatch, tmp_path
    ) -> None:
        from tok.cli import _cli_support

        def no_bridge(port) -> None:
            return None

        monkeypatch.setenv("TOK_API_BASE", "https://env.example.test/api")
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", no_bridge)
        for mod in (_cli_support,):
            monkeypatch.setattr(mod, "LOG_FILE", tmp_path / "bridge.log")
            monkeypatch.setattr(mod, "PID_FILE", tmp_path / "bridge.pid")

        captured = {}

        def _fake_run_bridge(**kwargs) -> None:
            captured.update(kwargs)

        monkeypatch.setattr("tok.gateway.run_bridge", _fake_run_bridge)

        result = runner.invoke(app, ["bridge", "start", "--foreground"])

        assert result.exit_code == 0
        assert captured["_api_base"] is None

    def test_bridge_start_subprocess_exports_custom_api_base(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _cli_support

        def no_bridge(port) -> None:
            return None

        def fake_memory_root():
            return tmp_path / ".tok"

        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", no_bridge)
        for mod in (_cli_support,):
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

    def test_bridge_start_subprocess_preserves_env_api_base_without_explicit_override(
        self, monkeypatch, tmp_path
    ) -> None:
        from tok.cli import _cli_support

        def no_bridge(port) -> None:
            return None

        def fake_memory_root():
            return tmp_path / ".tok"

        monkeypatch.setenv("TOK_API_BASE", "https://env.example.test/api")
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", no_bridge)
        for mod in (_cli_support,):
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

        result = runner.invoke(app, ["bridge", "start"])

        assert result.exit_code == 0
        assert captured["env"]["TOK_API_BASE"] == "https://env.example.test/api"
        assert captured["env"]["TOK_RESET_SESSION"] == "1"

    def test_savings_no_data(self) -> None:
        result = runner.invoke(app, ["stats"])
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
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda _port: None)

        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 0
        assert "Current Session" in result.output
        assert "Saved 100 tokens" in result.output
        assert "$0.0005 saved" in result.output
        assert "40.0% token savings" in result.output
        assert "Session degraded to baseline" in result.output
        assert "Strong savings" in result.output
        assert "Fallbacks" in result.output
        assert "Degraded to baseline" in result.output
        assert "Cost (with Tok / est. no Tok)" in result.output
        assert "est. no caching" not in result.output
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
        assert "Saved 1,000 tokens" in result.output
        assert "$0.0100 saved" in result.output
        assert "33.3% token savings" in result.output
        assert "1,000 tokens" in result.output
        assert "Fallbacks" in result.output
        assert "Baseline-only requests" in result.output

    def test_stats_share_prints_pasteable_savings_summary(self, tmp_path, monkeypatch) -> None:
        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n"
            "  sessions: 2\n"
            "  total_turns: 10\n"
            "  total_tokens: 2000\n"
            "  total_cost_usd: 0.020000\n"
            "  estimated_baseline_cost_usd: 0.070000\n"
            "  tokens_saved: 1000\n"
            "  cost_saved_usd: 0.050000\n"
            "  savings_pct: 33.3\n"
            "  tok_fallback_activated: 0\n"
            "  baseline_only_session: 0\n\n"
            "@per_session_log\n"
        )
        tracker = SavingsTracker(
            savings_file=str(tmp_path / "tok_savings.tok"),
            ledger_path=ledger,
        )
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=120,
            actual_output=30,
            cache_read=0,
            cache_write=0,
            input_saved=80,
            output_saved=20,
        )
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda _port: None)

        result = runner.invoke(app, ["stats", "--share"])

        assert result.exit_code == 0
        assert "Tok has saved an estimated $0.05 across 3 sessions." in result.output
        assert "1,100 tokens avoided" in result.output
        assert "Current session: estimated $" in result.output
        assert "Bridge not running; session quality:" in result.output
        assert "baseline-only: no." in result.output
        assert "Current Session" not in result.output
        assert "Cost (with Tok / est. no Tok)" not in result.output

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
        assert "Saved 700 tokens" in result.output
        assert "$0.0150 saved" in result.output
        assert "41.2% token savings" in result.output
        assert "Strong savings" in result.output
        assert "$0.0150" in result.output
        assert "2026-03-18T10:00:00Z" in result.output
        assert "Session quality" in result.output
        assert "Degradation reason" in result.output

    def test_stats_default_shows_last_completed_with_no_active_session(self, tmp_path, monkeypatch) -> None:
        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n  sessions: 2\n  total_turns: 10\n  total_tokens: 2000\n  total_cost_usd: 0.020000\n  estimated_baseline_cost_usd: 0.045000\n  tokens_saved: 1200\n  cost_saved_usd: 0.025000\n  savings_pct: 55.0\n\n@per_session_log\n  # format: date;session_id;turns;tokens;cost_usd;baseline_cost_usd;saved_usd;tokens_saved;invisible_pressure;non_tok_response\n  2026-03-17T10:00:00Z;aaa11111;5;1000;0.010000;0.020000;0.010000;500;3;1\n  2026-03-18T10:00:00Z;bbb22222;5;1000;0.010000;0.025000;0.015000;700;1;0"
            "\n"
        )
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))
        # Simulate bridge not running so the health-endpoint fallback does not fire
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda _port: None)

        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 0
        assert "Last Completed Session" in result.output
        assert "Saved 700 tokens" in result.output
        assert "$0.0150 saved" in result.output
        assert "41.2% token savings" in result.output
        assert "2026-03-18T10:00:00Z" in result.output

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
        assert "Saved 800 tokens" in result.output
        assert "$0.0120" in result.output
        assert "33.3% token savings" in result.output
        assert "Solid savings" in result.output
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
        assert "Saved 800 tokens" in result.output
        assert "$0.0120" in result.output
        assert "33.3% token savings" in result.output
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
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.setattr(_bridge, "get_running_bridge_pid", lambda port: 123)
        monkeypatch.setattr(_bridge, "PID_FILE", tmp_path / "bridge.pid")

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
        monkeypatch.setattr(_bridge, "_flush_bridge_ledger", lambda port: None)

        _bridge.bridge_stop()
        output = capsys.readouterr().out
        assert "Bridge stopped" in output
        assert "Last Session" in output
        assert "Saved 100 tokens" in output
        assert "Verdict" in output

    def test_bridge_stop_flushes_ledger_before_sigterm(self, monkeypatch, tmp_path, capsys) -> None:
        from tok.cli import _bridge

        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.setattr(_bridge, "get_running_bridge_pid", lambda port: 123)
        monkeypatch.setattr(_bridge, "PID_FILE", tmp_path / "bridge.pid")

        calls: list[str] = []

        def fake_flush(port: int) -> None:
            assert port == 9090
            calls.append("flush")

        def fake_kill(pid, sig) -> None:
            assert pid == 123
            if sig == signal.SIGTERM:
                calls.append("sigterm")
                return
            if sig == 0:
                raise ProcessLookupError

        monkeypatch.setattr(_bridge, "_flush_bridge_ledger", fake_flush)
        monkeypatch.setattr(_bridge.os, "kill", fake_kill)

        _bridge.bridge_stop()

        assert calls[:2] == ["flush", "sigterm"]
        assert "Bridge stopped" in capsys.readouterr().out

    def test_bridge_stop_refuses_in_self_bridged_context_without_force(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _bridge

        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:9090")
        monkeypatch.setenv("TOK_SELF_BRIDGED_SESSION", "1")
        monkeypatch.setattr(_bridge, "get_running_bridge_pid", lambda port: 123)
        monkeypatch.setattr(_bridge, "PID_FILE", tmp_path / "bridge.pid")
        monkeypatch.setattr(
            _bridge,
            "get_bridge_health_response",
            lambda *args, **kwargs: SimpleNamespace(status_code=200),
        )
        monkeypatch.setattr(
            _bridge.os,
            "kill",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("os.kill should not be called")),
        )

        result = runner.invoke(app, ["bridge", "stop"])

        assert result.exit_code == 2
        assert "Refusing to stop bridge from an active bridged Claude session." in result.output
        assert "tok bridge stop --force" in result.output

    def test_bridge_stop_force_allows_stop_in_self_bridged_context(self, monkeypatch, tmp_path, capsys) -> None:
        from tok.cli import _bridge

        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:9090")
        monkeypatch.setenv("TOK_SELF_BRIDGED_SESSION", "1")
        monkeypatch.setattr(_bridge, "get_running_bridge_pid", lambda port: 123)
        monkeypatch.setattr(_bridge, "PID_FILE", tmp_path / "bridge.pid")

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
        monkeypatch.setattr(_bridge, "_flush_bridge_ledger", lambda port: None)

        _bridge.bridge_stop(force=True)
        output = capsys.readouterr().out

        assert "Bridge stopped" in output

    def test_bridge_stop_does_not_refuse_without_self_bridged_marker(self, monkeypatch, tmp_path, capsys) -> None:
        from tok.cli import _bridge

        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:9090")
        monkeypatch.delenv("TOK_SELF_BRIDGED_SESSION", raising=False)
        monkeypatch.setattr(_bridge, "get_running_bridge_pid", lambda port: 123)
        monkeypatch.setattr(_bridge, "PID_FILE", tmp_path / "bridge.pid")

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
        monkeypatch.setattr(_bridge, "_flush_bridge_ledger", lambda port: None)

        _bridge.bridge_stop()
        output = capsys.readouterr().out

        assert "Bridge stopped" in output

    def test_bridge_stop_pid_file_permission_error_does_not_traceback(self, monkeypatch, tmp_path) -> None:
        from tok.cli import _bridge

        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.setattr(_bridge, "get_running_bridge_pid", lambda port: 123)

        class UnlinkDeniedPath:
            def unlink(self, *, missing_ok: bool = False) -> None:
                raise PermissionError("permission denied")

            def __str__(self) -> str:
                return str(tmp_path / "bridge.pid")

            def _print(self, *args, **kwargs) -> None:
                pass

        calls = {"checked": False}

        def fake_kill(pid, sig) -> None:
            assert pid == 123
            if sig == signal.SIGTERM:
                return
            if sig == 0 and not calls["checked"]:
                calls["checked"] = True
                raise ProcessLookupError
            return

        monkeypatch.setattr(_bridge, "PID_FILE", UnlinkDeniedPath())
        monkeypatch.setattr(_bridge.os, "kill", fake_kill)
        monkeypatch.setattr(_bridge, "_flush_bridge_ledger", lambda port: None)

        result = runner.invoke(app, ["bridge", "stop"])

        assert result.exit_code == 0
        assert "Could not remove bridge PID file" in result.output
        assert "Traceback" not in result.output

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

        result = runner.invoke(app, ["stats", "--breakdown"])
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

        result = runner.invoke(app, ["stats", "--total", "--trends"])
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
                    "api_base": "custom-api-host.internal",
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
        assert "Saved 120 tokens" in result.output
        assert "41.4%" in result.output
        assert "Session degraded to baseline" in result.output
        assert "Fallbacks" in result.output
        assert "Tok verdict:" in result.output
        assert "baseline" in result.output
        assert "API base" in result.output
        assert "custom-api-host.internal" in result.output
        assert "Recommendation:" in result.output
        assert "investigate degradation before trusting this session" in result.output

    def test_doctor_invalid_port_config_falls_back_without_traceback(self, monkeypatch) -> None:
        monkeypatch.setenv("TOK_BRIDGE_PORT", "bad")
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/claude")
        monkeypatch.setattr(
            "tok.cli._release.memory_root",
            lambda: Path("/tmp/nonexistent_tok"),
        )

        result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 1
        assert "Invalid integer config TOK_BRIDGE_PORT='bad'; using fallback 9090." in result.output
        assert "Traceback" not in result.output

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

    def test_doctor_report_redacts_secret_env_values(self, monkeypatch, tmp_path) -> None:
        memory_dir = tmp_path / ".tok"
        memory_dir.mkdir()
        (memory_dir / "bridge_memory.tok").write_text("state:present\n")

        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setattr("tok.cli._release.memory_root", lambda: memory_dir)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-123")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret-456")
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path / "project"))

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
                    "session_tokens_saved": 0,
                    "session_savings_pct": 0.0,
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
                    "smoothness_score": 100,
                    "labour_index": 0,
                    "current_mode": "FULL_TOK",
                    "stream_instability_events": 0,
                    "thinking_mutation_events": 0,
                    "task_score": 100,
                    "repeated_active_file_reads": 0,
                    "evidence_exact_observed_count": 4,
                    "evidence_non_exact_reference_count": 2,
                    "evidence_exact_reacquisition_required_count": 1,
                    "evidence_exact_reacquisition_satisfied_count": 1,
                    "evidence_compression_blocked_for_safety_count": 1,
                }

        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *args, **kwargs: FakeResponse())

        result = runner.invoke(app, ["doctor", "--report"])

        assert result.exit_code == 0
        assert "Tok Doctor report (safe to share)" in result.output
        assert "env_OPENAI_API_KEY=set" in result.output
        assert "env_OPENROUTER_API_KEY=set" in result.output
        assert "env_TOK_PROJECT_DIR=set" in result.output
        assert "evidence_exact_observed=4" in result.output
        assert "evidence_non_exact_reference=2" in result.output
        assert "evidence_exact_reacquisition_required=1" in result.output
        assert "evidence_exact_reacquisition_satisfied=1" in result.output
        assert "evidence_compression_blocked_for_safety=1" in result.output
        assert "sk-secret-123" not in result.output
        assert "or-secret-456" not in result.output

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
                self.total_token_delta = -35
                self.task_success_equal_or_better = True
                self.cost_delta_usd = None
                self.fairness_diagnostics: dict[str, Any] = {}
                self.token_savings_without_cost_savings = False

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
            lambda baseline, _comparisons: "# compare\n",
        )
        monkeypatch.setattr(
            "tok.testing.live_benchmark.select_preferred_mode",
            lambda _baseline, _comparisons: "tok-universal",
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


class TestStatsTotalNoDoubleCounting:
    """Regression: stats --total must not double-count inflight session tokens
    when lifetime_summary() already overlays inflight data."""

    def test_stats_total_no_double_count_with_bridge_running(self, tmp_path, monkeypatch) -> None:
        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n  sessions: 1\n  total_turns: 5\n  total_tokens: 1000\n"
            "  total_cost_usd: 0.010000\n  estimated_baseline_cost_usd: 0.020000\n"
            "  tokens_saved: 500\n  cost_saved_usd: 0.005000\n  savings_pct: 33.3\n"
            "  tok_fallback_activated: 0\n  baseline_only_session: 0\n\n"
            "@per_session_log\n"
            "  2026-05-01T10:00:00Z;aaa11111;5;1000;0.010000;0.020000;0.005000;500;0;0\n"
        )
        savings_file = tmp_path / "tok_savings.tok"
        tracker = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=ledger,
        )
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=800,
            actual_output=200,
            cache_read=0,
            cache_write=0,
            input_saved=300,
            output_saved=0,
        )

        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(savings_file))

        inflight_summary = tracker.lifetime_summary()
        assert inflight_summary is not None
        expected_sessions = inflight_summary["sessions"]
        expected_tokens = inflight_summary["actual_tokens"]

        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda _port: 9999)

        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "actual_tokens": 1000,
            "baseline_tokens": 1300,
            "session_tokens_saved": 300,
            "session_savings_pct": 23.1,
            "session_cost_savings_pct": 0.0,
            "actual_cost_usd": 0.01,
            "baseline_cost_usd": 0.015,
            "cost_saved_usd": 0.005,
            "baseline_only": False,
            "fallback_count": 0,
            "calls": 1,
            "session_count": 1,
            "session_quality": "clean",
            "last_degradation_reason": "",
            "request_policy": "",
            "baseline_prompt_tokens": 0,
            "prepared_prompt_tokens": 0,
            "saved_prompt_tokens": 0,
        }
        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *a, **kw: mock_resp)

        result = runner.invoke(app, ["stats", "--total"])
        assert result.exit_code == 0, result.output

        sessions_match = re.search(r"Sessions\s+(\d+)", result.output)
        assert sessions_match is not None, f"Could not find Sessions in output:\n{result.output}"
        displayed_sessions = int(sessions_match.group(1))
        assert displayed_sessions == expected_sessions, (
            f"Expected {expected_sessions} sessions (lifetime_summary already includes inflight), "
            f"got {displayed_sessions} — double-counting suspected"
        )

        tokens_match = re.search(r"Tokens \(with Tok / est\. no Tok\)\s+([\d,]+)\s*/\s*([\d,]+)", result.output)
        assert tokens_match is not None, f"Could not find token counts in output:\n{result.output}"
        displayed_actual = int(tokens_match.group(1).replace(",", ""))
        assert displayed_actual == expected_tokens, (
            f"Expected {expected_tokens} actual tokens, got {displayed_actual} — double-counting suspected"
        )

    def test_stats_total_uses_health_session_count_for_live_overlay(self, tmp_path, monkeypatch) -> None:
        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n  sessions: 10\n  total_turns: 20\n  total_tokens: 1000\n"
            "  total_cost_usd: 0.010000\n  estimated_baseline_cost_usd: 0.020000\n"
            "  tokens_saved: 500\n  cost_saved_usd: 0.005000\n  savings_pct: 33.3\n"
            "  tok_fallback_activated: 0\n  baseline_only_session: 0\n\n"
            "@per_session_log\n"
            "  2026-05-01T10:00:00Z;aaa11111;20;1000;0.010000;0.020000;0.005000;500;0;0\n"
        )
        savings_file = tmp_path / "tok_savings.tok"
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(savings_file))
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda _port: 9999)

        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "actual_tokens": 2000,
            "baseline_tokens": 3000,
            "session_tokens_saved": 1000,
            "session_savings_pct": 33.3,
            "session_cost_savings_pct": 50.0,
            "actual_cost_usd": 0.020,
            "baseline_cost_usd": 0.040,
            "cost_saved_usd": 0.020,
            "baseline_only": False,
            "fallback_count": 0,
            "calls": 5,
            "session_count": 3,
            "session_quality": "clean",
            "last_degradation_reason": "",
            "request_policy": "",
        }
        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *a, **kw: mock_resp)

        result = runner.invoke(app, ["stats", "--total"])

        assert result.exit_code == 0, result.output
        sessions_match = re.search(r"Sessions\s+(\d+)", result.output)
        assert sessions_match is not None, result.output
        assert int(sessions_match.group(1)) == 4

    def test_stats_total_uses_bridge_health_instead_of_stale_inflight_file(self, tmp_path, monkeypatch) -> None:
        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n  sessions: 1\n  total_turns: 5\n  total_tokens: 1000\n"
            "  total_cost_usd: 0.010000\n  estimated_baseline_cost_usd: 0.020000\n"
            "  tokens_saved: 500\n  cost_saved_usd: 0.005000\n  savings_pct: 33.3\n"
            "  tok_fallback_activated: 0\n  baseline_only_session: 0\n\n"
            "@per_session_log\n"
            "  2026-05-01T10:00:00Z;aaa11111;5;1000;0.010000;0.020000;0.005000;500;0;0\n"
        )
        savings_file = tmp_path / "tok_savings.tok"
        stale = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=ledger,
        )
        stale.record_call(
            model="claude-sonnet-4",
            actual_input=5000,
            actual_output=0,
            cache_read=0,
            cache_write=0,
            input_saved=2500,
            output_saved=0,
        )

        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(savings_file))
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda _port: 9999)

        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "actual_tokens": 2000,
            "baseline_tokens": 3000,
            "session_tokens_saved": 1000,
            "session_savings_pct": 33.3,
            "session_cost_savings_pct": 50.0,
            "actual_cost_usd": 0.020,
            "baseline_cost_usd": 0.040,
            "cost_saved_usd": 0.020,
            "baseline_only": False,
            "fallback_count": 0,
            "calls": 2,
            "session_count": 1,
            "session_quality": "clean",
            "last_degradation_reason": "",
            "request_policy": "",
        }
        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *a, **kw: mock_resp)

        result = runner.invoke(app, ["stats", "--total"])

        assert result.exit_code == 0, result.output
        tokens_match = re.search(r"Tokens \(with Tok / est\. no Tok\)\s+([\d,]+)\s*/\s*([\d,]+)", result.output)
        assert tokens_match is not None, result.output
        displayed_actual = int(tokens_match.group(1).replace(",", ""))
        displayed_baseline = int(tokens_match.group(2).replace(",", ""))
        assert displayed_actual == 3000
        assert displayed_baseline == 4500
        savings_file = tmp_path / "tok_savings.tok"
        tracker = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=ledger,
        )
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=800,
            actual_output=200,
            cache_read=0,
            cache_write=0,
            input_saved=300,
            output_saved=0,
        )

        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(savings_file))

        inflight_summary = tracker.lifetime_summary()
        assert inflight_summary is not None
        inflight_tokens = inflight_summary["actual_tokens"]

        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda _port: 9999)

        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "actual_tokens": 1000,
            "baseline_tokens": 1300,
            "session_tokens_saved": 300,
            "session_savings_pct": 23.1,
            "session_cost_savings_pct": 0.0,
            "actual_cost_usd": 0.01,
            "baseline_cost_usd": 0.015,
            "cost_saved_usd": 0.005,
            "baseline_only": False,
            "fallback_count": 0,
            "calls": 1,
            "session_quality": "clean",
            "last_degradation_reason": "",
            "request_policy": "",
            "baseline_prompt_tokens": 0,
            "prepared_prompt_tokens": 0,
            "saved_prompt_tokens": 0,
        }
        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *a, **kw: mock_resp)

        result = runner.invoke(app, ["stats", "--total"])
        assert result.exit_code == 0, result.output

        displayed_tokens_match = re.findall(r"([\d,]+)\s+tokens", result.output.replace("\n", " "))
        for token_str in displayed_tokens_match:
            val = int(token_str.replace(",", ""))
            assert val < inflight_tokens * 2, (
                f"Displayed token value {val} suggests double-counting (inflight was {inflight_tokens})"
            )
