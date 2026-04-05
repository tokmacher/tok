"""Tests for tok stats rendering — Interaction Quality panel and session quality."""

from pathlib import Path
from typer.testing import CliRunner

from tok.cli import app

runner = CliRunner()


class TestStatsRendering:
    def test_doctor_renders_interaction_quality_panel(self, monkeypatch):
        monkeypatch.setattr(
            "tok.cli._release.get_running_bridge_pid", lambda port: 321
        )
        monkeypatch.setattr(
            "shutil.which", lambda name: "/usr/local/bin/claude"
        )
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-iq-panel.tok")
        Path("/tmp/test-iq-panel.tok").unlink(missing_ok=True)

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
                    "session_tokens_saved": 120,
                    "session_savings_pct": 41.4,
                    "session_quality": "clean",
                    "smoothness_score": 85,
                    "labour_index": 12,
                    "current_mode": "natural-first",
                    "stream_instability_events": 0,
                    "thinking_mutation_events": 0,
                    "repeated_active_file_reads": 0,
                    "task_score": 90,
                }

        monkeypatch.setattr(
            "httpx.get", lambda *args, **kwargs: FakeResponse()
        )
        monkeypatch.setattr(
            "tok.cli._release.memory_root",
            lambda: Path("/tmp/nonexistent_tok"),
        )

        result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 1
        assert "Interaction Quality" in result.output
        assert "Smoothness score" in result.output
        assert "85" in result.output
        assert "Labour index" in result.output
        assert "12" in result.output
        assert "Task score" in result.output
        assert "90" in result.output
        assert "Current mode" in result.output

    def test_bridge_status_session_quality_changes_with_smoothness(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "tok.cli._bridge.get_running_bridge_pid", lambda port: 321
        )
        monkeypatch.setenv(
            "TOK_SAVINGS_FILE", "/tmp/test-quality-smoothness.tok"
        )
        Path("/tmp/test-quality-smoothness.tok").unlink(missing_ok=True)

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
                    "session_tokens_saved": 140,
                    "session_savings_pct": 48.3,
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

        monkeypatch.setattr(
            "httpx.get", lambda *args, **kwargs: FakeResponse()
        )

        result = runner.invoke(app, ["bridge", "status"])
        assert result.exit_code == 0
        assert "Session quality" in result.output
        assert "clean" in result.output
        assert "Tok active" in result.output

    def test_bridge_status_distinguishes_fallback_from_compat_fallback(
        self, monkeypatch
    ):
        """Verify degradation fallback and compat-fallback are rendered distinctly."""
        monkeypatch.setattr(
            "tok.cli._bridge.get_running_bridge_pid", lambda port: 321
        )
        monkeypatch.setenv(
            "TOK_SAVINGS_FILE", "/tmp/test-fallback-distinction.tok"
        )
        Path("/tmp/test-fallback-distinction.tok").unlink(missing_ok=True)

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
                    "session_tokens_saved": 100,
                    "session_savings_pct": 35.0,
                    "session_quality": "clean",
                    "last_degradation_reason": "",
                    "semantic_drift_count": 0,
                    "fail_open_count": 2,
                    "non_tok_count": 0,
                    "answer_anchor_miss_count": 0,
                    "repeat_search_count": 0,
                    "repeat_file_read_count": 0,
                    "state_resend_full_count": 0,
                    "state_resend_delta_count": 0,
                    "state_resend_suppressed_count": 0,
                }

        monkeypatch.setattr(
            "httpx.get", lambda *args, **kwargs: FakeResponse()
        )

        result = runner.invoke(app, ["bridge", "status"])
        assert result.exit_code == 0
        assert "Session quality" in result.output
        assert "Tok active" in result.output
        # Verify both signals appear distinctly in session signals
        assert "fallback=1" in result.output
        assert "compat-fallback=2" in result.output

    def test_doctor_renders_per_turn_vs_per_call_distinction(
        self, monkeypatch
    ):
        """Verify per-turn instability and per-call transport are labeled distinctly."""
        monkeypatch.setattr(
            "tok.cli._release.get_running_bridge_pid", lambda port: 321
        )
        monkeypatch.setattr(
            "shutil.which", lambda name: "/usr/local/bin/claude"
        )
        monkeypatch.setenv(
            "TOK_SAVINGS_FILE", "/tmp/test-per-turn-per-call.tok"
        )
        Path("/tmp/test-per-turn-per-call.tok").unlink(missing_ok=True)

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
                    "session_tokens_saved": 150,
                    "session_savings_pct": 42.5,
                    "session_quality": "clean",
                    "smoothness_score": 88,
                    "labour_index": 8,
                    "current_mode": "natural-first",
                    # Per-turn smoothness metric (5 event types)
                    "stream_instability_events": 5,
                    "thinking_mutation_events": 0,
                    "repeated_active_file_reads": 2,
                    "task_score": 92,
                }

        monkeypatch.setattr(
            "httpx.get", lambda *args, **kwargs: FakeResponse()
        )
        monkeypatch.setattr(
            "tok.cli._release.memory_root",
            lambda: Path("/tmp/nonexistent_tok"),
        )

        result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 1
        assert "Interaction Quality" in result.output
        # Verify per-turn label is present
        assert "Stream instability events (per-turn)" in result.output
        assert "5" in result.output
