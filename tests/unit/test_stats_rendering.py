"""Tests for tok stats rendering — Interaction Quality panel and session quality."""

from pathlib import Path

from typer.testing import CliRunner

from tok.cli import app
from tok.cli._cli_support import savings_diagnostic_note, session_signals_text, session_status_rows

runner = CliRunner()


class TestStatsRendering:
    def test_clean_session_omits_evidence_safety_noise(self) -> None:
        rows = session_status_rows(
            summary={
                "tokens_saved": 10,
                "actual_tokens": 100,
                "baseline_tokens": 110,
                "actual_cost_usd": 0.001,
                "baseline_cost_usd": 0.002,
                "cost_saved_usd": 0.001,
                "fallback_count": 0,
                "baseline_only": False,
                "session_quality": "clean",
                "last_degradation_reason": "",
            },
            tok_active=True,
            baseline_only=False,
            session_signals=session_signals_text({}),
        )

        assert ("Session signals", "clean") in rows
        assert not any(label == "Evidence safety" for label, _ in rows)
        assert not any(label == "Exact reacquisition" for label, _ in rows)

    def test_zero_savings_session_explains_short_session(self) -> None:
        rows = session_status_rows(
            summary={
                "tokens_saved": 0,
                "savings_pct": 0.0,
                "actual_tokens": 100,
                "baseline_tokens": 100,
                "actual_cost_usd": 0.001,
                "baseline_cost_usd": 0.001,
                "cost_saved_usd": 0.0,
                "fallback_count": 0,
                "calls": 1,
                "baseline_only": False,
                "session_quality": "clean",
                "last_degradation_reason": "",
            },
            tok_active=True,
            baseline_only=False,
            mode="tool-compatible",
            session_signals=session_signals_text({}),
        )

        assert (
            "Savings note",
            "Very short sessions often show no savings; recheck after sustained Claude Code work.",
        ) in rows

    def test_api_base_row_redacts_url_credentials(self) -> None:
        rows = session_status_rows(
            summary={
                "tokens_saved": 1,
                "actual_tokens": 100,
                "baseline_tokens": 101,
                "actual_cost_usd": 0.001,
                "baseline_cost_usd": 0.002,
                "cost_saved_usd": 0.001,
                "fallback_count": 0,
                "baseline_only": False,
                "session_quality": "clean",
                "last_degradation_reason": "",
            },
            tok_active=True,
            baseline_only=False,
            api_base="https://user:secret-token@api.example.test/v1?api_key=also-secret",
        )

        api_base_row = next(value for label, value in rows if label == "API base")
        assert "api.example.test" in api_base_row
        assert "secret-token" not in api_base_row
        assert "also-secret" not in api_base_row
        assert "<redacted>" in api_base_row

    def test_safe_block_zero_savings_explains_evidence_safety(self) -> None:
        note = savings_diagnostic_note(
            summary={
                "tokens_saved": 0,
                "savings_pct": 0.0,
                "calls": 6,
                "fallback_count": 0,
                "evidence_compression_blocked_for_safety_count": 2,
            },
            baseline_only=False,
            mode="tool-compatible",
        )

        assert note == "Tok blocked compression for evidence safety; exactness won over token savings."

    def test_non_exact_reacquisition_session_shows_evidence_safety_rows(self) -> None:
        summary = {
            "tokens_saved": 10,
            "actual_tokens": 100,
            "baseline_tokens": 110,
            "actual_cost_usd": 0.001,
            "baseline_cost_usd": 0.002,
            "cost_saved_usd": 0.001,
            "fallback_count": 0,
            "baseline_only": False,
            "session_quality": "clean",
            "last_degradation_reason": "",
            "evidence_exact_observed_count": 3,
            "evidence_non_exact_reference_count": 2,
            "evidence_non_exact_summary_count": 1,
            "evidence_non_exact_skeleton_count": 1,
            "evidence_exact_reacquisition_required_count": 2,
            "evidence_exact_reacquisition_satisfied_count": 2,
            "evidence_compression_blocked_for_safety_count": 1,
        }

        rows = session_status_rows(
            summary=summary,
            tok_active=True,
            baseline_only=False,
            session_signals=session_signals_text(summary),
        )

        assert ("Session signals", "exact=3, nonexact=2, reacq-safe=2/2, safe-block=1") in rows
        assert ("Evidence safety", "exact=3, non-exact=2, summaries=1, skeletons=1") in rows
        assert ("Exact reacquisition", "required=2, satisfied=2") in rows
        assert ("Compression safety blocks", "1") in rows

    def test_doctor_renders_interaction_quality_panel(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/claude")
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

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())
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

    def test_bridge_status_session_quality_changes_with_smoothness(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-quality-smoothness.tok")
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

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())

        result = runner.invoke(app, ["bridge", "status"])
        assert result.exit_code == 0
        assert "Session quality" in result.output
        assert "clean" in result.output
        assert "Tok active" in result.output

    def test_bridge_status_distinguishes_fallback_from_compat_fallback(self, monkeypatch) -> None:
        """Verify degradation fallback and compat-fallback are rendered distinctly."""
        monkeypatch.setattr("tok.cli._bridge.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-fallback-distinction.tok")
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

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())

        result = runner.invoke(app, ["bridge", "status"])
        assert result.exit_code == 0
        assert "Session quality" in result.output
        assert "Tok active" in result.output
        # Verify both signals appear distinctly in session signals
        assert "fallback=1" in result.output
        assert "compat-fallback=2" in result.output

    def test_doctor_renders_per_turn_vs_per_call_distinction(self, monkeypatch) -> None:
        """Verify per-turn instability and per-call transport are labeled distinctly."""
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 321)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/claude")
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-per-turn-per-call.tok")
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

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())
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
