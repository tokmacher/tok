"""Tests for tok stats rendering — Interaction Quality panel and session quality."""

from pathlib import Path

from typer.testing import CliRunner

from tok.cli import app
from tok.cli._cli_support import (
    format_savings_line,
    reliability_line,
    savings_diagnostic_note,
    session_signals_text,
    session_status_rows,
    status_sentence,
)

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
        assert "Stream instability events (per-turn)" in result.output
        assert "5" in result.output


class TestDefaultStatsRenderer:
    def test_active_zero_call_session_is_not_reported_inactive(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 4242)

        class FakeHealthResp:
            status_code = 200

            def json(self):
                return {
                    "calls": 0,
                    "actual_tokens": 0,
                    "baseline_tokens": 0,
                    "session_tokens_saved": 0,
                    "session_savings_pct": 0.0,
                    "actual_cost_usd": 0.0,
                    "baseline_cost_usd": 0.0,
                    "cost_saved_usd": 0.0,
                    "baseline_only": False,
                    "fallback_count": 0,
                    "session_quality": "clean",
                    "api_base": "https://api.anthropic.com",
                    "smoothness_score": 100,
                    "current_mode": "FULL_TOK",
                }

        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *args, **kwargs: FakeHealthResp())
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))

        result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "Bridge Status: Active" in result.output
        assert "API: Anthropic" in result.output
        assert "No completed calls recorded" in result.output
        assert "Tok is not active" not in result.output
        current_section = result.output.split("LIFETIME", maxsplit=1)[0]
        assert "0.0% less" not in current_section
        assert "no calls yet" in result.output
        assert "no completed calls yet" not in result.output
        assert "lifetime:" in result.output
        assert "  Cost Saved           $0.00                          $0.00" in result.output
        assert "  Tokens Saved         0                              0" in result.output

    def test_reliability_uses_current_session_fallbacks(self, tmp_path, monkeypatch) -> None:
        ledger = tmp_path / "global_savings.tok"
        ledger.write_text(
            "@lifetime_savings\n  sessions: 2\n  total_turns: 10\n  total_tokens: 2000\n"
            "  total_cost_usd: 0.020000\n  estimated_baseline_cost_usd: 0.030000\n"
            "  tokens_saved: 1000\n  cost_saved_usd: 0.010000\n  savings_pct: 33.3\n"
            "  tok_fallback_activated: 66\n  baseline_only_session: 0\n\n@per_session_log\n"
        )
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 4242)

        class FakeHealthResp:
            status_code = 200

            def json(self):
                return {
                    "calls": 0,
                    "fallback_count": 0,
                    "baseline_only": False,
                    "session_quality": "clean",
                    "api_base": "https://api.anthropic.com",
                    "smoothness_score": 100,
                    "current_mode": "FULL_TOK",
                }

        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *args, **kwargs: FakeHealthResp())
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))

        result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "Reliability:   100/100 smoothness · 0 fallbacks · 0 calls handled" in result.output
        assert "66 fallbacks" not in result.output

    def test_active_session_with_calls_keeps_meaningful_reductions(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 4242)

        class FakeHealthResp:
            status_code = 200

            def json(self):
                return {
                    "calls": 3,
                    "actual_tokens": 1000,
                    "baseline_tokens": 1500,
                    "session_tokens_saved": 500,
                    "session_net_tokens_saved": 450,
                    "reacquisition_cost_tokens": 50,
                    "session_savings_pct": 33.3,
                    "actual_cost_usd": 0.005,
                    "baseline_cost_usd": 0.010,
                    "cost_saved_usd": 0.005,
                    "baseline_only": False,
                    "fallback_count": 0,
                    "session_quality": "clean",
                    "api_base": "https://api.anthropic.com",
                    "smoothness_score": 100,
                    "current_mode": "FULL_TOK",
                }

        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *args, **kwargs: FakeHealthResp())
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))

        result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "Cost Reduction" in result.output
        assert "% less" in result.output
        assert "with Tok vs" in result.output
        assert "  Cost Reduction       33.3% less                     50.0% less" in result.output
        assert "  Token Reduction      33.3% less                     33.3% less" in result.output
        assert "Net Tokens Saved" in result.output
        assert "after reacq 50" in result.output
        assert "                       lifetime:" in result.output
        assert "\nbase" not in result.output
        assert "Tok is active and handling this session normally." in result.output

    def test_verbose_stats_exposes_raw_api_base(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 4242)

        class FakeHealthResp:
            status_code = 200

            def json(self):
                return {
                    "calls": 0,
                    "fallback_count": 0,
                    "baseline_only": False,
                    "session_quality": "clean",
                    "api_base": "https://api.anthropic.com",
                    "smoothness_score": 100,
                    "current_mode": "FULL_TOK",
                }

        monkeypatch.setattr("tok.cli._release.get_bridge_health_response", lambda *args, **kwargs: FakeHealthResp())
        monkeypatch.setenv("TOK_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("TOK_SAVINGS_FILE", str(tmp_path / "tok_savings.tok"))

        default_result = runner.invoke(app, ["stats"])
        verbose_result = runner.invoke(app, ["stats", "--verbose"])

        assert default_result.exit_code == 0
        assert verbose_result.exit_code == 0
        assert "Session:       clean (API: Anthropic)" in default_result.output
        assert "https://api.anthropic.com" not in default_result.output
        assert "API base: https://api.anthropic.com" in verbose_result.output

    def test_default_stats_contains_required_labels(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-default-stats.tok")
        Path("/tmp/test-default-stats.tok").unlink(missing_ok=True)

        from tok.stats import SavingsTracker

        tracker = SavingsTracker(savings_file="/tmp/test-default-stats.tok")
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=500,
            cache_read=0,
            cache_write=0,
            input_saved=300,
            output_saved=200,
        )

        result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "Bridge Status:" in result.output
        assert "Cost Reduction" in result.output
        assert "Token Reduction" in result.output
        assert "% less" in result.output
        assert "with Tok vs" in result.output
        assert "Cost Saved" in result.output
        assert "Tokens Saved" in result.output
        assert "Reliability:" in result.output
        assert "Status:" in result.output

    def test_default_stats_omits_diagnostics(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-no-diag.tok")
        Path("/tmp/test-no-diag.tok").unlink(missing_ok=True)

        from tok.stats import SavingsTracker

        tracker = SavingsTracker(savings_file="/tmp/test-no-diag.tok")
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=500,
            cache_read=0,
            cache_write=0,
            input_saved=300,
            output_saved=200,
        )

        result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "Degradation reason" not in result.output
        assert "Repeated active-file reads" not in result.output
        assert "Thinking mutation events" not in result.output
        assert "Labour index" not in result.output
        assert "Evidence safety" not in result.output

    def test_verbose_stats_includes_degradation_reason(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-verbose-diag.tok")
        Path("/tmp/test-verbose-diag.tok").unlink(missing_ok=True)

        from tok.stats import SavingsTracker

        tracker = SavingsTracker(savings_file="/tmp/test-verbose-diag.tok")
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=500,
            cache_read=0,
            cache_write=0,
            input_saved=300,
            output_saved=200,
        )

        result = runner.invoke(app, ["stats", "--verbose"])

        assert result.exit_code == 0

    def test_debug_stats_includes_interaction_quality(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: 4242)

        class FakeHealthResp:
            status_code = 200

            def json(self):
                return {
                    "calls": 5,
                    "actual_tokens": 1000,
                    "baseline_tokens": 1500,
                    "session_tokens_saved": 500,
                    "session_savings_pct": 33.3,
                    "actual_cost_usd": 0.005,
                    "baseline_cost_usd": 0.010,
                    "cost_saved_usd": 0.005,
                    "baseline_only": False,
                    "fallback_count": 0,
                    "session_quality": "clean",
                    "last_degradation_reason": "",
                    "smoothness_score": 85,
                    "labour_index": 10,
                    "current_mode": "tool-compatible",
                    "stream_instability_events": 0,
                    "thinking_mutation_events": 0,
                    "repeated_active_file_reads": 0,
                    "task_score": 90,
                }

        monkeypatch.setattr(
            "tok.cli._release.get_bridge_health_response",
            lambda *args, **kwargs: FakeHealthResp(),
        )
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-debug-iq.tok")
        Path("/tmp/test-debug-iq.tok").unlink(missing_ok=True)

        from tok.stats import SavingsTracker

        SavingsTracker(savings_file="/tmp/test-debug-iq.tok")

        result = runner.invoke(app, ["stats", "--debug"])

        assert result.exit_code == 0

    def test_status_sentence_rules(self) -> None:
        assert status_sentence(tok_active=True, baseline_only=False, fallback_count=0) == (
            "Tok is active and handling this session normally."
        )
        assert status_sentence(tok_active=True, baseline_only=False, fallback_count=2) == (
            "Tok is active, with fallback events recorded this session."
        )
        assert status_sentence(tok_active=True, baseline_only=False, fallback_count=0, calls=0) == (
            "Tok is active. No completed calls recorded for this session yet."
        )
        assert status_sentence(tok_active=True, baseline_only=True, fallback_count=0) == (
            "Tok has degraded to baseline for this session."
        )
        assert status_sentence(tok_active=False, baseline_only=False, fallback_count=0) == (
            "Tok is not active for this session."
        )

    def test_reliability_line_zero_fallbacks_green(self) -> None:
        line = reliability_line(smoothness_score=95, fallback_count=0, calls=10)
        assert "0 fallbacks" in line
        assert "[green]" in line

    def test_reliability_line_with_fallbacks_yellow(self) -> None:
        line = reliability_line(smoothness_score=80, fallback_count=3, calls=10)
        assert "3 fallbacks" in line
        assert "[yellow]" in line

    def test_format_savings_line_cost_colors(self) -> None:
        pct_good, _ = format_savings_line(pct=45.0, actual=2.51, baseline=4.30, is_cost=True)
        assert "green" in pct_good
        assert "45.0% less" in pct_good

        pct_mid, _ = format_savings_line(pct=25.0, actual=2.51, baseline=4.30, is_cost=True)
        assert "yellow" in pct_mid

        pct_low, _ = format_savings_line(pct=10.0, actual=2.51, baseline=4.30, is_cost=True)
        assert "red" in pct_low

    def test_default_stats_readable_without_colors(self, monkeypatch) -> None:
        monkeypatch.setattr("tok.cli._release.get_running_bridge_pid", lambda port: None)
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("TOK_SAVINGS_FILE", "/tmp/test-nocolor.tok")
        Path("/tmp/test-nocolor.tok").unlink(missing_ok=True)

        from tok.stats import SavingsTracker

        tracker = SavingsTracker(savings_file="/tmp/test-nocolor.tok")
        tracker.record_call(
            model="claude-sonnet-4",
            actual_input=1000,
            actual_output=500,
            cache_read=0,
            cache_write=0,
            input_saved=300,
            output_saved=200,
        )

        result = runner.invoke(app, ["stats"], catch_exceptions=False)
        assert result.exit_code == 0
        plain = result.output
        assert "Cost Reduction" in plain
        assert "Token Reduction" in plain
        assert "with Tok vs" in plain
        assert "Reliability:" in plain
        assert "Status:" in plain
