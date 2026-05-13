"""Tests for `tok stats --detail` rendering."""

from __future__ import annotations

from typer.testing import CliRunner

from tok.cli import app

runner = CliRunner()


class _FakeResp:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_stats_detail_renders_sections(monkeypatch) -> None:
    from tok.cli import _release as rel

    monkeypatch.setattr(rel, "get_running_bridge_pid", lambda port: 4242)
    monkeypatch.setattr(
        rel,
        "get_bridge_health_response",
        lambda port, timeout, attempts, backoff_seconds: _FakeResp(
            {
                "calls": 3,
                "actual_tokens": 100,
                "baseline_tokens": 120,
                "session_tokens_saved": 20,
                "session_savings_pct": 16.7,
                "actual_cost_usd": 0.001,
                "baseline_cost_usd": 0.002,
                "cost_saved_usd": 0.001,
                "baseline_only": False,
                "fallback_count": 0,
                "session_quality": "clean",
                "last_degradation_reason": "",
                "bloat_attribution": {
                    "request_footprint": {
                        "prepared": {"total_tokens": 900},
                        "baseline": {"total_tokens": 1100},
                    },
                    "tool_result_retention": {"message_count": 4, "tokens": 500, "heavy_block_count": 1},
                    "state_resend": {"mode": "delta", "delta_count": 2, "tokens": 123},
                    "history_retention": {"dropped_tokens": 50, "skip_reason": "pressure"},
                },
                "speculative_macros_injected": 2,
                "macro_savings_attributed": 1,
                "tok_prompt_bloat_detected": False,
            }
        ),
    )

    result = runner.invoke(app, ["stats", "--detail"])
    assert result.exit_code == 0
    assert "Detail" in result.output
    assert "Evidence forms" in result.output
    assert "Macro activity" in result.output
