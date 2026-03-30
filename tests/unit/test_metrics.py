"""Tests for metric commands and telemetry functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from tok.metrics import (
    fallback_trends,
    format_trend_display,
    health_summary,
    memory_trends,
    pressure_trends,
    savings_trends,
)


class TestTrendFormatting:
    """Test trend display formatting."""

    def test_format_trend_display_no_data(self):
        """Test formatting with no trend data."""
        trend = {"sessions_considered": 0}
        result = format_trend_display(trend, "test_metric")
        assert "No test_metric data available" in result

    def test_format_trend_display_with_data(self):
        """Test formatting with trend data."""
        trend = {
            "sessions_considered": 10,
            "direction": "improving",
            "avg_test_metric": 15.5,
            "test_metric_velocity": 2.1,
        }
        result = format_trend_display(trend, "test_metric")
        assert "10 sessions" in result
        assert "improving" in result
        assert "15.5" in result
        assert "+2.10/session" in result

    def test_format_trend_display_negative_velocity(self):
        """Test formatting with negative velocity."""
        trend = {
            "sessions_considered": 5,
            "direction": "regressing",
            "avg_pressure": 8.2,
            "pressure_velocity": -1.5,
        }
        result = format_trend_display(trend, "pressure")
        assert "-1.50/session" in result
        assert "📉" in result


class TestPressureTrends:
    """Test pressure trend analysis."""

    @patch("tok.utils.metrics.SavingsTracker")
    @patch("tok.utils.metrics.calculate_invisible_pressure")
    def test_pressure_trends_basic(self, mock_pressure, mock_tracker):
        """Test basic pressure trend functionality."""
        # Setup mocks
        mock_tracker.return_value.behavior_signals.return_value = {
            "repeat_file_read": 5,
            "repeat_search": 2,
        }
        mock_pressure.return_value = 7

        mock_tracker.return_value.trend_summary.return_value = {
            "sessions_considered": 5,
            "avg_invisible_pressure": 6.5,
            "pressure_velocity": 1.2,
        }

        # Test without export
        with patch("rich.console.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            pressure_trends(window=5)
            mock_console.print.assert_called()

        # Verify calls
        mock_tracker.return_value.behavior_signals.assert_called_once()
        mock_pressure.assert_called_once()
        mock_tracker.return_value.trend_summary.assert_called_once_with(
            recent_sessions=5
        )

    @patch("tok.utils.metrics.SavingsTracker")
    @patch("tok.utils.metrics.calculate_invisible_pressure")
    def test_pressure_trends_export(self, mock_pressure, mock_tracker):
        """Test pressure trends with export."""
        # Setup mocks
        mock_tracker.return_value.behavior_signals.return_value = {
            "repeat_file_read": 3
        }
        mock_pressure.return_value = 2

        mock_tracker.return_value.trend_summary.return_value = {
            "sessions_considered": 3,
            "avg_invisible_pressure": 1.5,
            "pressure_velocity": 0.5,
        }

        # Test with export
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            export_path = f.name

        try:
            with patch("rich.console.Console") as mock_console_class:
                mock_console = MagicMock()
                mock_console_class.return_value = mock_console
                pressure_trends(window=3, export=export_path)

            # Verify export file
            assert Path(export_path).exists()
            with open(export_path) as f:
                data = json.load(f)

            assert data["metric"] == "pressure"
            assert data["window"] == 3
            assert data["current"]["pressure_raw"] == 2
            assert data["trend"]["avg_pressure"] == 1.5
            assert data["status"] == "watch"
            assert data["avg_pressure"] == 1.5
            assert data["sessions_considered"] == 3
        finally:
            Path(export_path).unlink(missing_ok=True)


class TestHealthSummary:
    """Test health summary functionality."""

    @patch("tok.utils.metrics.SavingsTracker")
    @patch("tok.utils.metrics.calculate_invisible_pressure")
    def test_health_summary_healthy(self, mock_pressure, mock_tracker):
        """Test health summary with healthy metrics."""
        # Setup mocks for healthy system
        mock_tracker.return_value.behavior_signals.return_value = {
            "cold_start_structured_memory": 5,
            "durable_promotions": 3,
            "hot_promotions": 2,
        }
        mock_pressure.return_value = 0

        mock_tracker.return_value.trend_summary.return_value = {
            "sessions_considered": 10,
            "direction": "improving",
            "avg_savings_pct": 25.0,
            "avg_invisible_pressure": 0.5,
            "pressure_velocity": -0.5,
            "savings_velocity": 1.5,
        }

        with patch("rich.console.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            health_summary(window=10)
            mock_console.print.assert_called()

        # Verify raw pressure was computed (for diagnostic display)
        mock_pressure.assert_called_once()

    @patch("tok.utils.metrics.SavingsTracker")
    @patch("tok.utils.metrics.calculate_invisible_pressure")
    def test_health_summary_uses_bounded_pressure(
        self, mock_pressure, mock_tracker
    ):
        """Health verdict must use bounded avg pressure, not raw cumulative totals."""
        # Simulate a long session with inflated cumulative pressure (1137)
        mock_tracker.return_value.behavior_signals.return_value = {
            "repeat_file_read": 800,
            "repeat_search": 337,
        }
        mock_pressure.return_value = 1137  # This is the raw cumulative count

        # But avg across recent sessions is healthy (0.5 per session)
        mock_tracker.return_value.trend_summary.return_value = {
            "sessions_considered": 5,
            "direction": "improving",
            "avg_savings_pct": 45.0,
            "avg_invisible_pressure": 0.5,
            "pressure_velocity": -0.1,
            "savings_velocity": 1.0,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            export_path = f.name

        try:
            with patch("rich.console.Console") as mock_console_class:
                mock_console = MagicMock()
                mock_console_class.return_value = mock_console
                health_summary(window=5, export=export_path)

            with open(export_path) as f:
                data = json.load(f)

            # Pressure verdict must come from avg (0.5 → watch), NOT raw (1137 → unhealthy).
            # avg_pressure=0.5 is in the low/watch range, which is correct and much better
            # than the "unhealthy" verdict that raw cumulative 1137 would produce.
            pressure_metric = next(
                m for m in data["metrics"] if m["name"] == "Pressure"
            )
            assert (
                pressure_metric["status"] == "watch"
            ), f"Expected watch based on avg_pressure=0.5, got {pressure_metric['status']}"
            assert (
                pressure_metric["status"] != "unhealthy"
            ), "Raw cumulative count 1137 must not drive the verdict to unhealthy"
            # Raw pressure is exported separately for diagnostics
            assert data["current_pressure_raw"] == 1137
            assert data["avg_pressure"] == 0.5
        finally:
            Path(export_path).unlink(missing_ok=True)

    @patch("tok.utils.metrics.SavingsTracker")
    @patch("tok.utils.metrics.calculate_invisible_pressure")
    def test_health_summary_no_history_is_watch(
        self, mock_pressure, mock_tracker
    ):
        """With no session history, pressure verdict should be watch (unknown)."""
        mock_tracker.return_value.behavior_signals.return_value = {}
        mock_pressure.return_value = 0

        mock_tracker.return_value.trend_summary.return_value = {
            "sessions_considered": 0,
            "direction": "none",
            "avg_savings_pct": 0.0,
            "avg_invisible_pressure": 0.0,
            "pressure_velocity": 0.0,
            "savings_velocity": 0.0,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            export_path = f.name

        try:
            with patch("rich.console.Console") as mock_console_class:
                mock_console = MagicMock()
                mock_console_class.return_value = mock_console
                health_summary(window=5, export=export_path)

            with open(export_path) as f:
                data = json.load(f)

            pressure_metric = next(
                m for m in data["metrics"] if m["name"] == "Pressure"
            )
            assert (
                pressure_metric["status"] == "watch"
            ), "No history should yield watch, not healthy or unhealthy"
        finally:
            Path(export_path).unlink(missing_ok=True)

    @patch("tok.utils.metrics.SavingsTracker")
    @patch("tok.utils.metrics.calculate_invisible_pressure")
    def test_health_summary_export(self, mock_pressure, mock_tracker):
        """Test health summary with export."""
        # Setup mocks
        mock_tracker.return_value.behavior_signals.return_value = {
            "repeat_file_read": 10
        }
        mock_pressure.return_value = 15

        mock_tracker.return_value.trend_summary.return_value = {
            "sessions_considered": 5,
            "direction": "regressing",
            "avg_savings_pct": 5.0,
            "avg_invisible_pressure": 12.0,
            "pressure_velocity": 3.0,
            "savings_velocity": -2.0,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            export_path = f.name

        try:
            with patch("rich.console.Console") as mock_console_class:
                mock_console = MagicMock()
                mock_console_class.return_value = mock_console
                health_summary(window=5, export=export_path)

            # Verify export file
            assert Path(export_path).exists()
            with open(export_path) as f:
                data = json.load(f)

            assert data["metric"] == "health"
            assert data["window"] == 5
            assert data["overall_status"] == "attention_needed"
            assert (
                data["unhealthy_count"] >= 2
            )  # Savings and pressure both unhealthy
            assert (
                len(data["metrics"]) == 4
            )  # Savings, Pressure, Memory, Trend
        finally:
            Path(export_path).unlink(missing_ok=True)


class TestSavingsTrends:
    """Test savings trend analysis."""

    @patch("tok.utils.metrics.SavingsTracker")
    def test_savings_trends_basic(self, mock_tracker):
        """Test basic savings trend functionality."""
        # Setup mocks
        mock_tracker.return_value.format_session.return_value = (
            "Session: 10 calls | 1000 tokens"
        )
        mock_tracker.return_value.format_ledger.return_value = (
            "Lifetime: 100 sessions | 10000 tokens"
        )

        mock_tracker.return_value.trend_summary.return_value = {
            "sessions_considered": 8,
            "direction": "improving",
            "avg_savings_pct": 18.5,
            "avg_tokens_saved": 5000,
            "savings_velocity": 2.1,
        }

        with patch("rich.console.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            savings_trends(window=8)
            mock_console.print.assert_called()

        # Verify calls
        mock_tracker.return_value.format_session.assert_called_once()
        mock_tracker.return_value.format_ledger.assert_called_once()
        mock_tracker.return_value.trend_summary.assert_called_once_with(
            recent_sessions=8
        )


class TestMemoryTrends:
    """Test memory trend analysis."""

    @patch("tok.utils.metrics.SavingsTracker")
    def test_memory_trends_basic(self, mock_tracker):
        """Test basic memory trend functionality."""
        # Setup mocks
        mock_tracker.return_value.behavior_signals.return_value = {
            "cold_start_structured_memory": 8,
            "durable_promotions": 4,
            "hot_promotions": 6,
            "other_signal": 10,
        }

        mock_tracker.return_value.trend_summary.return_value = {
            "sessions_considered": 6,
            "memory_lift_velocity": 1.2,
        }

        with patch("rich.console.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            memory_trends(window=6)
            mock_console.print.assert_called()

        # Verify memory lift calculation
        mock_tracker.return_value.behavior_signals.assert_called_once()


class TestFallbackTrends:
    """Test fallback trend analysis."""

    @patch("tok.utils.metrics.SavingsTracker")
    def test_fallback_trends_with_data(self, mock_tracker):
        """Test fallback trends with session data."""
        # Setup mocks
        mock_tracker.return_value.behavior_signals.return_value = {
            "cold_start_wire_fallback": 12,
            "cold_start_structured_memory": 5,
        }

        # Mock session log entries
        mock_entries = [
            {"invisible_pressure": 8},
            {"invisible_pressure": 12},
            {"invisible_pressure": 6},
            {"invisible_pressure": 10},
            {"invisible_pressure": 9},
        ]
        mock_tracker.return_value._load_session_log_entries.return_value = (
            mock_entries
        )
        mock_tracker.return_value.trend_summary.return_value = {
            "pressure_velocity": 1.5
        }

        with patch("rich.console.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            fallback_trends(window=5)
            mock_console.print.assert_called()

        # Verify calls
        mock_tracker.return_value.behavior_signals.assert_called_once()
        mock_tracker.return_value._load_session_log_entries.assert_called_once()

    @patch("tok.utils.metrics.SavingsTracker")
    def test_fallback_trends_no_data(self, mock_tracker):
        """Test fallback trends with no session data."""
        # Setup mocks
        mock_tracker.return_value.behavior_signals.return_value = {}
        mock_tracker.return_value._load_session_log_entries.return_value = []

        with patch("rich.console.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            fallback_trends(window=3)
            mock_console.print.assert_called()

        # Should handle empty data gracefully
        mock_tracker.return_value.behavior_signals.assert_called_once()
        mock_tracker.return_value._load_session_log_entries.assert_called_once()


class TestIntegration:
    """Integration tests for metrics functionality."""

    def test_format_trend_display_edge_cases(self):
        """Test edge cases in trend formatting."""
        # Test with missing velocity
        trend = {
            "sessions_considered": 1,
            "direction": "flat",
            "avg_pressure": 5.0,
        }
        result = format_trend_display(trend, "pressure")
        assert "1 sessions" in result
        assert "5.0" in result
        assert (
            "Velocity" not in result
        )  # Should not show velocity for single session

        # Test with zero velocity
        trend["pressure_velocity"] = 0.0
        result = format_trend_display(trend, "pressure")
        assert "Velocity" not in result  # still suppressed with single session
        assert "➡️" in result

    @patch("tok.utils.metrics.SavingsTracker")
    def test_metrics_with_zero_values(self, mock_tracker):
        """Test metrics handling of zero values."""
        # Setup mocks with zeros
        mock_tracker.return_value.behavior_signals.return_value = {}
        mock_tracker.return_value.trend_summary.return_value = {
            "sessions_considered": 0,
            "direction": "none",
            "avg_savings_pct": 0.0,
            "avg_invisible_pressure": 0.0,
            "savings_velocity": 0.0,
            "pressure_velocity": 0.0,
            "memory_lift_velocity": 0.0,
        }
        mock_tracker.return_value._load_session_log_entries.return_value = []

        with patch("rich.console.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            # Should handle zeros gracefully
            savings_trends(window=5)
            pressure_trends(window=5)
            memory_trends(window=5)
            fallback_trends(window=5)
            health_summary(window=5)

        # All should complete without errors
        assert True  # If we get here, no exceptions were raised
