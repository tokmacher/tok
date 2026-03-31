"""Dedicated metric commands for Tok telemetry."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from typing import Any

from ..runtime.policy.semantic_validation import calculate_invisible_pressure
from ..stats import SavingsTracker


def format_trend_display(trend: dict[str, Any], metric_name: str) -> str:
    """Format trend data for display."""
    if trend["sessions_considered"] == 0:
        return f"[dim]No {metric_name} data available[/dim]"

    direction_icon = {"improving": "📈", "regressing": "📉", "flat": "➡️"}.get(
        trend["direction"], "❓"
    )

    result = (
        f"[bold]{metric_name.title()} Trend ({trend['sessions_considered']} sessions):[/bold] "
        f"{direction_icon} {trend['direction']}\n"
        f"  Average: {trend.get(f'avg_{metric_name}', 'N/A')}\n"
    )

    if trend["sessions_considered"] >= 2:
        velocity_key = f"{metric_name}_velocity"
        if velocity_key in trend:
            velocity = trend[velocity_key]
            velocity_icon = (
                "📈" if velocity > 0 else "📉" if velocity < 0 else "➡️"
            )
            result += f"  Velocity: {velocity_icon} {velocity:+.2f}/session\n"

    return result.strip()


def pressure_trends(window: int = 10, export: str = "") -> None:
    """Show invisible pressure trends over time."""
    from rich.console import Console

    console = Console()
    tracker = SavingsTracker()

    # Get current session raw signals (cumulative, for diagnostics only)
    current_signals = tracker.behavior_signals()
    current_pressure_raw = calculate_invisible_pressure(current_signals)

    console.print(
        f"[bold]Current Session Pressure (raw cumulative):[/bold] {current_pressure_raw}"
    )

    if current_signals:
        console.print("\n[bold]Current Session Signals:[/bold]")
        for signal, count in sorted(
            current_signals.items(), key=lambda x: (-x[1], x[0])
        ):
            console.print(f"  {signal:<25} {count:>3}")

    # Get trend data (per-session averages over recent completed sessions)
    trend = tracker.trend_summary(recent_sessions=window)

    # Extract pressure-specific data
    pressure_trend = {
        "sessions_considered": trend["sessions_considered"],
        "direction": (
            "regressing"
            if float(trend["pressure_velocity"]) > 0
            else (
                "improving"
                if float(trend["pressure_velocity"]) < 0
                else "flat"
            )
        ),
        "avg_pressure": trend["avg_invisible_pressure"],
        "pressure_velocity": trend["pressure_velocity"],
    }

    console.print(f"\n{format_trend_display(pressure_trend, 'pressure')}")

    # Status verdict uses bounded avg_invisible_pressure from recent sessions,
    # not the unbounded cumulative raw count from the current session.
    sessions_considered = trend.get("sessions_considered", 0)
    avg_pressure = trend.get("avg_invisible_pressure", 0.0)

    if sessions_considered == 0:
        console.print(
            "\n[yellow]⚠️  Status: Unknown - No completed session history[/yellow]"
        )
        status = "unknown"
    elif float(avg_pressure) == 0:
        console.print(
            "\n[green]✅ Status: Clean - No pressure in recent sessions[/green]"
        )
        status = "clean"
    elif float(avg_pressure) <= 3:
        console.print(
            f"\n[yellow]⚠️  Status: Watch - Low avg pressure ({avg_pressure:.1f})[/yellow]"
        )
        status = "watch"
    else:
        console.print(
            f"\n[red]❌ Status: Noisy - High avg pressure ({avg_pressure:.1f})[/red]"
        )
        status = "noisy"

    # Export functionality
    if export:
        export_data = {
            "metric": "pressure",
            "window": window,
            "current": {
                "pressure_raw": current_pressure_raw,
                "signals": current_signals,
            },
            "trend": pressure_trend,
            "avg_pressure": avg_pressure,
            "sessions_considered": sessions_considered,
            "status": status,
        }
        from pathlib import Path

        Path(export).write_text(json.dumps(export_data, indent=2))
        console.print(f"\n[dim]Data exported to: {export}[/dim]")


def memory_trends(window: int = 10) -> None:
    """Show memory lift trends over time."""
    from rich.console import Console

    console = Console()
    tracker = SavingsTracker()

    # Get current session memory lift
    current_signals = tracker.behavior_signals()
    memory_lift = sum(
        current_signals.get(name, 0)
        for name in (
            "cold_start_structured_memory",
            "durable_promotions",
            "hot_promotions",
            "file_snapshot_recorded",
            "search_snapshot_recorded",
        )
    )

    console.print(f"[bold]Current Memory Lift:[/bold] {memory_lift}")

    if current_signals:
        memory_signals = {
            k: v
            for k, v in current_signals.items()
            if any(name in k for name in ("cold_start", "durable", "hot"))
        }
        if memory_signals:
            console.print("\n[bold]Memory-related Signals:[/bold]")
            for signal, count in sorted(
                memory_signals.items(), key=lambda x: (-x[1], x[0])
            ):
                console.print(f"  {signal:<25} {count:>3}")

    # Get trend data
    trend = tracker.trend_summary(recent_sessions=window)

    memory_trend = {
        "sessions_considered": trend["sessions_considered"],
        "direction": (
            "improving"
            if float(trend["memory_lift_velocity"]) > 0
            else (
                "regressing"
                if float(trend["memory_lift_velocity"]) < 0
                else "flat"
            )
        ),
        "avg_memory_lift": trend.get("avg_memory_lift", memory_lift),
        "memory_lift_velocity": trend["memory_lift_velocity"],
    }

    console.print(f"\n{format_trend_display(memory_trend, 'memory_lift')}")

    # Status indicator
    if memory_lift > 0:
        console.print(
            f"\n[green]✅ Status: Active - Memory lift detected ({memory_lift})[/green]"
        )
    else:
        console.print(
            "\n[yellow]⚠️  Status: Inactive - No memory lift detected[/yellow]"
        )


def savings_trends(window: int = 10) -> None:
    """Show savings percentage trends over time."""
    from rich.console import Console

    console = Console()
    tracker = SavingsTracker()

    # Get current session savings
    session_stats = tracker.format_session()
    if session_stats:
        console.print(f"[bold]Current Session:[/bold] {session_stats}")
    else:
        console.print("[dim]No active session data[/dim]")

    # Get lifetime savings
    lifetime_stats = tracker.format_ledger()
    if lifetime_stats:
        console.print(f"[bold]Lifetime:[/bold] {lifetime_stats}")

    # Get trend data
    trend = tracker.trend_summary(recent_sessions=window)

    # Extract savings-specific data
    savings_trend = {
        "sessions_considered": trend["sessions_considered"],
        "direction": trend["direction"],
        "avg_savings_pct": trend["avg_savings_pct"],
        "savings_velocity": trend["savings_velocity"],
    }

    console.print(f"\n{format_trend_display(savings_trend, 'savings_pct')}")

    # Status indicator
    avg_savings = trend["avg_savings_pct"]
    if float(avg_savings) >= 20:
        console.print(
            f"\n[green]✅ Status: Excellent - {avg_savings}% average savings[/green]"
        )
    elif float(avg_savings) >= 10:
        console.print(
            f"\n[yellow]⚠️  Status: Good - {avg_savings}% average savings[/yellow]"
        )
    else:
        console.print(
            f"\n[red]❌ Status: Poor - {avg_savings}% average savings[/red]"
        )


def fallback_trends(window: int = 10) -> None:
    """Show cold-start fallback trends over time."""
    from rich.console import Console

    console = Console()
    tracker = SavingsTracker()

    # Get current session fallback signals
    current_signals = tracker.behavior_signals()
    fallback_signals = {
        k: v
        for k, v in current_signals.items()
        if "fallback" in k or "cold_start" in k
    }

    console.print("[bold]Current Session Fallback Activity:[/bold]")
    if fallback_signals:
        for signal, count in sorted(
            fallback_signals.items(), key=lambda x: (-x[1], x[0])
        ):
            console.print(f"  {signal:<25} {count:>3}")
    else:
        console.print("  [dim]No fallback activity detected[/dim]")

    # Get trend data
    trend = tracker.trend_summary(recent_sessions=window)

    # Calculate fallback trend from session logs
    entries = tracker._load_session_log_entries()
    if entries:
        recent = entries[-window:]
        fallback_counts = [
            entry.get("invisible_pressure", 0) for entry in recent
        ]  # Using pressure as proxy for fallback activity

        avg_fallback = (
            statistics.mean(fallback_counts) if fallback_counts else 0
        )

        fallback_trend = {
            "sessions_considered": len(recent),
            "direction": (
                "regressing"
                if avg_fallback > 3
                else "improving"
                if avg_fallback <= 1
                else "flat"
            ),
            "avg_fallback": round(avg_fallback, 1),
            "fallback_velocity": trend.get(
                "pressure_velocity", 0
            ),  # Using pressure velocity as proxy
        }

        console.print(f"\n{format_trend_display(fallback_trend, 'fallback')}")
    else:
        console.print("[dim]No trend data available[/dim]")


def health_summary(window: int = 10, export: str = "") -> None:
    """Show aggregated health metrics summary."""
    from rich.console import Console

    console = Console()
    tracker = SavingsTracker()

    console.print("[bold]Tok System Health Summary[/bold]")
    console.print("=" * 50)

    # Get trend data (recent completed sessions — bounded, per-session averages)
    trend = tracker.trend_summary(recent_sessions=window)

    # Current session raw signals — for diagnostics only, NOT for verdicting.
    # These are unbounded cumulative totals that grow with session length and
    # cannot meaningfully be compared against the small thresholds used for
    # single-event pressure checks.
    current_signals = tracker.behavior_signals()
    current_pressure_raw = calculate_invisible_pressure(current_signals)

    # Bounded pressure severity: use avg_invisible_pressure from recent completed
    # sessions. This is a per-session average, calibrated at the same scale as
    # the thresholds below. Falls back to 0 when no session history exists.
    sessions_considered = trend.get("sessions_considered", 0)
    avg_pressure = trend.get("avg_invisible_pressure", 0.0)

    # Memory lift from current session signals (counts positive memory events)
    memory_lift = sum(
        current_signals.get(name, 0)
        for name in (
            "cold_start_structured_memory",
            "durable_promotions",
            "hot_promotions",
            "file_snapshot_recorded",
            "search_snapshot_recorded",
        )
    )

    # Health status for each metric
    metrics = []

    # Savings health — from trend (avg over recent sessions)
    avg_savings = trend["avg_savings_pct"]
    savings_status = (
        "healthy"
        if avg_savings >= 20
        else "watch"
        if avg_savings >= 10
        else "unhealthy"
    )
    metrics.append(("Savings", f"{avg_savings}%", savings_status))

    # Pressure health — use bounded avg_invisible_pressure from recent sessions.
    # Thresholds: 0=healthy, ≤3=watch, >3=unhealthy.
    # When no session history exists, fall back to watch (unknown state).
    if sessions_considered == 0:
        pressure_status = "watch"
        pressure_display = "no history"
    else:
        pressure_display = f"{avg_pressure:.1f} avg/{sessions_considered}s"
        pressure_status = (
            "healthy"
            if avg_pressure == 0
            else "watch"
            if avg_pressure <= 3
            else "unhealthy"
        )
    metrics.append(("Pressure", pressure_display, pressure_status))

    # Memory health — current session signal count (positive indicator)
    memory_status = "healthy" if memory_lift > 0 else "watch"
    metrics.append(("Memory", str(memory_lift), memory_status))

    # Trend direction
    trend_status = (
        "healthy"
        if trend["direction"] == "improving"
        else "watch"
        if trend["direction"] == "flat"
        else "unhealthy"
    )
    metrics.append(("Trend", trend["direction"], trend_status))

    # Display metrics with status indicators
    for name, value, status in metrics:
        icon = {"healthy": "✅", "watch": "⚠️", "unhealthy": "❌"}.get(
            status, "❓"
        )
        color = {
            "healthy": "green",
            "watch": "yellow",
            "unhealthy": "red",
        }.get(status, "white")
        console.print(
            f"  {icon} {name:<12} {value:<15} [{color}]{status}[/{color}]"
        )

    # Show raw current-session pressure as a diagnostic footnote only
    if current_pressure_raw > 0:
        console.print(
            f"\n  [dim]Current session raw pressure: {current_pressure_raw} "
            f"(cumulative count, not used for verdict)[/dim]"
        )

    # Overall health assessment
    unhealthy_count = sum(
        1 for _, _, status in metrics if status == "unhealthy"
    )
    watch_count = sum(1 for _, _, status in metrics if status == "watch")

    console.print("\n" + "=" * 50)
    if unhealthy_count == 0 and watch_count <= 1:
        console.print("[green]🎉 Overall Status: HEALTHY[/green]")
        overall_status = "healthy"
    elif unhealthy_count == 0:
        console.print("[yellow]👀 Overall Status: MONITOR[/yellow]")
        overall_status = "monitor"
    else:
        console.print(
            f"[red]⚠️  Overall Status: ATTENTION NEEDED ({unhealthy_count} issues)[/red]"
        )
        overall_status = "attention_needed"

    # Export functionality
    if export:
        export_data = {
            "metric": "health",
            "window": window,
            "metrics": [
                {"name": name, "value": value, "status": status}
                for name, value, status in metrics
            ],
            "overall_status": overall_status,
            "unhealthy_count": unhealthy_count,
            "watch_count": watch_count,
            "avg_pressure": avg_pressure,
            "sessions_considered": sessions_considered,
            "current_pressure_raw": current_pressure_raw,
            "memory_lift": memory_lift,
            "trend": trend,
        }
        Path(export).write_text(json.dumps(export_data, indent=2))
        console.print(f"\n[dim]Health data exported to: {export}[/dim]")
