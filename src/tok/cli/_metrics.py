from __future__ import annotations

"""Telemetry and health reporting commands for the Tok CLI."""

from typing import Annotated

import typer


metrics_app = typer.Typer(help="Telemetry and health reporting commands")


@metrics_app.command("pressure")
def pressure(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
    export: Annotated[
        str, typer.Option("--export", "-e", help="Export results to file")
    ] = "",
) -> None:
    """Show invisible pressure trends and current status."""
    from ..metrics import pressure_trends

    pressure_trends(window, export)


@metrics_app.command("memory")
def memory(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
) -> None:
    """Show memory lift trends and current status."""
    from ..metrics import memory_trends

    memory_trends(window)


@metrics_app.command("savings-trend")
def savings_trend(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
) -> None:
    """Show savings percentage trends and current status."""
    from ..metrics import savings_trends

    savings_trends(window)


@metrics_app.command("fallback")
def fallback(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
) -> None:
    """Show cold-start fallback trends and current status."""
    from ..metrics import fallback_trends

    fallback_trends(window)


@metrics_app.command("health")
def health(
    window: Annotated[
        int,
        typer.Option(
            "--window",
            "-w",
            help="Number of recent sessions for trend analysis",
        ),
    ] = 10,
    export: Annotated[
        str, typer.Option("--export", "-e", help="Export results to file")
    ] = "",
) -> None:
    """Show aggregated health metrics summary."""
    from ..metrics import health_summary

    health_summary(window, export)
