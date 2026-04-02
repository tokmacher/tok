from __future__ import annotations

"""Bridge management commands for the Tok CLI."""

import os
import signal
import subprocess
import sys
import time
from typing import Annotated, Any

import typer

from ..stats import SavingsTracker

from ._cli_support import (
    COLLECTOR_PID_FILE,
    LOG_FILE,
    PID_FILE,
    TOK_DIR,
    bridge_url,
    console,
    get_running_bridge_pid,
    memory_root,
    read_collector_pid,
    render_stats_panel,
    runtime_verdict,
    savings_headline,
    savings_style,
    session_signals_text,
    session_status_rows,
    start_collector,
    status_border,
)

bridge_app = typer.Typer(help="Bridge management commands")


@bridge_app.command("start")
def bridge_start(
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port to listen on")
    ] = 9090,
    keep_turns: Annotated[
        int, typer.Option("--keep-turns", help="Human turns to keep verbatim")
    ] = 2,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
    foreground: Annotated[
        bool, typer.Option("--foreground", "-f", help="Run in foreground")
    ] = False,
    fail_open: Annotated[
        bool,
        typer.Option(
            "--fail-open/--no-fail-open", help="Pass through on errors"
        ),
    ] = True,
    capture: Annotated[
        bool,
        typer.Option(
            "--capture/--no-capture",
            help="Capture bridge sessions to the Tok sessions directory",
        ),
    ] = False,
    api_base: Annotated[
        str,
        typer.Option(
            "--api-base",
            help="Target API base URL (e.g., https://api.anthropic.com)",
        ),
    ] = "https://api.anthropic.com",
) -> None:
    """Start the Tok bridge server."""
    existing = get_running_bridge_pid(port)

    if existing:
        console.print(
            f"[yellow]Bridge already running on :{port} (PID {existing})[/yellow]"
        )
        raise typer.Exit(0)

    TOK_DIR.mkdir(parents=True, exist_ok=True)

    start_collector(_debug=debug)

    if foreground:
        from ..gateway import run_bridge

        if capture:
            os.environ["TOK_CAPTURE"] = "1"
        os.environ["TOK_RESET_SESSION"] = "1"

        run_bridge(
            port=port,
            keep_turns=keep_turns,
            debug=debug,
            fail_open=fail_open,
            _api_base=api_base,
        )
    else:
        env = os.environ.copy()
        env["TOK_BRIDGE_PORT"] = str(port)
        env["TOK_KEEP_TURNS"] = str(keep_turns)
        env["TOK_DEBUG"] = "1" if debug else "0"
        env["TOK_FAIL_OPEN"] = "1" if fail_open else "0"
        env["TOK_CAPTURE"] = "1" if capture else env.get("TOK_CAPTURE", "0")
        env["TOK_API_BASE"] = api_base
        env["TOK_RESET_SESSION"] = "1"

        log_file = open(LOG_FILE, "a")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "tok.gateway"],
                env=env,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
        finally:
            log_file.close()
        PID_FILE.write_text(str(proc.pid))

        for _ in range(15):
            time.sleep(0.2)
            try:
                import httpx

                r = httpx.get(bridge_url(port, "/health"), timeout=1.0)
                if r.status_code == 200:
                    console.print(
                        f"[green]Bridge started on :{port} (PID {proc.pid})[/green]"
                    )
                    console.print(f"Logs: {LOG_FILE}")
                    console.print(
                        "[dim]Next step: run `claude`, then `tok bridge status` or `tok doctor`.[/dim]"
                    )
                    if capture:
                        console.print(
                            f"Capture directory: {memory_root() / 'sessions'}"
                        )
                    return
            except Exception:
                pass

        console.print(
            f"[yellow]Bridge started (PID {proc.pid}) but health check pending[/yellow]"
        )
        console.print(f"Logs: {LOG_FILE}")
        console.print(
            "[dim]Next step: wait a moment, then run `tok bridge status`; if it still fails, restart with `tok bridge start --foreground`.[/dim]"
        )
        if capture:
            console.print(f"Capture directory: {memory_root() / 'sessions'}")


@bridge_app.command("stop")
def bridge_stop() -> None:
    """Stop the Tok bridge server."""
    port = int(os.getenv("TOK_BRIDGE_PORT", "9090"))
    pid = get_running_bridge_pid(port)
    tracker = SavingsTracker()

    if not pid:
        console.print("[yellow]Bridge not running[/yellow]")
        raise typer.Exit(0)

    for p in [pid]:
        try:
            os.kill(p, signal.SIGTERM)
            for _ in range(10):
                time.sleep(0.1)
                try:
                    os.kill(p, 0)
                except ProcessLookupError:
                    break
            else:
                os.kill(p, signal.SIGKILL)
            console.print(f"[green]Bridge stopped (PID {p})[/green]")
        except (ProcessLookupError, PermissionError):
            console.print(
                f"[yellow]Failed to stop PID {p} (gone or permission denied)[/yellow]"
            )

    PID_FILE.unlink(missing_ok=True)

    session_summary = tracker.session_summary()
    if session_summary:
        headline, headline_pct, subhead = savings_headline(session_summary)
        verdict, verdict_style = runtime_verdict(
            tok_active=not bool(session_summary["baseline_only"]),
            baseline_only=bool(session_summary["baseline_only"]),
            tokens_saved=int(session_summary["tokens_saved"]),
        )
        console.print(
            render_stats_panel(
                "Last Session",
                headline=f"{headline} • {headline_pct}",
                headline_style=savings_style(
                    float(session_summary["savings_pct"])
                ),
                subhead=f"{verdict} • {subhead}",
                rows=session_status_rows(
                    summary=session_summary,
                    tok_active=not bool(session_summary["baseline_only"]),
                    baseline_only=bool(session_summary["baseline_only"]),
                ),
                border_style=status_border(verdict_style),
            )
        )

    collector_pid = read_collector_pid()
    if collector_pid:
        try:
            os.kill(collector_pid, signal.SIGTERM)
            console.print(
                f"[green]Collector stopped (PID {collector_pid})[/green]"
            )
        except (ProcessLookupError, PermissionError):
            pass
    COLLECTOR_PID_FILE.unlink(missing_ok=True)


@bridge_app.command("status")
def bridge_status() -> None:
    """Check bridge status."""
    port = int(os.getenv("TOK_BRIDGE_PORT", "9090"))
    pid = get_running_bridge_pid(port)
    if pid is None:
        console.print("[yellow]Bridge not running[/yellow]")
        console.print(
            "[dim]Next step: run `tok bridge start`, then re-run `tok bridge status` or `tok doctor`.[/dim]"
        )
        raise typer.Exit(1)

    try:
        import httpx

        r = httpx.get(bridge_url(port, "/health"), timeout=2.0)
        if r.status_code == 200:
            payload = r.json()
            session_summary: dict[str, Any] = {
                "actual_tokens": int(payload.get("actual_tokens", 0)),
                "baseline_tokens": int(payload.get("baseline_tokens", 0)),
                "tokens_saved": int(payload.get("session_tokens_saved", 0)),
                "savings_pct": float(payload.get("session_savings_pct", 0.0)),
                "actual_cost_usd": float(payload.get("actual_cost_usd", 0.0)),
                "baseline_cost_usd": float(
                    payload.get("baseline_cost_usd", 0.0)
                ),
                "cost_saved_usd": float(payload.get("cost_saved_usd", 0.0)),
                "session_quality": str(
                    payload.get("session_quality", "clean")
                ),
                "last_degradation_reason": str(
                    payload.get("last_degradation_reason", "")
                ),
                "request_policy": str(payload.get("request_policy", "")),
                "preflight_block_original_payload_count": int(
                    payload.get("preflight_block_original_payload_count", 0)
                ),
                "preflight_block_rewritten_payload_count": int(
                    payload.get("preflight_block_rewritten_payload_count", 0)
                ),
                "stream_recovery_empty_success_count": int(
                    payload.get("stream_recovery_empty_success_count", 0)
                ),
                "stream_recovery_read_error_count": int(
                    payload.get("stream_recovery_read_error_count", 0)
                ),
                "request_policy_held_by_recovery_count": int(
                    payload.get("request_policy_held_by_recovery_count", 0)
                ),
            }
            baseline_only = bool(payload.get("baseline_only"))
            fallback_count = int(payload.get("fallback_count", 0))
            mode = str(payload.get("mode", "unknown"))
            tokens_saved = int(session_summary["tokens_saved"])
            verdict, verdict_style = runtime_verdict(
                tok_active=True,
                baseline_only=baseline_only,
                mode=mode,
                tokens_saved=tokens_saved,
                session_quality=str(payload.get("session_quality", "clean")),
            )
            headline, headline_pct, subhead = savings_headline(
                session_summary,
                savings_pct=float(payload.get("session_savings_pct", 0.0)),
                tokens_saved=int(payload.get("session_tokens_saved", 0)),
            )
            console.print(
                f"[green]Bridge running on :{port} (PID {pid})[/green]"
            )
            console.print(
                render_stats_panel(
                    "Bridge Status",
                    headline=f"{headline} • {headline_pct}",
                    headline_style=savings_style(
                        float(session_summary["savings_pct"])
                        if isinstance(
                            session_summary.get("savings_pct"),
                            int | float | str,
                        )
                        else 0.0
                    ),
                    subhead=f"{verdict} • {subhead}",
                    rows=session_status_rows(
                        summary=session_summary,
                        tok_active=True,
                        baseline_only=baseline_only,
                        mode=mode,
                        request_policy=str(payload.get("request_policy", ""))
                        or None,
                        fallback_count=fallback_count,
                        session_quality=str(
                            payload.get("session_quality", "clean")
                        ),
                        degradation_reason=str(
                            payload.get("last_degradation_reason", "")
                        ),
                        session_signals=session_signals_text(payload),
                    ),
                    border_style=status_border(verdict_style),
                )
            )
            if baseline_only:
                console.print(
                    "[dim]Next step: run `tok doctor`, then inspect `tok bridge logs 100` for the degradation reason.[/dim]"
                )
            elif mode == "baseline":
                console.print(
                    "[dim]Next step: restart without `TOK_MODE=baseline` if you want compression enabled.[/dim]"
                )
            elif str(payload.get("session_quality", "clean")) == "watch":
                console.print(
                    "[dim]Next step: keep Tok on, but watch `Fallbacks` and rerun `tok doctor` if they rise.[/dim]"
                )
            elif tokens_saved <= 0:
                console.print(
                    "[dim]Next step: keep working for a few turns, then run `tok stats --last-session` if savings are still unclear.[/dim]"
                )
            return
    except httpx.ConnectError:
        pass
    except Exception as exc:
        console.print(
            f"[dim]Status check error: {exc.__class__.__name__}: {exc}[/dim]"
        )

    console.print(
        f"[yellow]Bridge process alive (PID {pid}) but not responding[/yellow]"
    )
    console.print(
        "[dim]Next step: inspect `tok bridge logs 100` or restart with `tok bridge start --foreground`.[/dim]"
    )


@bridge_app.command("logs")
def bridge_logs(
    lines: int = typer.Argument(40, help="Number of lines to show"),
) -> None:
    """Tail the bridge log file."""
    if not LOG_FILE.exists():
        console.print("[yellow]No log file found[/yellow]")
        raise typer.Exit(1)

    content = LOG_FILE.read_text().splitlines()
    for line in content[-lines:]:
        console.print(line)
