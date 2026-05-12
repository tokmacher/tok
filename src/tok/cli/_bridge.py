"""Bridge management commands for the Tok CLI."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Annotated, Any
from urllib.parse import urlparse

import typer

from tok.stats import SavingsTracker

from ._cli_support import (
    LOG_FILE,
    PID_FILE,
    TOK_DIR,
    console,
    env_int,
    get_bridge_health_response,
    get_running_bridge_pid,
    json_envelope,
    memory_root,
    render_stats_panel,
    runtime_verdict,
    savings_headline,
    savings_style,
    session_signals_text,
    session_status_rows,
    status_border,
)

_LOCAL_HOST_ALIASES = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _normalized_host(host: str | None) -> str | None:
    if not host:
        return None
    value = host.strip().lower()
    return value or None


def _parsed_port(parsed_url: Any) -> int | None:
    if parsed_url.port is not None:
        return int(parsed_url.port)
    if parsed_url.scheme == "http":
        return 80
    if parsed_url.scheme == "https":
        return 443
    return None


def _is_self_bridged_invocation(port: int) -> bool:
    marker = os.getenv("TOK_SELF_BRIDGED_SESSION", "").strip().lower()
    if marker not in {"1", "true", "yes", "on"}:
        return False

    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if not base_url:
        return False

    parsed = urlparse(base_url)
    target_host = _normalized_host(parsed.hostname)
    target_port = _parsed_port(parsed)
    bridge_host = _normalized_host(os.getenv("TOK_BRIDGE_HOST", "localhost"))
    if not target_host or target_port is None or not bridge_host:
        return False
    if target_port != port:
        return False
    if target_host == bridge_host:
        return True
    return target_host in _LOCAL_HOST_ALIASES and bridge_host in _LOCAL_HOST_ALIASES


def bridge_start(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 9090,
    keep_turns: Annotated[int, typer.Option("--keep-turns", help="Human turns to keep verbatim")] = 2,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug logging")] = False,
    foreground: Annotated[bool, typer.Option("--foreground", "-f", help="Run in foreground")] = False,
    fail_open: Annotated[
        bool,
        typer.Option("--fail-open/--no-fail-open", help="Pass through on errors"),
    ] = True,
    capture: Annotated[
        bool,
        typer.Option(
            "--capture/--no-capture",
            help="Capture bridge sessions to the Tok sessions directory",
        ),
    ] = False,
    api_base: Annotated[
        str | None,
        typer.Option(
            "--api-base",
            help="Target API base URL (e.g., https://api.anthropic.com)",
        ),
    ] = None,
) -> None:
    """Start the Tok bridge server."""
    existing = get_running_bridge_pid(port)

    if existing:
        console.print(f"[yellow]Bridge already running on :{port} (PID {existing})[/yellow]")
        raise typer.Exit(0)

    TOK_DIR.mkdir(parents=True, exist_ok=True)

    if foreground:
        from tok.gateway import run_bridge

        previous_capture = os.environ.get("TOK_CAPTURE")
        previous_reset = os.environ.get("TOK_RESET_SESSION")
        if capture:
            os.environ["TOK_CAPTURE"] = "1"
        os.environ["TOK_RESET_SESSION"] = "1"

        try:
            run_bridge(
                port=port,
                keep_turns=keep_turns,
                debug=debug,
                fail_open=fail_open,
                _api_base=api_base,
            )
        finally:
            if previous_capture is None:
                os.environ.pop("TOK_CAPTURE", None)
            else:
                os.environ["TOK_CAPTURE"] = previous_capture
            if previous_reset is None:
                os.environ.pop("TOK_RESET_SESSION", None)
            else:
                os.environ["TOK_RESET_SESSION"] = previous_reset
    else:
        env = os.environ.copy()
        env["TOK_BRIDGE_PORT"] = str(port)
        env["TOK_KEEP_TURNS"] = str(keep_turns)
        env["TOK_DEBUG"] = "1" if debug else "0"
        env["TOK_FAIL_OPEN"] = "1" if fail_open else "0"
        env["TOK_CAPTURE"] = "1" if capture else env.get("TOK_CAPTURE", "0")
        if api_base is not None:
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
        except FileNotFoundError:
            console.print("[red]Failed to start bridge: Python interpreter not found.[/red]")
            raise typer.Exit(1) from None
        except PermissionError:
            console.print(f"[red]Failed to start bridge: permission denied writing to {LOG_FILE}.[/red]")
            raise typer.Exit(1) from None
        finally:
            log_file.close()
        PID_FILE.write_text(str(proc.pid))

        for _ in range(15):
            time.sleep(0.2)
            try:
                r = get_bridge_health_response(port, timeout=1.0, attempts=1, backoff_seconds=0.0)
                if r.status_code == 200:
                    console.print(f"[green]Bridge started on :{port} (PID {proc.pid})[/green]")
                    console.print(f"Logs: {LOG_FILE}")
                    console.print(
                        f"[dim]Next step: run `ANTHROPIC_BASE_URL=http://localhost:{port} claude`, then "
                        "`tok bridge status` or `tok doctor`.[/dim]"
                    )
                    if capture:
                        console.print(f"Capture directory: {memory_root() / 'sessions'}")
                    return
            except Exception:
                pass

        if proc.poll() is not None:
            console.print(f"[red]Bridge process exited unexpectedly (exit code {proc.returncode}).[/red]")
            console.print(f"Check logs: {LOG_FILE}")
            console.print("[dim]Try `tok bridge start --foreground` to see the error directly.[/dim]")
            raise typer.Exit(1)

        console.print(f"[yellow]Bridge started (PID {proc.pid}) but health check pending.[/yellow]")
        console.print(f"Logs: {LOG_FILE}")
        console.print(
            "[dim]Next step: wait a moment, then run `tok bridge status`; if it still fails, restart with `tok bridge start --foreground`.[/dim]"
        )
        if capture:
            console.print(f"Capture directory: {memory_root() / 'sessions'}")


def bridge_stop(force: bool = False) -> None:
    """Stop the Tok bridge server."""
    port = env_int("TOK_BRIDGE_PORT", 9090)
    pid = get_running_bridge_pid(port)
    tracker = SavingsTracker()

    if not pid:
        console.print("[yellow]Bridge not running[/yellow]")
        raise typer.Exit(0)

    if _is_self_bridged_invocation(port) and not force:
        try:
            health = get_bridge_health_response(port, timeout=0.8, attempts=1, backoff_seconds=0.0)
            health_ok = health.status_code == 200
        except Exception:
            health_ok = False
        if health_ok:
            console.print("[yellow]Refusing to stop bridge from an active bridged Claude session.[/yellow]")
            console.print(
                "[dim]Run `tok bridge stop --force` if intentional, or stop from a separate shell after this turn.[/dim]"
            )
            raise typer.Exit(2)

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
            console.print(f"[yellow]Failed to stop PID {p} (gone or permission denied)[/yellow]")

    try:
        PID_FILE.unlink(missing_ok=True)
    except PermissionError as exc:
        console.print(f"[yellow]Could not remove bridge PID file {PID_FILE}: {exc}[/yellow]")

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
                headline_style=savings_style(float(session_summary["savings_pct"])),
                subhead=f"{verdict} • {subhead}",
                rows=session_status_rows(
                    summary=session_summary,
                    tok_active=not bool(session_summary["baseline_only"]),
                    baseline_only=bool(session_summary["baseline_only"]),
                ),
                border_style=status_border(verdict_style),
            )
        )


def bridge_status(*, json_output: bool = False) -> None:
    """Check bridge status."""
    port = env_int("TOK_BRIDGE_PORT", 9090)
    pid = get_running_bridge_pid(port)
    if pid is None:
        if json_output:
            envelope = json_envelope(
                "tok bridge status",
                ok=False,
                status="error",
                data={"bridge_running": False, "port": port},
                warnings=["Bridge not running"],
                next_steps=["Run `tok bridge start`, then re-run `tok bridge status` or `tok doctor`."],
            )
            print(json.dumps(envelope, indent=2))
        else:
            console.print("[yellow]Bridge not running[/yellow]")
            console.print(
                "[dim]Next step: run `tok bridge start`, then re-run `tok bridge status` or `tok doctor`.[/dim]"
            )
        raise typer.Exit(1)

    try:
        r = get_bridge_health_response(port, timeout=2.0, attempts=2, backoff_seconds=0.2)
    except Exception:
        if json_output:
            envelope = json_envelope(
                "tok bridge status",
                ok=False,
                status="error",
                data={"bridge_running": True, "port": port, "pid": pid, "health_reachable": False},
                warnings=["Bridge process alive but not responding"],
                next_steps=["Inspect `tok bridge logs 100` or restart with `tok bridge start --foreground`."],
            )
            print(json.dumps(envelope, indent=2))
        else:
            console.print(f"[yellow]Bridge process alive (PID {pid}) but not responding[/yellow]")
            console.print(
                "[dim]Next step: inspect `tok bridge logs 100` or restart with `tok bridge start --foreground`.[/dim]"
            )
        return

    if r.status_code == 200:
        try:
            payload = r.json()
            session_summary: dict[str, Any] = {
                "actual_tokens": int(payload.get("actual_tokens", 0)),
                "baseline_tokens": int(payload.get("baseline_tokens", 0)),
                "tokens_saved": int(payload.get("session_tokens_saved", 0)),
                "savings_pct": float(payload.get("session_savings_pct", 0.0)),
                "actual_cost_usd": float(payload.get("actual_cost_usd", 0.0)),
                "baseline_cost_usd": float(payload.get("baseline_cost_usd", 0.0)),
                "cost_saved_usd": float(payload.get("cost_saved_usd", 0.0)),
                "session_quality": str(payload.get("session_quality", "clean")),
                "last_degradation_reason": str(payload.get("last_degradation_reason", "")),
                "request_policy": str(payload.get("request_policy", "")),
                "preflight_block_original_payload_count": int(payload.get("preflight_block_original_payload_count", 0)),
                "preflight_block_rewritten_payload_count": int(
                    payload.get("preflight_block_rewritten_payload_count", 0)
                ),
                "stream_recovery_empty_success_count": int(payload.get("stream_recovery_empty_success_count", 0)),
                "stream_recovery_read_error_count": int(payload.get("stream_recovery_read_error_count", 0)),
                "request_policy_held_by_recovery_count": int(payload.get("request_policy_held_by_recovery_count", 0)),
                "evidence_exact_observed_count": int(payload.get("evidence_exact_observed_count", 0)),
                "evidence_non_exact_reference_count": int(payload.get("evidence_non_exact_reference_count", 0)),
                "evidence_non_exact_summary_count": int(payload.get("evidence_non_exact_summary_count", 0)),
                "evidence_non_exact_skeleton_count": int(payload.get("evidence_non_exact_skeleton_count", 0)),
                "evidence_exact_reacquisition_required_count": int(
                    payload.get("evidence_exact_reacquisition_required_count", 0)
                ),
                "evidence_exact_reacquisition_satisfied_count": int(
                    payload.get("evidence_exact_reacquisition_satisfied_count", 0)
                ),
                "evidence_compression_blocked_for_safety_count": int(
                    payload.get("evidence_compression_blocked_for_safety_count", 0)
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
            capability = payload.get("capability")
            conformance = "unknown"
            if isinstance(capability, dict):
                conformance = str(capability.get("max_conformance_level", "unknown"))
            if json_output:
                envelope = json_envelope(
                    "tok bridge status",
                    ok=True,
                    status="ok",
                    data={
                        "bridge_running": True,
                        "port": port,
                        "pid": pid,
                        "health_reachable": True,
                        "tok_active": not baseline_only,
                        "mode": mode,
                        "conformance": conformance,
                        "baseline_only": baseline_only,
                        "degraded_to_baseline": baseline_only,
                        "fallback_count": fallback_count,
                        "session_quality": str(payload.get("session_quality", "clean")),
                        "tokens_saved": int(session_summary["tokens_saved"]),
                        "savings_pct": float(session_summary["savings_pct"]),
                        "cost_saved_usd": float(session_summary["cost_saved_usd"]),
                    },
                )
                print(json.dumps(envelope, indent=2))
                return
            console.print(f"[green]Bridge running on :{port} (PID {pid})[/green]")
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
                        request_policy=str(payload.get("request_policy", "")) or None,
                        api_base=str(payload.get("api_base", "")) or None,
                        fallback_count=fallback_count,
                        session_quality=str(payload.get("session_quality", "clean")),
                        degradation_reason=str(payload.get("last_degradation_reason", "")),
                        session_signals=session_signals_text(payload),
                    ),
                    border_style=status_border(verdict_style),
                )
            )
            capability = payload.get("capability")
            if isinstance(capability, dict):
                evidence_forms = capability.get("supported_evidence_forms", ())
                if isinstance(evidence_forms, list | tuple):
                    evidence_text = ", ".join(str(item) for item in evidence_forms)
                else:
                    evidence_text = str(evidence_forms or "")
                console.print("[bold]Bridge capability:[/bold]")
                console.print(f"  mode: {capability.get('bridge_mode', 'unknown')}")
                console.print(f"  trace: {capability.get('trace_version', 'unknown')}")
                console.print(f"  conformance: {capability.get('max_conformance_level', 'unknown')}")
                console.print(f"  evidence: {evidence_text}")
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
        except (TypeError, ValueError, KeyError):
            if json_output:
                envelope = json_envelope(
                    "tok bridge status",
                    ok=False,
                    status="error",
                    data={
                        "bridge_running": True,
                        "port": port,
                        "pid": pid,
                        "health_reachable": True,
                        "malformed_payload": True,
                    },
                    warnings=["Health payload malformed"],
                    next_steps=["Inspect `tok bridge logs 100` or restart with `tok bridge start --foreground`."],
                )
                print(json.dumps(envelope, indent=2))
            else:
                console.print(f"[yellow]Bridge process alive (PID {pid}) but health payload is malformed[/yellow]")
                console.print(
                    "[dim]Next step: inspect `tok bridge logs 100` and restart with `tok bridge start --foreground`.[/dim]"
                )
            raise typer.Exit(1) from None

    if json_output:
        envelope = json_envelope(
            "tok bridge status",
            ok=False,
            status="error",
            data={
                "bridge_running": True,
                "port": port,
                "pid": pid,
                "health_reachable": True,
                "http_status": r.status_code,
            },
            warnings=[f"Bridge returned HTTP {r.status_code}"],
            next_steps=["Inspect `tok bridge logs 100` or restart with `tok bridge start --foreground`."],
        )
        print(json.dumps(envelope, indent=2))
    else:
        console.print(f"[yellow]Bridge process alive (PID {pid}) but not responding[/yellow]")
        console.print(
            "[dim]Next step: inspect `tok bridge logs 100` or restart with `tok bridge start --foreground`.[/dim]"
        )
