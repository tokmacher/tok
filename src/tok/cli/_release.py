"""Release-facing CLI helper implementations."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from tok.runtime.pipeline.response_handling import evaluate_replay_gate
from tok.runtime.pipeline.tool_processing import (
    build_tool_use_id_to_context,
    collect_behavior_signals,
)
from tok.runtime.policy.semantic_validation import calculate_invisible_pressure
from tok.runtime.policy.smart_policy import (
    advance_state,
    initial_state,
    policy_for_model,
)
from tok.stats import SavingsTracker

from ._cli_support import (
    bridge_url,
    console,
    get_running_bridge_pid,
    interaction_quality_rows,
    memory_root,
    msg_text,
    render_stats_panel,
    runtime_verdict,
    savings_headline,
    savings_style,
    session_recommendation,
    session_signals_text,
    session_status_rows,
    status_border,
)
from ._gate import (
    evaluate_expected_failure,
    fixture_usage_weight,
    gate_release_summary,
    infer_fixture_format,
    is_common_path_fixture,
    load_fixture_files,
    load_gate_config,
    load_session_trend,
    read_fixture,
)


def stats_command(
    session: bool = False,
    total: bool = False,
    breakdown: bool = False,
    trends: bool = False,
    last_session: bool = False,
    recent: int | None = None,
    since: str | None = None,
    window: int = 5,
    reset: bool = False,
) -> None:
    """Show token savings and fallback state."""
    tracker = SavingsTracker()
    if reset:
        tracker.reset_ledger()
        console.print("[green]Lifetime stats have been reset.[/green]")
        return
    session_summary = tracker.session_summary()
    lifetime_summary = tracker.lifetime_summary()
    last_completed = tracker.last_session_summary()
    recent_completed = tracker.recent_summary(recent) if recent else None
    since_completed = tracker.since_summary(since) if since else None

    port = int(os.getenv("TOK_BRIDGE_PORT", "9090"))
    pid = get_running_bridge_pid(port)
    health_payload = None
    if pid:
        try:
            import httpx

            resp = httpx.get(bridge_url(port, "/health"), timeout=2.0)
            if resp.status_code == 200:
                health_payload = resp.json()
        except Exception:
            pass

    if not total:
        if session_summary:
            pct = float(session_summary["savings_pct"])
            headline, headline_pct, subhead = savings_headline(session_summary)
            verdict, verdict_style = runtime_verdict(
                tok_active=not bool(session_summary["baseline_only"]),
                baseline_only=bool(session_summary["baseline_only"]),
                tokens_saved=int(session_summary["tokens_saved"]),
                session_quality=str(session_summary.get("session_quality", "clean")),
            )
            console.print(
                render_stats_panel(
                    "Current Session",
                    headline=f"{headline} • {headline_pct}",
                    headline_style=savings_style(pct),
                    subhead=f"{verdict} • {subhead}",
                    rows=[
                        *session_status_rows(
                            summary=session_summary,
                            tok_active=not bool(session_summary["baseline_only"]),
                            baseline_only=bool(session_summary["baseline_only"]),
                            session_quality=str(session_summary.get("session_quality", "clean")),
                            degradation_reason=str(session_summary.get("last_degradation_reason", "")),
                        ),
                        ("Calls", str(session_summary["calls"])),
                    ],
                    border_style=status_border(verdict_style),
                )
            )

            iq_rows = interaction_quality_rows(
                smoothness_score=int(health_payload.get("smoothness_score", 0)) if health_payload else None,
                labour_index=int(health_payload.get("labour_index", 0)) if health_payload else None,
                current_mode=str(health_payload.get("current_mode", "")) if health_payload else None,
                stream_instability_events=int(health_payload.get("stream_instability_events", 0))
                if health_payload
                else None,
                thinking_mutation_events=int(health_payload.get("thinking_mutation_events", 0))
                if health_payload
                else None,
                repeated_active_file_reads=int(health_payload.get("repeated_active_file_reads", 0))
                if health_payload
                else None,
                task_score=int(health_payload.get("task_score", 0)) if health_payload else None,
            )
            if iq_rows:
                console.print(
                    render_stats_panel(
                        "Interaction Quality",
                        headline="Session flow metrics",
                        headline_style="bold cyan",
                        subhead="Lower labour index = smoother session",
                        rows=iq_rows,
                        border_style="cyan",
                    )
                )
        elif not session and last_session is False:
            if last_completed:
                pct = float(last_completed["savings_pct"])
                headline, headline_pct, subhead = savings_headline(last_completed)
                console.print(
                    render_stats_panel(
                        "Last Completed Session",
                        headline=f"{headline} • {headline_pct}",
                        headline_style=savings_style(pct),
                        subhead=subhead,
                        rows=[
                            ("Date", str(last_completed["date"])),
                            ("Turns", str(last_completed["turns"])),
                            (
                                "Session quality",
                                str(last_completed.get("session_quality", "clean")),
                            ),
                            (
                                "Degradation reason",
                                str(last_completed.get("last_degradation_reason", "") or "none"),
                            ),
                            (
                                "With Tok vs without Tok",
                                f"{int(last_completed['actual_tokens']):,} / {int(last_completed['baseline_tokens']):,} tokens",
                            ),
                            (
                                "Cost",
                                f"${float(last_completed['actual_cost_usd']):.4f} / ${float(last_completed['baseline_cost_usd']):.4f}",
                            ),
                        ],
                        border_style="cyan",
                    )
                )
            else:
                console.print("[dim]No active session data[/dim]")
        elif not session:
            if last_session:
                if last_completed:
                    pct = float(last_completed["savings_pct"])
                    headline, headline_pct, subhead = savings_headline(last_completed)
                    console.print(
                        render_stats_panel(
                            "Last Completed Session",
                            headline=f"{headline} • {headline_pct}",
                            headline_style=savings_style(pct),
                            subhead=subhead,
                            rows=[
                                ("Date", str(last_completed["date"])),
                                ("Turns", str(last_completed["turns"])),
                                (
                                    "Session quality",
                                    str(last_completed.get("session_quality", "clean")),
                                ),
                                (
                                    "Degradation reason",
                                    str(last_completed.get("last_degradation_reason", "") or "none"),
                                ),
                                (
                                    "With Tok vs without Tok",
                                    f"{int(last_completed['actual_tokens']):,} / {int(last_completed['baseline_tokens']):,} tokens",
                                ),
                                (
                                    "Cost",
                                    f"${float(last_completed['actual_cost_usd']):.4f} / ${float(last_completed['baseline_cost_usd']):.4f}",
                                ),
                            ],
                            border_style="cyan",
                        )
                    )
                else:
                    console.print("[dim]No completed session data yet[/dim]")

        if breakdown:
            if os.getenv("TOK_DEBUG", "0") == "1":
                console.print(f"[dim]Session file: {tracker.savings_file}[/dim]")
            stats = tracker.load_stats()
            bd: dict[str, int] = {}
            for m in stats.get("models", {}).values():
                for k, v in m.get("type_breakdown", {}).items():
                    bd[k] = bd.get(k, 0) + v
            if bd:
                console.print("\n[bold]Compression breakdown (chars saved):[/bold]")
                for k, v in sorted(bd.items(), key=lambda x: -x[1]):
                    console.print(f"  {k:<16} {v:>10,} chars  (~{v // 4:,} tokens)")
            signals = tracker.behavior_signals()
            summary = tracker.behavior_summary()
            console.print(
                "\n[bold]Tok health:[/bold] "
                f"status={summary['status']}  "
                f"invisible_pressure={summary['invisible_pressure']}  "
                f"memory_lift={summary['memory_lift']}"
            )
            input_saved = tracker.session_input_saved_tokens()
            console.print(f"[bold]Request-side savings:[/bold] input_saved_tokens={input_saved:,}")
            if signals:
                console.print("\n[bold]Behavior signals:[/bold]")
                for k, v in sorted(signals.items(), key=lambda x: (-x[1], x[0])):
                    console.print(f"  {k:<22} {v:>6}")

    if not session:
        if recent is not None:
            if recent_completed:
                pct = float(recent_completed["savings_pct"])
                headline, headline_pct, subhead = savings_headline(recent_completed)
                console.print(
                    render_stats_panel(
                        f"Recent Sessions ({int(recent_completed['sessions'])})",
                        headline=f"{headline} • {headline_pct}",
                        headline_style=savings_style(pct),
                        subhead=subhead,
                        rows=[
                            (
                                "Date range",
                                f"{recent_completed['date_start']} → {recent_completed['date_end']}",
                            ),
                            ("Turns", str(recent_completed["turns"])),
                            (
                                "With Tok vs without Tok",
                                f"{int(recent_completed['actual_tokens']):,} / {int(recent_completed['baseline_tokens']):,} tokens",
                            ),
                            (
                                "Cost",
                                f"${float(recent_completed['actual_cost_usd']):.4f} / ${float(recent_completed['baseline_cost_usd']):.4f}",
                            ),
                        ],
                        border_style="green" if pct >= 15 else "yellow",
                    )
                )
            else:
                console.print(f"[dim]No completed session data in the last {recent} sessions[/dim]")

        if since is not None:
            if since_completed:
                pct = float(since_completed["savings_pct"])
                headline, headline_pct, subhead = savings_headline(since_completed)
                console.print(
                    render_stats_panel(
                        str(since_completed["label"]),
                        headline=f"{headline} • {headline_pct}",
                        headline_style=savings_style(pct),
                        subhead=subhead,
                        rows=[
                            ("Sessions", str(since_completed["sessions"])),
                            (
                                "Date range",
                                f"{since_completed['date_start']} → {since_completed['date_end']}",
                            ),
                            ("Turns", str(since_completed["turns"])),
                            (
                                "With Tok vs without Tok",
                                f"{int(since_completed['actual_tokens']):,} / {int(since_completed['baseline_tokens']):,} tokens",
                            ),
                            (
                                "Cost",
                                f"${float(since_completed['actual_cost_usd']):.4f} / ${float(since_completed['baseline_cost_usd']):.4f}",
                            ),
                        ],
                        border_style="green" if pct >= 15 else "yellow",
                    )
                )
            else:
                console.print(f"[dim]No completed session data since {since}[/dim]")

        if lifetime_summary:
            pct = float(lifetime_summary["savings_pct"])
            headline, headline_pct, subhead = savings_headline(lifetime_summary)
            console.print(
                render_stats_panel(
                    "Lifetime",
                    headline=f"{headline} • {headline_pct}",
                    headline_style=savings_style(pct),
                    subhead=subhead,
                    rows=[
                        ("Sessions", str(lifetime_summary["sessions"])),
                        ("Turns", str(lifetime_summary["total_turns"])),
                        (
                            "With Tok vs without Tok",
                            f"{int(lifetime_summary['actual_tokens']):,} / {int(lifetime_summary['baseline_tokens']):,} tokens",
                        ),
                        (
                            "Cost",
                            f"${float(lifetime_summary['actual_cost_usd']):.4f} / ${float(lifetime_summary['baseline_cost_usd']):.4f}",
                        ),
                        (
                            "Fallbacks",
                            str(int(lifetime_summary["fallback_count"])),
                        ),
                        (
                            "Baseline-only requests",
                            str(int(lifetime_summary["baseline_only_requests"])),
                        ),
                    ],
                    border_style="blue",
                )
            )
            if trends:
                trend = tracker.trend_summary(recent_sessions=window)
                console.print(
                    "[bold]Trend:[/bold] "
                    f"sessions={trend['sessions_considered']}  "
                    f"direction={trend['direction']}  "
                    f"avg_savings={trend['avg_savings_pct']}%  "
                    f"avg_pressure={trend['avg_invisible_pressure']}  "
                    f"avg_saved_tokens={trend['avg_tokens_saved']}"
                )
                if trend["sessions_considered"] >= 2:
                    console.print(
                        "[bold]Velocity:[/bold] "
                        f"savings={trend['savings_velocity']:+.2f}/session  "
                        f"pressure={trend['pressure_velocity']:+.2f}/session  "
                        f"memory_lift={trend['memory_lift_velocity']:+.2f}/session"
                    )
        elif not total:
            console.print("[dim]No lifetime data yet[/dim]")


def replay_command(
    session_file: str,
    cost_per_mtok: float = 3.0,
    gate: bool = False,
) -> None:
    """Replay a captured session to measure compression savings offline."""
    import json as _json
    from pathlib import Path as _Path

    from tok.compression import compress_tool_results

    try:
        import tiktoken as _tiktoken

        _enc = _tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            return len(_enc.encode(text))

    except Exception:

        def _count(text: str) -> int:
            return len(text) // 4

    p = _Path(session_file)
    if not p.exists():
        console.print(f"[red]File not found: {session_file}[/red]")
        raise typer.Exit(1)

    meta_path = p.with_suffix(p.suffix + ".meta.json")
    replay_meta = None
    if meta_path.exists():
        replay_meta = _json.loads(meta_path.read_text())

    replay_model = str((replay_meta or {}).get("model", ""))
    replay_policy = policy_for_model(replay_model) if replay_model else None
    replay_state = initial_state(replay_policy) if replay_policy else None

    totals: dict[str, list[int]] = {}
    history_before = 0
    history_after = 0
    history_turns = 0
    file_cache: dict[str, tuple[str, str, float] | tuple[str, str] | tuple[str]] = {}
    behavior_totals: dict[str, int] = {}

    lines_read = 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        lines_read += 1
        messages = record.get("messages", [])
        if not messages:
            continue

        id_to_context = build_tool_use_id_to_context(messages)
        for key, value in collect_behavior_signals(messages, id_to_context).items():
            behavior_totals[key] = behavior_totals.get(key, 0) + value

        before = sum(_count(msg_text(m)) for m in messages)

        import copy

        msgs_copy = copy.deepcopy(messages)
        compression_level = "balanced"
        history_profile = None
        if replay_policy is not None and replay_state is not None:
            compression_level = replay_policy.tool_levels[replay_state.mode]
            history_profile = replay_policy.history_profiles[replay_state.mode]
        msgs_copy, bd = compress_tool_results(
            msgs_copy,
            result_cache=file_cache,
            tool_use_id_to_context=id_to_context,
            compression_level=compression_level,
        )

        for kind, chars in bd.items():
            toks = chars // 4
            if kind not in totals:
                totals[kind] = [0, 0, 0]
            totals[kind][0] += 1
            totals[kind][1] += toks
            totals[kind][2] += 0

        from tok.compression import compress_history

        recent_msgs, tok_state = compress_history(
            msgs_copy,
            keep_turns=2,
            profile=history_profile,
            prune_tool_results=True,
        )
        if tok_state:
            after_history = sum(_count(msg_text(m)) for m in recent_msgs) + _count(tok_state)
            history_turns += 1
            history_before += before
            history_after += after_history
        if replay_policy is not None and replay_state is not None:
            replay_state = advance_state(replay_policy, replay_state, behavior_totals)

    if lines_read == 0:
        console.print("[yellow]No records found in file.[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Replay: {lines_read} turns from {session_file}[/bold]\n")
    console.print(f"{'Content Type':<16} | {'Turns':>5} | {'Before':>12} | {'Saved':>10} | {'%':>5}")
    console.print("-" * 58)

    grand_before = 0
    grand_saved = 0

    for kind, (turns, before_tok, _) in sorted(totals.items(), key=lambda x: -x[1][1]):
        saved = before_tok
        pct = 100.0
        console.print(f"{kind:<16} | {turns:>5} | {before_tok:>12,} | {saved:>10,} | {pct:>4.0f}%")
        grand_before += before_tok
        grand_saved += saved

    console.print("-" * 58)
    if history_turns > 0:
        history_saved = max(0, history_before - history_after)
        history_pct = history_saved / history_before * 100 if history_before > 0 else 0.0
        console.print(
            f"{'history':<16} | {history_turns:>5} | {history_before:>12,} | {history_saved:>10,} | {history_pct:>4.1f}%"
        )
        grand_before += history_before
        grand_saved += history_saved
        console.print("-" * 58)
    grand_pct = grand_saved / grand_before * 100 if grand_before > 0 else 0.0
    console.print(f"{'TOTAL':<16} | {'':>5} | {grand_before:>12,} | {grand_saved:>10,} | {grand_pct:>4.1f}%")

    without_tok = grand_before / 1_000_000 * cost_per_mtok
    with_tok = (grand_before - grand_saved) / 1_000_000 * cost_per_mtok
    console.print(f"\n[dim]At ${cost_per_mtok:.2f}/MTok input:[/dim]")
    console.print(f"  Without tok:  ${without_tok:.4f}")
    console.print(f"  With tok:     ${with_tok:.4f}")
    console.print(f"  [green]Saved:        ${without_tok - with_tok:.4f}  ({grand_pct:.1f}%)[/green]")
    if behavior_totals:
        invisible_pressure = calculate_invisible_pressure(behavior_totals)
        console.print(f"\n[bold]Behavior replay:[/bold] invisible_pressure={invisible_pressure}")
        for key, value in sorted(behavior_totals.items(), key=lambda x: (-x[1], x[0])):
            console.print(f"  {key:<22} {value:>6}")
    else:
        invisible_pressure = 0

    if gate:
        if replay_meta is None:
            raise typer.Exit(1)
        gate_result = evaluate_replay_gate(
            replay_meta,
            savings_pct=grand_pct,
            behavior_signals=behavior_totals,
        )
        if not gate_result.passed:
            console.print(f"[red]Replay gate failed:[/red] {', '.join(gate_result.failed_checks)}")
            raise typer.Exit(1)


def _safe_env_flag(name: str) -> str:
    value = os.getenv(name)
    if not value:
        return "unset"
    return "set"


def _tok_version() -> str:
    try:
        from importlib import metadata

        return metadata.version("tok-protocol")
    except Exception:
        return "unknown"


def doctor_command(*, verbose: bool = False, report: bool = False) -> None:
    """Check bridge health and runtime contract conformance."""
    console.print("[bold]Tok Doctor — Runtime Health Check[/bold]")
    console.print("=" * 52)

    issues = False
    port = int(os.getenv("TOK_BRIDGE_PORT", "9090"))
    pid = get_running_bridge_pid(port)
    tracker = SavingsTracker()
    session_summary = tracker.session_summary()
    if pid:
        console.print(f"[green]✅ Bridge process: PID {pid}[/green]")
        try:
            import httpx

            resp = httpx.get(bridge_url(port, "/health"), timeout=2.0)
            if resp.status_code == 200:
                console.print(f"[green]✅ Health endpoint reachable on :{port}[/green]")
                payload = resp.json()
                baseline_only = bool(payload.get("baseline_only"))
                mode = str(payload.get("mode", "unknown"))
                fallback_count = int(payload.get("fallback_count", 0))
                tokens_saved = (
                    int(session_summary["tokens_saved"])
                    if session_summary
                    else int(payload.get("session_tokens_saved", 0))
                )
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
                session_view_summary: dict[str, int | float | str] = dict(session_summary or {})
                session_view_summary.update(
                    {
                        "actual_tokens": int(payload.get("actual_tokens", 0)),
                        "baseline_tokens": int(payload.get("baseline_tokens", 0)),
                        "tokens_saved": int(payload.get("session_tokens_saved", 0)),
                        "savings_pct": float(payload.get("session_savings_pct", 0.0)),
                        "actual_cost_usd": float(payload.get("actual_cost_usd", 0.0)),
                        "baseline_cost_usd": float(payload.get("baseline_cost_usd", 0.0)),
                        "cost_saved_usd": float(payload.get("cost_saved_usd", 0.0)),
                        "session_quality": str(payload.get("session_quality", "clean")),
                        "last_degradation_reason": str(payload.get("last_degradation_reason", "")),
                        "preflight_block_original_payload_count": int(
                            payload.get("preflight_block_original_payload_count", 0)
                        ),
                        "preflight_block_rewritten_payload_count": int(
                            payload.get("preflight_block_rewritten_payload_count", 0)
                        ),
                        "stream_recovery_empty_success_count": int(
                            payload.get("stream_recovery_empty_success_count", 0)
                        ),
                        "stream_recovery_read_error_count": int(payload.get("stream_recovery_read_error_count", 0)),
                        "request_policy_held_by_recovery_count": int(
                            payload.get("request_policy_held_by_recovery_count", 0)
                        ),
                    }
                )
                console.print(
                    render_stats_panel(
                        "Current Session",
                        headline=f"{headline} • {headline_pct}",
                        headline_style=(
                            savings_style(float(session_summary["savings_pct"])) if session_summary else "bold yellow"
                        ),
                        subhead=f"{verdict} • {subhead}",
                        rows=session_status_rows(
                            summary=session_view_summary,
                            tok_active=True,
                            baseline_only=baseline_only,
                            mode=mode,
                            fallback_count=fallback_count,
                            session_quality=str(payload.get("session_quality", "clean")),
                            degradation_reason=str(payload.get("last_degradation_reason", "")),
                            session_signals=session_signals_text(payload),
                        ),
                        border_style=status_border(verdict_style),
                    )
                )

                iq_rows = interaction_quality_rows(
                    smoothness_score=int(payload.get("smoothness_score", 0)),
                    labour_index=int(payload.get("labour_index", 0)),
                    current_mode=str(payload.get("current_mode", "")),
                    stream_instability_events=int(payload.get("stream_instability_events", 0)),
                    thinking_mutation_events=int(payload.get("thinking_mutation_events", 0)),
                    repeated_active_file_reads=int(payload.get("repeated_active_file_reads", 0)),
                    task_score=int(payload.get("task_score", 0)),
                )
                if iq_rows:
                    console.print(
                        render_stats_panel(
                            "Interaction Quality",
                            headline="Session flow metrics",
                            headline_style="bold cyan",
                            subhead="Lower labour index = smoother session",
                            rows=iq_rows,
                            border_style="cyan",
                        )
                    )
                if baseline_only:
                    console.print(
                        "[yellow]⚠️ Tok verdict:[/yellow] bridge is alive but the current session has degraded to baseline."
                    )
                    issues = True
                elif mode == "baseline":
                    console.print(
                        "[yellow]⚠️ Tok verdict:[/yellow] bridge is running in baseline mode, so compression is disabled by default."
                    )
                elif tokens_saved > 0:
                    console.print(
                        "[green]✅ Tok verdict:[/green] compression is active and saving tokens on the current session."
                    )
                else:
                    console.print(
                        "[yellow]⚠️ Tok verdict:[/yellow] bridge is healthy, but no current-session savings are visible yet."
                    )
                console.print(
                    f"[bold]Recommendation:[/bold] {session_recommendation(baseline_only=baseline_only, session_quality=str(payload.get('session_quality', 'clean'))).split(': ', 1)[1]}"
                )
            else:
                console.print(f"[red]❌ Health endpoint responded {resp.status_code} on :{port}[/red]")
                issues = True
        except Exception as exc:
            console.print(f"[red]❌ Unable to reach health endpoint on :{port} ({exc.__class__.__name__})[/red]")
            issues = True
    else:
        console.print("[red]❌ Bridge process not running[/red]")
        issues = True

    memory_dir = memory_root()
    structured_path = memory_dir / "bridge_memory.tok"
    fallback_path = memory_dir / "memory.tok"

    if not memory_dir.exists():
        console.print(f"[yellow]⚠️ Memory directory not initialized: {memory_dir}[/yellow]")
        issues = True
    elif structured_path.exists() and structured_path.stat().st_size > 0:
        console.print(f"[green]✅ Structured memory present: {structured_path}[/green]")
    elif fallback_path.exists() and fallback_path.stat().st_size > 0:
        console.print(f"[yellow]⚠️ Structured memory missing; wire fallback in use ({fallback_path})[/yellow]")
        issues = True
    else:
        console.print(f"[red]❌ No bridge memory files found in {memory_dir}[/red]")
        issues = True
    signals = tracker.behavior_signals()
    structured_hits = signals.get("cold_start_structured_memory", 0)
    fallback_hits = signals.get("cold_start_wire_fallback", 0)
    console.print(f"[bold]Cold-start signals:[/bold] structured={structured_hits} fallback={fallback_hits}")
    if fallback_hits > structured_hits:
        console.print("[yellow]⚠️ Wire fallback exceeded structured memory — check bridge state[/yellow]")
        issues = True

    if verbose and signals:
        console.print("\n[bold]Behavior signals (session):[/bold]")
        for key, value in sorted(signals.items(), key=lambda kv: (-kv[1], kv[0])):
            console.print(f"  {key:<32} {value:>4}")

    if report:
        collector_db = os.getenv("TOK_COLLECTOR_DB", "telemetry.db")
        report_lines = [
            "Tok Doctor report (safe to share)",
            f"tok_version={_tok_version()}",
            f"python={sys.version.split()[0]}",
            f"platform={platform.platform()}",
            f"cwd={Path.cwd()}",
            f"bridge_port={port}",
            f"bridge_pid={pid or 'none'}",
            f"memory_root={memory_dir}",
            f"memory_structured={'present' if structured_path.exists() else 'missing'}",
            f"memory_fallback={'present' if fallback_path.exists() else 'missing'}",
            f"collector_db={collector_db}",
            f"env_OPENAI_API_KEY={_safe_env_flag('OPENAI_API_KEY')}",
            f"env_OPENROUTER_API_KEY={_safe_env_flag('OPENROUTER_API_KEY')}",
            f"env_TOK_PROJECT_DIR={_safe_env_flag('TOK_PROJECT_DIR')}",
            f"env_TOK_COLLECTOR_DB={_safe_env_flag('TOK_COLLECTOR_DB')}",
            f"env_TOK_MODE={os.getenv('TOK_MODE', 'unset')}",
        ]
        console.print("\n[bold]Report:[/bold]")
        console.print("```")
        console.print("\n".join(str(line) for line in report_lines))
        console.print("```")

    if issues:
        console.print("\n[red]Doctor found issues — see above for remediation.[/red]")
        raise typer.Exit(1)

    console.print("\n[green]✅ All checks passed — runtime contract healthy.[/green]")


def gate_check_command(
    fixtures_dir: Path,
    fixtures: Path | None = None,
    export: Path | None = None,
    config: Path | None = None,
    continue_on_error: bool = False,
    fixture_set: str | None = None,
    emit_metrics: Path | None = None,
    stability_dir: Path | None = None,
    frontier_report: Path | None = None,
    benchmark_report: Path | None = None,
    required_benchmarks: str = "coding-loop-5,research-loop-5",
) -> None:
    """Run gate checks over a directory of replay fixtures."""
    if emit_metrics:
        export = emit_metrics

    gate_config = load_gate_config(config)
    tracker = SavingsTracker()
    trend_info = load_session_trend(tracker)

    if fixture_set is not None:
        from tok.cli._gate import select_fixture_set

        fixture_names = select_fixture_set(fixture_set)
        fixture_files = []
        for name in fixture_names:
            p = fixtures_dir / name
            if not p.exists() and not name.endswith(".jsonl"):
                p = fixtures_dir / (name + ".jsonl")
            fixture_files.append(p)
    elif fixtures is not None:
        try:
            data = json.loads(fixtures.read_text())
            fixture_names = data.get("fixtures", [])
            fixture_files = []
            for name in fixture_names:
                p = fixtures_dir / name
                if not p.exists() and not name.endswith(".jsonl"):
                    p = fixtures_dir / (name + ".jsonl")
                fixture_files.append(p)
        except Exception as exc:
            msg = f"Unable to read fixtures JSON: {exc}"
            raise typer.BadParameter(msg) from exc
    else:
        fixture_files = load_fixture_files(fixtures_dir)

    if not fixture_files:
        console.print(f"[yellow]No fixtures found in {fixtures_dir}[/yellow]")
        raise typer.Exit(1)

    def create_gate_table(title: str) -> Table:
        t = Table(title=title)
        t.add_column("Fixture", justify="left")
        t.add_column("Status", justify="center")
        t.add_column("Savings", justify="right")
        t.add_column("Pressure", justify="right")
        t.add_column("Failures", justify="left")
        return t

    perf_table = create_gate_table("Replay Performance (Compression)")
    conf_table = create_gate_table("Replay Conformance (Drift & Protocol)")

    results: list[dict[str, Any]] = []
    failed = 0

    for fixture_path in fixture_files:
        fixture_name = fixture_path.stem
        try:
            from tok.utils.replay_metrics import analyze_replay_fixture

            metrics = analyze_replay_fixture(fixture_path)
            records, meta = read_fixture(fixture_path)
            if not meta:
                msg = "Missing .meta.json next to fixture"
                raise ValueError(msg)

            behavior = metrics.behavior_totals
            actual_savings = metrics.savings_pct
            required_savings = meta.get("min_savings_pct", 0.0)

            gate_result = evaluate_replay_gate(
                meta,
                savings_pct=actual_savings,
                behavior_signals=behavior,
            )

            failures = list(gate_result.failed_checks)
            passed = not failures
            if meta.get("expected_failure", False):
                passed, failures = evaluate_expected_failure(
                    meta,
                    actual_failures=failures,
                    behavior_signals=behavior,
                )
            status = "✅ PASS"
            if not passed:
                status = "❌ FAIL"
                failed += 1

            target_table = perf_table if required_savings > 0 else conf_table
            target_table.add_row(
                fixture_name,
                status,
                f"{actual_savings:.1f}%",
                str(gate_result.invisible_pressure),
                ", ".join(failures) if failures else "None",
            )

            results.append(
                {
                    "fixture": fixture_name,
                    "passed": passed,
                    "savings_pct": actual_savings,
                    "required_savings_pct": required_savings,
                    "invisible_pressure": gate_result.invisible_pressure,
                    "failed_checks": failures,
                    "behavior_signals": behavior,
                    "records": len(records),
                    "fixture_format": infer_fixture_format(records),
                    "fixture_kind": str(meta.get("fixture_kind", "")),
                    "provenance": str(meta.get("provenance", "")),
                    "common_path": is_common_path_fixture(fixture_name, meta),
                    "usage_weight": fixture_usage_weight(fixture_name, meta),
                }
            )
            if not passed and not continue_on_error:
                break
        except Exception as exc:
            failed += 1
            conf_table.add_row(fixture_name, "❌ ERROR", "-", "-", str(exc))
            results.append(
                {
                    "fixture": fixture_name,
                    "passed": False,
                    "error": str(exc),
                }
            )
            if not continue_on_error:
                break

    if perf_table.row_count:
        console.print(perf_table)
    if conf_table.row_count:
        console.print(conf_table)

    common_path_rows = [row for row in results if row.get("common_path") and "error" not in row]
    common_path_summary = {
        "fixtures": len(common_path_rows),
        "total_weight": round(
            sum(float(row.get("usage_weight", 0.0)) for row in common_path_rows),
            1,
        ),
    }

    release_summary = gate_release_summary(
        results,
        tracker,
        trend_info,
    )
    if trend_info is not None:
        console.print(f"[bold]Set Trend:[/bold] status={trend_info.get('status', 'clean')}")
    required_fixture_names = gate_config.get("required_fixtures", []) if gate_config else []
    required_benchmark_list = [item.strip() for item in required_benchmarks.split(",") if item.strip()]
    stability_check = None
    if stability_dir is not None:
        from tok.testing.live_benchmark import check_stability_artifacts

        stability_check = check_stability_artifacts(stability_dir, required_benchmark_list)
    frontier_check = None
    if frontier_report is not None:
        from tok.testing.frontier import check_frontier_report

        frontier_check = check_frontier_report(frontier_report)
        release_summary["frontier_release_profile"] = str(frontier_check.get("release_profile", "baseline"))
        release_summary["frontier_status"] = "pass" if frontier_check.get("passed", False) else "fail"
        release_summary["frontier_probe_present"] = bool(frontier_check.get("openrouter_probe_present", False))
    benchmark_check = None
    if benchmark_report is not None:
        from tok.testing.benchmark_suite import check_benchmark_report

        benchmark_check = check_benchmark_report(benchmark_report)
        release_summary["benchmark_headline_lane"] = str(benchmark_check.get("headline_lane", ""))
        release_summary["benchmark_status"] = "pass" if benchmark_check.get("passed", False) else "fail"

    if export is not None:
        payload: dict[str, Any] = {
            "results": results,
            "required_fixtures": required_fixture_names,
            "release_summary": release_summary,
            "common_path_summary": common_path_summary,
        }
        if stability_check is not None:
            payload["stability_check"] = stability_check
        if frontier_check is not None:
            payload["frontier_check"] = frontier_check
        if benchmark_check is not None:
            payload["benchmark_check"] = benchmark_check
        export.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]Wrote gate results:[/green] {export}")

    if failed:
        raise typer.Exit(1)

    if stability_check is not None:
        failing_benchmarks = [name for name, row in stability_check.items() if not row.get("passed", False)]
        if failing_benchmarks:
            console.print(
                "[red]STABILITY FAIL[/red] "
                + ", ".join(
                    f"{name} ({stability_check[name].get('reason', 'criteria_failed')})" for name in failing_benchmarks
                )
            )
            console.print("[red]Stability gate: FAIL[/red]")
            console.print("[red]Stability artifact check failed:[/red] " + ", ".join(failing_benchmarks))
            raise typer.Exit(1)
        console.print("[green]STABILITY PASS[/green]")
        console.print("[green]Stability gate: PASS[/green]")

    if frontier_check is not None:
        if not frontier_check.get("passed", False):
            reason = frontier_check.get("reason", "criteria_failed")
            profile = frontier_check.get("release_profile", "baseline")
            console.print(f"[red]FRONTIER FAIL[/red] profile={profile} ({reason})")
            console.print("[red]Compression frontier gate: FAIL[/red]")
            raise typer.Exit(1)
        console.print(f"[green]FRONTIER PASS[/green] profile={frontier_check.get('release_profile', 'unknown')}")
        console.print("[green]Compression frontier gate: PASS[/green]")

    if benchmark_check is not None:
        if not benchmark_check.get("passed", False):
            reason = benchmark_check.get("reason", "headline_consistency_failed")
            console.print(
                f"[red]BENCHMARK FAIL[/red] lane={benchmark_check.get('headline_lane', 'unknown')} ({reason})"
            )
            console.print("[red]Production benchmark gate: FAIL[/red]")
            raise typer.Exit(1)
        console.print(f"[green]BENCHMARK PASS[/green] lane={benchmark_check.get('headline_lane', 'unknown')}")
        console.print("[green]Production benchmark gate: PASS[/green]")
