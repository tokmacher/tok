from __future__ import annotations

"""Release-facing CLI helper implementations."""

import json
import os
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from ..runtime.pipeline.response_handling import evaluate_replay_gate
from ..runtime.pipeline.tool_processing import (
    build_tool_use_id_to_context,
    collect_behavior_signals,
)
from ..runtime.policy.semantic_validation import calculate_invisible_pressure
from ..runtime.policy.smart_policy import (
    advance_state,
    initial_state,
    policy_for_model,
)
from ..stats import SavingsTracker
from . import _msg_text
from ._cli_support import (
    _get_running_bridge_pid,
    _memory_root,
    _render_stats_panel,
    _runtime_verdict,
    _savings_headline,
    _savings_style,
    _session_recommendation,
    _session_signals_text,
    _session_status_rows,
    _status_border,
    console,
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
) -> None:
    """Show token savings and fallback state."""
    tracker = SavingsTracker()
    session_summary = tracker.session_summary()
    lifetime_summary = tracker.lifetime_summary()
    last_completed = tracker.last_session_summary()
    recent_completed = tracker.recent_summary(recent) if recent else None
    since_completed = tracker.since_summary(since) if since else None

    if not total:
        if session_summary:
            pct = float(session_summary["savings_pct"])
            headline, headline_pct, subhead = _savings_headline(
                session_summary
            )
            verdict, verdict_style = _runtime_verdict(
                tok_active=not bool(session_summary["baseline_only"]),
                baseline_only=bool(session_summary["baseline_only"]),
                tokens_saved=int(session_summary["tokens_saved"]),
                session_quality=str(
                    session_summary.get("session_quality", "clean")
                ),
            )
            console.print(
                _render_stats_panel(
                    "Current Session",
                    headline=f"{headline} • {headline_pct}",
                    headline_style=_savings_style(pct),
                    subhead=f"{verdict} • {subhead}",
                    rows=_session_status_rows(
                        summary=session_summary,
                        tok_active=not bool(session_summary["baseline_only"]),
                        baseline_only=bool(session_summary["baseline_only"]),
                        session_quality=str(
                            session_summary.get("session_quality", "clean")
                        ),
                        degradation_reason=str(
                            session_summary.get("last_degradation_reason", "")
                        ),
                    )
                    + [("Calls", str(session_summary["calls"]))],
                    border_style=_status_border(verdict_style),
                )
            )
        elif not session and last_session is False:
            if last_completed:
                pct = float(last_completed["savings_pct"])
                headline, headline_pct, subhead = _savings_headline(
                    last_completed
                )
                console.print(
                    _render_stats_panel(
                        "Last Completed Session",
                        headline=f"{headline} • {headline_pct}",
                        headline_style=_savings_style(pct),
                        subhead=subhead,
                        rows=[
                            ("Date", str(last_completed["date"])),
                            ("Turns", str(last_completed["turns"])),
                            (
                                "Session quality",
                                str(
                                    last_completed.get(
                                        "session_quality", "clean"
                                    )
                                ),
                            ),
                            (
                                "Degradation reason",
                                str(
                                    last_completed.get(
                                        "last_degradation_reason", ""
                                    )
                                    or "none"
                                ),
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
                    headline, headline_pct, subhead = _savings_headline(
                        last_completed
                    )
                    console.print(
                        _render_stats_panel(
                            "Last Completed Session",
                            headline=f"{headline} • {headline_pct}",
                            headline_style=_savings_style(pct),
                            subhead=subhead,
                            rows=[
                                ("Date", str(last_completed["date"])),
                                ("Turns", str(last_completed["turns"])),
                                (
                                    "Session quality",
                                    str(
                                        last_completed.get(
                                            "session_quality", "clean"
                                        )
                                    ),
                                ),
                                (
                                    "Degradation reason",
                                    str(
                                        last_completed.get(
                                            "last_degradation_reason", ""
                                        )
                                        or "none"
                                    ),
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
                console.print(
                    f"[dim]Session file: {tracker.savings_file}[/dim]"
                )
            stats = tracker.load_stats()
            bd: dict[str, int] = {}
            for m in stats.get("models", {}).values():
                for k, v in m.get("type_breakdown", {}).items():
                    bd[k] = bd.get(k, 0) + v
            if bd:
                console.print(
                    "\n[bold]Compression breakdown (chars saved):[/bold]"
                )
                for k, v in sorted(bd.items(), key=lambda x: -x[1]):
                    console.print(
                        f"  {k:<16} {v:>10,} chars  (~{v // 4:,} tokens)"
                    )
            signals = tracker.behavior_signals()
            summary = tracker.behavior_summary()
            console.print(
                "\n[bold]Tok health:[/bold] "
                f"status={summary['status']}  "
                f"invisible_pressure={summary['invisible_pressure']}  "
                f"memory_lift={summary['memory_lift']}"
            )
            input_saved = tracker.session_input_saved_tokens()
            console.print(
                f"[bold]Request-side savings:[/bold] input_saved_tokens={input_saved:,}"
            )
            if signals:
                console.print("\n[bold]Behavior signals:[/bold]")
                for k, v in sorted(
                    signals.items(), key=lambda x: (-x[1], x[0])
                ):
                    console.print(f"  {k:<22} {v:>6}")

    if not session:
        if recent is not None:
            if recent_completed:
                pct = float(recent_completed["savings_pct"])
                headline, headline_pct, subhead = _savings_headline(
                    recent_completed
                )
                console.print(
                    _render_stats_panel(
                        f"Recent Sessions ({int(recent_completed['sessions'])})",
                        headline=f"{headline} • {headline_pct}",
                        headline_style=_savings_style(pct),
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
                console.print(
                    f"[dim]No completed session data in the last {recent} sessions[/dim]"
                )

        if since is not None:
            if since_completed:
                pct = float(since_completed["savings_pct"])
                headline, headline_pct, subhead = _savings_headline(
                    since_completed
                )
                console.print(
                    _render_stats_panel(
                        str(since_completed["label"]),
                        headline=f"{headline} • {headline_pct}",
                        headline_style=_savings_style(pct),
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
                console.print(
                    f"[dim]No completed session data since {since}[/dim]"
                )

        if lifetime_summary:
            pct = float(lifetime_summary["savings_pct"])
            headline, headline_pct, subhead = _savings_headline(
                lifetime_summary
            )
            console.print(
                _render_stats_panel(
                    "Lifetime",
                    headline=f"{headline} • {headline_pct}",
                    headline_style=_savings_style(pct),
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
                            str(
                                int(lifetime_summary["baseline_only_requests"])
                            ),
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

    from ..compression import compress_tool_results

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
    file_cache: dict[str, tuple[str, str]] = {}
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
        for key, value in collect_behavior_signals(
            messages, id_to_context
        ).items():
            behavior_totals[key] = behavior_totals.get(key, 0) + value

        before = sum(_count(_msg_text(m)) for m in messages)

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

        from ..compression import compress_history

        recent_msgs, tok_state = compress_history(
            msgs_copy,
            keep_turns=2,
            profile=history_profile,
            prune_tool_results=True,
        )
        if tok_state:
            after_history = sum(
                _count(_msg_text(m)) for m in recent_msgs
            ) + _count(tok_state)
            history_turns += 1
            history_before += before
            history_after += after_history
        if replay_policy is not None and replay_state is not None:
            replay_state = advance_state(
                replay_policy, replay_state, behavior_totals
            )

    if lines_read == 0:
        console.print("[yellow]No records found in file.[/yellow]")
        raise typer.Exit(0)

    console.print(
        f"\n[bold]Replay: {lines_read} turns from {session_file}[/bold]\n"
    )
    console.print(
        f"{'Content Type':<16} | {'Turns':>5} | {'Before':>12} | {'Saved':>10} | {'%':>5}"
    )
    console.print("-" * 58)

    grand_before = 0
    grand_saved = 0

    for kind, (turns, before_tok, _) in sorted(
        totals.items(), key=lambda x: -x[1][1]
    ):
        saved = before_tok
        pct = 100.0
        console.print(
            f"{kind:<16} | {turns:>5} | {before_tok:>12,} | {saved:>10,} | {pct:>4.0f}%"
        )
        grand_before += before_tok
        grand_saved += saved

    console.print("-" * 58)
    if history_turns > 0:
        history_saved = max(0, history_before - history_after)
        history_pct = (
            history_saved / history_before * 100 if history_before > 0 else 0.0
        )
        console.print(
            f"{'history':<16} | {history_turns:>5} | {history_before:>12,} | {history_saved:>10,} | {history_pct:>4.1f}%"
        )
        grand_before += history_before
        grand_saved += history_saved
        console.print("-" * 58)
    grand_pct = grand_saved / grand_before * 100 if grand_before > 0 else 0.0
    console.print(
        f"{'TOTAL':<16} | {'':>5} | {grand_before:>12,} | {grand_saved:>10,} | {grand_pct:>4.1f}%"
    )

    without_tok = grand_before / 1_000_000 * cost_per_mtok
    with_tok = (grand_before - grand_saved) / 1_000_000 * cost_per_mtok
    console.print(f"\n[dim]At ${cost_per_mtok:.2f}/MTok input:[/dim]")
    console.print(f"  Without tok:  ${without_tok:.4f}")
    console.print(f"  With tok:     ${with_tok:.4f}")
    console.print(
        f"  [green]Saved:        ${without_tok - with_tok:.4f}  ({grand_pct:.1f}%)[/green]"
    )
    if behavior_totals:
        invisible_pressure = calculate_invisible_pressure(behavior_totals)
        console.print(
            f"\n[bold]Behavior replay:[/bold] invisible_pressure={invisible_pressure}"
        )
        for key, value in sorted(
            behavior_totals.items(), key=lambda x: (-x[1], x[0])
        ):
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
            console.print(
                f"[red]Replay gate failed:[/red] {', '.join(gate_result.failed_checks)}"
            )
            raise typer.Exit(1)


def doctor_command(verbose: bool = False) -> None:
    """Check bridge health and runtime contract conformance."""
    console.print("[bold]Tok Doctor — Runtime Health Check[/bold]")
    console.print("=" * 52)

    issues = False
    port = int(os.getenv("TOK_BRIDGE_PORT", "9090"))
    pid = _get_running_bridge_pid(port)
    tracker = SavingsTracker()
    session_summary = tracker.session_summary()
    if pid:
        console.print(f"[green]✅ Bridge process: PID {pid}[/green]")
        try:
            import httpx

            resp = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            if resp.status_code == 200:
                console.print(
                    f"[green]✅ Health endpoint reachable on :{port}[/green]"
                )
                payload = resp.json()
                baseline_only = bool(payload.get("baseline_only"))
                mode = str(payload.get("mode", "unknown"))
                fallback_count = int(payload.get("fallback_count", 0))
                tokens_saved = (
                    int(session_summary["tokens_saved"])
                    if session_summary
                    else int(payload.get("session_tokens_saved", 0))
                )
                verdict, verdict_style = _runtime_verdict(
                    tok_active=True,
                    baseline_only=baseline_only,
                    mode=mode,
                    tokens_saved=tokens_saved,
                    session_quality=str(
                        payload.get("session_quality", "clean")
                    ),
                )
                headline, headline_pct, subhead = _savings_headline(
                    session_summary,
                    savings_pct=float(payload.get("session_savings_pct", 0.0)),
                    tokens_saved=int(payload.get("session_tokens_saved", 0)),
                )
                console.print(
                    _render_stats_panel(
                        "Current Session",
                        headline=f"{headline} • {headline_pct}",
                        headline_style=(
                            _savings_style(
                                float(session_summary["savings_pct"])
                            )
                            if session_summary
                            else "bold yellow"
                        ),
                        subhead=f"{verdict} • {subhead}",
                        rows=_session_status_rows(
                            summary=session_summary,
                            tok_active=True,
                            baseline_only=baseline_only,
                            mode=mode,
                            fallback_count=fallback_count,
                            session_quality=str(
                                payload.get("session_quality", "clean")
                            ),
                            degradation_reason=str(
                                payload.get("last_degradation_reason", "")
                            ),
                            session_signals=_session_signals_text(payload),
                        ),
                        border_style=_status_border(verdict_style),
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
                    f"[bold]Recommendation:[/bold] {_session_recommendation(baseline_only=baseline_only, session_quality=str(payload.get('session_quality', 'clean'))).split(': ', 1)[1]}"
                )
            else:
                console.print(
                    f"[red]❌ Health endpoint responded {resp.status_code} on :{port}[/red]"
                )
                issues = True
        except Exception as exc:
            console.print(
                f"[red]❌ Unable to reach health endpoint on :{port} ({exc.__class__.__name__})[/red]"
            )
            issues = True
    else:
        console.print("[red]❌ Bridge process not running[/red]")
        issues = True

    memory_dir = _memory_root()
    structured_path = memory_dir / "bridge_memory.tok"
    fallback_path = memory_dir / "memory.tok"

    if not memory_dir.exists():
        console.print(
            f"[yellow]⚠️ Memory directory not initialized: {memory_dir}[/yellow]"
        )
        issues = True
    elif structured_path.exists() and structured_path.stat().st_size > 0:
        console.print(
            f"[green]✅ Structured memory present: {structured_path}[/green]"
        )
    elif fallback_path.exists() and fallback_path.stat().st_size > 0:
        console.print(
            f"[yellow]⚠️ Structured memory missing; wire fallback in use ({fallback_path})[/yellow]"
        )
        issues = True
    else:
        console.print(
            f"[red]❌ No bridge memory files found in {memory_dir}[/red]"
        )
        issues = True
    signals = tracker.behavior_signals()
    structured_hits = signals.get("cold_start_structured_memory", 0)
    fallback_hits = signals.get("cold_start_wire_fallback", 0)
    console.print(
        f"[bold]Cold-start signals:[/bold] structured={structured_hits} fallback={fallback_hits}"
    )
    if fallback_hits > structured_hits:
        console.print(
            "[yellow]⚠️ Wire fallback exceeded structured memory — check bridge state[/yellow]"
        )
        issues = True

    if verbose and signals:
        console.print("\n[bold]Behavior signals (session):[/bold]")
        for key, value in sorted(
            signals.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            console.print(f"  {key:<32} {value:>4}")

    if issues:
        console.print(
            "\n[red]Doctor found issues — see above for remediation.[/red]"
        )
        raise typer.Exit(1)

    console.print(
        "\n[green]✅ All checks passed — runtime contract healthy.[/green]"
    )


def gate_check_command(
    fixtures_dir: Path,
    fixtures: Path | None = None,
    export: Path | None = None,
    config: Path | None = None,
    continue_on_error: bool = False,
    fixture_set: str | None = None,
    emit_metrics: Path | None = None,
    stability_dir: Path | None = None,
    required_benchmarks: str = "coding-loop-5,research-loop-5",
) -> None:
    """Run gate checks over a directory of replay fixtures."""
    if emit_metrics:
        export = emit_metrics

    gate_config = load_gate_config(config)
    tracker = SavingsTracker()
    trend_info = load_session_trend(tracker)

    if fixture_set is not None:
        from scripts.select_fixtures import select_fixtures

        fixture_names = select_fixtures(fixture_set)
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
            raise typer.BadParameter(
                f"Unable to read fixtures JSON: {exc}"
            ) from exc
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
            from ..utils.replay_metrics import analyze_replay_fixture

            metrics = analyze_replay_fixture(fixture_path)
            records, meta = read_fixture(fixture_path)
            if not meta:
                raise ValueError("Missing .meta.json next to fixture")

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

    common_path_rows = [
        row for row in results if row.get("common_path") and "error" not in row
    ]
    common_path_summary = {
        "fixtures": len(common_path_rows),
        "total_weight": round(
            sum(
                float(row.get("usage_weight", 0.0)) for row in common_path_rows
            ),
            1,
        ),
    }

    release_summary = gate_release_summary(
        results,
        tracker=tracker,
        trend_info=trend_info,
    )
    if trend_info is not None:
        console.print(
            "[bold]Set Trend:[/bold] "
            f"status={trend_info.get('status', 'clean')}"
        )
    required_fixture_names = (
        gate_config.get("required_fixtures", []) if gate_config else []
    )
    required_benchmark_list = [
        item.strip() for item in required_benchmarks.split(",") if item.strip()
    ]
    stability_check = None
    if stability_dir is not None:
        from ..testing.live_benchmark import check_stability_artifacts

        stability_check = check_stability_artifacts(
            stability_dir, required_benchmark_list
        )

    if export is not None:
        payload: dict[str, Any] = {
            "results": results,
            "required_fixtures": required_fixture_names,
            "release_summary": release_summary,
            "common_path_summary": common_path_summary,
        }
        if stability_check is not None:
            payload["stability_check"] = stability_check
        export.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]Wrote gate results:[/green] {export}")

    if failed:
        raise typer.Exit(1)

    if stability_check is not None:
        failing_benchmarks = [
            name
            for name, row in stability_check.items()
            if not row.get("passed", False)
        ]
        if failing_benchmarks:
            console.print(
                "[red]STABILITY FAIL[/red] "
                + ", ".join(
                    f"{name} ({stability_check[name].get('reason', 'criteria_failed')})"
                    for name in failing_benchmarks
                )
            )
            console.print("[red]Stability gate: FAIL[/red]")
            console.print(
                "[red]Stability artifact check failed:[/red] "
                + ", ".join(failing_benchmarks)
            )
            raise typer.Exit(1)
        console.print("[green]STABILITY PASS[/green]")
        console.print("[green]Stability gate: PASS[/green]")
