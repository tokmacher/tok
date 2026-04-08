"""Generate token trajectory plots from live-benchmark JSON artifacts.

This is intentionally a lightweight, local analysis utility:
- Reads `*_baseline.json` and `*_tok-universal.json` produced by `tok dev live-benchmark --mode compare`.
- Plots per-turn and cumulative token trajectories to visualize "token growth" across turns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TurnUsage:
    turn: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _load_turn_usage(path: Path) -> list[TurnUsage]:
    payload = json.loads(path.read_text())
    turns = payload.get("turns") or []
    result: list[TurnUsage] = []
    for entry in turns:
        usage = (entry or {}).get("provider_usage") or {}
        result.append(
            TurnUsage(
                turn=_safe_int((entry or {}).get("turn")),
                prompt_tokens=_safe_int(usage.get("prompt_tokens")),
                completion_tokens=_safe_int(usage.get("completion_tokens")),
                total_tokens=_safe_int(usage.get("total_tokens")),
                latency_ms=_safe_float(usage.get("latency_ms")),
            )
        )
    return [t for t in result if t.turn > 0]


def _cumulative(values: list[int]) -> list[int]:
    total = 0
    out: list[int] = []
    for v in values:
        total += int(v)
        out.append(total)
    return out


def _write_csv(
    out_csv: Path,
    *,
    benchmark: str,
    baseline: list[TurnUsage],
    tok: list[TurnUsage],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    max_turn = max([t.turn for t in baseline] + [t.turn for t in tok] + [0])
    baseline_by_turn = {t.turn: t for t in baseline}
    tok_by_turn = {t.turn: t for t in tok}

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "benchmark",
                "turn",
                "mode",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "latency_ms",
                "cumulative_total_tokens",
            ]
        )
        for mode, mapping in (("baseline", baseline_by_turn), ("tok-universal", tok_by_turn)):
            totals = []
            rows = []
            for turn in range(1, max_turn + 1):
                t = mapping.get(turn)
                if not t:
                    continue
                totals.append(t.total_tokens)
                rows.append((turn, t))
            cumul = _cumulative(totals)
            for idx, (turn, t) in enumerate(rows):
                w.writerow(
                    [
                        benchmark,
                        turn,
                        mode,
                        t.prompt_tokens,
                        t.completion_tokens,
                        t.total_tokens,
                        round(t.latency_ms, 2),
                        cumul[idx],
                    ]
                )


def _plot(
    out_svg: Path,
    *,
    title: str,
    turns_baseline: list[TurnUsage],
    turns_tok: list[TurnUsage],
) -> None:
    def _series(turns: list[TurnUsage]) -> tuple[list[int], list[int], list[int]]:
        x = [t.turn for t in turns]
        per_turn = [t.total_tokens for t in turns]
        cumul = _cumulative(per_turn)
        return x, per_turn, cumul

    xb, yb, cb = _series(turns_baseline)
    xt, yt, ct = _series(turns_tok)

    # Dependency-free SVG (per-turn + cumulative stacked).
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    max_turn = max([*xb, *xt, 1])
    max_per_turn = max([*yb, *yt, 1])
    max_cumul = max([*cb, *ct, 1])

    width = 1000
    height = 700
    pad = 60
    gap = 40
    panel_h = (height - pad * 2 - gap) // 2

    def _sx(turn: int) -> float:
        if max_turn <= 1:
            return float(pad)
        return pad + (width - pad * 2) * ((turn - 1) / (max_turn - 1))

    def _sy(value: int, *, panel: int, vmax: int) -> float:
        top = pad + (panel * (panel_h + gap))
        bottom = top + panel_h
        if vmax <= 0:
            return float(bottom)
        return float(bottom - (panel_h * (value / vmax)))

    def _polyline(xs: list[int], ys: list[int], *, panel: int, vmax: int) -> str:
        return " ".join(f"{_sx(x):.1f},{_sy(y, panel=panel, vmax=vmax):.1f}" for x, y in zip(xs, ys, strict=False))

    def _nice_ticks(vmax: int) -> list[int]:
        if vmax <= 0:
            return [0]
        step = max(1, int(math.ceil(vmax / 4)))
        return [0, step, step * 2, step * 3, step * 4]

    def _grid(panel: int, vmax: int, label: str) -> str:
        lines = []
        top = pad + (panel * (panel_h + gap))
        bottom = top + panel_h
        left = pad
        right = width - pad
        lines.append(f'<text x="{left}" y="{top - 12}" font-size="14" font-family="monospace">{label}</text>')
        for t in _nice_ticks(vmax):
            y = _sy(t, panel=panel, vmax=vmax)
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#ddd" stroke-width="1" />')
            lines.append(
                f'<text x="{left - 10}" y="{y:.1f}" text-anchor="end" dominant-baseline="middle" '
                f'font-size="12" font-family="monospace" fill="#666">{t}</text>'
            )
        if panel == 1:
            for turn in range(1, max_turn + 1):
                x = _sx(turn)
                lines.append(f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom + 5}" stroke="#999" />')
                lines.append(
                    f'<text x="{x:.1f}" y="{bottom + 20}" text-anchor="middle" '
                    f'font-size="12" font-family="monospace" fill="#666">{turn}</text>'
                )
        return "\n".join(lines)

    baseline_color = "#1f77b4"
    tok_color = "#2ca02c"

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />')
    svg.append(f'<text x="{pad}" y="{pad - 25}" font-size="18" font-family="monospace">{title}</text>')
    svg.append(_grid(0, max_per_turn, "Total tokens (per turn)"))
    svg.append(_grid(1, max_cumul, "Cumulative total tokens"))
    svg.append(
        f'<polyline fill="none" stroke="{baseline_color}" stroke-width="3" '
        f'points="{_polyline(xb, yb, panel=0, vmax=max_per_turn)}" />'
    )
    svg.append(
        f'<polyline fill="none" stroke="{tok_color}" stroke-width="3" '
        f'points="{_polyline(xt, yt, panel=0, vmax=max_per_turn)}" />'
    )
    svg.append(
        f'<polyline fill="none" stroke="{baseline_color}" stroke-width="3" '
        f'points="{_polyline(xb, cb, panel=1, vmax=max_cumul)}" />'
    )
    svg.append(
        f'<polyline fill="none" stroke="{tok_color}" stroke-width="3" '
        f'points="{_polyline(xt, ct, panel=1, vmax=max_cumul)}" />'
    )
    lx = width - pad
    ly = pad - 18
    svg.append(
        f'<text x="{lx}" y="{ly}" text-anchor="end" font-size="13" font-family="monospace" fill="{baseline_color}">'
        "baseline</text>"
    )
    svg.append(
        f'<text x="{lx}" y="{ly + 18}" text-anchor="end" font-size="13" font-family="monospace" fill="{tok_color}">'
        "tok-universal</text>"
    )
    svg.append("</svg>")
    out_svg.write_text("\n".join(svg))


def _find_benchmarks(dir_path: Path) -> list[str]:
    names: set[str] = set()
    for path in dir_path.glob("*_baseline.json"):
        names.add(path.name[: -len("_baseline.json")])
    return sorted(names)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "artifact_dir",
        type=Path,
        help="Directory containing `*_baseline.json` and `*_tok-universal.json` artifacts.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for plots/CSVs (default: <artifact_dir>/plots).",
    )
    args = ap.parse_args()

    artifact_dir: Path = args.artifact_dir
    out_dir: Path = args.out or (artifact_dir / "plots")

    benchmarks = _find_benchmarks(artifact_dir)
    if not benchmarks:
        raise SystemExit(f"No `*_baseline.json` artifacts found under: {artifact_dir}")

    for benchmark in benchmarks:
        baseline_path = artifact_dir / f"{benchmark}_baseline.json"
        tok_path = artifact_dir / f"{benchmark}_tok-universal.json"
        if not baseline_path.exists() or not tok_path.exists():
            # Skip partially-written runs.
            continue

        baseline_turns = _load_turn_usage(baseline_path)
        tok_turns = _load_turn_usage(tok_path)
        if not baseline_turns or not tok_turns:
            continue

        out_svg = out_dir / f"{benchmark}_token_trajectory.svg"
        out_csv = out_dir / f"{benchmark}_token_trajectory.csv"
        _plot(out_svg, title=f"{benchmark}: token trajectory", turns_baseline=baseline_turns, turns_tok=tok_turns)
        _write_csv(out_csv, benchmark=benchmark, baseline=baseline_turns, tok=tok_turns)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
