from __future__ import annotations

"""Run live benchmark frontier comparisons against an exported checkpoint."""

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


def _load_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        raise SystemExit("frontier checkpoint runner requires JSON on stdin")
    return json.loads(raw)


def main() -> None:
    payload = _load_payload()
    repo_root = Path(payload["repo_root"])
    sys.path.insert(0, str(repo_root / "src"))

    from tok.testing.live_benchmark import (  # type: ignore
        LiveBenchmarkRunner,
        compare_results,
        load_benchmark_definition,
    )

    profile = dict(payload["profile"])
    env = dict(profile.get("env", {}))
    previous = {key: None for key in env}
    import os

    for key in env:
        previous[key] = os.environ.get(key)
    os.environ.update(env)
    try:
        definition = load_benchmark_definition(str(payload["benchmark"]))
        with redirect_stdout(sys.stderr):
            runner = LiveBenchmarkRunner(
                model=str(payload["model"]),
                temperature=float(payload.get("temperature", 0.0)),
                max_tokens=int(payload.get("max_tokens", 300)),
                timeout=float(payload.get("timeout", 120.0)),
                pricing=payload.get("pricing"),
                provider_options=payload.get("provider_options"),
                api_key=payload.get("api_key"),
                api_base=payload.get("api_base"),
            )
            repeats = max(1, int(payload.get("repeats", 1)))
            turns = int(payload.get("turns", definition.default_turns))
            runs: list[dict[str, Any]] = []
            for _ in range(repeats):
                baseline = runner.run(definition, mode="baseline", turns=turns)
                candidate = runner.run(
                    definition,
                    mode=str(profile["mode"]),
                    turns=turns,
                )
                comparison = compare_results(baseline, candidate)
                runs.append(
                    {
                        "baseline": baseline.to_dict(),
                        "candidate": candidate.to_dict(),
                        "comparison": comparison.to_dict(),
                    }
                )
        sys.stdout.write(json.dumps({"runs": runs}))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
