#!/usr/bin/env python3

"""Run a bounded maintainer-facing release smoke sweep for Tok."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UV_RUN_PREFIX: tuple[str, ...] = ("uv", "run")
PYTHON_INLINE_PREFIX: tuple[str, ...] = (*UV_RUN_PREFIX, "python", "-c")


@dataclass(frozen=True)
class SmokeStep:
    name: str
    command: tuple[str, ...]


def _inline_python_step(name: str, statements: tuple[str, ...]) -> SmokeStep:
    return SmokeStep(name, (*PYTHON_INLINE_PREFIX, "; ".join(statements)))


def _pytest_step(name: str, *args: str) -> SmokeStep:
    return SmokeStep(name, (*UV_RUN_PREFIX, "pytest", *args))


IMPORT_CHECK = (
    "import tok.stats",
    "import tok.utils.savings_tracker",
    "import tok.universal_runtime",
    "import tok.runtime.core",
    "import tok.release_surface",
    "print('tok release smoke imports OK')",
)

# Release-surface contract gate (manifest + CLI declaration drift).
SURFACE_GATE_CHECK = (
    "from typer.testing import CliRunner",
    "from tok.cli import app",
    "from tok.release_surface import validate_release_surface",
    "import tok",
    "help_output = CliRunner().invoke(app, ['--help']).output",
    "failures = validate_release_surface(exported_names=tok.__all__, cli_help_output=help_output, root_app=app)",
    "print('surface_gate_failures=' + repr(failures))",
    "raise SystemExit(1 if failures else 0)",
)

# Boundary contract gate: malformed ingress fails locally without upstream execution.
VALIDATION_FAILURE_CHECK = (
    "import textwrap",
    """code = textwrap.dedent('''\
import asyncio
import json
import socket

import httpx
import uvicorn
from fastapi import FastAPI, Request

from tok.gateway import BridgeSession, create_app


class _UpstreamState:
    def __init__(self) -> None:
        self.call_count = 0
        self.lock = asyncio.Lock()

    async def increment(self) -> int:
        async with self.lock:
            self.call_count += 1
            return self.call_count

    async def get_count(self) -> int:
        async with self.lock:
            return self.call_count

    def reset(self) -> None:
        self.call_count = 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _create_synthetic_upstream(state: _UpstreamState) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok", "service": "validation-failure-upstream"}

    @app.post("/v1/messages")
    async def messages(request: Request) -> dict[str, object]:
        await state.increment()
        return {
            "type": "message",
            "content": [{"type": "text", "text": "upstream should not be reached"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    return app


async def _start_server(app: FastAPI, port: int, health_url: str) -> tuple[uvicorn.Server, asyncio.Task]:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    for _ in range(50):
        await asyncio.sleep(0.1)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(health_url)
                if resp.status_code == 200:
                    break
        except Exception:
            pass
    else:
        server.should_exit = True
        task.cancel()
        raise RuntimeError(f\"Server failed to start on port {port}\")

    return server, task


async def _stop_server(server: uvicorn.Server, task: asyncio.Task) -> None:
    server.should_exit = True
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except asyncio.TimeoutError:
        task.cancel()


async def _run() -> None:
    state = _UpstreamState()
    state.reset()

    upstream_port = _find_free_port()
    bridge_port = _find_free_port()
    upstream_base = f\"http://127.0.0.1:{upstream_port}\"

    upstream_app = _create_synthetic_upstream(state)
    upstream_server, upstream_task = await _start_server(
        upstream_app, upstream_port, f\"{upstream_base}/\"
    )

    bridge_session = BridgeSession(
        port=bridge_port,
        api_base=upstream_base,
        debug=False,
        fail_open=False,
    )
    bridge_app = create_app(bridge_session)
    bridge_server, bridge_task = await _start_server(
        bridge_app, bridge_port, f\"http://127.0.0.1:{bridge_port}/health\"
    )

    try:
        malformed_requests = [
            # Missing required field: omit model
            {
                \"max_tokens\": 16,
                \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}],
                \"stream\": False,
            },
            # Wrong type: messages is not a list
            {
                \"model\": \"claude-3-sonnet-20240229\",
                \"max_tokens\": 16,
                \"messages\": \"not-a-list\",
                \"stream\": False,
            },
        ]

        async with httpx.AsyncClient() as client:
            for body in malformed_requests:
                before = await state.get_count()

                resp = await client.post(
                    f\"http://127.0.0.1:{bridge_port}/v1/messages\",
                    json=body,
                    headers={\"x-api-key\": \"test-api-key\"},
                    timeout=15.0,
                )

                # Stable failure contract: accept 400/422 only.
                assert resp.status_code in (400, 422), (
                    f\"Unexpected status {resp.status_code}: {resp.text[:500]}\"
                )

                # No raw internal leakage.
                body_text = resp.text or \"\"
                assert \"Traceback\" not in body_text, body_text[:500]
                assert \"ValidationError\" not in body_text, body_text[:500]

                # Must be boundary failure: upstream must not be called.
                after = await state.get_count()
                assert after == before == 0, (
                    f\"Upstream executed during validation failure (before={before}, after={after}).\"
                )

                # JSON-ish response body (do not assert optional fields).
                try:
                    parsed = resp.json()
                except Exception as exc:
                    raise AssertionError(
                        f\"Expected JSON-ish error body, got: {body_text[:500]}\"
                    ) from exc
                assert isinstance(parsed, dict), (
                    f\"Expected dict-like error body, got: {type(parsed)}\"
                )

                if \"detail\" not in parsed and \"error\" not in parsed:
                    raise AssertionError(
                        \"Expected error body to include 'detail' or 'error' key. \"
                        f\"Got keys={sorted(parsed.keys())}\"
                    )

    finally:
        await _stop_server(bridge_server, bridge_task)
        await _stop_server(upstream_server, upstream_task)

    final_count = await state.get_count()
    assert final_count == 0, f\"Upstream should not be called; got {final_count}\"


asyncio.run(_run())
''')""",
    "exec(code, globals(), globals())",
)

# Drift guard for defended root export surface.
RELEASE_SURFACE_DRIFT_CHECK = (
    "import tok",
    "from tok.release_surface import (SUPPORTED_ROOT_EXPORTS, CANDIDATE_PENDING_PROOF, EXPERIMENTAL_ROOT_EXPORTS)",
    "effective = set(tok.__all__)",
    "declared = set(SUPPORTED_ROOT_EXPORTS)",
    "assert effective == declared, f'root_surface_drift: effective={sorted(effective)} declared={sorted(declared)}'",
    "candidate_leaks = sorted(set(CANDIDATE_PENDING_PROOF) & effective)",
    "assert not candidate_leaks, f'root_surface_candidate_leak: {candidate_leaks}'",
    "experimental_leaks = sorted(set(EXPERIMENTAL_ROOT_EXPORTS) & effective)",
    "assert not experimental_leaks, f'root_surface_experimental_leak: {experimental_leaks}'",
    "print('release-surface drift smoke OK')",
)

# Packaging/install contract gate for defended import surface.
CLEAN_INSTALL_IMPORT_CHECK = (
    "import textwrap",
    "code = textwrap.dedent('''\n"
    "import os, subprocess, sys, tempfile, tomllib, venv\n"
    "from pathlib import Path\n\n"
    "def _venv_python(v): bin_dir = 'Scripts' if os.name == 'nt' else 'bin'; return v / bin_dir / 'python'\n\n"
    "root = Path.cwd()\n"
    "with tempfile.TemporaryDirectory(prefix='tok-clean-import-') as td:\n"
    "    tmp = Path(td); dist_dir = tmp / 'dist'; dist_dir.mkdir(parents=True, exist_ok=True)\n"
    "    expected_version = tomllib.loads((root / 'pyproject.toml').read_text())['project']['version']\n"
    "    subprocess.run([sys.executable, '-m', 'build', '--wheel', '--sdist', '--outdir', str(dist_dir)], cwd=root, check=True)\n"
    "    wheels = sorted(dist_dir.glob('*.whl')); assert wheels, 'no wheel'; wheel = wheels[-1]\n"
    "    sdists = sorted(dist_dir.glob('*.tar.gz')); assert sdists, 'no sdist'\n"
    "    venv_dir = tmp / 'venv'; venv.EnvBuilder(with_pip=True).create(venv_dir)\n"
    "    vpy = _venv_python(venv_dir); assert vpy.exists(), 'venv missing'\n"
    "    subprocess.run([str(vpy), '-m', 'pip', 'install', '--quiet', str(wheel)], check=True)\n"
    "    check = f\"import tok; from importlib import metadata; from tok.release_surface import SUPPORTED_ROOT_EXPORTS; assert metadata.version('tok-protocol') == {expected_version!r}; assert set(tok.__all__) == set(SUPPORTED_ROOT_EXPORTS); print(1)\"\n"
    "    subprocess.run([str(vpy), '-c', check], check=True)\n"
    "''')",
    "exec(code, globals(), globals())",
)

SMOKE_LEGACY_BENCHMARKS: tuple[str, ...] = ("coding-loop-5", "research-loop-5")
SMOKE_CATALOG_TASKS: tuple[str, ...] = (
    "exec.click.option-precedence",
    "exec.rich.overflow-markup",
    "qa.click.option-precedence",
    "qa.pluggy.hook-discovery",
    "qa.rich.markup-pipeline",
    "qa.tok.api-base-plumbing",
)


SMOKE_STEPS: tuple[SmokeStep, ...] = (
    # Baseline command visibility.
    SmokeStep("CLI version", (*UV_RUN_PREFIX, "tok", "--version")),
    SmokeStep("CLI help", (*UV_RUN_PREFIX, "tok", "--help")),
    SmokeStep("Bridge help", (*UV_RUN_PREFIX, "tok", "bridge", "--help")),
    SmokeStep("Doctor help", (*UV_RUN_PREFIX, "tok", "doctor", "--help")),
    SmokeStep("Stats help", (*UV_RUN_PREFIX, "tok", "stats", "--help")),
    # Release-surface visibility and declaration checks.
    _inline_python_step(
        "Public imports",
        IMPORT_CHECK,
    ),
    _inline_python_step(
        "Release-surface gate",
        SURFACE_GATE_CHECK,
    ),
    # Focused baseline regression cluster.
    _pytest_step(
        "Focused bridge/runtime/compression smoke",
        "tests/unit/test_bridge_smoke.py",
        "tests/unit/test_packaging_smoke.py",
        "tests/unit/test_gateway.py::test_gateway_canonicalizes_tool_heavy_bridge_body_before_send",
        "tests/unit/test_universal_runtime.py::test_runtime_prepare_request_compression_in_tool_compatible_mode",
        "-q",
    ),
    _pytest_step(
        "Primary streaming bridge smoke",
        "tests/smoke/test_primary_streaming_smoke.py",
        "-q",
    ),
    _pytest_step("Claude live smoke matrix", "tests/smoke/test_live_claude_smoke_matrix.py", "-q"),
    # Ledger-promoted boundaries, in promotion order.
    _pytest_step(
        "Primary non-streaming bridge smoke",
        "tests/smoke/test_primary_non_streaming_smoke.py",
        "-q",
    ),
    _inline_python_step(
        "Validation-failure smoke",
        VALIDATION_FAILURE_CHECK,
    ),
    _inline_python_step(
        "Release-surface drift smoke",
        RELEASE_SURFACE_DRIFT_CHECK,
    ),
    _inline_python_step(
        "Clean install/import smoke",
        CLEAN_INSTALL_IMPORT_CHECK,
    ),
    # Packaging build sanity gate.
    SmokeStep(
        "Build smoke",
        (
            "uv",
            "run",
            "--with",
            "build",
            "--with",
            "hatchling",
            "python",
            "-m",
            "build",
        ),
    ),
    _inline_python_step(
        "Artifact metadata smoke",
        (
            "import subprocess",
            "from pathlib import Path",
            "files = sorted(str(path) for path in Path('dist').glob('*.whl')) + sorted(str(path) for path in Path('dist').glob('*.tar.gz'))",
            "assert files, 'no dist artifacts to validate'",
            "subprocess.run(['uv', 'run', '--with', 'twine', 'python', '-m', 'twine', 'check', *files], check=True)",
        ),
    ),
)


def _run_step(step: SmokeStep) -> int:
    completed = subprocess.run(step.command, cwd=ROOT, check=False)
    return completed.returncode


def _benchmark_steps(
    *,
    benchmark_mode: str,
    benchmark_output: Path,
    model: str,
    catalog_root: Path,
    repeats: int,
    pricing_prompt: float | None,
    pricing_completion: float | None,
    provider_options: str | None,
) -> tuple[SmokeStep, ...]:
    if benchmark_mode == "none":
        return ()

    benchmark_output = benchmark_output.resolve()
    legacy_benchmarks = ",".join(SMOKE_LEGACY_BENCHMARKS)
    live_benchmark_command: list[str] = [
        *UV_RUN_PREFIX,
        "tok",
        "dev",
        "live-benchmark",
        "--program",
        "both",
        "--mode",
        "compare",
        "--model",
        model,
        "--catalog-root",
        str(catalog_root.resolve()),
        "--output",
        str(benchmark_output),
        "--legacy-benchmarks",
        legacy_benchmarks,
        "--repeats",
        str(max(1, repeats)),
    ]
    if pricing_prompt is not None:
        live_benchmark_command.extend(["--pricing-prompt", str(pricing_prompt)])
    if pricing_completion is not None:
        live_benchmark_command.extend(["--pricing-completion", str(pricing_completion)])
    if provider_options:
        live_benchmark_command.extend(["--provider-options", provider_options])
    if benchmark_mode == "smoke":
        for task_id in SMOKE_CATALOG_TASKS:
            live_benchmark_command.extend(["--task", task_id])
    elif benchmark_mode == "public_full":
        live_benchmark_command.append("--public-release-only")
    else:
        raise ValueError(f"unsupported benchmark mode: {benchmark_mode}")
    gate_check_command = [
        *UV_RUN_PREFIX,
        "tok",
        "gate-check",
        "tests/fixtures/replay",
        "--stability-dir",
        str(benchmark_output / "replay"),
        "--benchmark-report",
        str(benchmark_output / "catalog" / "report.json"),
        "--emit-metrics",
        str(benchmark_output / "claims" / "gate_metrics.json"),
        "--continue-on-error",
    ]
    claims_check_command = [
        *UV_RUN_PREFIX,
        "python",
        "scripts/verify_release_claims.py",
        "--gate-metrics",
        str(benchmark_output / "claims" / "gate_metrics.json"),
        "--benchmark-report",
        str(benchmark_output / "catalog" / "report.json"),
        "--output",
        str(benchmark_output / "claims" / "claims_verification.json"),
        "--min-savings-pct",
        "45.0",
        "--max-savings-pct",
        "55.0",
    ]
    return (
        SmokeStep(f"Benchmark {benchmark_mode}", tuple(live_benchmark_command)),
        SmokeStep("Benchmark gate-check", tuple(gate_check_command)),
        SmokeStep("Claims verification", tuple(claims_check_command)),
    )


def build_steps(
    *,
    benchmark_mode: str,
    benchmark_output: Path,
    model: str,
    catalog_root: Path,
    repeats: int,
    pricing_prompt: float | None,
    pricing_completion: float | None,
    provider_options: str | None,
) -> tuple[SmokeStep, ...]:
    return (
        *SMOKE_STEPS,
        *_benchmark_steps(
            benchmark_mode=benchmark_mode,
            benchmark_output=benchmark_output,
            model=model,
            catalog_root=catalog_root,
            repeats=repeats,
            pricing_prompt=pricing_prompt,
            pricing_completion=pricing_completion,
            provider_options=provider_options,
        ),
    )


def _preflight_benchmark_environment(*, benchmark_mode: str, catalog_root: Path) -> tuple[bool, str]:
    if benchmark_mode == "none":
        return True, ""
    lanes_dir = catalog_root.resolve() / "lanes"
    if not lanes_dir.exists():
        return False, f"benchmark catalog lanes missing: {lanes_dir}"
    return True, ""


def _validate_benchmark_outputs(*, benchmark_mode: str, benchmark_output: Path, repeats: int) -> tuple[bool, str]:
    if benchmark_mode == "none":
        return True, ""
    required_paths = [
        benchmark_output / "catalog" / "report.json",
        benchmark_output / "replay" / "coding-loop-5_triage.json",
        benchmark_output / "replay" / "research-loop-5_triage.json",
        benchmark_output / "summary.md",
        benchmark_output / "claims" / "gate_metrics.json",
        benchmark_output / "claims" / "claims_verification.json",
    ]
    if repeats > 1:
        required_paths.extend(
            [
                benchmark_output / "replay" / "coding-loop-5_stability.json",
                benchmark_output / "replay" / "research-loop-5_stability.json",
            ]
        )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return False, "missing benchmark artifacts: " + ", ".join(missing)
    return True, ""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-mode",
        choices=("none", "smoke", "public_full"),
        default="none",
        help="Optional live benchmark sweep to append after the baseline smoke steps",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=ROOT / "tmp" / "benchmark-smoke",
        help="Artifact directory for benchmark smoke outputs",
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4.6",
        help="Model identifier to use for live benchmark smoke",
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=ROOT / "benchmarks",
        help="Benchmark catalog root for live benchmark runs",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Repeat count for live benchmark compare runs (release-grade default: 5)",
    )
    parser.add_argument(
        "--pricing-prompt",
        type=float,
        default=None,
        help="Prompt token price per 1M tokens (USD)",
    )
    parser.add_argument(
        "--pricing-completion",
        type=float,
        default=None,
        help="Completion token price per 1M tokens (USD)",
    )
    parser.add_argument(
        "--provider-options",
        default=None,
        help="JSON provider options passed to tok dev live-benchmark",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    preflight_ok, preflight_reason = _preflight_benchmark_environment(
        benchmark_mode=args.benchmark_mode,
        catalog_root=args.catalog_root,
    )
    if not preflight_ok:
        print(f"[release-smoke] benchmark preflight failed: {preflight_reason}")
        return 2

    for step in build_steps(
        benchmark_mode=args.benchmark_mode,
        benchmark_output=args.benchmark_output,
        model=args.model,
        catalog_root=args.catalog_root,
        repeats=max(1, int(args.repeats)),
        pricing_prompt=args.pricing_prompt,
        pricing_completion=args.pricing_completion,
        provider_options=args.provider_options,
    ):
        exit_code = _run_step(step)
        if exit_code != 0:
            return exit_code

    outputs_ok, outputs_reason = _validate_benchmark_outputs(
        benchmark_mode=args.benchmark_mode,
        benchmark_output=args.benchmark_output.resolve(),
        repeats=max(1, int(args.repeats)),
    )
    if not outputs_ok:
        print(f"[release-smoke] benchmark artifact validation failed: {outputs_reason}")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
