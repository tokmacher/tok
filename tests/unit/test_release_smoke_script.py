from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_module() -> object:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "run_release_smoke.py"
    spec = importlib.util.spec_from_file_location("run_release_smoke", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_smoke_mode_invokes_expected_live_benchmark_and_gate_commands(tmp_path, monkeypatch) -> None:
    module = _load_module()
    commands: list[tuple[str, ...]] = []
    output_root = tmp_path / "artifacts"
    catalog_root = tmp_path / "benchmarks"
    (catalog_root / "lanes").mkdir(parents=True, exist_ok=True)

    def _fake_run(command, cwd=None, check=False):  # type: ignore[no-untyped-def]
        command_tuple = tuple(str(part) for part in command)
        commands.append(command_tuple)
        if "live-benchmark" in command_tuple:
            output_index = command_tuple.index("--output") + 1
            live_output = Path(command_tuple[output_index])
            replay_output = live_output / "replay"
            replay_output.mkdir(parents=True, exist_ok=True)
            (live_output / "catalog").mkdir(parents=True, exist_ok=True)
            (live_output / "catalog" / "report.json").write_text("{}")
            (replay_output / "coding-loop-5_triage.json").write_text("{}")
            (replay_output / "research-loop-5_triage.json").write_text("{}")
            (replay_output / "coding-loop-5_stability.json").write_text("{}")
            (replay_output / "research-loop-5_stability.json").write_text("{}")
            (live_output / "summary.md").write_text("# summary\n")
        if "gate-check" in command_tuple and "--emit-metrics" in command_tuple:
            metrics_index = command_tuple.index("--emit-metrics") + 1
            metrics_path = Path(command_tuple[metrics_index])
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text('{"release_summary":{"avg_savings_pct":50.0}}')
        if any("verify_release_claims.py" in part for part in command_tuple) and "--output" in command_tuple:
            output_index = command_tuple.index("--output") + 1
            claims_output = Path(command_tuple[output_index])
            claims_output.parent.mkdir(parents=True, exist_ok=True)
            claims_output.write_text('{"passed":true}')
        return subprocess.CompletedProcess(command_tuple, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    exit_code = module.main(
        [
            "--benchmark-mode",
            "smoke",
            "--benchmark-output",
            str(output_root),
            "--model",
            "anthropic/test-model",
            "--catalog-root",
            str(catalog_root),
        ]
    )

    assert exit_code == 0

    live_command = next(command for command in commands if "live-benchmark" in command)
    gate_command = next(command for command in commands if "gate-check" in command)

    assert "--legacy-benchmarks" in live_command
    assert "coding-loop-5,research-loop-5" in live_command
    assert "--catalog-root" in live_command
    assert str(catalog_root) in live_command
    assert "--repeats" in live_command
    assert "5" in live_command
    assert live_command.count("--task") == 6
    assert "exec.click.option-precedence" in live_command
    assert "exec.rich.overflow-markup" in live_command
    assert "qa.click.option-precedence" in live_command
    assert "qa.pluggy.hook-discovery" in live_command
    assert "qa.rich.markup-pipeline" in live_command
    assert "qa.tok.api-base-plumbing" in live_command
    assert "--stability-dir" in gate_command
    assert str(output_root / "replay") in gate_command
    assert "--benchmark-report" in gate_command
    assert str(output_root / "catalog" / "report.json") in gate_command
    assert "--emit-metrics" in gate_command
    assert str(output_root / "claims" / "gate_metrics.json") in gate_command

    assert (output_root / "replay").exists()
    assert (output_root / "catalog" / "report.json").exists()
    assert (output_root / "claims" / "gate_metrics.json").exists()
    assert (output_root / "claims" / "claims_verification.json").exists()
    assert (output_root / "summary.md").exists()


def test_build_and_artifact_smoke_use_isolated_versioned_dist() -> None:
    module = _load_module()
    steps = module.build_steps(
        benchmark_mode="none",
        benchmark_output=Path("unused"),
        model="anthropic/test-model",
        catalog_root=Path("benchmarks"),
        repeats=1,
        pricing_prompt=None,
        pricing_completion=None,
        provider_options=None,
    )

    build_step = next(step for step in steps if step.name == "Build smoke")
    artifact_step = next(step for step in steps if step.name == "Artifact metadata smoke")
    build_command = " ".join(build_step.command)
    artifact_command = " ".join(artifact_step.command)

    assert "tmp/release-smoke-dist" in build_command
    assert "--outdir" in build_command
    assert "tmp/release-smoke-dist" in artifact_command
    assert "expected_version" in artifact_command
    assert "len(files) == 2" in artifact_command


def test_release_smoke_includes_tok_trace_spec_contract_gate() -> None:
    module = _load_module()
    steps = module.build_steps(
        benchmark_mode="none",
        benchmark_output=Path("unused"),
        model="anthropic/test-model",
        catalog_root=Path("benchmarks"),
        repeats=1,
        pricing_prompt=None,
        pricing_completion=None,
        provider_options=None,
    )

    commands = {" ".join(step.command) for step in steps}

    assert "uv run pytest tests/spec -q" in commands


def test_resolver_smoke_reconstructs_uri_from_digest_line() -> None:
    module = _load_module()
    steps = module.build_steps(
        benchmark_mode="none",
        benchmark_output=Path("unused"),
        model="anthropic/test-model",
        catalog_root=Path("benchmarks"),
        repeats=1,
        pricing_prompt=None,
        pricing_completion=None,
        provider_options=None,
    )

    resolver_step = next(step for step in steps if step.name == "Resolver smoke")
    resolver_command = " ".join(resolver_step.command)

    assert "digest_line = [l for l in put_out.output.splitlines() if l.startswith('Digest:')][0]" in resolver_command
    assert "put_uri = format_resolver_uri(put_digest)" in resolver_command
