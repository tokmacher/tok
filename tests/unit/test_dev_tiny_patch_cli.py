from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer

from tok.cli import _dev


def test_run_legacy_compare_benchmark_handles_patch_suites(monkeypatch, tmp_path) -> None:
    def _fake_run_patch_suite_benchmark(*, benchmark_name, runner, output_root, repeats, repo_root):
        assert benchmark_name == "small-patch"
        del runner, repeats, repo_root
        assert output_root == tmp_path
        return SimpleNamespace(selected_task_ids=("tiny.calc.add",)), {"runs": 2, "claimable": False}, "# tiny"

    monkeypatch.setattr("tok.testing.tiny_patch_benchmark.run_patch_suite_benchmark", _fake_run_patch_suite_benchmark)

    _dev._run_legacy_compare_benchmark(
        benchmark="small-patch",
        runner=object(),
        output=tmp_path,
        turns=1,
        repeats=1,
    )

    assert (tmp_path / "small-patch_summary.json").exists()
    assert (tmp_path / "small-patch_summary.md").read_text().startswith("# tiny")


def test_patch_suite_benchmark_name_maps_sizes() -> None:
    assert _dev._patch_suite_benchmark_name("tiny") == "tiny-patch"
    assert _dev._patch_suite_benchmark_name("small") == "small-patch"
    assert _dev._patch_suite_benchmark_name("medium") == "medium-patch"
    assert _dev._patch_suite_benchmark_name("large") == "large-patch"


def test_patch_suite_benchmark_name_rejects_unknown() -> None:
    with pytest.raises(typer.BadParameter):
        _dev._patch_suite_benchmark_name("xlarge")


def test_patch_benchmark_command_dispatches(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class _DummyRunner:
        def __init__(self, **kwargs) -> None:
            captured["runner_kwargs"] = kwargs

    def _fake_run_legacy_compare_benchmark(*, benchmark, runner, output, turns, repeats) -> None:
        captured["benchmark"] = benchmark
        captured["output"] = output
        captured["turns"] = turns
        captured["repeats"] = repeats
        captured["runner_type"] = type(runner).__name__

    monkeypatch.setattr("tok.testing.live_benchmark.LiveBenchmarkRunner", _DummyRunner)
    monkeypatch.setattr("tok.cli._dev._run_legacy_compare_benchmark", _fake_run_legacy_compare_benchmark)

    _dev.patch_benchmark(size="medium", output=tmp_path, repeats=2, timeout=33)

    assert captured["benchmark"] == "medium-patch"
    assert captured["output"] == tmp_path
    assert captured["turns"] is None
    assert captured["repeats"] == 2
    assert captured["runner_type"] == "_DummyRunner"
    assert captured["runner_kwargs"]["timeout"] == 33
