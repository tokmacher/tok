"""Tests for scripts/run_agent_smoke.py: verifies the agent smoke check is well-formed."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = ROOT / "scripts" / "run_agent_smoke.py"


def _load_smoke_module() -> object:
    spec = importlib.util.spec_from_file_location("run_agent_smoke", SMOKE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestSmokeScriptExists:
    def test_script_exists(self) -> None:
        assert SMOKE_SCRIPT.exists(), "scripts/run_agent_smoke.py missing"

    def test_shell_wrapper_exists(self) -> None:
        wrapper = ROOT / "scripts" / "agent_smoke.sh"
        assert wrapper.exists(), "scripts/agent_smoke.sh missing"


class TestSmokeSteps:
    def test_build_steps_returns_nonempty(self) -> None:
        module = _load_smoke_module()
        steps = module.build_steps()
        assert len(steps) > 0

    def test_build_steps_includes_cli_version(self) -> None:
        module = _load_smoke_module()
        names = [s.name for s in module.build_steps()]
        assert "CLI version" in names

    def test_build_steps_includes_cli_help(self) -> None:
        module = _load_smoke_module()
        names = [s.name for s in module.build_steps()]
        assert "CLI help" in names

    def test_build_steps_includes_unit_tests(self) -> None:
        module = _load_smoke_module()
        names = [s.name for s in module.build_steps()]
        assert any("unit" in n.lower() or "Unit" in n or "contract" in n.lower() or "test" in n.lower() for n in names)

    def test_build_steps_all_use_uv_run(self) -> None:
        module = _load_smoke_module()
        for step in module.build_steps():
            assert step.command[0] == "uv", f"Step '{step.name}' does not start with uv: {step.command}"
            assert step.command[1] == "run", f"Step '{step.name}' does not use 'uv run': {step.command}"

    def test_build_steps_commands_are_tuples(self) -> None:
        module = _load_smoke_module()
        for step in module.build_steps():
            assert isinstance(step.command, tuple), f"Step '{step.name}' command is not a tuple"


class TestSmokeMain:
    def test_main_returns_zero_on_all_pass(self, monkeypatch) -> None:
        module = _load_smoke_module()
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""),
        )
        exit_code = module.main([])
        assert exit_code == 0

    def test_main_returns_nonzero_on_failure(self, monkeypatch) -> None:
        module = _load_smoke_module()
        call_count = 0

        def _fail_on_second(*a, **_kw):
            nonlocal call_count
            call_count += 1
            return subprocess.CompletedProcess(
                args=a, returncode=1 if call_count == 2 else 0, stdout="", stderr="error"
            )

        monkeypatch.setattr(module.subprocess, "run", _fail_on_second)
        exit_code = module.main([])
        assert exit_code != 0

    def test_main_reports_per_step_status(self, monkeypatch, capsys) -> None:
        module = _load_smoke_module()
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""),
        )
        module.main([])
        output = capsys.readouterr().out
        assert "PASS" in output or "pass" in output.lower()


class TestSmokeReportFormat:
    def test_final_summary_line(self, monkeypatch, capsys) -> None:
        module = _load_smoke_module()
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *a, **_kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""),
        )
        module.main([])
        output = capsys.readouterr().out
        assert "Agent smoke:" in output


class TestAdversarialSmoke:
    def test_step_names_are_unique(self) -> None:
        module = _load_smoke_module()
        names = [s.name for s in module.build_steps()]
        assert len(names) == len(set(names)), f"Duplicate step names: {names}"

    def test_no_step_references_nonexistent_test_path(self) -> None:
        """Every pytest step should reference a path that exists or is a valid test selector."""
        module = _load_smoke_module()
        for step in module.build_steps():
            if "pytest" in step.command:
                for arg in step.command:
                    if arg.startswith("tests/"):
                        assert (ROOT / arg.split("::")[0]).exists(), (
                            f"Step '{step.name}' references nonexistent path: {arg}"
                        )

    def test_main_stops_on_first_failure(self, monkeypatch) -> None:
        module = _load_smoke_module()
        run_count = 0

        def _fail_once(*a, **_kw):
            nonlocal run_count
            run_count += 1
            rc = 1 if run_count == 1 else 0
            return subprocess.CompletedProcess(args=a, returncode=rc, stdout="", stderr="error")

        monkeypatch.setattr(module.subprocess, "run", _fail_once)
        module.main([])
        assert run_count == 1, "Smoke script should stop on first failure but continued"

    def test_bridge_step_is_labeled_clearly(self) -> None:
        """The bridge-related step must be easily identifiable by name."""
        module = _load_smoke_module()
        names = [s.name for s in module.build_steps()]
        bridge_names = [n for n in names if "bridge" in n.lower()]
        assert len(bridge_names) >= 1, "No bridge-related step found in agent smoke"

    def test_shell_wrapper_is_executable(self) -> None:
        wrapper = ROOT / "scripts" / "agent_smoke.sh"
        if wrapper.exists():
            assert wrapper.stat().st_mode & 0o111, "agent_smoke.sh is not executable"
