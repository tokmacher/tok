"""Agent docs contract: AGENTS.md and agent-contract.json must stay consistent with the real CLI surface."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS_MD = ROOT / "AGENTS.md"
AGENT_CONTRACT_JSON = ROOT / "docs" / "agent-contract.json"
PYPROJECT_TOML = ROOT / "pyproject.toml"

REQUIRED_CONTRACT_KEYS = (
    "project",
    "release_line",
    "supported_primary_path",
    "unsupported_claims",
    "required_verification_commands",
    "success_signals",
    "reporting_required_fields",
)

REQUIRED_AGENTS_MD_SECTIONS = (
    "## Project identity",
    "## Golden path",
    "## Bridge verification",
    "## Reporting rules",
    "## Safe editing rules",
    "## Useful files",
)


def _read_pyproject_version() -> str:
    import tomllib

    data = tomllib.loads(PYPROJECT_TOML.read_text())
    return data["project"]["version"]


def _parse_contract() -> dict:
    return json.loads(AGENT_CONTRACT_JSON.read_text())


class TestAgentsMdExists:
    def test_agents_md_exists(self) -> None:
        assert AGENTS_MD.exists(), "AGENTS.md missing from repository root"

    def test_agents_md_nonempty(self) -> None:
        assert AGENTS_MD.read_text().strip(), "AGENTS.md is empty"

    def test_agents_md_has_required_sections(self) -> None:
        content = AGENTS_MD.read_text()
        for section in REQUIRED_AGENTS_MD_SECTIONS:
            assert section in content, f"AGENTS.md missing section: {section}"


class TestAgentContractJson:
    def test_contract_exists(self) -> None:
        assert AGENT_CONTRACT_JSON.exists(), "docs/agent-contract.json missing"

    def test_contract_parses(self) -> None:
        data = _parse_contract()
        assert isinstance(data, dict)

    def test_contract_has_required_keys(self) -> None:
        data = _parse_contract()
        for key in REQUIRED_CONTRACT_KEYS:
            assert key in data, f"agent-contract.json missing key: {key}"

    def test_contract_project_name(self) -> None:
        assert _parse_contract()["project"] == "tok"

    def test_contract_release_line_matches_version(self) -> None:
        version = _read_pyproject_version()
        release_line = _parse_contract()["release_line"]
        prefix = release_line.rstrip(".x").rstrip(".")
        assert version.startswith(prefix), (
            f"agent-contract.json release_line '{release_line}' does not match pyproject.toml version '{version}'"
        )


class TestCommandsMatchReality:
    def test_every_contract_command_mentioned_in_agents_md(self) -> None:
        agents_content = AGENTS_MD.read_text()
        for cmd in _parse_contract()["required_verification_commands"]:
            assert cmd in agents_content, f"agent-contract.json command '{cmd}' not mentioned in AGENTS.md"

    def test_unsupported_claims_present_in_agents_md(self) -> None:
        agents_content = AGENTS_MD.read_text()
        for claim in _parse_contract()["unsupported_claims"]:
            assert claim.lower() in agents_content.lower(), f"Unsupported claim '{claim}' not present in AGENTS.md"

    def test_supported_path_mentions_claude_code_and_bridge(self) -> None:
        path = _parse_contract()["supported_primary_path"]
        assert "Claude Code" in path or "claude" in path.lower()
        assert "bridge" in path.lower()


class TestContractListsNonempty:
    def test_required_verification_commands_not_empty(self) -> None:
        cmds = _parse_contract()["required_verification_commands"]
        assert isinstance(cmds, list) and len(cmds) > 0

    def test_unsupported_claims_not_empty(self) -> None:
        claims = _parse_contract()["unsupported_claims"]
        assert isinstance(claims, list) and len(claims) > 0

    def test_success_signals_not_empty(self) -> None:
        signals = _parse_contract()["success_signals"]
        assert isinstance(signals, list) and len(signals) > 0

    def test_reporting_required_fields_not_empty(self) -> None:
        fields = _parse_contract()["reporting_required_fields"]
        assert isinstance(fields, list) and len(fields) > 0


class TestNoPhantomCommands:
    def test_contract_commands_are_real_cli_commands(self) -> None:
        """Every command in the contract must be a real supported CLI command."""
        from tok.release_surface import SUPPORTED_CLI_ROOT_COMMANDS

        supported = set(SUPPORTED_CLI_ROOT_COMMANDS)
        bridge_subcommands = {"bridge status", "bridge start", "bridge stop", "bridge logs"}
        for cmd in _parse_contract()["required_verification_commands"]:
            bare = cmd.removeprefix("tok ").strip()
            parts = bare.split()
            if len(parts) >= 2 and parts[0] == "bridge":
                sub = f"bridge {parts[1]}"
                assert sub in bridge_subcommands, f"Unknown bridge subcommand in contract: {cmd}"
            elif bare in ("--version", "--help"):
                continue
            else:
                assert parts[0] in supported, f"Command '{cmd}' not in SUPPORTED_CLI_ROOT_COMMANDS"


class TestAdversarialContractDrift:
    """Regression tests: catch specific rot patterns that have happened before."""

    def test_contract_json_is_not_truncated(self) -> None:
        raw = AGENT_CONTRACT_JSON.read_text()
        parsed = _parse_contract()
        assert raw.strip().endswith("}"), "agent-contract.json appears truncated"
        assert isinstance(parsed["unsupported_claims"], list)
        assert len(parsed["unsupported_claims"]) >= 3

    def test_agents_md_mentions_core_diagnostic_commands(self) -> None:
        """Ensure the four core diagnostics are named, not just referenced generically."""
        content = AGENTS_MD.read_text()
        for cmd in ("tok bridge status", "tok doctor", "tok stats", "tok audit"):
            assert cmd in content, f"AGENTS.md missing explicit mention of '{cmd}'"

    def test_agents_md_do_not_claim_section_present(self) -> None:
        content = AGENTS_MD.read_text()
        assert "Do not" in content or "do not" in content
        assert "Do not describe" in content or "Do not claim" in content

    def test_contract_success_signals_match_doctor_output(self) -> None:
        """Success signals should reference real tok doctor / bridge status output."""
        signals = _parse_contract()["success_signals"]
        signal_text = " ".join(signals).lower()
        assert "bridge" in signal_text
        assert "fallback" in signal_text or "fallbacks" in signal_text
        assert "degraded" in signal_text

    def test_no_duplicate_commands_in_contract(self) -> None:
        cmds = _parse_contract()["required_verification_commands"]
        assert len(cmds) == len(set(cmds)), f"Duplicate commands in contract: {cmds}"

    def test_no_duplicate_unsupported_claims(self) -> None:
        claims = _parse_contract()["unsupported_claims"]
        assert len(claims) == len(set(claims)), f"Duplicate unsupported claims: {claims}"

    def test_contract_reporting_fields_include_claude_available(self) -> None:
        fields = _parse_contract()["reporting_required_fields"]
        assert "claude_available" in fields, "Missing 'claude_available' in reporting_required_fields"

    def test_contract_reporting_fields_include_bridge_running(self) -> None:
        fields = _parse_contract()["reporting_required_fields"]
        assert "bridge_running" in fields

    def test_agents_md_useful_files_point_to_real_paths(self) -> None:
        """Every file listed in Useful files section should exist or be a known docs path."""
        content = AGENTS_MD.read_text()
        in_section = False
        for line in content.splitlines():
            if line.startswith("## Useful files"):
                in_section = True
                continue
            if in_section and line.startswith("## "):
                break
            if in_section and line.strip().startswith("- `"):
                path_str = line.split("`")[1]
                resolved = ROOT / path_str
                if resolved.exists():
                    continue
                assert path_str.endswith("/"), f"Useful file '{path_str}' does not exist and is not a directory marker"

    def test_agents_md_golden_path_uses_uv_run(self) -> None:
        content = AGENTS_MD.read_text()
        assert "uv sync --frozen --extra dev" in content
        assert "uv run tok --version" in content
        assert "uv run pytest" in content

    def test_contract_does_not_mention_hidden_commands(self) -> None:
        """Agent contract should only reference visible, supported commands."""
        from tok.release_surface import EXPERIMENTAL_CLI_ROOT_COMMANDS

        cmds_text = " ".join(_parse_contract()["required_verification_commands"])
        for hidden in EXPERIMENTAL_CLI_ROOT_COMMANDS:
            assert hidden not in cmds_text, f"Agent contract references hidden command '{hidden}'"
