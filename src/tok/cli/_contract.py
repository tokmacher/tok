"""Hidden agent-contract verification command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from ._cli_support import console

_JSON_OPT = typer.Option(False, "--json", help="Emit machine-readable JSON.")


def verify_contract(
    *,
    contract_path: Path | None = None,
    help_text: str | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    path = contract_path or root / "docs" / "agent-contract.json"
    if not path.exists():
        return {"ok": False, "violations": ["contract_missing"], "contract": {}}
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ok": False, "violations": ["contract_invalid_json"], "contract": {}}

    violations: list[str] = []
    visible_help = help_text if help_text is not None else _root_help_text()
    lowered_help = visible_help.lower()

    for claim in contract.get("forbidden_claims", []):
        if str(claim).lower() in lowered_help:
            violations.append(f"forbidden_claim_visible:{claim}")

    for claim in contract.get("unsupported_claims", []):
        if str(claim).lower() in lowered_help:
            violations.append(f"unsupported_claim_visible:{claim}")

    layers = contract.get("implemented_protocol_layers", {})
    if isinstance(layers, dict):
        for layer, status in layers.items():
            if status == "deferred":
                layer_name = str(layer)
                public_phrases = {
                    layer_name.replace("_", "-"),
                    layer_name.replace("_", " "),
                    layer_name.replace("_to_", "-to-").replace("_", " "),
                }
                if any(phrase in lowered_help for phrase in public_phrases):
                    violations.append(f"deferred_layer_visible:{layer}")

    _check_release_surface(violations)
    _check_required_commands(contract, violations)

    return {
        "ok": not violations,
        "violations": violations,
        "contract": {
            "schema": contract.get("schema"),
            "release_line": contract.get("release_line"),
            "supported_primary_path": contract.get("supported_primary_path"),
        },
    }


def register(app: typer.Typer) -> None:
    @app.command("verify-contract", hidden=True)
    def verify_contract_command(json_output: bool = _JSON_OPT) -> None:
        """Verify the local agent contract against the CLI surface."""
        result = verify_contract()
        if json_output:
            console.print(json.dumps(result, sort_keys=True))
        else:
            if result["ok"]:
                console.print("[green]Agent contract verification passed.[/green]")
            else:
                console.print("[red]Agent contract verification failed.[/red]")
                for violation in result["violations"]:
                    console.print(f"- {violation}")
        if not result["ok"]:
            raise typer.Exit(1)


def _root_help_text() -> str:
    from tok.cli import app
    from tok.release_surface import _collect_visible_cli_names

    return "\n".join(sorted(_collect_visible_cli_names(app)))


def _check_release_surface(violations: list[str]) -> None:
    from tok.cli import app
    from tok.release_surface import EXPERIMENTAL_CLI_ROOT_COMMANDS, _collect_visible_cli_names

    visible = _collect_visible_cli_names(app)
    for name in EXPERIMENTAL_CLI_ROOT_COMMANDS:
        if name in visible:
            violations.append(f"experimental_command_visible:{name}")


def _check_required_commands(contract: dict[str, Any], violations: list[str]) -> None:
    from tok.release_surface import SUPPORTED_CLI_ROOT_COMMANDS

    supported = set(SUPPORTED_CLI_ROOT_COMMANDS)
    bridge_subcommands = {"status", "start", "stop", "logs"}
    for command in contract.get("required_verification_commands", []):
        bare = str(command).removeprefix("tok ").strip()
        parts = bare.split()
        if bare in {"--version", "--help"}:
            continue
        if len(parts) >= 2 and parts[0] == "bridge":
            if parts[1] not in bridge_subcommands:
                violations.append(f"unknown_bridge_command:{command}")
        elif not parts or parts[0] not in supported:
            violations.append(f"unknown_required_command:{command}")


__all__ = ["register", "verify_contract"]
