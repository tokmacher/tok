"""
Release-surface manifest for Tok 0.1.

This manifest defines the defended 0.1.0 public surface.
Anything not listed here is explicitly not part of the defended surface.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

SUPPORTED_ROOT_EXPORTS: tuple[str, ...] = ("Bridge",)

CANDIDATE_PENDING_PROOF: tuple[str, ...] = (
    "BlockSchema",
    "BridgeUnavailableError",
    "CompressionError",
    "DEFAULT_SCHEMA",
    "DocumentTransformer",
    "explore_file",
    "explore_module",
    "get_file_overview",
    "InvalidSessionStateError",
    "list_large_files",
    "PreparedRuntimeRequest",
    "ProcessedRuntimeResponse",
    "ReplayGateError",
    "RuntimeRequest",
    "RuntimeSession",
    "Sifter",
    "SessionError",
    "TokEncoder",
    "TokError",
    "TokNode",
    "TokParser",
    "TokRegistry",
    "TokSchema",
    "tok_to_dict",
    "tok_to_tok",
    "serialize",
    "UniversalTokRuntime",
)

EXPERIMENTAL_ROOT_EXPORTS: tuple[str, ...] = (
    "ClaudeBridgeAdapter",
    "OpenAIChatAdapter",
    "OrchestratorAdapter",
    "TextLoopAdapter",
    "process",
    "wrap",
)

SUPPORTED_CLI_ROOT_COMMANDS: tuple[str, ...] = (
    "bridge",
    "install",
    "doctor",
    "stats",
    "savings",
)

EXPERIMENTAL_CLI_ROOT_COMMANDS: tuple[str, ...] = (
    "metrics",
    "dev",
    "capture-summary",
    "capture-review",
    "evidence-gap",
    "convert",
    "parse",
    "pressure",
    "memory",
    "savings-trend",
    "fallback",
    "health",
    "generate-fixture",
    "live-benchmark",
    "stress-language",
    "jit-check",
    "gate-check",
)


def _collect_visible_cli_names(root_app: object | None) -> set[str]:
    """Extract visible CLI command/group names from root app."""
    visible: set[str] = set()
    if root_app is None:
        return visible

    for command in getattr(root_app, "registered_commands", []):
        if getattr(command, "hidden", False):
            continue
        name = getattr(command, "name", None)
        if not name:
            callback = getattr(command, "callback", None)
            name = getattr(callback, "__name__", "")
        if name:
            visible.add(str(name).replace("_", "-"))

    for group in getattr(root_app, "registered_groups", []):
        if getattr(group, "hidden", False):
            continue
        name = getattr(group, "name", None)
        if name:
            visible.add(str(name))

    return visible


def validate_release_surface(
    *,
    exported_names: Iterable[str],
    cli_help_output: str,
    root_app: object | None = None,
) -> list[str]:
    """Return release-surface violations for supported public entrypoints."""
    failures: list[str] = []
    exported = set(exported_names)

    for name in SUPPORTED_ROOT_EXPORTS:
        if name not in exported:
            failures.append(f"missing_supported_root_export:{name}")

    for name in EXPERIMENTAL_ROOT_EXPORTS:
        if name in exported:
            failures.append(f"experimental_root_export_exposed:{name}")

    normalized_help = " " + re.sub(r"\s+", " ", cli_help_output) + " "
    for command in SUPPORTED_CLI_ROOT_COMMANDS:
        if command not in normalized_help:
            failures.append(f"missing_supported_cli_command:{command}")

    visible_cli_names = _collect_visible_cli_names(root_app)

    for command in EXPERIMENTAL_CLI_ROOT_COMMANDS:
        if root_app is not None:
            if command in visible_cli_names:
                failures.append(f"experimental_cli_command_exposed:{command}")
        elif f" {command} " in normalized_help:
            failures.append(f"experimental_cli_command_exposed:{command}")

    return failures
