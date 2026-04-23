"""Release coverage contract: supported modules must not be excluded from the coverage gate."""

from __future__ import annotations

from pathlib import Path

import tomllib


def test_supported_bridge_modules_are_covered() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    omit_list = data["tool"]["coverage"]["run"]["omit"]

    supported_modules = [
        "src/tok/gateway/__init__.py",
        "src/tok/protocol/models.py",
        "src/tok/protocol/parser.py",
        "src/tok/protocol/encoder.py",
        "src/tok/protocol/schema.py",
        "src/tok/protocol/format_bridge.py",
        "src/tok/compression/__init__.py",
        "src/tok/runtime/core.py",
        "src/tok/runtime/types.py",
        "src/tok/cli/__init__.py",
        "src/tok/cli/_release.py",
        "src/tok/cli/_release_commands.py",
        "src/tok/cli/_bridge_commands.py",
        "src/tok/cli/_bridge.py",
        "src/tok/cli/_cli_support.py",
        "src/tok/exceptions.py",
        "src/tok/stats.py",
        "src/tok/release_surface.py",
    ]

    leaks = [m for m in supported_modules if m in omit_list]
    assert not leaks, f"Supported modules incorrectly omitted from coverage: {leaks}"


def test_protocol_submodule_not_glob_omitted() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    omit_list = data["tool"]["coverage"]["run"]["omit"]

    protocol_glob_omissions = [o for o in omit_list if "protocol" in o and o.endswith("/*")]
    assert not protocol_glob_omissions, (
        f"Protocol module glob omissions found (would exclude supported IDL): {protocol_glob_omissions}"
    )


def test_format_bridge_backward_compat_is_covered() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    omit_list = data["tool"]["coverage"]["run"]["omit"]

    backward_compat_facades = [
        "src/tok/format_bridge.py",
        "src/tok/universal_runtime.py",
        "src/tok/gateway.py",
    ]
    leaks = [m for m in backward_compat_facades if m in omit_list]
    assert not leaks, f"Backward-compatibility facades incorrectly omitted from coverage: {leaks}"
