"""Adapter discipline and parity tests.

Verifies that:
1. adapters.py contains no direct imports of compression/bridge_memory/stats modules.
2. OrchestratorAdapter and ClaudeBridgeAdapter (via RuntimeAdapter) produce identical
   PreparedRuntimeRequest bodies for the same inputs.
3. Switching adapter_kind doesn't change the output body.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tok.adapters import (
    ClaudeBridgeAdapter,
    OrchestratorAdapter,
    RuntimeAdapter,
)
from tok.universal_runtime import RuntimeSession


# ---------------------------------------------------------------------------
# 1. Adapter discipline: no forbidden imports
# ---------------------------------------------------------------------------

_FORBIDDEN_MODULES = {"compression", "bridge_memory", "stats"}
_ADAPTERS_PATH = (
    Path(__file__).parent.parent.parent
    / "src"
    / "tok"
    / "adapters"
    / "adapters.py"
)


def _direct_imports_in_file(path: Path) -> set[str]:
    """Return set of directly imported module names (last segment) from a .py file."""
    source = path.read_text()
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[-1])
    return names


def test_adapters_module_does_not_import_compression():
    imports = _direct_imports_in_file(_ADAPTERS_PATH)
    forbidden = imports & _FORBIDDEN_MODULES
    assert (
        not forbidden
    ), f"adapters.py must not import {_FORBIDDEN_MODULES}, found: {forbidden}"


# ---------------------------------------------------------------------------
# 2. OrchestratorAdapter and ClaudeBridgeAdapter produce equivalent bodies
# ---------------------------------------------------------------------------

_MODEL = "google/gemini-2.0-flash-lite-001"
_MESSAGES = [{"role": "user", "content": "Audit the codebase"}]
_SYSTEM = "test system prompt"


def test_orchestrator_and_bridge_adapters_produce_identical_body(tmp_path):
    orch_session = RuntimeSession(memory_dir=tmp_path / "orch")
    bridge_session = RuntimeSession(memory_dir=tmp_path / "bridge")

    orch_adapter = OrchestratorAdapter(session=orch_session)
    _, orch_prepared = orch_adapter.prepare_turn(
        model=_MODEL,
        system_prompt=_SYSTEM,
        dynamic_messages=_MESSAGES,
    )

    bridge_adapter = ClaudeBridgeAdapter(session=bridge_session)
    bridge_prepared = bridge_adapter.prepare(
        model=_MODEL,
        messages=_MESSAGES,
        system=_SYSTEM,
        tool_compatible=False,
    )

    assert orch_prepared.body == bridge_prepared.body


def test_adapter_kind_does_not_affect_prepared_body(tmp_path):
    """Two RuntimeAdapters differing only in adapter_kind must produce the same body."""
    session_a = RuntimeSession(memory_dir=tmp_path / "a")
    session_b = RuntimeSession(memory_dir=tmp_path / "b")

    adapter_a = RuntimeAdapter(adapter_kind="orchestrator", session=session_a)
    adapter_b = RuntimeAdapter(adapter_kind="claude-bridge", session=session_b)

    prepared_a = adapter_a.prepare(
        model=_MODEL, messages=_MESSAGES, system=_SYSTEM, tool_compatible=False
    )
    prepared_b = adapter_b.prepare(
        model=_MODEL, messages=_MESSAGES, system=_SYSTEM, tool_compatible=False
    )

    assert prepared_a.body == prepared_b.body


def test_adapters_finalize_produces_consistent_mode(tmp_path):
    """Both adapter surfaces must classify the same response text the same way."""
    text = ">>> turns:1|goal:audit\n@msg role:assistant\n  |> ok"

    orch_session = RuntimeSession(memory_dir=tmp_path / "orch")
    bridge_session = RuntimeSession(memory_dir=tmp_path / "bridge")

    orch_adapter = OrchestratorAdapter(session=orch_session)
    bridge_adapter = ClaudeBridgeAdapter(session=bridge_session)

    orch_processed = orch_adapter.finalize(text=text, model=_MODEL)
    bridge_processed = bridge_adapter.finalize(text=text, model=_MODEL)

    assert orch_processed.mode == bridge_processed.mode
    assert orch_processed.content_blocks == bridge_processed.content_blocks


def test_adapter_memory_state_is_independent_per_session(tmp_path):
    """Two adapters with different sessions must not share memory state."""
    orch_session = RuntimeSession(memory_dir=tmp_path / "orch")
    bridge_session = RuntimeSession(memory_dir=tmp_path / "bridge")

    payload = ">>> turns:1|goal:orch_only\n@msg role:assistant\n  |> step done"
    orch_adapter = OrchestratorAdapter(session=orch_session)
    orch_adapter.finalize(text=payload, model=_MODEL)

    bridge_wire = bridge_session.load_memory()
    assert "orch_only" not in bridge_wire
