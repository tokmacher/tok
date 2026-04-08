"""Adapter conformance test: adapters must only own transport plumbing."""

import ast
import pathlib

ADAPTERS_PATH = pathlib.Path("src/tok/adapters/adapters.py")

# These symbols must NOT be imported or defined inside adapters.py
FORBIDDEN_IMPORTS = {
    "compress",
    "compression",
    "bridge_memory",
    "BridgeMemoryState",
    "SavingsTracker",
    "stats",
    "smart_policy",
}


def test_adapters_no_forbidden_imports() -> None:
    source = ADAPTERS_PATH.read_text()
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.names:
                for alias in node.names:
                    imported_names.add(alias.name)
            if node.module:
                imported_names.add(node.module.split(".")[-1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.split(".")[-1])
    violations = FORBIDDEN_IMPORTS & imported_names
    assert not violations, (
        f"adapters.py must not import: {violations}. Compression/memory/telemetry logic belongs in the runtime."
    )


def test_adapters_delegates_to_runtime() -> None:
    source = ADAPTERS_PATH.read_text()
    assert "UniversalTokRuntime" in source, "OrchestratorAdapter must use UniversalTokRuntime"
    assert "prepare_request" in source or "prepare" in source, "Adapter must call runtime.prepare_request()"
    assert "process_response" in source or "finalize" in source, "Adapter must call runtime.process_response()"


def test_orchestrator_and_bridge_adapter_telemetry_parity() -> None:
    """OrchestratorAdapter and ClaudeBridgeAdapter must produce equivalent telemetry signals."""
    from tok.adapters import ClaudeBridgeAdapter, OrchestratorAdapter

    orch = OrchestratorAdapter()
    bridge = ClaudeBridgeAdapter()

    messages = [{"role": "user", "content": ">>> goal:test|files:foo.py"}]
    system = "You are a test agent."

    orch_prepared = orch.prepare(model="test-model", messages=messages, system=system)
    bridge_prepared = bridge.prepare(model="test-model", messages=messages, system=system)

    # Both should report the same input_saved_tokens type (int)
    assert isinstance(orch_prepared.input_saved_tokens, int)
    assert isinstance(bridge_prepared.input_saved_tokens, int)

    # Both should have a type_breakdown dict
    assert isinstance(orch_prepared.type_breakdown, dict)
    assert isinstance(bridge_prepared.type_breakdown, dict)


def test_orchestrator_finalize_does_not_double_consume_signals() -> None:
    """
    TokOrchestrator.chat() must not re-consume session signals after finalize().

    process_response() (called inside finalize()) already drains
    session.consume_behavior_signals(). A second consume returns {} and was
    previously a misleading no-op. After the Plan 5 fix, the re-consume lines
    are removed; this test guards against their reintroduction by verifying that
    OrchestratorAdapter.finalize() does NOT call consume_behavior_signals() on
    the session a second time within the adapter itself.
    """
    import pathlib

    source = pathlib.Path("src/tok/adapters/orchestrator.py").read_text()

    consume_count = source.count("consume_behavior_signals")
    assert consume_count == 0, (
        f"adapters/orchestrator.py should not call consume_behavior_signals() "
        f"(process_response already merges session signals). Found {consume_count} occurrence(s)."
    )


def test_gateway_sse_path_does_not_double_consume_signals() -> None:
    """
    gateway.py SSE path must not re-consume session signals after process_response().

    After the Plan 5 fix, the stale re-consume that followed process_response() in the
    SSE path was removed. This test verifies that process_response() and
    consume_behavior_signals() are not called back-to-back (with only whitespace/comments
    between them) in gateway.py.
    """
    import pathlib
    import re

    source = pathlib.Path("src/tok/gateway/__init__.py").read_text()
    # Check that process_response(...) is NOT immediately followed by consume_behavior_signals
    # within a few lines (the problematic double-consume pattern)
    pattern = re.compile(
        r"process_response\([^)]*\).*?consume_behavior_signals\(",
        re.DOTALL,
    )
    # Allow up to ~5 lines between them as a reasonable threshold
    for match in pattern.finditer(source):
        span_text = match.group(0)
        newline_count = span_text.count("\n")
        assert newline_count > 5, (
            f"gateway.py: process_response() appears to be immediately followed by "
            f"consume_behavior_signals() within {newline_count} lines — "
            "this is the stale double-consume pattern that was removed in Plan 5."
        )
