"""Regression tests for BridgeSession delegation removal (Plan 6).

Verifies that BridgeSession no longer has shallow delegation methods
and that callers access runtime_session directly where needed.
"""

from __future__ import annotations


def test_bridge_session_has_no_shallow_delegation_methods():
    import dataclasses

    from tok.gateway import BridgeSession

    fields = {f.name for f in dataclasses.fields(BridgeSession)}
    assert "runtime_session" in fields

    shallow_methods = [
        "write_memory",
        "policy_snapshot",
        "load_memory",
        "refresh_hot_memory",
        "update_family_mode",
        "consume_behavior_signals",
        "_bump_signals",
        "_save_bridge_memory",
    ]
    for method_name in shallow_methods:
        assert not hasattr(BridgeSession, method_name) or isinstance(getattr(BridgeSession, method_name), property), (
            f"BridgeSession still has shallow delegation method: {method_name}"
        )
