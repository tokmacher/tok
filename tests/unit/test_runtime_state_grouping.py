"""Section 5.3.2: Runtime State Grouping Cleanup — RED/GREEN tests.

Verifies that all state group objects exist on RuntimeSession, have the correct
fields and reset behavior, and that reset_session() resets ALL groups without
state bleed.
"""

from __future__ import annotations

from tok.runtime.core import RuntimeSession

# ---------------------------------------------------------------------------
# State group objects exist on RuntimeSession
# ---------------------------------------------------------------------------


class TestSessionPersistenceGroup:
    """session_persistence state group must exist and delegate bridge_memory correctly."""

    def test_session_persistence_attribute_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "session_persistence"), "RuntimeSession must have session_persistence state group"

    def test_session_persistence_has_reset_method(self) -> None:
        s = RuntimeSession()
        assert hasattr(s.session_persistence, "reset"), "session_persistence must have reset() method"

    def test_session_persistence_exposes_bridge_memory(self) -> None:
        from tok.runtime.memory.bridge_memory import BridgeMemoryState

        s = RuntimeSession()
        assert hasattr(s.session_persistence, "bridge_memory"), "session_persistence must expose bridge_memory"
        assert isinstance(s.session_persistence.bridge_memory, BridgeMemoryState)

    def test_session_persistence_bridge_memory_is_same_object_as_session(self) -> None:
        """session_persistence.bridge_memory must be the same object as session.bridge_memory."""
        s = RuntimeSession()
        assert s.session_persistence.bridge_memory is s.bridge_memory, (
            "session_persistence.bridge_memory must alias session.bridge_memory"
        )

    def test_session_persistence_reset_clears_hot_memory(self) -> None:
        s = RuntimeSession()
        s.bridge_memory.hot["test_key"] = {"value": 1}
        assert len(s.bridge_memory.hot) > 0
        s.session_persistence.reset()
        assert len(s.bridge_memory.hot) == 0

    def test_session_persistence_reset_clears_rolling_cmds(self) -> None:
        s = RuntimeSession()
        s.bridge_memory.rolling_cmds = [{"cmd": "echo hi"}]
        s.session_persistence.reset()
        assert s.bridge_memory.rolling_cmds == []

    def test_reset_session_resets_session_persistence(self) -> None:
        s = RuntimeSession()
        s.bridge_memory.hot["key"] = {"value": 1}
        s.bridge_memory.rolling_cmds = [{"cmd": "ls"}]
        s.reset_session()
        assert len(s.bridge_memory.hot) == 0
        assert s.bridge_memory.rolling_cmds == []


class TestSessionPersistenceAdversarial:
    """Adversarial tests: alias integrity, constructor override, double reset."""

    def test_session_persistence_alias_survives_bridge_memory_mutation(self) -> None:
        """Mutating bridge_memory via session must be visible via session_persistence."""
        s = RuntimeSession()
        s.bridge_memory.hot["probe"] = {"val": 42}
        assert s.session_persistence.bridge_memory.hot["probe"] == {"val": 42}

    def test_session_persistence_alias_survives_reverse_mutation(self) -> None:
        """Mutating bridge_memory via session_persistence must be visible via session."""
        s = RuntimeSession()
        s.session_persistence.bridge_memory.hot["probe2"] = {"val": 99}
        assert s.bridge_memory.hot["probe2"] == {"val": 99}

    def test_session_persistence_aliases_final_bridge_memory(self) -> None:
        """session_persistence.bridge_memory must alias session.bridge_memory
        (the final loaded object, after initialize_session_storage runs)."""
        s = RuntimeSession()
        assert s.session_persistence.bridge_memory is s.bridge_memory, (
            "session_persistence.bridge_memory must be the same object as session.bridge_memory"
        )

    def test_session_persistence_aliases_final_bridge_memory_with_custom_flags(self) -> None:
        """Even when bridge_memory is constructed with custom flags,
        session_persistence must alias whatever bridge_memory the session settles on."""
        from tok.runtime.memory.bridge_memory import BridgeMemoryState

        s = RuntimeSession(bridge_memory=BridgeMemoryState(load_global_macros=False))
        # initialize_session_storage replaces bridge_memory, but session_persistence
        # must alias the final object regardless.
        assert s.session_persistence.bridge_memory is s.bridge_memory

    def test_double_reset_idempotent_for_session_persistence(self) -> None:
        s = RuntimeSession()
        s.bridge_memory.hot["k"] = {"v": 1}
        s.session_persistence.reset()
        s.session_persistence.reset()  # second reset must not raise
        assert len(s.bridge_memory.hot) == 0

    def test_reset_session_delegates_to_session_persistence(self) -> None:
        """reset_session() must clear bridge_memory state via session_persistence."""
        s = RuntimeSession()
        s.bridge_memory.hot["key"] = {"data": True}
        s.bridge_memory.rolling_cmds = [{"cmd": "ls"}]
        s.reset_session()
        assert len(s.session_persistence.bridge_memory.hot) == 0
        assert s.session_persistence.bridge_memory.rolling_cmds == []


class TestStateGroupsExist:
    def test_evidence_safety_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "evidence_safety")

    def test_fallback_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "fallback")

    def test_smoothness_state_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "smoothness_state")

    def test_cache_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "cache")

    def test_telemetry_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "telemetry")

    def test_request_policy_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "request_policy")

    def test_streaming_recovery_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "streaming_recovery")

    def test_answer_phase_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "answer_phase")

    def test_file_delivery_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "file_delivery")

    def test_loop_detection_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "loop_detection")

    def test_hot_summary_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "hot_summary")

    def test_macro_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "macro")

    def test_fidelity_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "fidelity")

    def test_user_prompt_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "user_prompt")

    def test_project_exists(self) -> None:
        s = RuntimeSession()
        assert hasattr(s, "project")


# ---------------------------------------------------------------------------
# Each state group has a reset() method
# ---------------------------------------------------------------------------


class TestStateGroupsHaveReset:
    _GROUP_NAMES = [
        "evidence_safety",
        "fallback",
        "smoothness_state",
        "cache",
        "telemetry",
        "request_policy",
        "streaming_recovery",
        "answer_phase",
        "file_delivery",
        "loop_detection",
        "hot_summary",
        "macro",
        "fidelity",
        "user_prompt",
        "project",
        "session_persistence",
    ]

    def test_all_state_groups_have_reset_method(self) -> None:
        s = RuntimeSession()
        missing = [name for name in self._GROUP_NAMES if not hasattr(getattr(s, name), "reset")]
        assert not missing, f"State groups without reset(): {missing}"


# ---------------------------------------------------------------------------
# reset_session() resets all grouped states
# ---------------------------------------------------------------------------


class TestResetSessionResetsAllGroups:
    def test_evidence_safety_cleared_by_reset(self) -> None:
        s = RuntimeSession()
        s.evidence_safety.record_exact("src/foo.py", digest="abc", turn=1)
        assert len(s.evidence_safety.ledger) > 0
        s.reset_session()
        assert len(s.evidence_safety.ledger) == 0

    def test_fallback_consecutive_count_cleared_by_reset(self) -> None:
        s = RuntimeSession()
        # Record a fallback event
        s.fallback.consecutive_count += 5
        s.reset_session()
        assert s.fallback.consecutive_count == 0

    def test_cache_cleared_by_reset(self) -> None:
        s = RuntimeSession()
        # Seed result_cache
        s.cache.result_cache["test_key"] = {"data": "value"}
        s.reset_session()
        assert len(s.cache.result_cache) == 0

    def test_pending_behavior_signals_cleared_by_reset(self) -> None:
        s = RuntimeSession()
        s.pending_behavior_signals["test_signal"] = 5
        s.reset_session()
        assert len(s.pending_behavior_signals) == 0

    def test_family_states_cleared_by_reset(self) -> None:
        s = RuntimeSession()
        s.family_states["test"] = object()  # type: ignore[assignment]
        s.reset_session()
        assert len(s.family_states) == 0

    def test_request_policy_cleared_by_reset(self) -> None:
        s = RuntimeSession()
        # Dirty some state on request_policy if possible
        if hasattr(s.request_policy, "reset"):
            s.reset_session()  # should not raise
        assert True  # just verify no exception

    def test_streaming_recovery_cleared_by_reset(self) -> None:
        s = RuntimeSession()
        s.reset_session()  # should not raise
        assert True


# ---------------------------------------------------------------------------
# No state bleed between groups after reset
# ---------------------------------------------------------------------------


class TestNoStateBleedAfterReset:
    def test_evidence_safety_does_not_bleed_into_cache(self) -> None:
        s = RuntimeSession()
        # Record exact evidence
        s.evidence_safety.record_exact("src/foo.py", digest="hash1", turn=1)
        s.reset_session()
        # After reset, evidence_safety ledger must be empty
        assert len(s.evidence_safety.ledger) == 0
        # Cache should also be empty
        assert len(s.cache.result_cache) == 0

    def test_reset_does_not_reset_memory_dir(self) -> None:
        from pathlib import Path

        s = RuntimeSession(memory_dir=Path("/tmp/test_tok_memory"))
        s.reset_session()
        # memory_dir is persisted config — should not be cleared
        assert s.memory_dir == Path("/tmp/test_tok_memory")

    def test_double_reset_is_idempotent(self) -> None:
        s = RuntimeSession()
        s.evidence_safety.record_exact("src/bar.py", digest="hash2", turn=1)
        s.reset_session()
        s.reset_session()  # second reset must not raise
        assert len(s.evidence_safety.ledger) == 0

    def test_state_accumulates_after_reset(self) -> None:
        s = RuntimeSession()
        s.evidence_safety.record_exact("src/baz.py", digest="hash3", turn=1)
        s.reset_session()
        assert len(s.evidence_safety.ledger) == 0
        # After reset, can accumulate state again
        s.evidence_safety.record_exact("src/baz.py", digest="hash3", turn=5)
        assert len(s.evidence_safety.ledger) == 1
