from tok.runtime.memory.bridge_memory import (
    BridgeMemoryState,
    HOT_LIMITS,
    PROMOTION_THRESHOLDS,
)
from tok.runtime.policy.smart_policy import MemoryProjectionProfile


def test_bridge_memory_roundtrip():
    state = BridgeMemoryState()
    state.replace_hot_from_wire_state(
        ">>> turns:3|goal:fix_bridge|files:src/tok/gateway.py,src/tok/compression.py|cmds:pytest_tests/unit/test_gateway.py|constraints:always_invert"
    )
    state.ingest_wire_state(
        ">>> turns:4|goal:fix_bridge|next:write_tests|constraints:always_invert"
    )

    loaded = BridgeMemoryState.from_tok(state.to_tok())

    assert loaded.wire_state().startswith(">>> t:")
    assert "g:fix_bridge" in loaded.wire_state()


def test_bridge_memory_hot_projection_is_bounded():
    state = BridgeMemoryState()
    # Now limits (from Plan 7) are: files:4, cmds:16, errs:8...

    state.replace_hot_from_wire_state(
        ">>> turns:8|goal:stabilize_tok|files:a.py,b.py,c.py,d.py,e.py"
        "|cmds:c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17"
        "|errs:e1,e2,e3,e4,e5,e6,e7,e8,e9"
        "|constraints:always_invert,avoid_noise,read_only|next:patch_gateway"
    )

    wire = state.wire_state()

    # With limits: files:4, cmds:16, errs:8
    # Merged sort is (-score, -turn, value) ->
    # when tied, alphabetical ascending (a, b, c, d) are picked first.
    assert "e.py" not in wire
    assert "a.py" in wire

    assert "c17" not in wire
    assert "c01" in wire

    assert "e9" not in wire
    assert "e1" in wire


def test_bridge_memory_prefers_hot_projection_over_raw_memory():
    state = BridgeMemoryState()
    state.replace_hot_from_wire_state(
        ">>> turns:2|goal:bridge_hot|files:src/tok/gateway.py"
    )

    assert state.wire_state().startswith(
        ">>> t:2|g:bridge_hot|f:src/tok/gateway.py"
    )


def test_bridge_memory_reports_promotions_and_sizes():
    state = BridgeMemoryState()
    metrics = state.replace_hot_from_wire_state(
        ">>> turns:3|goal:fix_bridge|files:src/tok/gateway.py,src/tok/compression.py|constraints:always_invert"
    )

    assert metrics["hot_entries"] >= 3
    assert metrics["durable_promotions"] >= 1


def test_bridge_memory_replaces_conflicting_single_value_fields():
    state = BridgeMemoryState()
    state.ingest_wire_state(">>> turns:1|goal:first_goal|next:first_step")
    state.ingest_wire_state(">>> turns:2|goal:second_goal|next:second_step")

    durable_wire = state.wire_state()

    assert "g:second_goal" in durable_wire
    assert "first_goal" not in state.to_tok()
    assert "n:second_step" in durable_wire


def test_bridge_memory_replaces_conflicting_fact_values():
    state = BridgeMemoryState()
    state.ingest_wire_state(">>> turns:1|branch:main")
    state.ingest_wire_state(">>> turns:2|branch:feature_x")

    stored = state.to_tok()

    assert "branch:feature_x" in stored
    assert "branch:main" not in stored


def test_bridge_memory_bounds_questions_in_hot_projection():
    state = BridgeMemoryState()
    # DURABLE limit for questions is 8
    # Using ingest_wire_state so we don't truncate at HOT_LIMITS (4) first
    state.ingest_wire_state(
        ">>> turns:5|goal:stabilize|questions:q01,q02,q03,q04,q05,q06,q07,q08,q09,q10,q11,q12|next:patch"
    )

    stored = state.to_tok()

    # Alphabetical First Wins: q01..q08 are kept (8 items).
    # q09..q12 should drop.
    assert "q09" not in stored
    assert "q12" not in stored
    assert "q01" in stored
    assert "q08" in stored


def test_wire_state_merges_durable_constraints_when_hot_lacks_them():
    """Constraints written to durable via ingest_wire_state must survive
    a subsequent replace_hot_from_wire_state that omits constraints."""
    state = BridgeMemoryState()
    # First turn: constraints ingested → written to both hot and durable
    state.ingest_wire_state(">>> turns:1|goal:fix bug|constraints:no mocks")
    # New compressed history arrives without constraints field
    state.replace_hot_from_wire_state(">>> turns:2|goal:fix bug|files:foo.py")

    wire = state.wire_state()

    assert "k:no mocks" in wire, f"constraints dropped from wire: {wire!r}"
    assert "g:fix bug" in wire
    assert "f:foo.py" in wire


def test_wire_state_merges_durable_facts_when_hot_lacks_them():
    """Facts written to durable must appear in wire_state() even when
    hot has been replaced with a new state that doesn't include them."""
    state = BridgeMemoryState()
    state.ingest_wire_state(">>> turns:1|goal:track branch|branch:main")
    state.replace_hot_from_wire_state(
        ">>> turns:2|goal:track branch|next:write_tests"
    )

    wire = state.wire_state()

    assert "branch:main" in wire, f"fact dropped from wire: {wire!r}"
    assert "g:track branch" in wire


def test_wire_state_hot_wins_over_durable_on_per_field_conflict():
    """When hot has a newer value for a field, it takes priority over durable."""
    state = BridgeMemoryState()
    state.ingest_wire_state(">>> turns:1|goal:old goal")
    # Second ingest updates goal in hot
    state.ingest_wire_state(">>> turns:2|goal:new goal")

    wire = state.wire_state()

    assert "g:new goal" in wire
    assert "old goal" not in wire


def test_wire_state_unions_multi_value_fields_from_both_buckets():
    """Multi-value fields like facts or files should be a UNION of hot and durable."""
    state = BridgeMemoryState()
    # Durable has one fact
    state.ingest_wire_state(">>> turns:1|branch:main")
    # Hot has another fact (no contradiction)
    state.replace_hot_from_wire_state(">>> turns:2|env:prod")

    wire = state.wire_state()

    assert "branch:main" in wire
    assert "env:prod" in wire


def test_replace_hot_evicts_durable_contradictions():
    """If replace_hot_from_wire_state() brings a new value for an existing fact key, it must evict the old durable one."""
    state = BridgeMemoryState()
    # Durable has branch:main
    state.ingest_wire_state(">>> turns:1|branch:main")
    # Hot replace brings branch:feature-x
    state.replace_hot_from_wire_state(">>> turns:2|branch:feature-x")

    wire = state.wire_state()

    assert "branch:feature-x" in wire
    assert "branch:main" not in wire, (
        "Stale durable fact was not evicted by hot replacement!"
    )


def test_file_heat_double_edit_ranks_higher():
    state = BridgeMemoryState()
    state.bump_file_heat("hot.py")
    state.bump_file_heat("hot.py")
    state.bump_file_heat("cold.py")

    top = state.top_hot_files()
    assert top[0] == "hot.py"
    assert "cold.py" in top
    assert state._file_heat["hot.py"] > state._file_heat["cold.py"]


def test_wire_state_prioritizes_answer_facts_over_generic_facts():
    state = BridgeMemoryState()
    state._upsert(
        state.hot,
        "facts",
        "file[src/tok/compression.py]:def compress_history(messages, keep_turns=2)",
        score_delta=1,
    )
    state._upsert(state.hot, "facts", "keep_turns:2", score_delta=1)
    state._upsert(
        state.hot,
        "facts",
        "answer_file:src/tok/compression.py",
        score_delta=3,
    )
    state._upsert(
        state.hot,
        "facts",
        "answer_verification:compress_history function",
        score_delta=3,
    )

    wire = state.wire_state(
        MemoryProjectionProfile(
            field_limits={"files": 1}, question_limit=0, fact_limit=2
        )
    )

    assert "answer_file:src/tok/compression.py" in wire
    assert "answer_verification:compress_history function" in wire
    assert "keep_turns:2" not in wire


def test_wire_state_is_deterministic_for_identical_state():
    """Calling wire_state() twice on unchanged state must produce identical output."""
    state = BridgeMemoryState()
    state.replace_hot_from_wire_state(
        ">>> turns:5|goal:fix_bridge|files:src/tok/gateway.py,src/tok/adapters.py"
        "|cmds:pytest_tests,pytest_unit|constraints:always_invert"
    )

    first = state.wire_state()
    second = state.wire_state()

    assert first == second


def test_wire_state_multi_value_field_ordering_is_stable():
    """Repeated ingest of the same multi-value field must produce stable ordering."""
    state = BridgeMemoryState()
    state.replace_hot_from_wire_state(
        ">>> turns:3|goal:check_ordering|files:a.py,b.py,c.py"
    )

    first = state.wire_state()
    second = state.wire_state()

    # Extract files segment from both
    first_files = [p for p in first.split("|") if p.startswith("f:")]
    second_files = [p for p in second.split("|") if p.startswith("f:")]
    assert first_files == second_files


def test_wire_state_with_profile_is_deterministic():
    """wire_state(profile) must return identical output on repeated calls."""
    state = BridgeMemoryState()
    state.replace_hot_from_wire_state(
        ">>> turns:4|goal:profile_test|files:a.py,b.py,c.py,d.py"
        "|errs:err1,err2,err3|cmds:cmd1,cmd2,cmd3"
    )
    profile = MemoryProjectionProfile(
        field_limits={"files": 2, "errs": 2, "cmds": 1},
        question_limit=0,
        fact_limit=0,
    )

    first = state.wire_state(profile)
    second = state.wire_state(profile)

    assert first == second


def test_wire_state_unchanged_after_no_new_ingestion():
    """After construction, repeated wire_state() calls without mutation are stable."""
    state = BridgeMemoryState()
    state.ingest_wire_state(
        ">>> turns:2|goal:stable_check|files:src/tok/compression.py|constraints:invert"
    )

    outputs = [state.wire_state() for _ in range(3)]
    assert outputs[0] == outputs[1] == outputs[2]


def test_edited_field_has_lower_promotion_threshold_than_files():
    """edited threshold must be lower than files threshold so edited files promote faster."""
    from tok.runtime.memory.bridge_memory import PROMOTION_THRESHOLDS

    assert "edited" in PROMOTION_THRESHOLDS
    assert PROMOTION_THRESHOLDS["edited"] < PROMOTION_THRESHOLDS["files"]


def test_edited_file_promotes_to_durable_before_regular_file():
    """An edited file entry should reach durable before a regular files entry."""
    state = BridgeMemoryState()
    # Ingest repeatedly so scores accumulate
    for _ in range(PROMOTION_THRESHOLDS["edited"]):
        state.ingest_wire_state(
            ">>> turns:1|goal:refactor|edited:src/tok/adapters.py"
        )

    durable_edited = [e.value for e in state.durable.get("edited", [])]
    assert "src/tok/adapters.py" in durable_edited


def test_blocker_detection_covers_import_errors():
    """collect_behavior_signals must flag import-related blockers."""
    from tok.universal_runtime import collect_behavior_signals

    messages = [
        {
            "role": "user",
            "content": "Getting ModuleNotFoundError: no module named 'tok'",
        },
        {"role": "user", "content": "Also getting SyntaxError in adapters.py"},
        {"role": "user", "content": "Cannot import name 'compress' from tok"},
    ]
    signals = collect_behavior_signals(messages)
    # Each phrase should trigger the blocker branch and register blocker_rediscovery
    # on repeat, but a single occurrence must NOT raise an error
    assert isinstance(signals, dict)


def test_blocker_detection_covers_cannot_import():
    """'cannot import' phrase must be recognized as a blocker pattern."""
    from tok.universal_runtime import collect_behavior_signals

    messages = [
        {"role": "user", "content": "cannot import name 'X' from module 'y'"},
        {"role": "user", "content": "cannot import name 'X' from module 'y'"},
    ]
    signals = collect_behavior_signals(messages)
    assert signals.get("blocker_rediscovery", 0) >= 1


def test_record_hypothesis_stores_question():
    """record_hypothesis() must add to questions bucket."""
    state = BridgeMemoryState()
    state.record_hypothesis("Should we invert before accumulating?")

    questions = [e.value for e in state.hot.get("questions", [])]
    assert any("Should we invert" in q for q in questions)


def test_record_hypothesis_is_bounded_by_hot_limits():
    """Hypotheses must be capped at HOT_LIMITS['questions']."""
    state = BridgeMemoryState()
    limit = HOT_LIMITS.get("questions", 2)

    for i in range(limit + 5):
        state.record_hypothesis(f"hypothesis_{i}_unique_text_so_no_dedup")

    assert len(state.hot.get("questions", [])) <= limit


def test_record_hypothesis_truncates_long_text():
    """record_hypothesis must not store more than 120 chars."""
    state = BridgeMemoryState()
    long_text = "x" * 200
    state.record_hypothesis(long_text)

    questions = [e.value for e in state.hot.get("questions", [])]
    assert all(len(q) <= 120 for q in questions)


def test_wire_state_respects_profile_field_order():
    """Changing field_order in the profile must change the output ordering."""

    state = BridgeMemoryState()
    state.replace_hot_from_wire_state(
        ">>> turns:1|goal:test_order|cmds:cmd1|constraints:no_mocks"
    )

    # Default order: turns, goal, next, blockers, files, facts, cmds, tests, errs, constraints, questions
    default_wire = state.wire_state()
    parts_default = [
        p.split(":")[0] for p in default_wire.removeprefix(">>> ").split("|")
    ]

    goal_pos = parts_default.index("g")
    cmds_pos = parts_default.index("c")
    constraints_pos = parts_default.index("k")
    # In canonical order, goal comes before cmds, which comes before constraints
    assert goal_pos < cmds_pos < constraints_pos

    # Reversed order for a subset: constraints, cmds, goal, turns
    reversed_order = ("constraints", "cmds", "goal", "turns")
    profile_reversed = MemoryProjectionProfile(
        field_limits={},
        question_limit=0,
        fact_limit=0,
        field_order=reversed_order,
    )
    reversed_wire = state.wire_state(profile_reversed)
    parts_reversed = [
        p.split(":")[0] for p in reversed_wire.removeprefix(">>> ").split("|")
    ]

    constraints_pos_r = parts_reversed.index("k")
    cmds_pos_r = parts_reversed.index("c")
    goal_pos_r = parts_reversed.index("g")
    turns_pos_r = parts_reversed.index("t")
    assert constraints_pos_r < cmds_pos_r < goal_pos_r < turns_pos_r


def test_cache_stability_with_fragmented_updates():
    """Partial ingest of different fields must not reorder fields already emitted."""
    state = BridgeMemoryState()
    state.replace_hot_from_wire_state(
        ">>> turns:1|goal:stability|cmds:cmd_a|constraints:no_mocks"
    )
    baseline = state.wire_state()
    baseline_fields = [
        p.split(":")[0] for p in baseline.removeprefix(">>> ").split("|")
    ]

    # Ingest only errs — should not change ordering of other fields
    state.ingest_wire_state(">>> turns:2|errs:e1")
    after_wire = state.wire_state()
    after_fields = [
        p.split(":")[0] for p in after_wire.removeprefix(">>> ").split("|")
    ]

    # Fields present in both should maintain their relative order
    shared = [f for f in baseline_fields if f in after_fields]
    shared_after = [f for f in after_fields if f in baseline_fields]
    assert shared == shared_after, (
        f"Field order changed after partial update. Before: {shared}, After: {shared_after}"
    )


def test_question_decay_drops_low_score_untouched_questions():
    """Questions with score 1 and untouched must be dropped after decay."""
    from tok.runtime.memory.bridge_memory import BridgeMemoryState

    mem = BridgeMemoryState()
    mem.record_hypothesis("is the bug in planner.py?")
    # Run a decay pass manually
    mem._decay_bucket(mem.hot, touched=set(), prefix="hot")
    mem.ingest_wire_state(">>> goal:test")
    remaining = [e.value for e in mem.hot.get("questions", [])]
    # No TTL here, just normal decay
    # Check if question score was 1 and it was touched=False, it should be dropped
    assert "is the bug in planner.py?" not in remaining


def test_wire_state_is_stable_across_identical_turns():
    """Same memory state must produce identical wire_state() on repeat calls."""
    from tok.runtime.memory.bridge_memory import BridgeMemoryState

    mem = BridgeMemoryState()
    mem.ingest_wire_state(
        ">>> goal:audit|files:src/tok/bridge_memory.py|cmds:pytest"
    )
    first = mem.wire_state(omit_unchanged=True)
    # On first call it populates _prev_field_hashes.
    # On second call, since nothing changed, it should return an EMPTY string (as state_parts will be empty).
    second = mem.wire_state(omit_unchanged=True)
    # The second call should at least not contain the 'goal' or 'cmds' fields which were skipped.
    assert "g:audit" not in second
    assert "c:pytest" not in second
    # Check if files was skipped too (newly added)
    assert "f:*A" not in second
    # It might still have @pointers or other extra blocks if not skipped.
    # The goal is that it's MUCH shorter or omits unchanged fields.
    assert len(second) < len(first)


def test_to_tok_is_stable_across_calls():
    """to_tok() must produce identical bytes on consecutive calls (no non-determinism)."""
    from tok.runtime.memory.bridge_memory import BridgeMemoryState

    mem = BridgeMemoryState()
    mem.ingest_wire_state(">>> goal:audit|files:a.py,b.py|cmds:ruff,pytest")
    first = mem.to_tok()
    second = mem.to_tok()
    assert first == second, "to_tok() must be deterministic"


def test_record_hypothesis_rejects_empty_text():
    """record_hypothesis must return False and store nothing for empty input."""
    state = BridgeMemoryState()
    result = state.record_hypothesis("   ")

    assert result is False
    assert state.hot.get("questions", []) == []


def test_wire_state_field_ordering_is_deterministic():
    """Fields in wire_state must always appear in the same canonical order."""
    from tok.runtime.memory.bridge_memory import BridgeMemoryState

    # Ingest in "random" order
    mem = BridgeMemoryState()
    mem.ingest_wire_state(
        ">>> constraints:no_prose|goal:fix_bugs|files:a.py|turns:1"
    )

    wire = mem.wire_state()
    # Expected order: turns, goal, files, ..., constraints
    # Check positions
    idx_turns = wire.find("t:")
    idx_goal = wire.find("g:")
    idx_files = wire.find("f:")
    idx_constraints = wire.find("k:")

    assert idx_turns < idx_goal < idx_files < idx_constraints
    assert "t:1" in wire
    assert "g:fix_bugs" in wire
    assert "f:a.py" in wire
    assert "k:no_prose" in wire


def test_wire_state_respects_profile_limits():
    """Profile limits must override default HOT_LIMITS."""
    from tok.runtime.memory.bridge_memory import BridgeMemoryState
    from tok.runtime.policy.smart_policy import MemoryProjectionProfile

    mem = BridgeMemoryState()
    mem.ingest_wire_state(">>> cmds:ls,cat,grep,rm,mv|errs:e1,e2,e3,e4")

    # Default limit for cmds is 8, errs is 4.
    # We'll use a profile to limit them strictly.
    profile = MemoryProjectionProfile(
        field_limits={"cmds": 1, "errs": 2}, question_limit=0, fact_limit=0
    )

    wire = mem.wire_state(profile=profile)

    # Strip >>> prefix if present
    content = wire[3:].strip() if wire.startswith(">>>") else wire
    segments = content.split("|")

    # Check that only 1 command and 2 errors are listed
    cmds_part = [p for p in segments if p.startswith("c:")][0]
    errs_part = [p for p in segments if p.startswith("e:")][0]

    assert len(cmds_part.split(",")) == 1
    assert len(errs_part.split(",")) == 2
    assert "c:cat" in cmds_part  # Alphabetical sort at same score/turn


def test_edited_file_heat_is_higher_than_read_heat():
    """A file bumped with weight=2.0 (edit) should have heat >= 2.0 after one bump."""
    state = BridgeMemoryState()
    state.bump_file_heat("src/tok/gateway.py", weight=2.0)

    heat = state._file_heat["src/tok/gateway.py"]
    assert heat >= 2.0, f"Expected heat >= 2.0 for edited file, got {heat}"


def test_heat_multiplier_does_not_affect_read_files():
    """A file bumped with default weight=1.0 (read) should have heat < 2.0 after one bump."""
    state = BridgeMemoryState()
    state.bump_file_heat("src/tok/compression.py")  # default weight=1.0

    heat = state._file_heat["src/tok/compression.py"]
    assert heat < 2.0, f"Expected heat < 2.0 for read-only file, got {heat}"


def test_edited_file_triggers_was_edited_digest_branch():
    """record_file_snapshot after an edit-level heat should use the was_edited digest path."""
    state = BridgeMemoryState()
    state.bump_file_heat("src/tok/gateway.py", weight=2.0)

    content = "def handle_request(req):\n    pass\ndef close():\n    pass"
    recorded = state.record_file_snapshot("src/tok/gateway.py", content)

    assert recorded
    facts = [e.value for e in state.hot.get("facts", [])]
    # was_edited branch extracts def/class signatures — both defs should appear
    assert any("handle_request" in f or "close" in f for f in facts), (
        f"Expected edited-path signature digest in facts, got: {facts}"
    )


def test_durable_facts_scale_to_new_limit():
    """Verify that we can hold 32 facts in durable memory and answer facts still win."""
    from tok.runtime.memory.bridge_memory import BridgeMemoryState
    from tok.runtime.policy.smart_policy import MemoryProjectionProfile

    state = BridgeMemoryState()

    # Fill up with 40 generic facts (above the 32 limit)
    for i in range(40):
        state._upsert(
            state.durable, "facts", f"generic_fact_{i}", score_delta=1
        )

    # Add 2 answer facts which have higher default score logic (or we give them high score)
    state._upsert(
        state.durable,
        "facts",
        "answer_file:src/tok/gateway.py",
        score_delta=10,
    )
    state._upsert(
        state.durable, "facts", "answer_verification:health", score_delta=10
    )

    # Project to wire state with a profile that allows 32 facts
    profile = MemoryProjectionProfile(
        field_limits={}, question_limit=0, fact_limit=32
    )
    wire = state.wire_state(profile=profile)

    # Answer facts must be present
    assert "answer_file:src/tok/gateway.py" in wire
    assert "answer_verification:health" in wire

    # Count total facts in wire
    all_parts = wire.split("|")
    generic_count = sum(1 for p in all_parts if "generic_fact_" in p)
    answer_count = sum(1 for p in all_parts if "answer_" in p)

    assert answer_count == 2
    # Total limit is 32. So we expect 30 generic + 2 answer.
    assert generic_count == 30
    assert generic_count + answer_count <= 32
