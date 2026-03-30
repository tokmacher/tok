from tok.smart_policy import (
    advance_state,
    identify_model_family,
    initial_state,
    policy_for_model,
    pressure_score,
)


def test_identify_model_family_maps_known_families():
    assert identify_model_family("claude-sonnet-4").key == "anthropic:claude"
    assert identify_model_family("gpt-5.4").key == "openai:gpt"
    assert (
        identify_model_family("google/gemini-2.0-flash").key == "google:gemini"
    )


def test_unknown_family_defaults_to_balanced():
    policy = policy_for_model("mystery/foo-1")
    state = initial_state(policy)

    assert policy.family.key == "mystery:foo"
    assert state.mode == "balanced"


def test_family_policies_can_differ():
    claude = policy_for_model("claude-sonnet-4")
    gemini = policy_for_model("google/gemini-2.0-flash")

    assert claude.default_mode == "aggressive"
    assert gemini.default_mode == "balanced"


def test_adaptive_state_shifts_and_relaxes_by_family_thresholds():
    gemini = policy_for_model("google/gemini-2.0-flash")
    state = initial_state(gemini)
    state = advance_state(
        gemini, state, {"repeat_file_read": 1, "repeat_search": 1}
    )
    assert state.mode == "balanced"

    claude = policy_for_model("claude-sonnet-4")
    state = initial_state(claude)
    state = advance_state(
        claude, state, {"repeat_file_read": 2, "repeat_search": 1}
    )
    assert state.mode == "recovery"
    state = advance_state(claude, state, {})
    state = advance_state(claude, state, {})
    assert state.mode == "aggressive"


def test_tool_contract_failure_contributes_to_family_pressure():
    assert pressure_score({"tool_contract_failure": 1}) == 2

    gemini = policy_for_model("google/gemini-2.0-flash")
    state = initial_state(gemini)
    state = advance_state(gemini, state, {"tool_contract_failure": 1})

    assert state.mode == "balanced"


def test_answer_anchor_reacquisition_attempt_is_telemetry_only_for_family_pressure():
    assert pressure_score({"answer_anchor_reacquisition_attempt": 1}) == 0

    gemini = policy_for_model("google/gemini-2.0-flash")
    state = initial_state(gemini)
    state = advance_state(
        gemini, state, {"answer_anchor_reacquisition_attempt": 1}
    )

    assert state.mode == gemini.default_mode
