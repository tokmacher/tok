from tok.smart_policy import (
    advance_state,
    detect_task_type,
    identify_model_family,
    initial_state,
    policy_for_model,
    pressure_score,
    select_optimal_mode,
)


def test_identify_model_family_maps_known_families():
    assert identify_model_family("claude-sonnet-4").key == "anthropic:claude"
    assert identify_model_family("gpt-5.4").key == "openai:gpt"
    assert (
        identify_model_family("google/gemini-2.0-flash").key == "google:gemini"
    )
    assert identify_model_family("deepseek-v3").key == "deepseek:deepseek"


def test_unknown_family_defaults_to_balanced():
    policy = policy_for_model("mystery/foo-1")
    state = initial_state(policy)

    assert policy.family.key == "universal:universal"
    assert state.mode == "tok-universal"


def test_family_policies_can_differ():
    claude = policy_for_model("claude-sonnet-4")
    gemini = policy_for_model("google/gemini-2.0-flash")

    assert claude.default_mode == "tok-universal"
    assert gemini.default_mode == "tok-universal"
    assert claude.memory_profiles.keys() == gemini.memory_profiles.keys()


def test_adaptive_state_shifts_and_relaxes_by_family_thresholds():
    gemini = policy_for_model("google/gemini-2.0-flash")
    state = initial_state(gemini)
    state = advance_state(
        gemini, state, {"repeat_file_read": 1, "repeat_search": 1}
    )
    assert state.mode == "tok-universal"

    claude = policy_for_model("claude-sonnet-4")
    state = initial_state(claude)
    state = advance_state(
        claude, state, {"repeat_file_read": 2, "repeat_search": 1}
    )
    assert state.mode == "tok-universal"
    state = advance_state(claude, state, {})
    state = advance_state(claude, state, {})
    assert state.mode == "tok-universal"


def test_tool_contract_failure_contributes_to_family_pressure():
    assert pressure_score({"tool_contract_failure": 1}) == 2

    gemini = policy_for_model("google/gemini-2.0-flash")
    state = initial_state(gemini)
    state = advance_state(gemini, state, {"tool_contract_failure": 1})

    assert state.mode == "tok-universal"


def test_answer_anchor_reacquisition_attempt_is_telemetry_only_for_family_pressure():
    assert pressure_score({"answer_anchor_reacquisition_attempt": 1}) == 0

    gemini = policy_for_model("google/gemini-2.0-flash")
    state = initial_state(gemini)
    state = advance_state(
        gemini, state, {"answer_anchor_reacquisition_attempt": 1}
    )

    assert state.mode == gemini.default_mode


def test_detect_task_type_identifies_coding():
    coding_tools = ["write_file", "edit_file", "run_shell", "pytest"]
    task_type, confidence = detect_task_type(coding_tools)
    assert task_type == "coding"
    assert confidence > 0.5


def test_detect_task_type_identifies_research():
    research_tools = ["grep_search", "view_file", "list_dir", "code_search"]
    task_type, confidence = detect_task_type(research_tools)
    assert task_type == "research"
    assert confidence > 0.5


def test_detect_task_type_returns_mixed_for_empty():
    task_type, confidence = detect_task_type([])
    assert task_type == "mixed"
    assert confidence == 0.0


def test_select_optimal_mode_by_family_and_task():
    assert select_optimal_mode("claude-sonnet-4", "coding") == "tok-universal"
    assert select_optimal_mode("gpt-4", "research") == "tok-universal"
    assert select_optimal_mode("deepseek-v3", "coding") == "tok-universal"
    assert select_optimal_mode("deepseek-v3", "research") == "tok-universal"
    assert select_optimal_mode("unknown-model", "mixed") == "tok-universal"


def test_advance_state_updates_task_type():
    claude = policy_for_model("claude-sonnet-4")
    state = initial_state(claude)
    assert state.task_type == "mixed"

    # Advance with coding tools
    state = advance_state(
        claude, state, {}, tool_names=["write_file", "edit_file", "run_shell"]
    )
    assert state.task_type == "coding"
    assert state.task_confidence > 0
