"""Tests for RuntimeSession state group dataclasses."""

from __future__ import annotations


class TestFallbackState:
    def test_defaults(self) -> None:
        from tok.runtime._fallback_state import FallbackState

        s = FallbackState()
        assert s.consecutive_count == 0
        assert s.baseline_only is False
        assert s.persistence_failures == 0

    def test_reset_clears(self) -> None:
        from tok.runtime._fallback_state import FallbackState

        s = FallbackState()
        s.consecutive_count = 5
        s.baseline_only = True
        s.persistence_failures = 3
        s.reset()
        assert s.consecutive_count == 0
        assert s.baseline_only is False
        assert s.persistence_failures == 0

    def test_record_fallback_event_increments(self) -> None:
        from tok.runtime._fallback_state import FallbackState

        s = FallbackState()
        s.record_fallback_event()
        assert s.consecutive_count == 1
        assert s.baseline_only is False

    def test_record_fallback_event_activates_baseline_at_threshold(self) -> None:
        from tok.runtime._fallback_state import _FALLBACK_THRESHOLD, FallbackState

        s = FallbackState()
        for _ in range(_FALLBACK_THRESHOLD):
            s.record_fallback_event()
        assert s.baseline_only is True

    def test_reset_fallback_count(self) -> None:
        from tok.runtime._fallback_state import FallbackState

        s = FallbackState()
        s.consecutive_count = 5
        s.baseline_only = True
        s.reset_fallback_count()
        assert s.consecutive_count == 0
        assert s.baseline_only is False


class TestSmoothnessState:
    def test_defaults(self) -> None:
        from tok.runtime._smoothness_state import SmoothnessState
        from tok.runtime.smoothness.models import TokMode

        s = SmoothnessState()
        assert s.latest_turn_score == 100
        assert s.latest_turn_labour_index == 0
        assert s.current_task_score == 100
        assert s.current_task_labour_index == 0
        assert s.current_tok_mode == TokMode.FULL_TOK
        assert s.event_counts == {}

    def test_reset_clears(self) -> None:
        from tok.runtime._smoothness_state import SmoothnessState
        from tok.runtime.smoothness.models import TokMode

        s = SmoothnessState()
        s.latest_turn_score = 50
        s.latest_turn_labour_index = 3
        s.current_task_score = 60
        s.current_task_labour_index = 2
        s.event_counts["some_event"] = 5
        s.reset()
        assert s.latest_turn_score == 100
        assert s.latest_turn_labour_index == 0
        assert s.current_task_score == 100
        assert s.current_task_labour_index == 0
        assert s.current_tok_mode == TokMode.FULL_TOK
        assert s.event_counts == {}


class TestLoopDetectionState:
    def test_defaults(self) -> None:
        from tok.runtime._loop_detection_state import LoopDetectionState

        s = LoopDetectionState()
        assert s.window == []
        assert s.detected is False

    def test_reset_clears(self) -> None:
        from tok.runtime._loop_detection_state import LoopDetectionState

        s = LoopDetectionState()
        s.window.append(("bash", "bash:ls"))
        s.detected = True
        s.reset()
        assert s.window == []
        assert s.detected is False

    def test_consume_loop_detected(self) -> None:
        from tok.runtime._loop_detection_state import LoopDetectionState

        s = LoopDetectionState()
        s.detected = True
        result = s.consume_loop_detected()
        assert result is True
        assert s.detected is False


class TestCacheState:
    def test_defaults(self) -> None:
        from tok.runtime._cache_state import CacheState

        s = CacheState()
        assert s.result_cache == {}
        assert s.semantic_hash_cache == {}
        assert s.observed_tool_result_ids == {}
        assert s.prepared_prompt_token_cache == {}
        assert s.predictive_cache_warm_keys == set()

    def test_reset_clears(self) -> None:
        from tok.runtime._cache_state import CacheState

        s = CacheState()
        s.result_cache["k"] = "v"
        s.semantic_hash_cache["a"] = "b"
        s.observed_tool_result_ids["x"] = None
        s.prepared_prompt_token_cache["p"] = 10
        s.predictive_cache_warm_keys.add("w")
        s.reset()
        assert s.result_cache == {}
        assert s.semantic_hash_cache == {}
        assert s.observed_tool_result_ids == {}
        assert s.prepared_prompt_token_cache == {}
        assert s.predictive_cache_warm_keys == set()


class TestHotSummaryState:
    def test_defaults(self) -> None:
        from tok.runtime._hot_summary_state import HotSummaryState

        s = HotSummaryState()
        assert s.recent_repeat_target_events == []
        assert s.records == {}
        assert s.hints_loaded_from_disk == 0

    def test_reset_clears(self) -> None:
        from tok.runtime._hot_summary_state import HotSummaryState

        s = HotSummaryState()
        s.recent_repeat_target_events.append("event")
        s.records["key"] = "val"
        s.hints_loaded_from_disk = 5
        s.reset()
        assert s.recent_repeat_target_events == []
        assert s.records == {}
        assert s.hints_loaded_from_disk == 0


class TestTelemetryState:
    def test_defaults(self) -> None:
        from tok.runtime._telemetry_state import TelemetryState

        s = TelemetryState()
        assert s.step_count == 0
        assert s.tool_names_seen == set()
        assert s.token_count == 0
        assert s.tool_density == 0.0
        assert s.context_char_count == 0
        assert s.invisible_pressure == 0
        assert s.active_tools == []
        assert s.last_tool_compatible_state == ""
        assert s.last_tool_compatible_state_fields == {}
        assert s.suppressed_failure_markers == frozenset()
        assert s.response_word_samples == []

    def test_reset_clears(self) -> None:
        from tok.runtime._telemetry_state import TelemetryState

        s = TelemetryState()
        s.step_count = 10
        s.tool_names_seen.add("bash")
        s.token_count = 500
        s.tool_density = 0.5
        s.context_char_count = 1000
        s.invisible_pressure = 3
        s.active_tools.append("bash")
        s.last_tool_compatible_state = "some_state"
        s.last_tool_compatible_state_fields["x"] = ["y"]
        s.suppressed_failure_markers = frozenset({"m1"})
        s.response_word_samples.append(100)
        s.reset()
        assert s.step_count == 0
        assert s.tool_names_seen == set()
        assert s.token_count == 0
        assert s.tool_density == 0.0
        assert s.context_char_count == 0
        assert s.invisible_pressure == 0
        assert s.active_tools == []
        assert s.last_tool_compatible_state == ""
        assert s.last_tool_compatible_state_fields == {}
        assert s.suppressed_failure_markers == frozenset()
        assert s.response_word_samples == []


class TestMacroState:
    def test_defaults(self) -> None:
        from tok.runtime._macro_state import MacroState

        s = MacroState()
        assert s.load_global_macros is True
        assert s.pending_heal == ""
        assert s.pending_heal_turn == 0

    def test_reset_clears(self) -> None:
        from tok.runtime._macro_state import MacroState

        s = MacroState()
        s.pending_heal = "my_macro"
        s.pending_heal_turn = 5
        s.reset()
        assert s.pending_heal == ""
        assert s.pending_heal_turn == 0

    def test_reset_preserves_load_global_macros(self) -> None:
        from tok.runtime._macro_state import MacroState

        s = MacroState()
        s.load_global_macros = False
        s.reset()
        assert s.load_global_macros is False


class TestFidelityState:
    def test_defaults(self) -> None:
        from tok.runtime._fidelity_state import FidelityState

        s = FidelityState()
        assert s.overrides == {}
        assert s.file_reads_by_turn == {}
        assert s.last_elevated_path == ""
        assert s.tool_required_latch_streak == 0

    def test_reset_clears(self) -> None:
        from tok.runtime._fidelity_state import FidelityState

        s = FidelityState()
        s.overrides["path"] = 2
        s.file_reads_by_turn["path"] = 1
        s.last_elevated_path = "/some/path"
        s.tool_required_latch_streak = 3
        s.reset()
        assert s.overrides == {}
        assert s.file_reads_by_turn == {}
        assert s.last_elevated_path == ""
        assert s.tool_required_latch_streak == 0


class TestUserPromptState:
    def test_defaults(self) -> None:
        from tok.runtime._user_prompt_state import UserPromptState

        s = UserPromptState()
        assert s.last_text == ""
        assert s.last_labels == ()
        assert s.request_has_tools is False
        assert s.hint_last_turn == {}

    def test_reset_clears(self) -> None:
        from tok.runtime._user_prompt_state import UserPromptState

        s = UserPromptState()
        s.last_text = "hello"
        s.last_labels = ("label1",)
        s.request_has_tools = True
        s.hint_last_turn["key"] = 3
        s.reset()
        assert s.last_text == ""
        assert s.last_labels == ()
        assert s.request_has_tools is False
        assert s.hint_last_turn == {}


class TestProjectState:
    def test_defaults(self) -> None:
        from tok.runtime._project_state import ProjectState

        s = ProjectState()
        assert s.markers == frozenset()
        assert s.files_read == set()
        assert s.files_fully_delivered == {}
        assert s.recently_edited_files == {}
        assert s.skeleton_delivered_paths == set()

    def test_reset_clears(self) -> None:
        from tok.runtime._project_state import ProjectState

        s = ProjectState()
        s.markers = frozenset({"go.mod"})
        s.files_read.add("/some/file.py")
        s.files_fully_delivered["/f"] = 1
        s.recently_edited_files["/f"] = 2
        s.skeleton_delivered_paths.add("/g")
        s.reset()
        assert s.files_read == set()
        assert s.files_fully_delivered == {}
        assert s.recently_edited_files == {}
        assert s.skeleton_delivered_paths == set()

    def test_mark_file_edited(self) -> None:
        from tok.runtime._project_state import ProjectState

        s = ProjectState()
        s.mark_file_edited("path/to/file.py", step_count=5)
        assert s.recently_edited_files["path/to/file.py"] == 5

    def test_is_recently_edited(self) -> None:
        from tok.runtime._project_state import ProjectState

        s = ProjectState()
        s.mark_file_edited("path/to/file.py", step_count=5)
        assert s.is_recently_edited("path/to/file.py", step_count=6)
        assert not s.is_recently_edited("path/to/file.py", step_count=8)
        assert not s.is_recently_edited("other.py", step_count=6)
