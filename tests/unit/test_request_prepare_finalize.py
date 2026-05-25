from __future__ import annotations

from tok.runtime.pipeline._prepare_finalize import FinalizeResult, prepare_finalize
from tok.runtime.types import PreparedRuntimeRequest


class TestFinalizeResultDefaults:
    def test_finalize_result_has_prepared_request_field(self) -> None:
        r = FinalizeResult()
        assert hasattr(r, "prepared_request")
        assert r.prepared_request is None


class TestPrepareFinalize:
    def test_prepare_finalize_returns_finalize_result(self) -> None:
        result = prepare_finalize()
        assert isinstance(result, FinalizeResult)
        assert isinstance(result.prepared_request, PreparedRuntimeRequest)

    def test_body_defaults_to_empty_dict(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.body == {}

    def test_compressed_defaults_false(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.compressed is False

    def test_saved_tokens_defaults_zero(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.input_saved_tokens == 0

    def test_type_breakdown_defaults_empty_dict(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.type_breakdown == {}

    def test_behavior_signals_defaults_empty_dict(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.behavior_signals == {}

    def test_mode_defaults_empty_string(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.mode == ""

    def test_effective_tool_compatible_defaults_false(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.effective_tool_compatible is False

    def test_request_policy_defaults_legacy(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.request_policy == "legacy_tool_compatible"

    def test_normalized_tool_events_defaults_empty_list(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.normalized_tool_events == []

    def test_baseline_prompt_tokens_defaults_zero(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.baseline_prompt_tokens == 0

    def test_prepared_prompt_tokens_defaults_zero(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.prepared_prompt_tokens == 0

    def test_saved_prompt_tokens_defaults_zero(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.saved_prompt_tokens == 0

    def test_hot_hint_tokens_added_defaults_zero(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.hot_hint_tokens_added == 0

    def test_reacquisition_tokens_avoided_estimate_defaults_zero(self) -> None:
        result = prepare_finalize()
        assert result.prepared_request.reacquisition_tokens_avoided_estimate == 0

    def test_behavior_signals_passed_through(self) -> None:
        result = prepare_finalize(behavior_signals={"test_signal": 42})
        assert result.prepared_request.behavior_signals.get("test_signal") == 42

    def test_saved_tokens_accumulated(self) -> None:
        result = prepare_finalize(saved_tokens=100)
        assert result.prepared_request.input_saved_tokens == 100

    def test_type_breakdown_passed_through(self) -> None:
        breakdown = {"text": 500, "thinking": 200}
        result = prepare_finalize(type_breakdown=breakdown)
        assert result.prepared_request.type_breakdown == breakdown

    def test_mode_passed_through(self) -> None:
        result = prepare_finalize(mode="test_mode")
        assert result.prepared_request.mode == "test_mode"

    def test_request_policy_passed_through(self) -> None:
        result = prepare_finalize(request_policy="natural_first")
        assert result.prepared_request.request_policy == "natural_first"

    def test_effective_tool_compatible_passed_through(self) -> None:
        result = prepare_finalize(effective_tool_compatible=True)
        assert result.prepared_request.effective_tool_compatible is True

    def test_request_policy_escalated_passed_through(self) -> None:
        result = prepare_finalize(request_policy_escalated=True)
        assert result.prepared_request.request_policy_escalated is True

    def test_hot_hint_metrics_accumulated(self) -> None:
        hot_hint_metrics = {"hot_hint_tokens_added": 50, "reacquisition_tokens_avoided_estimate": 30}
        result = prepare_finalize(hot_hint_metrics=hot_hint_metrics)
        assert result.prepared_request.hot_hint_tokens_added == 50
        assert result.prepared_request.reacquisition_tokens_avoided_estimate == 30

    def test_body_passed_through(self) -> None:
        body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = prepare_finalize(body=body)
        assert result.prepared_request.body == body

    def test_original_body_passed_through(self) -> None:
        original_body = {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hello"}]}
        result = prepare_finalize(original_body=original_body, body=original_body)
        assert result.prepared_request.body == original_body
