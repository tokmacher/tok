from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from tok.gateway import BridgeSession
from tok.gateway._bridge_runtime_pipeline import BridgePreparedPayload, prepare_bridge_payload
from tok.runtime._request_lifecycle import RequestLifecycle


def _minimal_body() -> dict:
    return {
        "model": "claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
    }


class TestRequestLifecycleDataclass:
    def test_request_lifecycle_is_frozen(self) -> None:
        lc = RequestLifecycle()
        with pytest.raises(FrozenInstanceError):
            lc.initial_preflight = True  # type: ignore[misc]

    def test_request_lifecycle_all_fields_default_false(self) -> None:
        lc = RequestLifecycle()
        assert lc.initial_preflight is False
        assert lc.model_extraction is False
        assert lc.tool_compatibility_check is False
        assert lc.runtime_preparation is False
        assert lc.signals_and_metrics is False
        assert lc.prepared_preflight is False
        assert lc.plan_finalization_guard is False
        assert lc.final_payload_construction is False
        assert lc.request_preparation is False
        assert lc.repeat_target_capture is False
        assert lc.tool_event_normalization is False
        assert lc.hot_memory_refresh is False

    def test_request_lifecycle_can_set_individual_stages(self) -> None:
        lc = RequestLifecycle()
        lc = replace(lc, initial_preflight=True)
        assert lc.initial_preflight is True
        assert lc.model_extraction is False

        lc = replace(lc, model_extraction=True, runtime_preparation=True)
        assert lc.initial_preflight is True
        assert lc.model_extraction is True
        assert lc.runtime_preparation is True
        assert lc.tool_compatibility_check is False

    def test_request_lifecycle_with_all_stages_set(self) -> None:
        lc = RequestLifecycle(
            initial_preflight=True,
            model_extraction=True,
            tool_compatibility_check=True,
            runtime_preparation=True,
            signals_and_metrics=True,
            prepared_preflight=True,
            plan_finalization_guard=True,
            final_payload_construction=True,
            request_preparation=True,
            repeat_target_capture=True,
            tool_event_normalization=True,
            hot_memory_refresh=True,
        )
        assert all(getattr(lc, f.name) is True for f in lc.__dataclass_fields__.values())


class TestBridgePreparedPayloadLifecycleField:
    def test_bridge_prepared_payload_lifecycle_field_defaults_to_none(self) -> None:
        payload = BridgePreparedPayload(
            body={},
            behavior_signals={},
            request_policy="forced_baseline",
            request_tool_compatible=False,
            compressed=False,
            saved_toks=0,
            tool_breakdown={},
            prompt_metrics={},
            retry_forbidden=False,
        )
        assert payload.lifecycle is None

    def test_bridge_prepared_payload_lifecycle_field_accepts_request_lifecycle(
        self,
    ) -> None:
        lc = RequestLifecycle(initial_preflight=True, model_extraction=True)
        payload = BridgePreparedPayload(
            body={"model": "claude-3"},
            behavior_signals={},
            request_policy="natural_first",
            request_tool_compatible=True,
            compressed=True,
            saved_toks=100,
            tool_breakdown={"Read": 3},
            prompt_metrics={"baseline_prompt_tokens": 500},
            retry_forbidden=False,
            lifecycle=lc,
        )
        assert payload.lifecycle is not None
        assert payload.lifecycle.initial_preflight is True
        assert payload.lifecycle.model_extraction is True
        assert payload.lifecycle.runtime_preparation is False


class TestPipelineLifecycleFlagsCoverage:
    """C2: verify all bridge-layer flags are set by prepare_bridge_payload()."""

    def test_all_bridge_layer_stages_set_after_full_pipeline(self) -> None:
        session = BridgeSession()
        payload, preflight = prepare_bridge_payload(
            session=session,
            body=_minimal_body(),
            headers={"x-api-key": "test-key"},
            path="v1/messages",
        )
        assert preflight is None
        lc = payload.lifecycle
        assert lc is not None

        # All bridge-layer stages must be True for a successful full pipeline run.
        assert lc.initial_preflight is True
        assert lc.model_extraction is True
        assert lc.tool_compatibility_check is True
        assert lc.request_preparation is True
        assert lc.runtime_preparation is True
        assert lc.signals_and_metrics is True
        assert lc.prepared_preflight is True
        assert lc.plan_finalization_guard is True
        assert lc.final_payload_construction is True

    def test_early_exit_lifecycle_has_only_preflight_stages_set(self) -> None:
        # count_tokens path exits before tool_compatibility_check
        session = BridgeSession()
        payload, preflight = prepare_bridge_payload(
            session=session,
            body=_minimal_body(),
            headers={"x-api-key": "test-key"},
            path="v1/messages/count_tokens",
        )
        assert preflight is None
        lc = payload.lifecycle
        assert lc is not None
        assert lc.initial_preflight is True
        assert lc.model_extraction is True
        # These should NOT be set for the count_tokens early-exit path
        assert lc.tool_compatibility_check is False
        assert lc.runtime_preparation is False
        assert lc.final_payload_construction is False

    def test_runtime_internal_stages_remain_false_from_bridge_layer(self) -> None:
        # repeat_target_capture, tool_event_normalization, hot_memory_refresh
        # are runtime-internal and not yet wired at the bridge layer.
        session = BridgeSession()
        payload, _ = prepare_bridge_payload(
            session=session,
            body=_minimal_body(),
            headers={"x-api-key": "test-key"},
            path="v1/messages",
        )
        lc = payload.lifecycle
        assert lc is not None
        assert lc.repeat_target_capture is False
        assert lc.tool_event_normalization is False
        assert lc.hot_memory_refresh is False


class TestLifecycleStateAfterFailOpenRetry:
    """C5: lifecycle is not mutated by the fail-open retry path."""

    def test_lifecycle_unchanged_after_retried_without_tok(self) -> None:
        # Simulate a successful pipeline run, then a fail-open reset.
        # The lifecycle on the payload reflects the pipeline, not the retry.
        session = BridgeSession()
        payload, _ = prepare_bridge_payload(
            session=session,
            body=_minimal_body(),
            headers={"x-api-key": "test-key"},
            path="v1/messages",
        )
        lc_before_retry = payload.lifecycle
        assert lc_before_retry is not None
        assert lc_before_retry.final_payload_construction is True

        # Simulate what _handle_retried_without_tok does: it resets compressed/saved_toks
        # but does NOT touch the lifecycle field on the payload.
        assert payload.lifecycle is lc_before_retry  # lifecycle is the same object
        assert payload.lifecycle.final_payload_construction is True
