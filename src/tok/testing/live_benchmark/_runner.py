from __future__ import annotations

import copy
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Literal, cast

from openai import OpenAI

from tok.gateway import BridgeSession
from tok.gateway._request_policy import default_request_policy
from tok.provider_optimizations import apply_provider_optimizations
from tok.runtime.core import (
    RuntimeRequest,
    RuntimeSession,
    UniversalTokRuntime,
    apply_schema_adaptations,
    calculate_invisible_pressure,
)
from tok.runtime.pipeline.response_processing import response_contract_for_mode
from tok.utils.config import API_BASE

from ._evaluation import (
    _evaluate_repo_grounded_research_success,
    _evaluate_task_success,
    _is_research_benchmark,
    _message_shape_forensics,
)
from ._fixtures import (
    _chunk_messages,
    _provider_safe_chat_messages,
    _turn_prompts,
    load_fixture_messages,
    normalize_fixture_messages,
    normalize_fixture_messages_for_bridge,
)
from ._models import (
    BenchmarkDefinition,
    BenchmarkResult,
    ConversationTurnResult,
    ProviderUsageSnapshot,
)
from ._openai_tools import (
    _adapt_tool_results_for_openai,
    _build_openai_tools_param,
    _convert_openai_tool_calls,
    _detect_tool_protocol_retry_reason,
)
from ._prompting import _minimalize_system_prompt, _system_breakdown
from ._utils import _content_text, _estimate_tokens, _sum_warning_signals, _system_to_messages


class LiveBenchmarkRunner:
    def __init__(
        self,
        *,
        model: str,
        provider: str = "openrouter",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 300,
        timeout: float = 120.0,
        client: OpenAI | None = None,
        pricing: dict[str, float] | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_base = api_base or API_BASE
        self.pricing = pricing
        self.provider_options = provider_options
        self.client = client or OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=3,
        )

    def run_conversation_step(
        self,
        *,
        conversation: list[dict[str, Any]],
        system_prompt: str,
        mode: str,
        session: RuntimeSession,
        bridge_session: BridgeSession | None = None,
        adapter_kind: str = "text-loop",
        allowed_tools: tuple[str, ...] | None = None,
    ) -> ConversationTurnResult:
        canonical_mode = "tok-universal" if mode in {"tok-tool-compatible", "tok-universal"} else mode
        use_openai_tools = bool(allowed_tools) and self.provider.lower() != "anthropic"
        original_system_tokens = _estimate_tokens(system_prompt)
        normalized_messages_tokens = _estimate_tokens(conversation)
        baseline_prompt_estimate = original_system_tokens + normalized_messages_tokens
        runtime = UniversalTokRuntime()
        tool_compatible = canonical_mode in {"tok-universal", "tok-minimal"}
        request_policy = default_request_policy() if canonical_mode == "tok-universal" else "legacy_tool_compatible"

        prepared = None
        prepared_body: dict[str, Any] | None = None
        turn_tool_compatible = False
        if canonical_mode == "baseline":
            chat_messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                *conversation,
            ]
            compression_metrics: dict[str, Any] = {
                "input_saved_tokens": 0,
                "output_saved_tokens": 0,
                "total_saved_tokens": 0,
                "input_behavior_signals": {},
                "type_breakdown": {},
            }
            prompt_metrics: dict[str, Any] = {
                "system_prompt_tokens": original_system_tokens,
                "normalized_messages_tokens": normalized_messages_tokens,
                "prepared_messages_tokens": normalized_messages_tokens,
                "system_tokens_estimate": original_system_tokens,
                "directive_tokens_estimate": 0,
                "state_payload_tokens_estimate": 0,
                "tok_system_additions_tokens": 0,
                "tok_overhead_tokens": 0,
                "estimated_prompt_delta_tokens": 0,
                "outbound_prompt_estimate_tokens": baseline_prompt_estimate,
            }
            response_metrics: dict[str, Any] = {
                "response_behavior_signals": {},
                "invisible_pressure": 0,
                "reacquisition_cost_tokens": 0,
                "family_mode": "",
                "response_mode": "baseline",
            }
            diagnostics: dict[str, Any] = {
                "tool_compatible_requested": False,
                "request_policy": "forced_baseline",
                "request_messages_before": len(conversation),
                "request_messages_after": len(chat_messages),
            }
            outbound_payload: dict[str, Any] = {
                "system": system_prompt,
                "messages": conversation.copy(),
            }
        else:
            if canonical_mode == "tok-universal":
                active_bridge_session = bridge_session or BridgeSession()
                bridge_request_body = {
                    "model": self.model,
                    "messages": copy.deepcopy(conversation),
                    "system": system_prompt,
                    "max_tokens": self.max_tokens,
                }
                if use_openai_tools:
                    openai_tools = _build_openai_tools_param(allowed_tools or ())
                    if openai_tools:
                        bridge_request_body["tools"] = openai_tools
                from tok.testing import live_benchmark as live_benchmark_pkg

                bridge_payload, preflight_response = live_benchmark_pkg.prepare_bridge_payload(
                    session=active_bridge_session,
                    body=bridge_request_body,
                    headers={},
                    path="v1/messages",
                    allowed_tools=allowed_tools,
                )
                if preflight_response is not None:
                    status_code = getattr(preflight_response, "status_code", "unknown")
                    msg = f"tok-universal benchmark bridge preflight rejected payload (status={status_code})"
                    raise RuntimeError(msg)

                prepared_body, _ = apply_provider_optimizations(
                    adapter_kind="claude-bridge",
                    body=copy.deepcopy(bridge_payload.body),
                )
                request_policy = bridge_payload.request_policy
                turn_tool_compatible = bridge_payload.request_tool_compatible
                chat_messages = _system_to_messages(prepared_body.get("system")) + prepared_body.get("messages", [])
                prepared_system_tokens = _estimate_tokens(prepared_body.get("system"))
                prepared_messages_tokens = _estimate_tokens(prepared_body.get("messages", []))
                sys_val2 = prepared_body.get("system")
                (
                    system_tokens_estimate,
                    directive_tokens_estimate,
                    state_payload_tokens_estimate,
                ) = _system_breakdown(
                    system_prompt,
                    sys_val2 if isinstance(sys_val2, str | list | dict) else "",
                )
                outbound_prompt_estimate = prepared_system_tokens + prepared_messages_tokens
                tok_system_additions_tokens = max(0, prepared_system_tokens - original_system_tokens)
                tok_overhead_tokens = max(
                    0,
                    outbound_prompt_estimate - baseline_prompt_estimate + bridge_payload.saved_toks,
                )
                compression_metrics = {
                    "input_saved_tokens": bridge_payload.saved_toks,
                    "output_saved_tokens": 0,
                    "total_saved_tokens": bridge_payload.saved_toks,
                    "input_behavior_signals": dict(bridge_payload.behavior_signals),
                    "type_breakdown": dict(bridge_payload.tool_breakdown),
                }
                prompt_metrics = {
                    "system_prompt_tokens": original_system_tokens,
                    "normalized_messages_tokens": normalized_messages_tokens,
                    "prepared_messages_tokens": prepared_messages_tokens,
                    "system_tokens_estimate": system_tokens_estimate,
                    "directive_tokens_estimate": directive_tokens_estimate,
                    "state_payload_tokens_estimate": state_payload_tokens_estimate,
                    "tok_system_additions_tokens": tok_system_additions_tokens,
                    "tok_overhead_tokens": tok_overhead_tokens,
                    "estimated_prompt_delta_tokens": outbound_prompt_estimate - baseline_prompt_estimate,
                    "outbound_prompt_estimate_tokens": outbound_prompt_estimate,
                }
                response_metrics = {
                    "response_behavior_signals": {},
                    "invisible_pressure": 0,
                    "reacquisition_cost_tokens": int(
                        bridge_payload.behavior_signals.get("reacquisition_cost_tokens", 0)
                    ),
                    "family_mode": "",
                    "response_mode": canonical_mode,
                }
                diagnostics = {
                    "tool_compatible_requested": turn_tool_compatible,
                    "request_policy": request_policy,
                    "request_messages_before": len(conversation),
                    "request_messages_after": len(chat_messages),
                    "runtime_mode": "claude-bridge",
                    "execution_path": "claude-bridge",
                    "bridge_preflight_applied": 1,
                }
                if diagnostics["execution_path"] != "claude-bridge":
                    msg = "tok-universal benchmark must execute through the claude-bridge path"
                    raise RuntimeError(msg)
                outbound_payload = {
                    "system": prepared_body.get("system"),
                    "messages": prepared_body.get("messages", []),
                }
            else:
                prepared = runtime.prepare_request(
                    RuntimeRequest(
                        model=self.model,
                        messages=conversation,
                        system=system_prompt,
                        adapter_kind=adapter_kind,
                        tool_compatible=tool_compatible,
                        request_policy=cast(
                            "Literal['legacy_tool_compatible', 'natural_first', 'forced_baseline']",
                            request_policy,
                        ),
                        allowed_tools=allowed_tools,
                    ),
                    session,
                )
                prepared_body = dict(prepared.body)
                if canonical_mode == "tok-minimal":
                    sys_val = prepared_body.get("system")
                    prepared_body["system"] = _minimalize_system_prompt(
                        sys_val if isinstance(sys_val, str | list | dict) else "",
                        system_prompt,
                    )
                turn_tool_compatible = tool_compatible
                chat_messages = _system_to_messages(prepared_body.get("system")) + prepared_body.get("messages", [])
                prepared_system_tokens = _estimate_tokens(prepared_body.get("system"))
                prepared_messages_tokens = _estimate_tokens(prepared_body.get("messages", []))
                sys_val2 = prepared_body.get("system")
                (
                    system_tokens_estimate,
                    directive_tokens_estimate,
                    state_payload_tokens_estimate,
                ) = _system_breakdown(
                    system_prompt,
                    sys_val2 if isinstance(sys_val2, str | list | dict) else "",
                )
                outbound_prompt_estimate = prepared_system_tokens + prepared_messages_tokens
                tok_system_additions_tokens = max(0, prepared_system_tokens - original_system_tokens)
                tok_overhead_tokens = max(
                    0,
                    outbound_prompt_estimate - baseline_prompt_estimate + prepared.input_saved_tokens,
                )
                compression_metrics = {
                    "input_saved_tokens": prepared.input_saved_tokens,
                    "output_saved_tokens": 0,
                    "total_saved_tokens": prepared.input_saved_tokens,
                    "input_behavior_signals": dict(prepared.behavior_signals),
                    "type_breakdown": dict(prepared.type_breakdown),
                }
                prompt_metrics = {
                    "system_prompt_tokens": original_system_tokens,
                    "normalized_messages_tokens": normalized_messages_tokens,
                    "prepared_messages_tokens": prepared_messages_tokens,
                    "system_tokens_estimate": system_tokens_estimate,
                    "directive_tokens_estimate": directive_tokens_estimate,
                    "state_payload_tokens_estimate": state_payload_tokens_estimate,
                    "tok_system_additions_tokens": tok_system_additions_tokens,
                    "tok_overhead_tokens": tok_overhead_tokens,
                    "estimated_prompt_delta_tokens": outbound_prompt_estimate - baseline_prompt_estimate,
                    "outbound_prompt_estimate_tokens": outbound_prompt_estimate,
                }
                response_metrics = {
                    "response_behavior_signals": {},
                    "invisible_pressure": 0,
                    "reacquisition_cost_tokens": int(prepared.behavior_signals.get("reacquisition_cost_tokens", 0)),
                    "family_mode": "",
                    "response_mode": canonical_mode,
                }
                diagnostics = {
                    "tool_compatible_requested": tool_compatible,
                    "request_policy": request_policy,
                    "request_messages_before": len(conversation),
                    "request_messages_after": len(chat_messages),
                    "runtime_mode": prepared.mode,
                    "execution_path": "text-loop",
                    "bridge_preflight_applied": 0,
                }
                outbound_payload = {
                    "system": prepared_body.get("system"),
                    "messages": prepared_body.get("messages", []),
                }

        started = time.time()
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.provider_options:
            create_kwargs["extra_body"] = self.provider_options

        shape_before = _message_shape_forensics(chat_messages)
        adapted_chat_messages = apply_schema_adaptations(chat_messages)
        shape_after = _message_shape_forensics(adapted_chat_messages)

        if use_openai_tools:
            provider_messages = _adapt_tool_results_for_openai(adapted_chat_messages)
        else:
            provider_messages = _provider_safe_chat_messages(adapted_chat_messages, self.provider)

        provider_shape = _message_shape_forensics(provider_messages)
        compatibility_warnings: list[str] = []
        if self.provider.lower() != "anthropic" and (
            shape_after["tool_use_blocks"] > 0 or shape_after["tool_result_blocks"] > 0
        ):
            compatibility_warnings.append("non_anthropic_tool_block_payload")
        if use_openai_tools:
            compatibility_warnings.append("openai_tools_param_active")
        outbound_system = prepared_body.get("system") if prepared_body is not None else system_prompt
        outbound_payload = {
            "system": outbound_system,
            "messages": provider_messages,
        }
        diagnostics["schema_forensics"] = {
            "provider": self.provider,
            "before": shape_before,
            "after": shape_after,
            "provider_after": provider_shape,
            "compatibility_warnings": compatibility_warnings,
        }
        diagnostics["provider_message_normalization_path"] = (
            "apply_schema_adaptations -> _adapt_tool_results_for_openai"
            if use_openai_tools
            else (
                "apply_schema_adaptations -> passthrough_messages"
                if self.provider.lower() == "anthropic"
                else "apply_schema_adaptations -> _provider_safe_chat_messages"
            )
        )
        diagnostics["tool_block_adaptation_warnings"] = list(compatibility_warnings)
        diagnostics["tool_protocol_retry_count"] = 0
        diagnostics["tool_protocol_retry_success"] = 0
        diagnostics["tool_protocol_retry_reason"] = ""
        diagnostics["tool_protocol_retry_mode"] = ""
        create_kwargs["messages"] = provider_messages

        if use_openai_tools:
            openai_tools = _build_openai_tools_param(allowed_tools)
            if openai_tools:
                create_kwargs["tools"] = openai_tools
                create_kwargs["tool_choice"] = "auto"

        response = None
        try:
            response = self.client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            retry_reason = _detect_tool_protocol_retry_reason(exc) if use_openai_tools else None
            if retry_reason is None:
                raise
            diagnostics["tool_protocol_retry_count"] = 1
            diagnostics["tool_protocol_retry_reason"] = retry_reason
            diagnostics["tool_protocol_retry_mode"] = "safe_provider_text_history"
            safe_provider_messages = _provider_safe_chat_messages(adapted_chat_messages, self.provider)
            diagnostics["provider_message_normalization_path"] = (
                "apply_schema_adaptations -> _adapt_tool_results_for_openai -> retry_safe_provider_text_history"
            )
            diagnostics["schema_forensics"]["provider_retry_after"] = _message_shape_forensics(safe_provider_messages)
            retry_kwargs = dict(create_kwargs)
            retry_kwargs["messages"] = safe_provider_messages
            retry_kwargs.pop("tools", None)
            outbound_payload = {
                "system": outbound_system,
                "messages": safe_provider_messages,
            }
            create_kwargs = retry_kwargs
            try:
                response = self.client.chat.completions.create(**retry_kwargs)
            except Exception:
                diagnostics["tool_protocol_retry_success"] = 0
                raise
            diagnostics["tool_protocol_retry_success"] = 1
            compatibility_warnings.append("tool_protocol_retry_safe_provider_text_history")
            diagnostics["tool_block_adaptation_warnings"] = list(compatibility_warnings)
            diagnostics["schema_forensics"]["compatibility_warnings"] = list(compatibility_warnings)
        if response is None:
            msg = "provider response missing after tool protocol retry handling"
            raise RuntimeError(msg)
        latency_ms = (time.time() - started) * 1000
        message_obj = response.choices[0].message if response.choices else None
        raw_response = _content_text(message_obj.content) if message_obj and message_obj.content is not None else ""
        visible_response = raw_response

        openai_tool_call_blocks: list[dict[str, Any]] = []
        if use_openai_tools and message_obj is not None:
            if hasattr(message_obj, "tool_calls") and message_obj.tool_calls:
                openai_tool_call_blocks = _convert_openai_tool_calls(message_obj.tool_calls)
                if not raw_response and openai_tool_call_blocks:
                    raw_response = " ".join(f"Tool use ({b.get('name', '')})" for b in openai_tool_call_blocks)
                    visible_response = raw_response

        if canonical_mode != "baseline":
            processed = runtime.process_response(
                raw_response,
                model=self.model,
                session=session,
                behavior_signals=compression_metrics["input_behavior_signals"],
                tool_compatible=turn_tool_compatible,
            )
            content_blocks = processed.content_blocks
            visible_response = (
                "\n".join(block.get("text", "") for block in content_blocks if block.get("type") == "text").strip()
                or raw_response
            )
            compression_metrics["output_saved_tokens"] = processed.output_saved_tokens
            compression_metrics["total_saved_tokens"] = (
                compression_metrics["input_saved_tokens"] + processed.output_saved_tokens
            )
            response_metrics["response_behavior_signals"] = dict(processed.behavior_signals)
            response_metrics["family_mode"] = processed.family_mode
            response_metrics["invisible_pressure"] = calculate_invisible_pressure(processed.behavior_signals)
            response_metrics["reacquisition_cost_tokens"] = int(
                processed.behavior_signals.get("reacquisition_cost_tokens", 0)
                or response_metrics["reacquisition_cost_tokens"]
            )
            response_metrics["response_mode"] = processed.mode
            if canonical_mode == "tok-universal" and response_metrics["response_mode"] == "tok-universal":
                msg = "tok-universal benchmark must execute through runtime response-contract processing"
                raise RuntimeError(msg)
            if openai_tool_call_blocks:
                content_blocks = openai_tool_call_blocks + [b for b in content_blocks if b.get("type") != "tool_use"]
        else:
            content_blocks = response_contract_for_mode(
                raw_response, tool_compatible=False, session=session
            ).content_blocks
            if openai_tool_call_blocks:
                content_blocks = openai_tool_call_blocks + [b for b in content_blocks if b.get("type") != "tool_use"]

        usage = response.usage
        provider_usage = ProviderUsageSnapshot(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0)),
            completion_tokens=int(getattr(usage, "completion_tokens", 0)),
            total_tokens=int(getattr(usage, "total_tokens", 0)),
            latency_ms=round(latency_ms, 2),
        )
        diagnostics["response_warning_signal_count"] = _sum_warning_signals(
            response_metrics["response_behavior_signals"]
        )
        return ConversationTurnResult(
            mode=canonical_mode,
            provider_usage=provider_usage,
            compression_metrics=compression_metrics,
            prompt_metrics=prompt_metrics,
            response_metrics=response_metrics,
            diagnostics=diagnostics,
            outbound_payload=outbound_payload,
            raw_response=raw_response,
            visible_response=visible_response,
            content_blocks=content_blocks,
        )

    def run(self, definition: BenchmarkDefinition, *, mode: str, turns: int = 3) -> BenchmarkResult:
        if turns < 1:
            msg = "turns must be >= 1"
            raise ValueError(msg)
        canonical_mode = "tok-universal" if mode in {"tok-tool-compatible", "tok-universal"} else mode
        raw_messages = load_fixture_messages(definition.fixture_path)
        normalized = (
            normalize_fixture_messages_for_bridge(raw_messages, definition.followup_prompt)
            if canonical_mode == "tok-universal"
            else normalize_fixture_messages(raw_messages, definition.followup_prompt)
        )
        context_messages = normalized[:-1]
        turn_prompts = _turn_prompts(definition, turns)
        message_chunks = _chunk_messages(context_messages, turns)
        original_system_tokens = _estimate_tokens(definition.system_prompt)
        runtime = UniversalTokRuntime()
        tool_compatible = canonical_mode in {"tok-universal", "tok-minimal"}

        # Identity logging for pointer registry continuity
        id(runtime)

        if canonical_mode not in {
            "baseline",
            "tok-native",
            "tok-universal",
            "tok-minimal",
            "tok-neuro",
        }:
            msg = f"Unknown benchmark mode: {mode}"
            raise ValueError(msg)

        # Set Pattern Reactor toggle
        if canonical_mode == "tok-neuro":
            os.environ["TOK_NEURO_REACTOR"] = "1"
        else:
            os.environ["TOK_NEURO_REACTOR"] = "0"

        with tempfile.TemporaryDirectory(prefix="tok_live_benchmark_") as tmpdir:
            bridge_session = BridgeSession(memory_dir=Path(tmpdir))
            session = bridge_session.runtime_session
            session.model = self.model
            conversation: list[dict[str, Any]] = []
            turn_results: list[dict[str, Any]] = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_latency_ms = 0.0
            total_input_saved = 0
            total_output_saved = 0
            total_type_breakdown: dict[str, int] = {}
            aggregate_input_behavior_signals: dict[str, int] = {}
            aggregate_response_signals: dict[str, int] = {}
            final_visible_response = ""
            final_raw_response = ""

            from tok.compression import compress_history
            from tok.runtime.policy.smart_policy import policy_for_model

            past_messages: list[dict[str, Any]] = []
            for idx, chunk in enumerate(message_chunks):
                # Ingest previous assistant turns to prime the Reactor
                if canonical_mode == "tok-neuro" and idx > 0:
                    h_profile = dict(policy_for_model(self.model).history_profiles["balanced"])
                    h_profile["_no_pointers"] = True
                    _, tok_state, _ = compress_history(
                        past_messages,
                        keep_turns=1,
                        profile=h_profile,
                    )
                    if tok_state:
                        session.write_memory(tok_state)

                conversation.extend(chunk)
                past_messages.extend(chunk)
                user_msg = {"role": "user", "content": turn_prompts[idx]}
                conversation.append(user_msg)
                past_messages.append(user_msg)

                step_result = self.run_conversation_step(
                    conversation=conversation,
                    system_prompt=definition.system_prompt,
                    mode=canonical_mode,
                    session=session,
                    bridge_session=bridge_session,
                )
                prompt_tokens = step_result.provider_usage.prompt_tokens
                completion_tokens = step_result.provider_usage.completion_tokens
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_latency_ms += step_result.provider_usage.latency_ms
                total_input_saved += int(step_result.compression_metrics["input_saved_tokens"])
                total_output_saved += int(step_result.compression_metrics["output_saved_tokens"])
                for key, value in step_result.compression_metrics["type_breakdown"].items():
                    total_type_breakdown[key] = total_type_breakdown.get(key, 0) + int(value)
                for key, value in step_result.compression_metrics["input_behavior_signals"].items():
                    aggregate_input_behavior_signals[key] = aggregate_input_behavior_signals.get(key, 0) + int(value)
                for key, value in step_result.response_metrics["response_behavior_signals"].items():
                    aggregate_response_signals[key] = aggregate_response_signals.get(key, 0) + int(value)
                turn_results.append(
                    {
                        "turn": idx + 1,
                        "provider_usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": step_result.provider_usage.total_tokens,
                            "latency_ms": step_result.provider_usage.latency_ms,
                        },
                        "compression_metrics": step_result.compression_metrics,
                        "prompt_metrics": step_result.prompt_metrics,
                        "response_metrics": step_result.response_metrics,
                        "diagnostics": step_result.diagnostics,
                        "outbound_payload": step_result.outbound_payload,
                        "visible_response": step_result.visible_response,
                        "raw_response": step_result.raw_response,
                    }
                )
                final_visible_response = step_result.visible_response
                final_raw_response = step_result.raw_response
                conversation.append({"role": "assistant", "content": step_result.raw_response})

            cost_usd: float | None = None
            if self.pricing is not None:
                from tok.runtime.metrics import calculate_usage_cost

                cost_usd = calculate_usage_cost(total_prompt_tokens, total_completion_tokens, self.pricing)
            provider_usage = ProviderUsageSnapshot(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
                latency_ms=round(total_latency_ms, 2),
                cost_usd=cost_usd,
            )
            task_success, matched_terms, failures = _evaluate_task_success(
                definition, final_visible_response, session=session
            )
            notes = list(failures)
            if canonical_mode != "baseline" and _sum_warning_signals(aggregate_response_signals):
                notes.append("response_contract_friction_detected")
            repo_grounded_task_success = task_success
            repo_grounded_failures: list[str] = []
            repo_grounded_warnings: list[str] = []
            if _is_research_benchmark(definition):
                (
                    repo_grounded_task_success,
                    repo_grounded_failures,
                    repo_grounded_warnings,
                ) = _evaluate_repo_grounded_research_success(
                    definition,
                    final_visible_response,
                    repo_root=Path.cwd(),
                    session=session,
                )
                notes.extend(f"repo_grounded:{reason}" for reason in repo_grounded_failures)

            prompt_metrics = {
                "system_prompt_tokens": original_system_tokens,
                "normalized_messages_tokens": sum(
                    int(turn["prompt_metrics"]["normalized_messages_tokens"]) for turn in turn_results
                ),
                "prepared_messages_tokens": sum(
                    int(turn["prompt_metrics"]["prepared_messages_tokens"]) for turn in turn_results
                ),
                "system_tokens_estimate": sum(
                    int(turn["prompt_metrics"]["system_tokens_estimate"]) for turn in turn_results
                ),
                "directive_tokens_estimate": sum(
                    int(turn["prompt_metrics"]["directive_tokens_estimate"]) for turn in turn_results
                ),
                "state_payload_tokens_estimate": sum(
                    int(turn["prompt_metrics"]["state_payload_tokens_estimate"]) for turn in turn_results
                ),
                "tok_system_additions_tokens": sum(
                    int(turn["prompt_metrics"]["tok_system_additions_tokens"]) for turn in turn_results
                ),
                "tok_overhead_tokens": sum(int(turn["prompt_metrics"]["tok_overhead_tokens"]) for turn in turn_results),
                "estimated_prompt_delta_tokens": sum(
                    int(turn["prompt_metrics"]["estimated_prompt_delta_tokens"]) for turn in turn_results
                ),
                "outbound_prompt_estimate_tokens": sum(
                    int(turn["prompt_metrics"]["outbound_prompt_estimate_tokens"]) for turn in turn_results
                ),
            }
            response_metrics = {
                "response_behavior_signals": aggregate_response_signals,
                "invisible_pressure": calculate_invisible_pressure(aggregate_response_signals),
                "reacquisition_cost_tokens": int(aggregate_response_signals.get("reacquisition_cost_tokens", 0)),
                "family_mode": (turn_results[-1]["response_metrics"]["family_mode"] if turn_results else ""),
                "response_mode": (
                    turn_results[-1]["response_metrics"]["response_mode"] if turn_results else canonical_mode
                ),
            }
            diagnostics = {
                "tool_compatible_requested": tool_compatible,
                "request_messages_before": len(context_messages),
                "request_messages_after": (
                    turn_results[-1]["diagnostics"]["request_messages_after"] if turn_results else 0
                ),
                "response_warning_signal_count": _sum_warning_signals(aggregate_response_signals),
                "session_turns": turns,
                "cumulative_prompt_tokens": total_prompt_tokens,
                "cumulative_completion_tokens": total_completion_tokens,
                "state_resend_suppressed_turns": aggregate_input_behavior_signals.get(
                    "state_resend_suppressed_turn", 0
                ),
                "state_resend_delta_turns": aggregate_input_behavior_signals.get("state_resend_delta_turn", 0),
                "state_resend_full_turns": aggregate_input_behavior_signals.get("state_resend_full_turn", 0),
                "legacy_task_success": task_success,
                "repo_grounded_task_success": repo_grounded_task_success,
                "repo_grounded_failures": repo_grounded_failures,
                "repo_grounded_warnings": repo_grounded_warnings,
                "message_normalization_path": (
                    "normalize_fixture_messages_for_bridge"
                    if canonical_mode == "tok-universal"
                    else "normalize_fixture_messages"
                ),
                "provider_message_normalization_path": (
                    turn_results[-1]["diagnostics"].get("provider_message_normalization_path", "")
                    if turn_results
                    else ""
                ),
            }
            if canonical_mode != "baseline" and turn_results:
                diagnostics["runtime_mode"] = turn_results[-1]["diagnostics"].get("runtime_mode", "")

            compression_metrics = {
                "input_saved_tokens": total_input_saved,
                "output_saved_tokens": total_output_saved,
                "total_saved_tokens": total_input_saved + total_output_saved,
                "input_behavior_signals": aggregate_input_behavior_signals,
                "type_breakdown": total_type_breakdown,
            }

            return BenchmarkResult(
                benchmark=definition.name,
                mode=canonical_mode,
                model=self.model,
                provider=self.provider,
                fixture_path=str(definition.fixture_path),
                provider_usage=provider_usage,
                compression_metrics=compression_metrics,
                prompt_metrics=prompt_metrics,
                response_metrics=response_metrics,
                diagnostics=diagnostics,
                task_success=task_success,
                matched_success_terms=matched_terms,
                request_messages=(turn_results[-1]["diagnostics"]["request_messages_after"] if turn_results else 0),
                turn_count=turns,
                turns=turn_results,
                visible_response=final_visible_response,
                raw_response=final_raw_response,
                notes=notes,
            )
