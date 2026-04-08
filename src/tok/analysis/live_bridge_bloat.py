"""Live-bridge prompt bloat attribution and audit helpers."""

from __future__ import annotations

import copy
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from tok.compression import (
    TOK_OUTPUT_DIRECTIVE_MINIMAL,
    TOK_OUTPUT_DIRECTIVE_REINFORCED,
    TOK_PROTOCOL_LAW,
    TOK_TOOL_COMPAT_DIRECTIVE,
)
from tok.runtime.config import ANSWER_READY_RUNTIME_HINT
from tok.runtime.pipeline.tool_processing import count_tokens

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tok.runtime.types import PreparedRuntimeRequest

_HEALING_BLOCK = (
    "\n\n[PROTOCOL HEALING] Your previous response drifted from Tok grammar."
    " Remember: use >>> headers and @msg role:assistant blocks exclusively."
)


def _system_text(value: str | list[Any] | None) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    if value is None:
        return ""
    return str(value)


def _token_size(value: str | dict[str, Any] | list[Any] | None) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return count_tokens(value)
    return count_tokens(json.dumps(value, sort_keys=True))


def _char_size(value: str | dict[str, Any] | list[Any] | None) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, sort_keys=True))


def _tool_result_retention(messages: list[dict[str, str]]) -> dict[str, int]:
    tool_messages = 0
    blocks = 0
    tokens = 0
    chars = 0
    heavy_blocks = 0

    for message in messages:
        if message.get("role") == "tool_result":
            tool_messages += 1
            content = message.get("content", "")
            text = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
            tok = count_tokens(text)
            tokens += tok
            chars += len(text)
            if tok >= 200:
                heavy_blocks += 1
            continue

        content_val = message.get("content")
        if not isinstance(content_val, list):
            continue
        for block in content_val:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            blocks += 1
            raw = block.get("content", "")
            text = raw if isinstance(raw, str) else json.dumps(raw, sort_keys=True)
            tok = count_tokens(text)
            tokens += tok
            chars += len(text)
            if tok >= 200:
                heavy_blocks += 1

    return {
        "message_count": tool_messages,
        "embedded_block_count": blocks,
        "tokens": tokens,
        "chars": chars,
        "heavy_block_count": heavy_blocks,
    }


def _resend_mode(resend_signals: dict[str, int]) -> str:
    if resend_signals.get("state_resend_full_turn"):
        return "full"
    if resend_signals.get("state_resend_delta_turn"):
        return "delta"
    if resend_signals.get("state_resend_suppressed_turn"):
        return "suppressed"
    return "none"


def build_prepared_request_bloat_attribution(
    *,
    original_body: dict[str, Any],
    prepared_body: dict[str, Any],
    tool_compatible: bool,
    pressure: int,
    behavior_signals: dict[str, int],
    runtime_hints: list[str] | None = None,
    state_payload: str = "",
    resend_signals: dict[str, int] | None = None,
    history_skip_reason: str = "",
    healing_applied: bool = False,
) -> dict[str, Any]:
    """Summarize bloat-relevant request details at the prepared-request boundary."""
    runtime_hints = [hint.strip() for hint in (runtime_hints or []) if str(hint).strip()]
    resend_signals = resend_signals or {}

    original_system = _system_text(original_body.get("system", ""))
    prepared_system = _system_text(prepared_body.get("system", ""))
    original_messages = original_body.get("messages", [])
    prepared_messages = prepared_body.get("messages", [])

    original_messages_tokens = _token_size(original_messages)
    prepared_messages_tokens = _token_size(prepared_messages)
    original_total_tokens = _token_size(original_body)
    prepared_total_tokens = _token_size(prepared_body)

    if tool_compatible:
        directive_variant = "tool-compatible"
        directive_text = TOK_TOOL_COMPAT_DIRECTIVE
    elif pressure > 50 or behavior_signals.get("semantic_drift_detected"):
        directive_variant = "reinforced"
        directive_text = f"=== MODE: TOK-NATIVE ===\n\n{TOK_PROTOCOL_LAW}\n\n{TOK_OUTPUT_DIRECTIVE_REINFORCED}"
    elif pressure > 1:
        directive_variant = "minimal+law"
        directive_text = f"=== MODE: TOK-NATIVE ===\n\n{TOK_PROTOCOL_LAW}\n\n{TOK_OUTPUT_DIRECTIVE_MINIMAL}"
    else:
        directive_variant = "minimal"
        directive_text = f"=== MODE: TOK-NATIVE ===\n\n{TOK_OUTPUT_DIRECTIVE_MINIMAL}"

    runtime_hint_text = "\n".join(runtime_hints)
    state_block_text = f">>>\n{state_payload}" if state_payload else ""
    healing_block_text = _HEALING_BLOCK if healing_applied else ""

    system_addition_tokens = max(0, _token_size(prepared_system) - _token_size(original_system))
    system_addition_chars = max(0, _char_size(prepared_system) - _char_size(original_system))

    minimal_equivalent_body = {
        "model": original_body.get("model"),
        "messages": copy.deepcopy(prepared_messages),
    }
    if "system" in original_body:
        minimal_equivalent_body["system"] = copy.deepcopy(original_body.get("system"))
    minimal_equivalent_tokens = _token_size(minimal_equivalent_body)

    return {
        "request_footprint": {
            "original": {
                "system_tokens": _token_size(original_system),
                "messages_tokens": original_messages_tokens,
                "total_tokens": original_total_tokens,
                "message_count": len(original_messages),
            },
            "prepared": {
                "system_tokens": _token_size(prepared_system),
                "messages_tokens": prepared_messages_tokens,
                "total_tokens": prepared_total_tokens,
                "message_count": len(prepared_messages),
            },
            "delta_tokens_vs_original": prepared_total_tokens - original_total_tokens,
            "delta_tokens_vs_minimal_equivalent": (prepared_total_tokens - minimal_equivalent_tokens),
            "minimal_equivalent_tokens": minimal_equivalent_tokens,
        },
        "system_additions": {
            "tokens": system_addition_tokens,
            "chars": system_addition_chars,
            "directive_variant": directive_variant,
            "directive_tokens": _token_size(directive_text),
            "directive_chars": _char_size(directive_text),
            "protocol_law_tokens": (_token_size(TOK_PROTOCOL_LAW) if (not tool_compatible and pressure > 1) else 0),
            "tok_state_tokens": _token_size(state_block_text),
            "tok_state_chars": _char_size(state_block_text),
            "runtime_hint_tokens": _token_size(runtime_hint_text),
            "runtime_hint_chars": _char_size(runtime_hint_text),
            "runtime_hint_count": len(runtime_hints),
            "healing_tokens": _token_size(healing_block_text),
            "healing_chars": _char_size(healing_block_text),
        },
        "state_resend": {
            "mode": _resend_mode(resend_signals),
            "payload_tokens": _token_size(state_payload),
            "payload_chars": _char_size(state_payload),
            "answer_anchor_present": bool(behavior_signals.get("answer_anchor_present", 0)),
        },
        "history_retention": {
            "skipped": bool(behavior_signals.get("tok_history_compression_skipped", 0)),
            "skip_reason": history_skip_reason,
            "original_message_tokens": original_messages_tokens,
            "prepared_message_tokens": prepared_messages_tokens,
            "retained_tokens": prepared_messages_tokens,
            "retained_message_count": len(prepared_messages),
            "dropped_tokens": max(0, original_messages_tokens - prepared_messages_tokens),
        },
        "tool_result_retention": _tool_result_retention(prepared_messages),
        "runtime_hints": {
            "items": runtime_hints,
            "tokens": _token_size(runtime_hint_text),
            "chars": _char_size(runtime_hint_text),
        },
        "pressure": pressure,
        "tool_compatible": tool_compatible,
    }


def _make_tool_use(tool_id: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": payload}],
    }


def _make_tool_result(tool_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool_result", "tool_use_id": tool_id, "content": content}


def _snapshot(
    prepared: object,
    *,
    description: str,
    counterfactual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    typed_prepared = cast("PreparedRuntimeRequest", prepared)
    attribution = typed_prepared.bloat_attribution
    return {
        "description": description,
        "behavior_signals": dict(typed_prepared.behavior_signals),
        "type_breakdown": dict(typed_prepared.type_breakdown),
        "request_footprint": attribution.get("request_footprint", {}),
        "system_additions": attribution.get("system_additions", {}),
        "state_resend": attribution.get("state_resend", {}),
        "history_retention": attribution.get("history_retention", {}),
        "tool_result_retention": attribution.get("tool_result_retention", {}),
        "runtime_hints": attribution.get("runtime_hints", {}),
        "counterfactual": counterfactual or {},
    }


@contextmanager
def _patched_history_skip() -> Iterator[None]:
    from tok.runtime import core as core_module

    original = core_module._should_skip_history_rewrite
    core_module._should_skip_history_rewrite = lambda messages, normalized_tool_events, *, tool_compatible: (
        False,
        "",
    )
    try:
        yield
    finally:
        core_module._should_skip_history_rewrite = original


@contextmanager
def _forced_history_skip(reason: str) -> Iterator[None]:
    from tok.runtime import core as core_module

    original = core_module._should_skip_history_rewrite
    core_module._should_skip_history_rewrite = lambda messages, normalized_tool_events, *, tool_compatible: (
        True,
        reason,
    )
    try:
        yield
    finally:
        core_module._should_skip_history_rewrite = original


def _new_runtime(tmp_path: Path) -> tuple[Any, Any, Any]:
    from tok.runtime.core import RuntimeSession, UniversalTokRuntime

    memory_dir = tmp_path / ".tok"
    return (
        UniversalTokRuntime(),
        RuntimeSession(memory_dir=memory_dir),
        memory_dir,
    )


def measure_live_bridge_bloat_scenarios() -> dict[str, Any]:
    """Run a fixed live-bridge scenario suite against the actual runtime path."""
    from tok.runtime.core import RuntimeRequest

    scenarios: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="tok-bridge-bloat-") as tmp:
        tmp_path = Path(tmp)

        runtime, session, _ = _new_runtime(tmp_path / "cold")
        cold_request = RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[{"role": "user", "content": "Confirm the gateway entry point."}],
        )
        cold_prepared = runtime.prepare_request(cold_request, session)
        scenarios["cold_start"] = _snapshot(
            cold_prepared,
            description="Cold start, tool-compatible, no memory.",
        )

        runtime, session, _ = _new_runtime(tmp_path / "warm")
        warm_request = RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Help me inspect these files"},
                _make_tool_use("t0", "view_file", {"path": "src/tok/file_0.py"}),
                _make_tool_result("t0", "file 0 content " * 100),
                {"role": "user", "content": "What changed?"},
            ],
        )
        warm_first = runtime.prepare_request(warm_request, session)
        warm_second = runtime.prepare_request(warm_request, session)
        scenarios["warm_unchanged_state_first"] = _snapshot(
            warm_first,
            description="First warm tool-compatible turn with compressible history.",
        )
        scenarios["warm_unchanged_state_second"] = _snapshot(
            warm_second,
            description="Second identical warm turn to verify state suppression/delta.",
        )

        runtime, session, _ = _new_runtime(tmp_path / "answer")
        session.bridge_memory._upsert(
            session.bridge_memory.hot,
            "facts",
            "answer_file:src/tok/gateway.py",
            score_delta=3,
        )
        session.bridge_memory._upsert(
            session.bridge_memory.hot,
            "facts",
            "answer_verification:health",
            score_delta=3,
        )
        answer_request = RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[{"role": "user", "content": "confirm the gateway entry point"}],
        )
        answer_first = runtime.prepare_request(answer_request, session)
        answer_second = runtime.prepare_request(answer_request, session)
        scenarios["answer_anchor_first"] = _snapshot(
            answer_first,
            description="First answer-anchor turn with one-time full state resend.",
        )
        scenarios["answer_anchor_second"] = _snapshot(
            answer_second,
            description="Second identical answer-anchor turn after suppression.",
        )

        runtime, session, _ = _new_runtime(tmp_path / "coding")
        coding_messages: list[dict[str, Any]] = [{"role": "user", "content": "Help me with these files"}]
        for idx in range(4):
            code_lines = []
            for line_idx in range(60):
                code_lines.append(f"def func_{idx}_{line_idx}():\n")
                code_lines.append("    x = 1\n")
                code_lines.append("    y = 2\n")
                code_lines.append("    return x + y\n")
            coding_messages.append(_make_tool_use(f"c{idx}", "view_file", {"path": f"src/tok/file_{idx}.py"}))
            coding_messages.append(_make_tool_result(f"c{idx}", "".join(code_lines)))
            if idx < 3:
                coding_messages.append({"role": "user", "content": f"Continue with step {idx}"})
        coding_messages.append({"role": "assistant", "content": "I'll help with more files."})
        coding_request = RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=coding_messages,
        )
        coding_prepared = runtime.prepare_request(coding_request, session)
        scenarios["moderate_coding"] = _snapshot(
            coding_prepared,
            description="Moderate coding turn with compressible history and retained recent tool results.",
        )

        runtime, session, _ = _new_runtime(tmp_path / "skip")
        skip_messages: list[dict[str, Any]] = [{"role": "user", "content": "Audit the recent file reads."}]
        for idx in range(10):
            skip_messages.append(_make_tool_use(f"s{idx}", "view_file", {"path": f"src/tok/file_{idx}.py"}))
            skip_messages.append(_make_tool_result(f"s{idx}", f"skip file {idx} content " * 90))
            if idx < 9:
                skip_messages.append(
                    {
                        "role": "user",
                        "content": f"Continue reviewing file batch {idx}.",
                    }
                )
        skip_messages.append({"role": "user", "content": "Summarize the state."})
        skip_request = RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=skip_messages,
        )
        skip_actual = runtime.prepare_request(skip_request, session)

        runtime_cf, session_cf, _ = _new_runtime(tmp_path / "skip_cf")
        with _forced_history_skip("tool_use_count_high"):
            skip_counterfactual = runtime_cf.prepare_request(skip_request, session_cf)
        scenarios["history_skip"] = _snapshot(
            skip_actual,
            description="Tool-heavy bridge turn that used to skip rewrite under the legacy tool-use threshold.",
            counterfactual={
                "prepared_total_tokens": skip_counterfactual.bloat_attribution["request_footprint"]["prepared"][
                    "total_tokens"
                ],
                "savings_tokens_vs_legacy_skip": (
                    skip_counterfactual.bloat_attribution["request_footprint"]["prepared"]["total_tokens"]
                    - skip_actual.bloat_attribution["request_footprint"]["prepared"]["total_tokens"]
                ),
            },
        )

        runtime, session, _ = _new_runtime(tmp_path / "answer_ready")
        session.bridge_memory._upsert(
            session.bridge_memory.hot,
            "facts",
            "answer_file:src/tok/gateway.py",
            score_delta=3,
        )
        session.bridge_memory._upsert(
            session.bridge_memory.hot,
            "facts",
            "answer_verification:health",
            score_delta=3,
        )
        answer_ready_request = RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[{"role": "user", "content": "confirm the gateway entry point"}],
        )
        answer_ready_prepared = runtime.prepare_request(answer_ready_request, session)
        scenarios["answer_ready_hint"] = _snapshot(
            answer_ready_prepared,
            description="Answer-ready turn that injects the no-tools answer hint.",
        )

        runtime, session, _ = _new_runtime(tmp_path / "stable")
        large_output = "file content line\n" * 50
        stable_request = RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                _make_tool_use("r1", "view_file", {"path": "src/tok/gateway.py"}),
                _make_tool_result("r1", large_output),
            ],
        )
        runtime.prepare_request(stable_request, session)
        stable_second = runtime.prepare_request(stable_request, session)
        scenarios["stable_result_hint"] = _snapshot(
            stable_second,
            description="Repeated identical file read that injects the stable-result explanation hint.",
        )

        runtime, session, _ = _new_runtime(tmp_path / "strict")
        strict_request = RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=False,
            messages=[
                _make_tool_use("f1", "view_file", {"path": "src/tok/gateway.py"}),
                _make_tool_use("f2", "view_file", {"path": "src/tok/gateway.py"}),
                _make_tool_use(
                    "g1",
                    "grep_search",
                    {"query": "health", "search_path": "src/tok"},
                ),
                _make_tool_use(
                    "g2",
                    "grep_search",
                    {"query": "health", "search_path": "src/tok"},
                ),
                {
                    "role": "user",
                    "content": "Summarize the gateway health entry point.",
                },
            ],
        )
        strict_low_request = RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=False,
            messages=[
                _make_tool_use("f1", "view_file", {"path": "src/tok/gateway.py"}),
                _make_tool_use(
                    "g1",
                    "grep_search",
                    {"query": "health", "search_path": "src/tok"},
                ),
                {
                    "role": "user",
                    "content": "Summarize the gateway health entry point.",
                },
            ],
        )
        strict_low = runtime.prepare_request(strict_low_request, session)
        strict_high = runtime.prepare_request(strict_request, session)
        scenarios["strict_pressure"] = _snapshot(
            strict_high,
            description="Bridge opt-out path in strict mode with enough repeat pressure to trigger the protocol law.",
            counterfactual={
                "prepared_total_tokens_without_pressure": strict_low.bloat_attribution["request_footprint"]["prepared"][
                    "total_tokens"
                ],
                "protocol_law_delta_tokens": (
                    strict_high.bloat_attribution["request_footprint"]["prepared"]["total_tokens"]
                    - strict_low.bloat_attribution["request_footprint"]["prepared"]["total_tokens"]
                ),
            },
        )

    return scenarios


def rank_live_bridge_bloat_suspects(
    scenarios: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create a ranked suspect list from the fixed scenario suite."""
    cold = scenarios["cold_start"]
    answer_first = scenarios["answer_anchor_first"]
    moderate = scenarios["moderate_coding"]
    answer_ready = scenarios["answer_ready_hint"]
    stable = scenarios["stable_result_hint"]
    strict_pressure = scenarios["strict_pressure"]

    suspects = [
        {
            "name": "Retained recent tool results after compression",
            "classification": "likely accidental / too eager",
            "trigger": "Moderate coding turns keep recent tool_result payloads in the final message window.",
            "code_path": "src/tok/compression/__init__.py::compress_recent_window and compress_tool_results",
            "request_area": "tool-result retention",
            "estimated_overhead_tokens": moderate["tool_result_retention"]["tokens"],
            "frequency": "common on coding turns with file reads",
            "why": "Even after history compression, retained tool payloads still dominate the live request footprint.",
            "fix_direction": "Further compress or summarize recent tool_result payloads when answer anchors or cached/stable results already preserve the needed evidence.",
        },
        {
            "name": "Answer-ready runtime hint",
            "classification": "expected floor",
            "trigger": "Turns where Tok believes Claude already has enough File=/Verification= evidence and should answer.",
            "code_path": "src/tok/runtime/pipeline/request_preparation.py::_runtime_hints_for_turn",
            "request_area": "system additions",
            "estimated_overhead_tokens": answer_ready["runtime_hints"]["tokens"],
            "frequency": "conditional; answer-ready turns only",
            "why": "Small, intentional guidance to stop further tool use. Not a primary bloat driver by itself.",
            "fix_direction": "Leave as-is unless multiple overlapping runtime hints begin stacking on the same turn.",
        },
        {
            "name": "First answer-anchor state resend",
            "classification": "expected floor",
            "trigger": "First warm turn where answer/file verification facts are present.",
            "code_path": "src/tok/runtime/core.py::maybe_suppress_tool_compatible_state",
            "request_area": "state resend",
            "estimated_overhead_tokens": answer_first["state_resend"]["payload_tokens"],
            "frequency": "one-time per new answer anchor",
            "why": "Intentional one-time resend so Claude sees new File=/Verification= facts before later suppression kicks in.",
            "fix_direction": "Keep behavior, but guard against repeated full resends when the comparable state is unchanged.",
        },
        {
            "name": "Tool-compatible directive floor",
            "classification": "expected floor",
            "trigger": "Every default live-bridge prompt in tool-compatible mode.",
            "code_path": "src/tok/gateway/__init__.py -> src/tok/runtime/core.py -> src/tok/compression/__init__.py::inject_system_additions",
            "request_area": "system additions",
            "estimated_overhead_tokens": cold["system_additions"]["tokens"],
            "frequency": "every bridge prompt",
            "why": "Intentional baseline. This is the compact floor Tok adds even on a cold start.",
            "fix_direction": "Do not optimize first unless duplication appears; treat as the baseline floor in later rankings.",
        },
        {
            "name": "Stable-result explanation hint",
            "classification": "regression risk",
            "trigger": "Repeated identical file reads that hit semantic dedup and inject explanatory guidance.",
            "code_path": "src/tok/runtime/core.py semantic_dedup path + src/tok/compression/__init__.py::_STABLE_RESULT_EXPLANATION",
            "request_area": "system additions",
            "estimated_overhead_tokens": stable["runtime_hints"]["tokens"],
            "frequency": "repeat-read sessions with dedup hits",
            "why": "Helpful, but it adds narrative prompt tax precisely when Tok has already compressed the underlying tool output.",
            "fix_direction": "Consider replacing the explanatory sentence with a shorter symbolic cue or only emitting it on the first dedup hit per session.",
        },
        {
            "name": "Strict-mode protocol law in bridge opt-out path",
            "classification": "regression risk",
            "trigger": "Live bridge is forced out of tool-compatible mode and repeat-read pressure exceeds the law threshold.",
            "code_path": "src/tok/compression/__init__.py::inject_system_additions",
            "request_area": "system additions",
            "estimated_overhead_tokens": strict_pressure["counterfactual"]["protocol_law_delta_tokens"],
            "frequency": "rare; strict opt-out only",
            "why": "Not on the default bridge path, but it is the largest strict-mode prompt spike this audit observed.",
            "fix_direction": "Keep out of the default bridge path and compress the law block further if strict mode becomes more common.",
        },
    ]

    suspects = [suspect for suspect in suspects if suspect["estimated_overhead_tokens"] > 0]
    suspects.sort(key=lambda item: item["estimated_overhead_tokens"], reverse=True)
    for index, suspect in enumerate(suspects, start=1):
        suspect["rank"] = index
    return suspects


def generate_live_bridge_bloat_report() -> dict[str, Any]:
    scenarios = measure_live_bridge_bloat_scenarios()
    suspects = rank_live_bridge_bloat_suspects(scenarios)
    avoidable = [
        suspect
        for suspect in suspects
        if suspect["classification"] in {"likely accidental / too eager", "regression risk"}
    ]
    return {
        "scenarios": scenarios,
        "ranked_suspects": suspects,
        "largest_absolute_overhead": suspects[:5],
        "largest_avoidable_overhead": avoidable[:5],
        "notes": [
            "Scope is the live Claude bridge path only.",
            "Measurements are taken at the final PreparedRuntimeRequest boundary.",
            "Default bridge mode is tool-compatible unless explicitly opted out.",
        ],
    }


def render_live_bridge_bloat_markdown(report: dict[str, Any]) -> str:
    suspects = report["ranked_suspects"]
    scenarios = report["scenarios"]

    lines = [
        "# Live Bridge Prompt Bloat Audit",
        "",
        "Generated from the live Claude bridge request path.",
        "",
        "## Executive Summary",
        "",
        "This audit measures prompt bloat at the final prepared-request boundary used by the live bridge.",
        "It separates system additions, state resend behavior, history retention, and retained tool results.",
        "",
        "### Largest Absolute Overhead",
        "",
        "| Rank | Suspect | Class | Tokens | Frequency |",
        "|------|---------|-------|-------:|-----------|",
    ]
    for suspect in report["largest_absolute_overhead"]:
        lines.append(
            f"| {suspect['rank']} | {suspect['name']} | {suspect['classification']} | {suspect['estimated_overhead_tokens']} | {suspect['frequency']} |"
        )

    lines += [
        "",
        "### Largest Avoidable Overhead",
        "",
        "| Rank | Suspect | Class | Tokens | Frequency |",
        "|------|---------|-------|-------:|-----------|",
    ]
    for suspect in report["largest_avoidable_overhead"]:
        lines.append(
            f"| {suspect['rank']} | {suspect['name']} | {suspect['classification']} | {suspect['estimated_overhead_tokens']} | {suspect['frequency']} |"
        )

    lines += [
        "",
        "## Scenario Measurements",
        "",
        "| Scenario | System Tokens | Message Tokens | Total Tokens | Delta vs Minimal | Key Signal |",
        "|----------|--------------:|---------------:|-------------:|-----------------:|-----------|",
    ]
    for name, scenario in scenarios.items():
        footprint = scenario["request_footprint"]["prepared"]
        delta = scenario["request_footprint"].get("delta_tokens_vs_minimal_equivalent", 0)
        signal = next(
            (key for key, value in scenario["behavior_signals"].items() if value),
            "none",
        )
        lines.append(
            f"| {name} | {footprint['system_tokens']} | {footprint['messages_tokens']} | {footprint['total_tokens']} | {delta} | {signal} |"
        )

    lines += [
        "",
        "## Ranked Suspects",
        "",
        "| Rank | Suspect | Trigger Condition | Exact Code Path | Request Area | Estimated Overhead | Why It Looks Intentional vs Accidental | Recommended Fix Direction |",
        "|------|---------|-------------------|-----------------|--------------|-------------------:|----------------------------------------|---------------------------|",
    ]
    for suspect in suspects:
        lines.append(
            f"| {suspect['rank']} | {suspect['name']} | {suspect['trigger']} | `{suspect['code_path']}` | {suspect['request_area']} | {suspect['estimated_overhead_tokens']}t | {suspect['why']} | {suspect['fix_direction']} |"
        )

    lines += [
        "",
        "## Key Scenario Notes",
        "",
        f"- `cold_start`: default bridge floor is {scenarios['cold_start']['system_additions']['tokens']} tokens and contains no compressed history.",
        f"- `warm_unchanged_state_second`: prepared system shrinks to {scenarios['warm_unchanged_state_second']['request_footprint']['prepared']['system_tokens']} tokens after resend suppression/delta.",
        f"- `answer_anchor_second`: state resend mode is `{scenarios['answer_anchor_second']['state_resend']['mode']}` after the one-time full resend.",
        f"- `history_skip`: the default tool-heavy bridge turn now stays compressed and saves {scenarios['history_skip']['counterfactual']['savings_tokens_vs_legacy_skip']} tokens versus the legacy forced-skip behavior.",
        f"- `strict_pressure`: bridge default mode avoids the protocol-law spike; the strict opt-out path adds {scenarios['strict_pressure']['counterfactual']['protocol_law_delta_tokens']} tokens over a low-pressure strict turn.",
        "",
        "## Assumptions",
        "",
        "- Scope is the live bridge path only.",
        "- Prompt bloat is measured on the final prepared request body, not intermediate state.",
        "- The tool-compatible directive is treated as the intentional baseline floor unless duplicated.",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "ANSWER_READY_RUNTIME_HINT",
    "build_prepared_request_bloat_attribution",
    "generate_live_bridge_bloat_report",
    "measure_live_bridge_bloat_scenarios",
    "rank_live_bridge_bloat_suspects",
    "render_live_bridge_bloat_markdown",
]
