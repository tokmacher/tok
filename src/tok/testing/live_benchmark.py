"""Live benchmark harness for comparing baseline vs Tok runtime behavior."""

from __future__ import annotations

import json
import os
import re
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..utils.config import API_BASE
from ..runtime.core import (
    RuntimeRequest,
    RuntimeSession,
    UniversalTokRuntime,
    apply_schema_adaptations,
    calculate_invisible_pressure,
    count_tokens,
)


def _system_to_messages(
    system: str | list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    if not system:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    messages: list[dict[str, str]] = []
    for block in system:
        if isinstance(block, dict):
            messages.append(
                {"role": "system", "content": block.get("text", "")}
            )
    return messages


def _estimate_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return count_tokens(value)
    return count_tokens(json.dumps(value, sort_keys=True))


def _sum_warning_signals(signals: dict[str, int]) -> int:
    return sum(
        int(signals.get(key, 0))
        for key in (
            "non_tok_response",
            "fail_open_compat_response",
            "malformed_tok_response",
            "tok_drift_healed",
        )
    )


@dataclass(frozen=True)
class BenchmarkDefinition:
    name: str
    fixture_path: Path
    system_prompt: str
    followup_prompt: str
    success_terms: tuple[str, ...]
    min_success_terms: int = 2
    expected_file_terms: tuple[str, ...] = ()
    expected_verification_terms: tuple[str, ...] = ()
    default_turns: int = 3
    prompt_sequence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderUsageSnapshot:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str
    mode: str
    model: str
    provider: str
    fixture_path: str
    provider_usage: ProviderUsageSnapshot
    compression_metrics: dict[str, Any]
    prompt_metrics: dict[str, Any]
    response_metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    task_success: bool
    matched_success_terms: list[str]
    request_messages: int
    turn_count: int
    turns: list[dict[str, Any]]
    visible_response: str
    raw_response: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provider_usage"] = asdict(self.provider_usage)
        return data


@dataclass(frozen=True)
class BenchmarkComparison:
    benchmark: str
    model: str
    candidate_mode: str
    baseline: BenchmarkResult
    candidate: BenchmarkResult
    prompt_token_delta: int
    completion_token_delta: int
    total_token_delta: int
    total_token_delta_pct: float | None
    latency_delta_ms: float
    reacquisition_delta_tokens: int
    pressure_delta: int
    task_success_equal_or_better: bool
    provider_total_token_winner: str
    diagnosis: str
    tok_improved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "candidate_mode": self.candidate_mode,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "prompt_token_delta": self.prompt_token_delta,
            "completion_token_delta": self.completion_token_delta,
            "total_token_delta": self.total_token_delta,
            "total_token_delta_pct": self.total_token_delta_pct,
            "latency_delta_ms": self.latency_delta_ms,
            "reacquisition_delta_tokens": self.reacquisition_delta_tokens,
            "pressure_delta": self.pressure_delta,
            "task_success_equal_or_better": self.task_success_equal_or_better,
            "provider_total_token_winner": self.provider_total_token_winner,
            "diagnosis": self.diagnosis,
            "tok_improved": self.tok_improved,
        }


DEFAULT_BENCHMARKS: dict[str, BenchmarkDefinition] = {
    "coding-loop": BenchmarkDefinition(
        name="coding-loop",
        fixture_path=Path("tests/fixtures/replay/claude_coding_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed coding session. "
            "Answer briefly and cite exact filenames when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the file that was changed>\n"
            "Verification=<the command or result that verified the fix>"
        ),
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "1 passed", "pytest"),
        default_turns=3,
    ),
    "coding-loop-5": BenchmarkDefinition(
        name="coding-loop-5",
        fixture_path=Path("tests/fixtures/replay/claude_coding_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed coding session. "
            "Answer briefly and cite exact filenames when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the file that was changed>\n"
            "Verification=<the command or result that verified the fix>"
        ),
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "1 passed", "pytest"),
        default_turns=5,
    ),
    "research-loop": BenchmarkDefinition(
        name="research-loop",
        fixture_path=Path("tests/fixtures/replay/research_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed codebase research session. "
            "Answer briefly and cite exact filenames and identifiers when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>"
        ),
        success_terms=("compression.py", "compress_history"),
        min_success_terms=2,
        expected_file_terms=("compression.py",),
        expected_verification_terms=("compress_history",),
        default_turns=3,
        prompt_sequence=(
            "Respond in exactly one line: File=<the primary file that answered the original question>",
            "Respond in exactly one line: Verification=<the function, class, or finding that supports the answer>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
        ),
    ),
    "research-loop-5": BenchmarkDefinition(
        name="research-loop-5",
        fixture_path=Path("tests/fixtures/replay/research_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed codebase research session. "
            "Answer briefly and cite exact filenames and identifiers when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>"
        ),
        success_terms=("compression.py", "compress_history"),
        min_success_terms=2,
        expected_file_terms=("compression.py",),
        expected_verification_terms=("compress_history",),
        default_turns=5,
        prompt_sequence=(
            "Respond in exactly one line: File=<the primary file that answered the original question>",
            "Respond in exactly one line: Verification=<the function, class, or finding that supports the answer>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
            "Respond in exactly one line: Related=<the related file or class mentioned during the investigation>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
        ),
    ),
    # --- 8-turn probes: not part of the validated 5-turn release set ---
    "coding-loop-8": BenchmarkDefinition(
        name="coding-loop-8",
        fixture_path=Path("tests/fixtures/replay/claude_coding_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed coding session. "
            "Answer briefly and cite exact filenames when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the file that was changed>\n"
            "Verification=<the command or result that verified the fix>"
        ),
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "1 passed", "pytest"),
        default_turns=8,
    ),
    "research-loop-8": BenchmarkDefinition(
        name="research-loop-8",
        fixture_path=Path("tests/fixtures/replay/research_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed codebase research session. "
            "Answer briefly and cite exact filenames and identifiers when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>"
        ),
        success_terms=("compression.py", "compress_history"),
        min_success_terms=2,
        expected_file_terms=("compression.py",),
        expected_verification_terms=("compress_history",),
        default_turns=8,
        prompt_sequence=(
            "Respond in exactly one line: File=<the primary file that answered the original question>",
            "Respond in exactly one line: Verification=<the function, class, or finding that supports the answer>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
            "Respond in exactly one line: Related=<the related file or class mentioned during the investigation>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
            "Respond in exactly one line: File=<the primary file that answered the original question>",
            "Respond in exactly one line: Verification=<the function, class, or finding that supports the answer>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
        ),
    ),
    "neuro-loop": BenchmarkDefinition(
        name="neuro-loop",
        fixture_path=Path("tests/fixtures/replay/neuro_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed coding session. "
            "Answer briefly and cite exact filenames when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>"
        ),
        success_terms=("parser.py", "parse_error"),
        min_success_terms=2,
        expected_file_terms=("parser.py",),
        expected_verification_terms=("parse_error",),
        default_turns=3,
        prompt_sequence=(
            "Respond in exactly one line: File=<the primary file that answered the original question>",
            "Respond in exactly one line: Verification=<the function, class, or finding that supports the answer>",
        ),
    ),
    "jit-loop": BenchmarkDefinition(
        name="jit-loop",
        fixture_path=Path("tests/fixtures/replay/jit_loop.jsonl"),
        system_prompt=(
            "You are evaluating a repetitive coding task. "
            "Identify patterns and use JIT macros when offered to save time."
        ),
        followup_prompt=("Check src/tok/cli.py for the same pattern."),
        success_terms=("jit_executed", "JIT Execution Result"),
        min_success_terms=1,
        expected_file_terms=("cli.py",),
        expected_verification_terms=(),
        default_turns=1,
    ),
    "grammar_drift": BenchmarkDefinition(
        name="grammar_drift",
        fixture_path=Path("tests/fixtures/replay/grammar_drift.jsonl"),
        system_prompt="Analyze the following session for grammar drift.",
        followup_prompt="Has the grammar drifted?",
        success_terms=("yes", "no"),
        min_success_terms=1,
        default_turns=3,
    ),
}


DEFAULT_MULTI_TURN_PROMPTS: tuple[str, ...] = (
    "Respond in exactly one line: File=<the file that was changed>",
    "Respond in exactly one line: Verification=<the command or result that verified the fix>",
    "Based on the conversation so far, respond in exactly two lines:\n"
    "File=<the file that was changed>\n"
    "Verification=<the command or result that verified the fix>",
)


def load_benchmark_definition(name: str) -> BenchmarkDefinition:
    try:
        return DEFAULT_BENCHMARKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark: {name}") from exc


def load_fixture_messages(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        if isinstance(record.get("messages"), list):
            records.extend(record["messages"])
        elif "role" in record and "content" in record:
            records.append(record)
    return records


def normalize_fixture_messages(
    messages: list[dict[str, Any]], followup_prompt: str
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        if role == "tool_result":
            tool_id = str(msg.get("tool_use_id", "")).strip() or "unknown"
            normalized.append(
                {
                    "role": "user",
                    "content": f"Tool result ({tool_id}): {_content_text(msg.get('content', ''))}",
                }
            )
            continue

        content = msg.get("content", "")
        if isinstance(content, str):
            if content.strip():
                normalized.append({"role": role or "user", "content": content})
            continue
        if not isinstance(content, list):
            continue

        new_content: list[dict[str, str]] = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append({"type": "text", "text": str(block)})
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    new_content.append({"type": "text", "text": text})
            elif block_type in ("tool_use", "tool_result"):
                # Preserve structured blocks for Tok runtime analysis but stringify for baseline simplicity in tests
                if role == "assistant" and block_type == "tool_use":
                    tool_name = block.get("name", "unknown")
                    new_content.append(
                        {"type": "text", "text": f"Tool use ({tool_name})"}
                    )
                new_content.append(block)

        if new_content:
            # Flatten to string if possible for broader provider compatibility (e.g. Bedrock)
            if all(block.get("type") == "text" for block in new_content):
                text_content = "\n".join(
                    block.get("text", "") for block in new_content
                ).strip()
                normalized.append(
                    {"role": role or "user", "content": text_content}
                )
            else:
                normalized.append(
                    {"role": role or "user", "content": new_content}
                )

    # Flatten the final followup_prompt as well
    if isinstance(followup_prompt, str):
        normalized.append({"role": "user", "content": followup_prompt})
    else:
        normalized.append({"role": "user", "content": followup_prompt})
    return normalized


def _turn_prompts(definition: BenchmarkDefinition, turns: int) -> list[str]:
    if definition.prompt_sequence:
        prompts = list(definition.prompt_sequence[:turns])
        while len(prompts) < turns:
            prompts.append(definition.followup_prompt)
        if turns > 0:
            prompts[-1] = definition.followup_prompt
        return prompts
    prompts = list(DEFAULT_MULTI_TURN_PROMPTS[:turns])
    while len(prompts) < turns:
        prompts.append(definition.followup_prompt)
    if turns > 0:
        prompts[-1] = definition.followup_prompt
    return prompts


def _chunk_messages(
    messages: list[dict[str, Any]], turns: int
) -> list[list[dict[str, Any]]]:
    if turns <= 1:
        return [messages]
    if not messages:
        return [[] for _ in range(turns)]

    def _is_tool_result_message(message: dict[str, Any]) -> bool:
        content = str(message.get("content", "")).strip()
        return bool(
            message.get("role") == "user"
            and content.startswith("Tool result (")
        )

    def _is_user_authored_message(message: dict[str, Any]) -> bool:
        return bool(
            message.get("role") == "user"
            and not _is_tool_result_message(message)
        )

    units: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if _is_user_authored_message(message) and current:
            units.append(current)
            current = []
        current.append(message)
    if current:
        units.append(current)

    if not units:
        return [messages] + [[] for _ in range(turns - 1)]

    chunks: list[list[dict[str, Any]]] = []
    previous_target = 0
    for idx in range(turns):
        target = round(((idx + 1) * len(units)) / turns)
        target = max(previous_target, min(target, len(units)))
        chunk_units = units[previous_target:target]
        chunk: list[dict[str, Any]] = []
        for unit in chunk_units:
            chunk.extend(unit)
        chunks.append(chunk)
        previous_target = target

    if previous_target < len(units):
        chunks[-1].extend(
            message for unit in units[previous_target:] for message in unit
        )
    return chunks


def _minimalize_system_prompt(system: Any, original_system_prompt: str) -> Any:
    minimal_directive = (
        "Use plain text only. History may be compressed. "
        "Prefer exact filenames and verification results."
    )

    # Extract @pointers and @macros from the 'system' returned by runtime
    system_text = ""
    if isinstance(system, str):
        system_text = system
    elif isinstance(system, list):
        system_text = "\n".join(
            block.get("text", "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )

    extra_blocks = []
    for block_name in ["@pointers", "@macros"]:
        if block_name in system_text:
            lines = system_text.splitlines()
            in_block = False
            block_lines = []
            for line in lines:
                if line.strip() == block_name:
                    in_block = True
                    block_lines.append(line)
                    continue
                if in_block:
                    if (
                        line.strip().startswith("@")
                        and not line.strip().startswith("@pointers")
                        and not line.strip().startswith("@macros")
                    ):
                        in_block = False
                        continue
                    if line.strip().startswith("|> ") or not line.strip():
                        block_lines.append(line)
                    else:
                        in_block = False
            if block_lines:
                extra_blocks.append("\n".join(block_lines))

    additions = minimal_directive
    if extra_blocks:
        additions += "\n\n" + "\n\n".join(extra_blocks)

    if isinstance(system, str):
        return (
            original_system_prompt + "\n\n" + additions
            if original_system_prompt
            else additions
        )

    # Return as list of blocks if original was likely a list or specifically requested
    return (
        original_system_prompt + "\n\n" + additions
        if original_system_prompt
        else additions
    )


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in system
            if isinstance(block, dict) and str(block.get("text", "")).strip()
        )
    return ""


def _system_breakdown(
    original_system_prompt: str, system: Any
) -> tuple[int, int, int]:
    system_text = _system_text(system)
    if not system_text:
        return 0, 0, 0
    total_system_tokens = _estimate_tokens(system_text)
    state_tokens = 0
    marker = "[Tok compressed history]"
    if marker in system_text:
        state_fragment = system_text.split(marker, 1)[1].strip()
        state_tokens = _estimate_tokens(state_fragment)
    directive_tokens = max(
        0,
        total_system_tokens
        - _estimate_tokens(original_system_prompt)
        - state_tokens,
    )
    return total_system_tokens, directive_tokens, state_tokens


def _extract_labeled_fields(
    text: str, session: RuntimeSession | None = None
) -> dict[str, str]:
    fields: dict[str, str] = {}
    # Search for labels anywhere in the text (last occurrence wins)
    labels = ["file", "verification", "related"]
    for label in labels:
        # Match "File=..." or "file: ..." or "|> File=..."
        # Capture until end of line or pipe separator, allowing spaces
        pattern = rf"(?:\|>\s*)?{label}\s*[:=]\s*([^|\n]+)"
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for m in matches:
            fields[label.lower()] = m.group(1).strip()

    # Fallback to line-by-line for non-standard keys or if regex missed something
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("|>"):
            cleaned = cleaned[2:].strip()
        if "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        k = key.strip().lower()
        if k not in fields:
            fields[k] = value.strip()

    # Resolve Tok v7 Macro Pointers (*A, *B, etc.)
    if session and session.bridge_memory:
        pointers = session.bridge_memory.pointers
        for key, val in fields.items():
            if str(val).startswith("*"):
                resolved = pointers.resolve(val)
                if resolved:
                    fields[key] = resolved
    return fields


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    return any(
        marker in lowered
        for marker in (
            "<the",
            "<the specific",
            "<the command",
            "<the file",
            "<the result",
            "<the specific filename",
            "specific filename from the code change",
            "command run or test outcome",
        )
    )


def _evaluate_task_success(
    definition: BenchmarkDefinition,
    visible_response: str,
    session: RuntimeSession | None = None,
) -> tuple[bool, list[str], list[str]]:
    if session:
        session_id = id(session)
        pointers_id = (
            id(session.bridge_memory.pointers)
            if session.bridge_memory
            else "none"
        )
        print(
            f"DEBUG: _evaluate_task_success session={session_id} pointers={pointers_id}"
        )
    fields = _extract_labeled_fields(visible_response, session=session)
    # Combine the visible text with resolved values for term matching
    search_space = visible_response.lower()
    for val in fields.values():
        search_space += f" {val.lower()}"

    matched_terms = [
        term
        for term in definition.success_terms
        if term.lower() in search_space
    ]
    failures: list[str] = []
    file_value = fields.get("file", "")
    verification_value = fields.get("verification", "")

    if not file_value:
        failures.append("missing_file_field")
    elif _looks_like_placeholder(file_value):
        failures.append("placeholder_file_field")
    elif definition.expected_file_terms and not any(
        term.lower() in file_value.lower()
        for term in definition.expected_file_terms
    ):
        failures.append("unexpected_file_field")

    if not verification_value:
        failures.append("missing_verification_field")
    elif _looks_like_placeholder(verification_value):
        failures.append("placeholder_verification_field")
    elif definition.expected_verification_terms and not any(
        term.lower() in verification_value.lower()
        for term in definition.expected_verification_terms
    ):
        failures.append("unexpected_verification_field")

    structured_valid = not failures
    if (
        not structured_valid
        and len(matched_terms) < definition.min_success_terms
    ):
        failures.append("response_missing_success_terms")

    return not failures, matched_terms, failures


def _diagnose_comparison(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    *,
    total_delta: int,
    reacquisition_delta: int,
    pressure_delta: int,
) -> str:
    if baseline.task_success and not candidate.task_success:
        return "lost_on_task_success"
    if total_delta < 0 and candidate.task_success:
        return "won_on_prompt_reduction"
    if reacquisition_delta > 0 and total_delta > 0:
        return "lost_on_reacquisition"

    response_signals = candidate.response_metrics.get(
        "response_behavior_signals", {}
    )
    if _sum_warning_signals(response_signals) > 0 and total_delta > 0:
        return "lost_on_response_drift"

    tok_overhead = int(candidate.prompt_metrics.get("tok_overhead_tokens", 0))
    total_saved = int(
        candidate.compression_metrics.get("total_saved_tokens", 0)
    )
    if total_delta > 0 and tok_overhead >= total_saved:
        return "lost_on_bootstrap_overhead"

    if total_delta > 0 and pressure_delta > 0:
        return "lost_on_response_drift"

    return "mixed_result"


def compare_results(
    baseline: BenchmarkResult, candidate: BenchmarkResult
) -> BenchmarkComparison:
    total_delta = (
        candidate.provider_usage.total_tokens
        - baseline.provider_usage.total_tokens
    )
    total_pct = None
    if baseline.provider_usage.total_tokens > 0:
        total_pct = round(
            (total_delta / baseline.provider_usage.total_tokens) * 100.0, 1
        )

    baseline_reacq = int(
        baseline.response_metrics.get("reacquisition_cost_tokens", 0)
    )
    candidate_reacq = int(
        candidate.response_metrics.get("reacquisition_cost_tokens", 0)
    )
    baseline_pressure = int(
        baseline.response_metrics.get("invisible_pressure", 0)
    )
    candidate_pressure = int(
        candidate.response_metrics.get("invisible_pressure", 0)
    )
    task_success_equal_or_better = candidate.task_success and (
        baseline.task_success == candidate.task_success
        or not baseline.task_success
    )
    provider_total_token_winner = (
        candidate.mode
        if candidate.provider_usage.total_tokens
        < baseline.provider_usage.total_tokens
        else "baseline"
    )
    diagnosis = _diagnose_comparison(
        baseline,
        candidate,
        total_delta=total_delta,
        reacquisition_delta=candidate_reacq - baseline_reacq,
        pressure_delta=candidate_pressure - baseline_pressure,
    )
    tok_improved = task_success_equal_or_better and total_delta <= 0

    return BenchmarkComparison(
        benchmark=candidate.benchmark,
        model=candidate.model,
        candidate_mode=candidate.mode,
        baseline=baseline,
        candidate=candidate,
        prompt_token_delta=candidate.provider_usage.prompt_tokens
        - baseline.provider_usage.prompt_tokens,
        completion_token_delta=candidate.provider_usage.completion_tokens
        - baseline.provider_usage.completion_tokens,
        total_token_delta=total_delta,
        total_token_delta_pct=total_pct,
        latency_delta_ms=round(
            candidate.provider_usage.latency_ms
            - baseline.provider_usage.latency_ms,
            2,
        ),
        reacquisition_delta_tokens=candidate_reacq - baseline_reacq,
        pressure_delta=candidate_pressure - baseline_pressure,
        task_success_equal_or_better=task_success_equal_or_better,
        provider_total_token_winner=provider_total_token_winner,
        diagnosis=diagnosis,
        tok_improved=tok_improved,
    )


def select_preferred_mode(
    baseline: BenchmarkResult, comparisons: list[BenchmarkComparison]
) -> str:
    viable = [
        comparison
        for comparison in comparisons
        if comparison.candidate.task_success
    ]
    if not viable:
        return "baseline" if baseline.task_success else "none"
    best = min(
        viable,
        key=lambda comparison: (
            comparison.candidate.provider_usage.total_tokens
        ),
    )
    if not baseline.task_success:
        return best.candidate.mode
    if (
        best.candidate.provider_usage.total_tokens
        < baseline.provider_usage.total_tokens
    ):
        return best.candidate.mode
    return "baseline"


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
        client: Any | None = None,
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
            max_retries=0,
        )

    def run(
        self, definition: BenchmarkDefinition, *, mode: str, turns: int = 3
    ) -> BenchmarkResult:
        if turns < 1:
            raise ValueError("turns must be >= 1")
        raw_messages = load_fixture_messages(definition.fixture_path)
        normalized = normalize_fixture_messages(
            raw_messages, definition.followup_prompt
        )
        context_messages = normalized[:-1]
        turn_prompts = _turn_prompts(definition, turns)
        message_chunks = _chunk_messages(context_messages, turns)
        original_system_tokens = _estimate_tokens(definition.system_prompt)
        runtime = UniversalTokRuntime()
        tool_compatible = mode in {"tok-tool-compatible", "tok-minimal"}

        # Identity logging for pointer registry continuity
        session_id = id(runtime)
        pointers_id = (
            id(runtime._state.pointers)
            if hasattr(runtime, "_state")
            and hasattr(runtime._state, "pointers")
            else "unknown"
        )
        print(
            f"DEBUG: LiveBenchmarkRunner.run starting session={session_id} pointers={pointers_id}"
        )

        if mode not in {
            "baseline",
            "tok-native",
            "tok-tool-compatible",
            "tok-minimal",
            "tok-neuro",
        }:
            raise ValueError(f"Unknown benchmark mode: {mode}")

        # Set Pattern Reactor toggle
        if mode == "tok-neuro":
            os.environ["TOK_NEURO_REACTOR"] = "1"
        else:
            os.environ["TOK_NEURO_REACTOR"] = "0"

        with tempfile.TemporaryDirectory(
            prefix="tok_live_benchmark_"
        ) as tmpdir:
            session = RuntimeSession(memory_dir=Path(tmpdir))
            conversation: list[dict[str, str]] = []
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

            from ..compression import compress_history
            from ..runtime.policy.smart_policy import policy_for_model

            past_messages: list[dict[str, str]] = []
            for idx, chunk in enumerate(message_chunks):
                # Ingest previous assistant turns to prime the Reactor
                if mode == "tok-neuro" and idx > 0:
                    h_profile = dict(
                        policy_for_model(self.model).history_profiles[
                            "balanced"
                        ]
                    )
                    h_profile["_no_pointers"] = True
                    _, tok_state = compress_history(
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

                normalized_messages_tokens = _estimate_tokens(conversation)
                baseline_prompt_estimate = (
                    original_system_tokens + normalized_messages_tokens
                )
                prepared = None
                prepared_body: dict[str, Any] | None = None

                if mode == "baseline":
                    chat_messages = [
                        {
                            "role": "system",
                            "content": definition.system_prompt,
                        },
                        *conversation,
                    ]
                    turn_compression_metrics: dict[str, Any] = {
                        "input_saved_tokens": 0,
                        "output_saved_tokens": 0,
                        "total_saved_tokens": 0,
                        "input_behavior_signals": {},
                        "type_breakdown": {},
                    }
                    turn_prompt_metrics: dict[str, Any] = {
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
                    turn_response_metrics: dict[str, Any] = {
                        "response_behavior_signals": {},
                        "invisible_pressure": 0,
                        "reacquisition_cost_tokens": 0,
                        "family_mode": "",
                        "response_mode": "baseline",
                    }
                    turn_diagnostics: dict[str, Any] = {
                        "tool_compatible_requested": False,
                        "request_messages_before": len(conversation),
                        "request_messages_after": len(chat_messages),
                    }
                    outbound_payload: dict[str, Any] = {
                        "system": definition.system_prompt,
                        "messages": conversation.copy(),
                    }
                else:
                    prepared = runtime.prepare_request(
                        RuntimeRequest(
                            model=self.model,
                            messages=conversation,
                            system=definition.system_prompt,
                            adapter_kind="text-loop",
                            tool_compatible=tool_compatible,
                        ),
                        session,
                    )
                    prepared_body = dict(prepared.body)
                    if mode == "tok-minimal":
                        prepared_body["system"] = _minimalize_system_prompt(
                            prepared_body.get("system"),
                            definition.system_prompt,
                        )
                    chat_messages = _system_to_messages(
                        prepared_body.get("system")
                    ) + prepared_body.get("messages", [])
                    prepared_system_tokens = _estimate_tokens(
                        prepared_body.get("system")
                    )
                    prepared_messages_tokens = _estimate_tokens(
                        prepared_body.get("messages", [])
                    )
                    (
                        system_tokens_estimate,
                        directive_tokens_estimate,
                        state_payload_tokens_estimate,
                    ) = _system_breakdown(
                        definition.system_prompt, prepared_body.get("system")
                    )
                    outbound_prompt_estimate = (
                        prepared_system_tokens + prepared_messages_tokens
                    )
                    tok_system_additions_tokens = max(
                        0, prepared_system_tokens - original_system_tokens
                    )
                    tok_overhead_tokens = max(
                        0,
                        outbound_prompt_estimate
                        - baseline_prompt_estimate
                        + prepared.input_saved_tokens,
                    )
                    turn_compression_metrics = {
                        "input_saved_tokens": prepared.input_saved_tokens,
                        "output_saved_tokens": 0,
                        "total_saved_tokens": prepared.input_saved_tokens,
                        "input_behavior_signals": dict(
                            prepared.behavior_signals
                        ),
                        "type_breakdown": dict(prepared.type_breakdown),
                    }
                    turn_prompt_metrics = {
                        "system_prompt_tokens": original_system_tokens,
                        "normalized_messages_tokens": normalized_messages_tokens,
                        "prepared_messages_tokens": prepared_messages_tokens,
                        "system_tokens_estimate": system_tokens_estimate,
                        "directive_tokens_estimate": directive_tokens_estimate,
                        "state_payload_tokens_estimate": state_payload_tokens_estimate,
                        "tok_system_additions_tokens": tok_system_additions_tokens,
                        "tok_overhead_tokens": tok_overhead_tokens,
                        "estimated_prompt_delta_tokens": outbound_prompt_estimate
                        - baseline_prompt_estimate,
                        "outbound_prompt_estimate_tokens": outbound_prompt_estimate,
                    }
                    turn_response_metrics = {
                        "response_behavior_signals": {},
                        "invisible_pressure": 0,
                        "reacquisition_cost_tokens": int(
                            prepared.behavior_signals.get(
                                "reacquisition_cost_tokens", 0
                            )
                        ),
                        "family_mode": "",
                        "response_mode": mode,
                    }
                    turn_diagnostics = {
                        "tool_compatible_requested": tool_compatible,
                        "request_messages_before": len(conversation),
                        "request_messages_after": len(chat_messages),
                        "runtime_mode": prepared.mode,
                    }
                    outbound_payload = {
                        "system": prepared_body.get("system"),
                        "messages": prepared_body.get("messages", []),
                    }

                started = time.time()
                create_kwargs: dict[str, Any] = dict(
                    model=self.model,
                    messages=chat_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if self.provider_options:
                    create_kwargs["extra_body"] = self.provider_options

                # Final safety pass for Bedrock/Gemini/OpenRouter compatibility
                chat_messages = apply_schema_adaptations(chat_messages)
                create_kwargs["messages"] = chat_messages

                response = self.client.chat.completions.create(**create_kwargs)
                latency_ms = (time.time() - started) * 1000

                raw_response = response.choices[0].message.content or ""
                visible_response = raw_response
                if mode != "baseline":
                    processed = runtime.process_response(
                        raw_response,
                        model=self.model,
                        session=session,
                        behavior_signals=turn_compression_metrics[
                            "input_behavior_signals"
                        ],
                        tool_compatible=tool_compatible,
                    )
                    visible_response = (
                        "\n".join(
                            block.get("text", "")
                            for block in processed.content_blocks
                            if block.get("type") == "text"
                        ).strip()
                        or raw_response
                    )
                    turn_compression_metrics["output_saved_tokens"] = (
                        processed.output_saved_tokens
                    )
                    turn_compression_metrics["total_saved_tokens"] = (
                        turn_compression_metrics["input_saved_tokens"]
                        + processed.output_saved_tokens
                    )
                    turn_response_metrics["response_behavior_signals"] = dict(
                        processed.behavior_signals
                    )
                    turn_response_metrics["family_mode"] = (
                        processed.family_mode
                    )
                    turn_response_metrics["invisible_pressure"] = (
                        calculate_invisible_pressure(
                            processed.behavior_signals
                        )
                    )
                    turn_response_metrics["reacquisition_cost_tokens"] = int(
                        processed.behavior_signals.get(
                            "reacquisition_cost_tokens", 0
                        )
                        or turn_response_metrics["reacquisition_cost_tokens"]
                    )
                    turn_response_metrics["response_mode"] = processed.mode

                usage = response.usage
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
                completion_tokens = int(getattr(usage, "completion_tokens", 0))
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_latency_ms += latency_ms
                total_input_saved += int(
                    turn_compression_metrics["input_saved_tokens"]
                )
                total_output_saved += int(
                    turn_compression_metrics["output_saved_tokens"]
                )
                for key, value in turn_compression_metrics[
                    "type_breakdown"
                ].items():
                    total_type_breakdown[key] = total_type_breakdown.get(
                        key, 0
                    ) + int(value)
                for key, value in turn_compression_metrics[
                    "input_behavior_signals"
                ].items():
                    aggregate_input_behavior_signals[key] = (
                        aggregate_input_behavior_signals.get(key, 0)
                        + int(value)
                    )
                for key, value in turn_response_metrics[
                    "response_behavior_signals"
                ].items():
                    aggregate_response_signals[key] = (
                        aggregate_response_signals.get(key, 0) + int(value)
                    )

                turn_diagnostics["response_warning_signal_count"] = (
                    _sum_warning_signals(
                        turn_response_metrics["response_behavior_signals"]
                    )
                )
                turn_results.append(
                    {
                        "turn": idx + 1,
                        "provider_usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": int(
                                getattr(usage, "total_tokens", 0)
                            ),
                            "latency_ms": round(latency_ms, 2),
                        },
                        "compression_metrics": turn_compression_metrics,
                        "prompt_metrics": turn_prompt_metrics,
                        "response_metrics": turn_response_metrics,
                        "diagnostics": turn_diagnostics,
                        "outbound_payload": outbound_payload,
                        "visible_response": visible_response,
                        "raw_response": raw_response,
                    }
                )
                final_visible_response = visible_response
                final_raw_response = raw_response
                conversation.append(
                    {"role": "assistant", "content": raw_response}
                )

            cost_usd: float | None = None
            if self.pricing is not None:
                from ..runtime.metrics import calculate_usage_cost

                cost_usd = calculate_usage_cost(
                    total_prompt_tokens, total_completion_tokens, self.pricing
                )
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
            if mode != "baseline" and _sum_warning_signals(
                aggregate_response_signals
            ):
                notes.append("response_contract_friction_detected")

            prompt_metrics = {
                "system_prompt_tokens": original_system_tokens,
                "normalized_messages_tokens": sum(
                    int(turn["prompt_metrics"]["normalized_messages_tokens"])
                    for turn in turn_results
                ),
                "prepared_messages_tokens": sum(
                    int(turn["prompt_metrics"]["prepared_messages_tokens"])
                    for turn in turn_results
                ),
                "system_tokens_estimate": sum(
                    int(turn["prompt_metrics"]["system_tokens_estimate"])
                    for turn in turn_results
                ),
                "directive_tokens_estimate": sum(
                    int(turn["prompt_metrics"]["directive_tokens_estimate"])
                    for turn in turn_results
                ),
                "state_payload_tokens_estimate": sum(
                    int(
                        turn["prompt_metrics"]["state_payload_tokens_estimate"]
                    )
                    for turn in turn_results
                ),
                "tok_system_additions_tokens": sum(
                    int(turn["prompt_metrics"]["tok_system_additions_tokens"])
                    for turn in turn_results
                ),
                "tok_overhead_tokens": sum(
                    int(turn["prompt_metrics"]["tok_overhead_tokens"])
                    for turn in turn_results
                ),
                "estimated_prompt_delta_tokens": sum(
                    int(
                        turn["prompt_metrics"]["estimated_prompt_delta_tokens"]
                    )
                    for turn in turn_results
                ),
                "outbound_prompt_estimate_tokens": sum(
                    int(
                        turn["prompt_metrics"][
                            "outbound_prompt_estimate_tokens"
                        ]
                    )
                    for turn in turn_results
                ),
            }
            response_metrics = {
                "response_behavior_signals": aggregate_response_signals,
                "invisible_pressure": calculate_invisible_pressure(
                    aggregate_response_signals
                ),
                "reacquisition_cost_tokens": int(
                    aggregate_response_signals.get(
                        "reacquisition_cost_tokens", 0
                    )
                ),
                "family_mode": (
                    turn_results[-1]["response_metrics"]["family_mode"]
                    if turn_results
                    else ""
                ),
                "response_mode": (
                    turn_results[-1]["response_metrics"]["response_mode"]
                    if turn_results
                    else mode
                ),
            }
            diagnostics = {
                "tool_compatible_requested": tool_compatible,
                "request_messages_before": len(context_messages),
                "request_messages_after": (
                    turn_results[-1]["diagnostics"]["request_messages_after"]
                    if turn_results
                    else 0
                ),
                "response_warning_signal_count": _sum_warning_signals(
                    aggregate_response_signals
                ),
                "session_turns": turns,
                "cumulative_prompt_tokens": total_prompt_tokens,
                "cumulative_completion_tokens": total_completion_tokens,
                "state_resend_suppressed_turns": aggregate_input_behavior_signals.get(
                    "state_resend_suppressed_turn", 0
                ),
                "state_resend_delta_turns": aggregate_input_behavior_signals.get(
                    "state_resend_delta_turn", 0
                ),
                "state_resend_full_turns": aggregate_input_behavior_signals.get(
                    "state_resend_full_turn", 0
                ),
            }
            if mode != "baseline" and turn_results:
                diagnostics["runtime_mode"] = turn_results[-1][
                    "diagnostics"
                ].get("runtime_mode", "")

            compression_metrics = {
                "input_saved_tokens": total_input_saved,
                "output_saved_tokens": total_output_saved,
                "total_saved_tokens": total_input_saved + total_output_saved,
                "input_behavior_signals": aggregate_input_behavior_signals,
                "type_breakdown": total_type_breakdown,
            }

            return BenchmarkResult(
                benchmark=definition.name,
                mode=mode,
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
                request_messages=(
                    turn_results[-1]["diagnostics"]["request_messages_after"]
                    if turn_results
                    else 0
                ),
                turn_count=turns,
                turns=turn_results,
                visible_response=final_visible_response,
                raw_response=final_raw_response,
                notes=notes,
            )


def write_result(
    path: Path, payload: BenchmarkResult | BenchmarkComparison
) -> None:
    path.write_text(json.dumps(payload.to_dict(), indent=2))


def render_comparison_markdown(
    baseline: BenchmarkResult, comparisons: list[BenchmarkComparison]
) -> str:
    lines = [
        f"# Live Benchmark: {baseline.benchmark}",
        "",
        f"- Model: `{baseline.model}`",
        f"- Baseline total tokens: `{baseline.provider_usage.total_tokens}`",
        f"- Session turns: `{baseline.turn_count}`",
        "",
        "| Mode | Success | Total | Prompt | Completion | Tok saved | Tok overhead | Pressure | Reacquisition | Diagnosis |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| baseline | {baseline.task_success} | {baseline.provider_usage.total_tokens} | "
            f"{baseline.provider_usage.prompt_tokens} | {baseline.provider_usage.completion_tokens} | "
            f"0 | 0 | 0 | 0 | baseline |"
        ),
    ]
    for comparison in comparisons:
        candidate = comparison.candidate
        lines.append(
            f"| {candidate.mode} | {candidate.task_success} | {candidate.provider_usage.total_tokens} | "
            f"{candidate.provider_usage.prompt_tokens} | {candidate.provider_usage.completion_tokens} | "
            f"{candidate.compression_metrics.get('total_saved_tokens', 0)} | "
            f"{candidate.prompt_metrics.get('tok_overhead_tokens', 0)} | "
            f"{candidate.response_metrics.get('invisible_pressure', 0)} | "
            f"{candidate.response_metrics.get('reacquisition_cost_tokens', 0)} | "
            f"{comparison.diagnosis} |"
        )
    lines.extend(["", "## Comparisons", ""])
    for comparison in comparisons:
        pct = (
            f"{comparison.total_token_delta_pct:+.1f}%"
            if comparison.total_token_delta_pct is not None
            else "n/a"
        )
        lines.extend(
            [
                f"### {comparison.candidate.mode}",
                "",
                f"- Total token delta: `{comparison.total_token_delta}` ({pct})",
                f"- Prompt token delta: `{comparison.prompt_token_delta}`",
                f"- Completion token delta: `{comparison.completion_token_delta}`",
                f"- Directive tokens estimate: `{comparison.candidate.prompt_metrics.get('directive_tokens_estimate', 0)}`",
                f"- State payload tokens estimate: `{comparison.candidate.prompt_metrics.get('state_payload_tokens_estimate', 0)}`",
                f"- Latency delta (ms): `{comparison.latency_delta_ms}`",
                f"- Reacquisition delta (tokens): `{comparison.reacquisition_delta_tokens}`",
                f"- Pressure delta: `{comparison.pressure_delta}`",
                f"- Task success equal or better: `{comparison.task_success_equal_or_better}`",
                f"- Candidate task success: `{comparison.candidate.task_success}`",
                f"- Candidate notes: `{', '.join(comparison.candidate.notes) or 'none'}`",
                f"- Provider total token winner: `{comparison.provider_total_token_winner}`",
                f"- Diagnosis: `{comparison.diagnosis}`",
                "",
            ]
        )
    lines.append(
        f"- Preferred mode: `{select_preferred_mode(baseline, comparisons)}`"
    )
    lines.append("")
    return "\n".join(lines)


def summarize_compare_runs(
    repeated_results: list[dict[str, BenchmarkResult]],
) -> dict[str, Any]:
    if not repeated_results:
        return {
            "runs": 0,
            "preferred_mode_counts": {},
            "mode_summaries": {},
        }

    mode_order = (
        "baseline",
        "tok-minimal",
        "tok-native",
        "tok-tool-compatible",
        "tok-neuro",
    )
    preferred_mode_counts: dict[str, int] = {}
    mode_summaries: dict[str, Any] = {}

    for run in repeated_results:
        baseline = run["baseline"]
        comparisons = []
        for m in [
            "tok-minimal",
            "tok-native",
            "tok-tool-compatible",
            "tok-neuro",
        ]:
            if m in run:
                comparisons.append(compare_results(baseline, run[m]))
        if not comparisons:
            continue
        preferred = select_preferred_mode(baseline, comparisons)
        preferred_mode_counts[preferred] = (
            preferred_mode_counts.get(preferred, 0) + 1
        )

    for mode in mode_order:
        results = [run[mode] for run in repeated_results if mode in run]
        if not results:
            continue
        total_tokens = [
            result.provider_usage.total_tokens for result in results
        ]
        prompt_tokens = [
            result.provider_usage.prompt_tokens for result in results
        ]
        completion_tokens = [
            result.provider_usage.completion_tokens for result in results
        ]
        latency_ms = [result.provider_usage.latency_ms for result in results]
        successes = sum(1 for result in results if result.task_success)

        mode_summaries[mode] = {
            "runs": len(results),
            "success_rate": round(successes / len(results), 3),
            "success_count": successes,
            "median_total_tokens": int(statistics.median(total_tokens)),
            "min_total_tokens": min(total_tokens),
            "max_total_tokens": max(total_tokens),
            "median_prompt_tokens": int(statistics.median(prompt_tokens)),
            "median_completion_tokens": int(
                statistics.median(completion_tokens)
            ),
            "median_latency_ms": round(statistics.median(latency_ms), 2),
        }

    return {
        "runs": len(repeated_results),
        "preferred_mode_counts": preferred_mode_counts,
        "mode_summaries": mode_summaries,
    }


def render_stability_markdown(
    benchmark: str,
    model: str,
    summary: dict[str, Any],
) -> str:
    lines = [
        f"# Live Benchmark Stability: {benchmark}",
        "",
        f"- Model: `{model}`",
        f"- Repeats: `{summary.get('runs', 0)}`",
        "",
        "## Preferred Mode Counts",
        "",
    ]
    preferred_counts = summary.get("preferred_mode_counts", {})
    if preferred_counts:
        for mode, count in sorted(preferred_counts.items()):
            lines.append(f"- `{mode}`: `{count}`")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Median Metrics",
            "",
            "| Mode | Success Rate | Median Total | Min | Max | Median Prompt | Median Completion | Median Latency (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, metrics in summary.get("mode_summaries", {}).items():
        lines.append(
            f"| {mode} | {metrics['success_rate']:.3f} | {metrics['median_total_tokens']} | "
            f"{metrics['min_total_tokens']} | {metrics['max_total_tokens']} | "
            f"{metrics['median_prompt_tokens']} | {metrics['median_completion_tokens']} | "
            f"{metrics['median_latency_ms']} |"
        )

    lines.append("")
    return "\n".join(lines)


def check_stability_artifacts(
    stability_dir: Path,
    required_benchmarks: list[str],
) -> dict[str, dict[str, Any]]:
    """Validate checked-in stability artifacts for the release gate."""
    results: dict[str, dict[str, Any]] = {}

    for benchmark in required_benchmarks:
        artifact = stability_dir / f"{benchmark}_stability.json"
        row: dict[str, Any] = {
            "path": str(artifact),
            "passed": False,
        }

        if not artifact.exists():
            row["reason"] = "missing"
            results[benchmark] = row
            continue

        try:
            payload = json.loads(artifact.read_text())
        except Exception as exc:
            row["reason"] = "invalid_json"
            row["error"] = str(exc)
            results[benchmark] = row
            continue

        mode_summaries = payload.get("mode_summaries", {})
        preferred_mode_counts = payload.get("preferred_mode_counts", {})
        tok_summary = mode_summaries.get("tok-tool-compatible", {})
        runs = int(payload.get("runs", 0))
        preferred_count = int(preferred_mode_counts.get("tok-tool-compatible", 0))
        success_rate = float(tok_summary.get("success_rate", 0.0))

        passed = (
            benchmark == str(payload.get("benchmark", ""))
            and runs > 0
            and success_rate == 1.0
            and preferred_count == runs
        )

        row.update(
            {
                "benchmark": str(payload.get("benchmark", "")),
                "runs": runs,
                "success_rate": success_rate,
                "preferred_mode": preferred_count,
                "passed": passed,
            }
        )
        if not passed:
            row["reason"] = "criteria_failed"

        results[benchmark] = row

    return results


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_text(item) for item in content)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return _content_text(content["content"])
    return str(content)
