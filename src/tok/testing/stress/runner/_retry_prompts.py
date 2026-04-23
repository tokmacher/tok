from __future__ import annotations

from typing import Any

from ..classification import (
    _early_retry_contract_stage,
    _late_retry_contract_stage,
    _retry_prompt_shape,
)
from ._turn_validation import fields_match_expected


def retry_prompt(
    *,
    task: Any,
    expected_fields: dict[str, str],
    observed_fields: dict[str, str],
    attempt_tool_count: int,
    attempt_tool_names: set[str],
    validated_reacquisition: bool,
    target_already_validated: bool,
    payload_pressure_ready: bool,
    validated_target_exact_reacquired: bool,
    validated_target_reconfirmation_attempt: bool,
    mixed_answer_tool_event: bool,
    toolless_fresh_answer_event: bool,
    unsupported_tool_event: bool,
    bad_tool_args_event: bool,
    prior_turn_has_valid_supporting_tool_backing: bool,
    current_turn_was_tool_only_retry: bool,
    current_turn_satisfied_tool_only_stage: bool,
    retry_index: int,
) -> tuple[str, dict[str, int]]:
    prompt_shape = _retry_prompt_shape(
        task=task,
        retry_index=retry_index,
        target_already_validated=target_already_validated,
        payload_pressure_ready=payload_pressure_ready,
        validated_target_exact_reacquired=validated_target_exact_reacquired,
        validated_target_reconfirmation_attempt=validated_target_reconfirmation_attempt,
        mixed_answer_tool_event=mixed_answer_tool_event,
        toolless_fresh_answer_event=toolless_fresh_answer_event,
        unsupported_tool_event=unsupported_tool_event,
        bad_tool_args_event=bad_tool_args_event,
    )
    diagnostics = {f"retry_prompt_shape_{prompt_shape}": 1}
    expected_file = expected_fields.get("file", "<missing>")
    expected_verification = expected_fields.get("verification", "<missing>")
    staged_contract = _late_retry_contract_stage(
        task=task,
        prompt_shape=prompt_shape,
        payload_pressure_ready=payload_pressure_ready,
        validated_target_exact_reacquired=validated_target_exact_reacquired,
        unsupported_tool_event=unsupported_tool_event,
        bad_tool_args_event=bad_tool_args_event,
        prior_turn_has_valid_supporting_tool_backing=prior_turn_has_valid_supporting_tool_backing,
        current_turn_was_tool_only_retry=current_turn_was_tool_only_retry,
        current_turn_satisfied_tool_only_stage=current_turn_satisfied_tool_only_stage,
    )
    early_staged_contract = _early_retry_contract_stage(
        task=task,
        payload_pressure_ready=payload_pressure_ready,
        validated_target_exact_reacquired=validated_target_exact_reacquired,
        unsupported_tool_event=unsupported_tool_event,
        bad_tool_args_event=bad_tool_args_event,
        mixed_answer_tool_event=mixed_answer_tool_event,
        prior_turn_has_valid_supporting_tool_backing=prior_turn_has_valid_supporting_tool_backing,
        current_turn_was_tool_only_retry=current_turn_was_tool_only_retry,
        current_turn_satisfied_tool_only_stage=current_turn_satisfied_tool_only_stage,
    )
    if task.phase_name == "tool-contract" and target_already_validated and payload_pressure_ready:
        if prompt_shape == "mixed_turn":
            diagnostics["retry_prompt_no_exact_reread"] = 1
            if task.require_fresh_evidence:
                diagnostics["retry_prompt_requires_supporting_tool"] = 1
                return (
                    (
                        f"Retry {retry_index}: your previous turn mixed tool use with a final answer. "
                        f"CRITICAL: Perform ONLY ONE action. If gathering evidence, emit ONLY @Tool blocks. "
                        f"If answering, emit ONLY the File=/Verification= block. Do NOT do both in one turn."
                        "The target is already validated. Do not reopen the exact target again on this retry. "
                        "You must use exactly one supported read-only tool before answering, and it must gather only supporting evidence that is not the exact validated target. "
                        "Then end in exactly two lines:\n"
                        f"File={expected_file}\n"
                        f"Verification={expected_verification}"
                    ),
                    diagnostics,
                )
            return (
                (
                    f"Retry {retry_index}: your previous turn mixed tool use with a final answer. "
                    f"CRITICAL: Perform ONLY ONE action. If gathering evidence, emit ONLY @Tool blocks. "
                    f"If answering, emit ONLY the File=/Verification= block. Do NOT do both in one turn."
                    "The target is already validated. Do not reopen the exact target again on this retry. "
                    "Either use already validated evidence to answer directly, or if you truly need fresh support, "
                    "gather only supporting evidence that is not the exact validated target. When you answer, emit exactly two lines:\n"
                    f"File={expected_file}\n"
                    f"Verification={expected_verification}"
                ),
                diagnostics,
            )
        if prompt_shape == "toolless_fresh":
            diagnostics["retry_prompt_no_exact_reread"] = 1
            diagnostics["retry_prompt_requires_supporting_tool"] = 1
            return (
                (
                    f"Retry {retry_index}: you answered without satisfying the fresh-evidence requirement, "
                    "but the target is already validated. Do not reopen the exact validated target on this retry. "
                    "You must use exactly one supported read-only tool before answering, and it must gather only supporting evidence that is not the exact validated target. "
                    "Then answer in exactly two lines:\n"
                    f"File={expected_file}\n"
                    f"Verification={expected_verification}"
                ),
                diagnostics,
            )
        if prompt_shape == "exact_target_reread":
            diagnostics["retry_prompt_no_exact_reread"] = 1
            return (
                (
                    f"Retry {retry_index}: you reopened an already validated exact target. "
                    "Do not read or search the exact validated target again on this retry. "
                    "Answer from already validated evidence unless the session truly lost the fact. "
                    "When you answer, emit exactly two lines:\n"
                    f"File={expected_file}\n"
                    f"Verification={expected_verification}"
                ),
                diagnostics,
            )
    if staged_contract == "tool_only":
        diagnostics["late_retry_contract_stage_tool_only"] = 1
        prompt_lines = [
            (
                f"Retry {retry_index}: do not answer yet. Use exactly one supported "
                "read-only tool in this turn and nothing else. Do not include File= "
                "or Verification=. Do not mix tool use with a final answer."
            )
        ]
        if target_already_validated:
            diagnostics["late_retry_no_exact_target"] = 1
            prompt_lines.append("Do not reopen the exact validated target.")
        if task.require_fresh_evidence and target_already_validated:
            prompt_lines.append("Use the tool only for supporting evidence, not the exact validated target.")
        return ("\n".join(prompt_lines), diagnostics)
    if staged_contract == "answer_only":
        diagnostics["late_retry_contract_stage_answer_only"] = 1
        opening = f"Retry {retry_index}: enough evidence is already available. Do not call tools in this turn."
        if target_already_validated:
            diagnostics["late_retry_no_exact_target"] = 1
            opening += " Do not reopen the exact validated target."
        prompt_lines = [
            opening + " Reply in exactly two lines:",
            f"File={expected_file}",
            f"Verification={expected_verification}",
        ]
        return ("\n".join(prompt_lines), diagnostics)
    if early_staged_contract == "tool_only_bad_args":
        diagnostics["early_retry_contract_stage_tool_only"] = 1
        diagnostics["early_retry_bad_args_tool_only"] = 1
        return (
            (
                f"Retry {retry_index}: your previous tool call used invalid arguments. "
                "Do not answer yet. Use exactly one supported read-only tool with valid "
                "arguments in this turn and nothing else. Do not include File= or Verification=."
            ),
            diagnostics,
        )
    if early_staged_contract == "tool_only_mixed":
        diagnostics["early_retry_contract_stage_tool_only"] = 1
        return (
            (
                f"Retry {retry_index}: your previous turn mixed tool use with a final answer. "
                f"CRITICAL: Perform ONLY ONE action. If gathering evidence, emit ONLY @Tool blocks. "
                f"If answering, emit ONLY the File=/Verification= block. Do NOT do both in one turn."
                "Do not answer yet. Use exactly one supported read-only tool in this turn "
                "and nothing else. Do not include File= or Verification=. Do not mix tool "
                "use with a final answer."
            ),
            diagnostics,
        )
    if early_staged_contract == "answer_only_mixed":
        diagnostics["early_retry_contract_stage_answer_only"] = 1
        return (
            (
                f"Retry {retry_index}: enough evidence is already available. Do not call tools "
                "in this turn. Reply in exactly two lines:\n"
                f"File={expected_file}\n"
                f"Verification={expected_verification}"
            ),
            diagnostics,
        )
    if early_staged_contract == "answer_only":
        diagnostics["early_retry_contract_stage_answer_only"] = 1
        return (
            (
                f"Retry {retry_index}: enough evidence is already available. Do not call tools "
                "in this turn. Reply in exactly two lines:\n"
                f"File={expected_file}\n"
                f"Verification={expected_verification}"
            ),
            diagnostics,
        )
    reasons: list[str] = []
    if task.forbid_reacquisition and validated_reacquisition:
        reasons.append("you reacquired a previously validated target instead of reusing memory")
    if not fields_match_expected(expected_fields, observed_fields):
        reasons.append("your last answer did not match the grounded target")
    if task.require_fresh_evidence and attempt_tool_count < max(task.require_tool_count, 1):
        reasons.append("you answered without enough fresh read-only evidence")
    if task.required_tool_names and not (set(task.required_tool_names) & set(attempt_tool_names)):
        reasons.append(f"this task requires a direct file read via {', '.join(task.required_tool_names)}")
    reason_text = "; ".join(reasons) or "your previous answer was not acceptable"
    return (
        (
            f"Retry {retry_index}: {reason_text}. "
            "Stay grounded in real repo facts. "
            f"Focus on the expected target class: {expected_verification} in {expected_file}. "
            "If fresh evidence is required, use the read-only tools first. "
            "If this is a reuse task, answer from validated memory unless the session truly lost the fact. "
            "When you answer, emit exactly two lines:\n"
            f"File={expected_file}\n"
            f"Verification={expected_verification}"
        ),
        diagnostics,
    )


def seed_synthesis_prompt(expected_fields: dict[str, str], *, evidence_sufficient: bool) -> str:
    if evidence_sufficient:
        return (
            "Use the evidence you just retrieved. Do not call another tool unless the evidence is insufficient. "
            "Answer now in exactly two lines:\n"
            f"File={expected_fields.get('file', '<missing>')}\n"
            f"Verification={expected_fields.get('verification', '<missing>')}"
        )
    return (
        "Use the evidence you just retrieved. If it is still insufficient, do one narrow search or one direct file read next. "
        "Once you have enough evidence, stop searching and answer in exactly two lines:\n"
        f"File={expected_fields.get('file', '<missing>')}\n"
        f"Verification={expected_fields.get('verification', '<missing>')}"
    )


def checkpoint_prompt(anchor_history: list[Any]) -> str:
    if not anchor_history:
        return (
            "Checkpoint: summarize the most recent grounded answer in exactly two lines with no extra prose:\n"
            "File=<the primary file>\nVerification=<the function or symbol>"
        )
    oldest = anchor_history[0]
    latest = anchor_history[-1]
    return (
        "Checkpoint: recover the oldest validated anchor, not the latest one. "
        "Do not switch to newer facts. Use no new tools unless the session truly lost it. "
        "When you answer, emit exactly two lines:\n"
        f"File={oldest.file or latest.file}\nVerification={oldest.verification or latest.verification}"
    )
