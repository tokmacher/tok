"""Report rendering and artifact writing for stress harness."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .classification import required_class_coverage
from .models import (
    _PATH_PATTERN,
    EXCLUDED_GROUNDED_PATH_FRAGMENTS,
    StressBreakpoint,
    StressRunResult,
)
from .utils import _extract_labeled_fields, _normalize_extracted_path

if TYPE_CHECKING:
    from tok.runtime import RuntimeSession


def render_stress_report(result: StressRunResult, session: RuntimeSession | None = None) -> str:
    coverage = required_class_coverage(
        {item.breakpoint_class for item in result.breakpoints},
        result.required_classes,
    )
    lines = [
        "# Tok Stress Report",
        "",
        f"- Model: `{result.model}`",
        f"- Provider: `{result.provider}`",
        f"- Started: `{result.started_at}`",
        f"- Completed: `{result.completed_at}`",
        f"- Tasks completed: `{result.tasks_completed}`",
        f"- Total tokens: `{result.total_tokens}`",
        f"- Baseline-only reached: `{result.baseline_only}`",
        f"- Required coverage complete: `{coverage['complete']}`",
        f"- Covered classes: `{', '.join(coverage['covered']) or 'none'}`",
        f"- Missing classes: `{', '.join(coverage['missing']) or 'none'}`",
        f"- Validated anchors: `{getattr(result, 'validated_anchor_count', 0)}`",
        f"- Reuse checks run: `{getattr(result, 'reuse_checks_run', 0)}`",
        f"- Checkpoint checks run: `{getattr(result, 'checkpoint_checks_run', 0)}`",
        f"- Anchors before baseline: `{getattr(result, 'anchors_before_baseline', 0)}`",
        f"- Seed searches: `{getattr(result, 'seed_searches', 0)}`",
        f"- Seed direct reads: `{getattr(result, 'seed_direct_reads', 0)}`",
        f"- Seed answer attempts: `{getattr(result, 'seed_answer_attempts', 0)}`",
        f"- Seed evidence sufficient: `{getattr(result, 'seed_evidence_sufficient', False)}`",
        f"- First-anchor failure mode: `{getattr(result, 'first_anchor_failure_mode', 'unknown')}`",
        f"- Reuse probes: `attempts={getattr(result, 'reuse_probe_attempts', 0)} successes={getattr(result, 'reuse_probe_successes', 0)} reacquisitions={getattr(result, 'reacquisition_events_seen', 0)}`",
        f"- Reacquisition diagnostics: `validated_target_exact={getattr(result, 'validated_target_exact_reacquisition_events_seen', getattr(result, 'validated_target_reacquisition_events_seen', 0))} validated_target_reconfirmation={getattr(result, 'validated_target_reconfirmation_events_seen', 0)} anchor_backed={getattr(result, 'answer_anchor_reacquisition_events_seen', 0)} answer_ready={getattr(result, 'answer_ready_reacquisition_events_seen', 0)} repair_phase={getattr(result, 'repair_phase_reacquisition_events_seen', 0)} benign_reverification={getattr(result, 'benign_reverification_events_seen', 0)}`",
        f"- Retention probes: `attempts={getattr(result, 'retention_probe_attempts', 0)} successes={getattr(result, 'retention_probe_successes', 0)} substitutions={getattr(result, 'retention_substitution_events_seen', 0)}`",
        f"- Late retention probes: `attempts={getattr(result, 'late_retention_probe_attempts', 0)} successes={getattr(result, 'late_retention_probe_successes', 0)}`",
        f"- Tool contract probes: `attempts={getattr(result, 'tool_contract_probe_attempts', 0)} failure_events={getattr(result, 'tool_contract_failure_events_seen', 0)}`",
        f"- Tool contract signals: `mixed={getattr(result, 'mixed_answer_tool_events_seen', 0)} unsupported={getattr(result, 'unsupported_tool_events_seen', 0)} bad_args={getattr(result, 'bad_tool_args_events_seen', 0)} toolless_fresh={getattr(result, 'toolless_fresh_answer_events_seen', 0)}`",
        f"- Early retention probe ran: `{any(getattr(turn, 'phase_name', '') == 'retention-probe' and 'retention_probe_early' in getattr(turn, 'task_id', '') for turn in getattr(result, 'turns', []))}`",
        f"- Late retention probe ran: `{any(getattr(turn, 'phase_name', '') == 'retention-probe' and 'retention_probe_early' not in getattr(turn, 'task_id', '') for turn in getattr(result, 'turns', []))}`",
        f"- Tool-backed turns: `{getattr(result, 'tool_backed_turns', 0)}/{len(getattr(result, 'turns', []))}`",
        f"- Resend modes seen: `{', '.join(getattr(result, 'resend_modes_seen', [])) or 'none'}`",
        f"- Payload pressure reached: `{getattr(result, 'payload_pressure_reached', False)}`",
        f"- Compaction eligibility: `{getattr(result, 'compaction_eligible', False)}`",
        f"- Run diagnosis: `{getattr(result, 'run_diagnosis', 'unknown')}`",
        f"- Weak-run reasons: `{', '.join(getattr(result, 'weak_run_reasons', [])) or 'none'}`",
        "",
        "## First-Anchor Failure Mode",
        "",
        f"- Failure mode: `{getattr(result, 'first_anchor_failure_mode', 'unknown')}`",
        f"- Seed searches: `{getattr(result, 'seed_searches', 0)}`",
        f"- Seed direct reads: `{getattr(result, 'seed_direct_reads', 0)}`",
        f"- Seed answer attempts: `{getattr(result, 'seed_answer_attempts', 0)}`",
        f"- Seed evidence sufficient: `{getattr(result, 'seed_evidence_sufficient', False)}`",
        "",
        "## Memory Probe Coverage",
        "",
        f"- Reuse probes: `attempts={getattr(result, 'reuse_probe_attempts', 0)} successes={getattr(result, 'reuse_probe_successes', 0)} reacquisitions={getattr(result, 'reacquisition_events_seen', 0)}`",
        f"- Reacquisition diagnostics: `validated_target_exact={getattr(result, 'validated_target_exact_reacquisition_events_seen', getattr(result, 'validated_target_reacquisition_events_seen', 0))} validated_target_reconfirmation={getattr(result, 'validated_target_reconfirmation_events_seen', 0)} anchor_backed={getattr(result, 'answer_anchor_reacquisition_events_seen', 0)} answer_ready={getattr(result, 'answer_ready_reacquisition_events_seen', 0)} repair_phase={getattr(result, 'repair_phase_reacquisition_events_seen', 0)} benign_reverification={getattr(result, 'benign_reverification_events_seen', 0)}`",
        f"- Retention probes: `attempts={getattr(result, 'retention_probe_attempts', 0)} successes={getattr(result, 'retention_probe_successes', 0)} substitutions={getattr(result, 'retention_substitution_events_seen', 0)}`",
        f"- Late retention probes: `attempts={getattr(result, 'late_retention_probe_attempts', 0)} successes={getattr(result, 'late_retention_probe_successes', 0)}`",
        f"- Tool contract probes: `attempts={getattr(result, 'tool_contract_probe_attempts', 0)} failure_events={getattr(result, 'tool_contract_failure_events_seen', 0)}`",
        f"- Tool contract signals: `mixed={getattr(result, 'mixed_answer_tool_events_seen', 0)} unsupported={getattr(result, 'unsupported_tool_events_seen', 0)} bad_args={getattr(result, 'bad_tool_args_events_seen', 0)} toolless_fresh={getattr(result, 'toolless_fresh_answer_events_seen', 0)}`",
        f"- Early retention probe ran: `{any(getattr(turn, 'phase_name', '') == 'retention-probe' and 'retention_probe_early' in getattr(turn, 'task_id', '') for turn in getattr(result, 'turns', []))}`",
        f"- Late retention probe ran: `{any(getattr(turn, 'phase_name', '') == 'retention-probe' and 'retention_probe_early' not in getattr(turn, 'task_id', '') for turn in getattr(result, 'turns', []))}`",
        f"- Compaction eligibility: `{getattr(result, 'compaction_eligible', False)}`",
        "",
        "## Frontier Diagnostics",
        "",
        f"- First payload-pressure-ready turn: `turn={getattr(result, 'first_payload_pressure_turn', None)} task={getattr(result, 'first_payload_pressure_task', '') or 'none'}`",
        f"- First compaction-eligible turn: `turn={getattr(result, 'first_compaction_eligible_turn', None)} task={getattr(result, 'first_compaction_eligible_task', '') or 'none'}`",
        f"- First baseline-fallback turn: `turn={getattr(result, 'first_baseline_fallback_turn', None)} task={getattr(result, 'first_baseline_fallback_task', '') or 'none'}`",
        f"- Fallback after payload pressure: `{getattr(result, 'fallback_after_payload_pressure', False)}` turns=`{getattr(result, 'baseline_fallback_turns_after_payload_pressure', 0)}`",
        f"- Fallback after compaction eligibility: `{getattr(result, 'fallback_after_compaction_eligible', False)}` turns=`{getattr(result, 'baseline_fallback_turns_after_compaction_eligible', 0)}`",
        f"- Answer-ready repair totals: `requested={getattr(result, 'answer_ready_repair_requested_count', 0)} active={getattr(result, 'answer_ready_repair_active_count', 0)} resolved={getattr(result, 'answer_ready_repair_resolved_count', 0)} failed={getattr(result, 'answer_ready_repair_failed_count', 0)}`",
        f"- Late freshness handoff: `promoted={getattr(result, 'late_freshness_signal_promoted_count', 0)} consumed={getattr(result, 'late_freshness_signal_consumed_by_tok_count', 0)}`",
        f"- Late mixed handoff: `promoted={getattr(result, 'late_mixed_signal_promoted_count', 0)} consumed={getattr(result, 'late_mixed_signal_consumed_by_tok_count', 0)}`",
        f"- Late mixed answer-only repair: `requested={getattr(result, 'late_answer_assembly_repair_answer_only_requested_count', 0)} resolved={getattr(result, 'late_answer_assembly_repair_answer_only_resolved_count', 0)} failed={getattr(result, 'late_answer_assembly_repair_answer_only_failed_count', 0)}`",
        f"- Late answer follow-through: `requested={getattr(result, 'late_answer_followthrough_requested_count', 0)} active={getattr(result, 'late_answer_followthrough_active_count', 0)} resolved={getattr(result, 'late_answer_followthrough_resolved_count', 0)} failed={getattr(result, 'late_answer_followthrough_failed_count', 0)}`",
        f"- Late answer follow-through after tool-only repair: `{getattr(result, 'late_answer_followthrough_after_tool_only_repair_count', 0)}`",
        f"- Late answer follow-through blocked (insufficient evidence): `{getattr(result, 'late_answer_followthrough_blocked_insufficient_evidence_count', 0)}`",
        f"- Late tool-contract grace: `reconfirmation={getattr(result, 'late_tool_contract_reconfirmation_grace_count', 0)} mixed={getattr(result, 'late_tool_contract_mixed_grace_count', 0)} toolless={getattr(result, 'late_tool_contract_toolless_grace_count', 0)}`",
        f"- Late tool-contract retry failures: `reconfirmation={getattr(result, 'late_tool_contract_reconfirmation_retry_failure_count', 0)} mixed={getattr(result, 'late_tool_contract_mixed_retry_failure_count', 0)} toolless={getattr(result, 'late_tool_contract_toolless_retry_failure_count', 0)}`",
        f"- Fallback pressure totals: `incremented={getattr(result, 'fallback_pressure_incremented_count', 0)} suppressed={getattr(result, 'fallback_pressure_suppressed_count', 0)}`",
        f"- Fallback pressure causes: `exact_reacquisition={getattr(result, 'fallback_pressure_cause_exact_reacquisition_count', 0)} mixed_turn={getattr(result, 'fallback_pressure_cause_mixed_turn_count', 0)} toolless_fresh={getattr(result, 'fallback_pressure_cause_toolless_fresh_count', 0)} bad_args={getattr(result, 'fallback_pressure_cause_bad_args_count', 0)} unsupported_tool={getattr(result, 'fallback_pressure_cause_unsupported_tool_count', 0)}`",
        f"- Retry prompt shapes: `exact_target_reread={getattr(result, 'retry_prompt_shape_exact_target_reread_count', 0)} mixed_turn={getattr(result, 'retry_prompt_shape_mixed_turn_count', 0)} toolless_fresh={getattr(result, 'retry_prompt_shape_toolless_fresh_count', 0)} unsupported_tool={getattr(result, 'retry_prompt_shape_unsupported_tool_count', 0)} bad_args={getattr(result, 'retry_prompt_shape_bad_args_count', 0)} generic_retry={getattr(result, 'retry_prompt_shape_generic_retry_count', 0)}`",
        f"- Retry no-exact-reread: `prompts={getattr(result, 'retry_prompt_no_exact_reread_count', 0)} exact_rereads_after={getattr(result, 'exact_target_reread_after_no_exact_retry_count', 0)}`",
        f"- Retry supporting-tool totals: `required={getattr(result, 'retry_prompt_requires_supporting_tool_count', 0)} satisfied={getattr(result, 'retry_prompt_supporting_tool_satisfied_count', 0)} missed={getattr(result, 'retry_prompt_supporting_tool_missed_count', 0)}`",
        f"- Retry supporting-tool misses: `mixed={getattr(result, 'retry_prompt_supporting_tool_missed_mixed_count', 0)} toolless={getattr(result, 'retry_prompt_supporting_tool_missed_toolless_count', 0)}`",
        f"- Early staged retries: `tool_only={getattr(result, 'early_retry_contract_stage_tool_only_count', 0)} answer_only={getattr(result, 'early_retry_contract_stage_answer_only_count', 0)}`",
        f"- Early bad-args tool-only: `count={getattr(result, 'early_retry_bad_args_tool_only_count', 0)}`",
        f"- Early tool-only outcomes: `satisfied={getattr(result, 'early_retry_tool_only_satisfied_count', 0)} mixed_fail={getattr(result, 'early_retry_tool_only_failed_mixed_count', 0)} toolless_fail={getattr(result, 'early_retry_tool_only_failed_toolless_count', 0)}`",
        f"- Early answer-only outcomes: `satisfied={getattr(result, 'early_retry_answer_only_satisfied_count', 0)} tool_fail={getattr(result, 'early_retry_answer_only_failed_tool_count', 0)}`",
        f"- Staged late retries: `tool_only={getattr(result, 'late_retry_contract_stage_tool_only_count', 0)} answer_only={getattr(result, 'late_retry_contract_stage_answer_only_count', 0)}`",
        f"- Tool-only stage outcomes: `satisfied={getattr(result, 'late_retry_tool_only_satisfied_count', 0)} mixed_fail={getattr(result, 'late_retry_tool_only_failed_mixed_count', 0)} toolless_fail={getattr(result, 'late_retry_tool_only_failed_toolless_count', 0)}`",
        f"- Answer-only stage outcomes: `satisfied={getattr(result, 'late_retry_answer_only_satisfied_count', 0)} tool_fail={getattr(result, 'late_retry_answer_only_failed_tool_count', 0)}`",
        f"- Staged retries with no-exact-target: `prompts={getattr(result, 'late_retry_no_exact_target_count', 0)} exact_rereads_after={getattr(result, 'exact_target_reread_after_late_retry_no_exact_target_count', 0)}`",
        f"- Reacquisition totals: `validated_target_exact={getattr(result, 'validated_target_exact_reacquisition_events_seen', getattr(result, 'validated_target_reacquisition_events_seen', 0))} validated_target_reconfirmation={getattr(result, 'validated_target_reconfirmation_events_seen', 0)} anchor_backed={getattr(result, 'answer_anchor_reacquisition_events_seen', 0)} answer_ready={getattr(result, 'answer_ready_reacquisition_events_seen', 0)} repair_phase={getattr(result, 'repair_phase_reacquisition_events_seen', 0)} benign_reverification={getattr(result, 'benign_reverification_events_seen', 0)}`",
        "",
        "### Failure Locus",
        "",
        f"- Dominant failure locus: `{getattr(result, 'dominant_failure_locus', 'mixed')}`",
        f"- First failed task: `phase={getattr(result, 'first_failed_phase', '') or 'none'} task={getattr(result, 'first_failed_task', '') or 'none'}`",
        f"- First irreversible miss kind: `{getattr(result, 'first_irreversible_miss_kind', 'first_miss_unknown')}`",
        f"- Failed tasks by retry family: `before_any={getattr(result, 'failed_tasks_before_any_retry_contract_count', 0)} generic={getattr(result, 'failed_tasks_after_generic_retry_only_count', 0)} early_staged={getattr(result, 'failed_tasks_after_early_staged_retry_count', 0)} late_staged={getattr(result, 'failed_tasks_after_late_staged_retry_count', 0)} validated_target={getattr(result, 'failed_tasks_after_validated_target_retry_count', 0)}`",
        "",
        "## Resend Analysis",
        "",
    ]
    resend_analysis = [
        f"- Turn `{getattr(turn, 'turn_index', 0)}` task `{getattr(turn, 'task_id', '')}` stayed `full`: "
        f"reason=`{getattr(turn, 'resend_decision_reason', 'unknown')}` "
        f"payload_ready=`{getattr(turn, 'payload_pressure_ready', False)}` "
        f"state_payload_chars=`{getattr(turn, 'state_payload_chars', 0)}` "
        f"tool_volume_chars=`{getattr(turn, 'tool_result_volume_chars', 0)}` "
        f"tool_dense=`{getattr(turn, 'tool_dense_session', False)}` "
        f"answer_facts=`{getattr(turn, 'answer_fact_projection_present', False)}`"
        for turn in getattr(result, "turns", [])
        if getattr(result, "compaction_eligible", False) and getattr(turn, "resend_mode", "") == "full"
    ]
    lines.extend(resend_analysis or ["- none"])
    if getattr(result, "compaction_eligible", False):
        first_compaction_turn = getattr(result, "first_compaction_eligible_turn", None)
        if first_compaction_turn is not None:
            timing = (
                "after"
                if getattr(result, "fallback_after_compaction_eligible", False)
                else "before_or_without_baseline"
            )
            lines.extend(
                [
                    f"- Compaction first became eligible on turn `{first_compaction_turn}` task `{getattr(result, 'first_compaction_eligible_task', '')}`.",
                    f"- Baseline fallback happened `{timing}` compaction eligibility materially appeared.",
                ]
            )
    lines.extend(
        [
            "",
            "## Breakpoints",
            "",
        ]
    )
    missing_classes = set(coverage["missing"])
    if "reacquisition_loop" in missing_classes:
        if getattr(result, "reuse_probe_attempts", 0) == 0:
            lines.append("- Reacquisition loop missing because reuse probes were not fairly exercised yet.")
        else:
            lines.append(
                "- Reacquisition loop missing despite reuse probes running; treat this as provisional resistance, not a harness gap."
            )
    if "tool_contract_failure" in missing_classes:
        if getattr(result, "tool_contract_probe_attempts", 0) == 0:
            lines.append(
                "- Tool contract failure missing because dedicated tool-contract probes were not fairly exercised yet."
            )
        elif getattr(result, "tool_contract_failure_events_seen", 0) == 0:
            lines.append(
                "- Tool contract failure missing even though dedicated tool-contract probes ran and held; treat this as provisional contract resistance."
            )
        else:
            lines.append(
                "- Tool contract failure missing despite contract-like events being observed; inspect the dedicated tool-contract turns for classifier or sequencing drift."
            )
    if "retention_loss" in missing_classes:
        if getattr(result, "retention_probe_attempts", 0) == 0:
            lines.append("- Retention loss missing because retention probes were not fairly exercised yet.")
        elif (
            getattr(result, "late_retention_probe_attempts", 0) == 0
            and getattr(result, "retention_probe_successes", 0) > 0
        ):
            lines.append(
                "- Retention loss missing because only the early retention probe ran and held; the later harder retention surface never ran."
            )
        elif (
            any(
                getattr(turn, "phase_name", "") == "retention-probe"
                and "retention_probe_early" in getattr(turn, "task_id", "")
                for turn in getattr(result, "turns", [])
            )
            and getattr(result, "late_retention_probe_successes", 0) > 0
        ):
            lines.append(
                "- Retention loss missing even though a later retention probe ran and held; treat this as provisional late-stage retention resistance."
            )
        else:
            lines.append(
                "- Retention loss missing because later retention probes were skipped or failed before producing a fair late-stage hold or substitution outcome."
            )
    if coverage["missing"]:
        lines.append("")
    if not result.breakpoints:
        lines.append("- none")
        lines.append("")
    implicated = summarize_implicated_files(result.breakpoints, session=session)
    if implicated:
        lines.extend(["## Implicated Files", ""])
        for imp_item in implicated:
            lines.append(f"- `{imp_item['path']}` (`{imp_item['count']}` mentions)")
        lines.append("")

    grouped: dict[str, list[StressBreakpoint]] = {}
    for breakpoint in result.breakpoints:
        grouped.setdefault(breakpoint.breakpoint_class, []).append(breakpoint)
    for breakpoint_class, items in grouped.items():
        lines.append(f"### {breakpoint_class}")
        lines.append("")
        lines.append(f"- Refactor target: `{items[0].refactor_target}`")
        lines.append(f"- Likely cause: {items[0].inferred_cause}")
        for item in items:
            lines.append(
                f"- Task `{item.task_id}` turn `{item.turn_index}` resend=`{item.resend_mode}` "
                f"state_payload_chars=`{item.state_payload_chars}` tools=`{', '.join(item.active_tools) or 'none'}`"
            )
            if item.visible_response:
                lines.append(f"- Evidence: `{item.visible_response[:180]}`")
        lines.append("")
    return "\n".join(lines)


def render_language_refactor_plan(result: StressRunResult) -> str:
    lines = [
        "# Language Refactor Plan",
        "",
        "## Summary",
        "",
    ]
    if not result.breakpoints:
        lines.append("- No distinct breakpoint classes were recorded.")
        lines.append("")

    themes: dict[str, list[str]] = {}
    for item in result.breakpoints:
        themes.setdefault(item.refactor_target, []).append(item.breakpoint_class)

    for target, classes in themes.items():
        lines.append(f"### {target}")
        lines.append("")
        unique_classes = sorted(set(classes))
        lines.append(f"- Driven by breakpoint classes: `{', '.join(unique_classes)}`")
        if target == "anchor persistence":
            lines.append(
                "- Strengthen checkpoint-era answer anchor carry rules so late turns can restate grounded file and verification facts."
            )
        elif target == "directive density":
            lines.append(
                "- Add stronger reuse cues and validated-target replay rules so the model stops reacquiring facts it already proved."
            )
        elif target == "state projection":
            lines.append(
                "- Preserve answer-bearing fields and grounded anchors when evidence volume rises enough to trigger compaction."
            )
        elif target == "response contract pressure":
            lines.append(
                "- Treat repeated wrong structured answers as contract drift pressure even when the outer Tok shape is preserved."
            )
        elif target == "fallback policy clarity":
            lines.append(
                "- Make grounded-discipline failure accumulation and degradation thresholds easier to observe during long runs."
            )
        elif target == "tool contract constraints":
            lines.append(
                "- Tighten the read-only evidence contract so unsupported or tool-less grounded answers fail faster and more clearly."
            )
        lines.append("")

    if str(getattr(result, "run_diagnosis", "")).startswith("early_contract_collapse:"):
        lines.extend(
            [
                "## First Anchor Hardening",
                "",
                f"- The first anchor collapsed at the `{getattr(result, 'first_anchor_failure_mode', 'unknown')}` layer before memory surfaces were reached.",
                "- Prioritize early grounded synthesis and answer assembly before tuning retention or compaction behavior.",
                "",
            ]
        )
    if getattr(result, "run_diagnosis", "") == "weak_harness_pressure":
        lines.extend(
            [
                "## Harness Follow-Up",
                "",
                f"- Weak-run reasons: `{', '.join(getattr(result, 'weak_run_reasons', [])) or 'none'}`",
                "- Increase evidence-bearing turns or payload pressure before treating low coverage as Tok resilience.",
                "",
            ]
        )
    if getattr(result, "run_diagnosis", "") == "memory_surface_reached_but_not_broken":
        lines.extend(
            [
                "## Memory Probe Follow-Up",
                "",
                "- Reuse and retention probes both ran, but neither missing class broke cleanly in this run.",
                "- Tighten the specific probe prompts and payload mix before concluding Tok resisted these surfaces.",
                "",
            ]
        )
    if getattr(result, "run_diagnosis", "") == "retention_surface_held_early_only":
        lines.extend(
            [
                "## Retention Follow-Up",
                "",
                "- Only the early retention probe ran and held.",
                "- Add or preserve a later harder retention probe before reading this as strong long-horizon resistance.",
                "",
            ]
        )
    if getattr(result, "run_diagnosis", "") == "retention_surface_held":
        lines.extend(
            [
                "## Retention Follow-Up",
                "",
                "- A later retention probe ran and held without substitution.",
                "- Treat missing retention loss as provisional long-horizon retention resistance unless an even harder later probe breaks.",
                "",
            ]
        )
    return "\n".join(lines)


def summarize_implicated_files(
    breakpoints: list[StressBreakpoint],
    session: RuntimeSession | None = None,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in breakpoints:
        for path in extract_breakpoint_paths(item, session=session):
            counts[path] = counts.get(path, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [{"path": path, "count": count} for path, count in ranked]


def extract_breakpoint_paths(item: StressBreakpoint | Any, session: RuntimeSession | None = None) -> list[str]:
    paths: set[str] = set()
    for source in (
        getattr(item, "prompt", ""),
        getattr(item, "visible_response", ""),
        json.dumps(getattr(item, "transcript_slice", []), sort_keys=True),
    ):
        for match in _PATH_PATTERN.findall(source):
            normalized = _normalize_extracted_path(match)
            if _is_excluded_grounded_path(normalized):
                continue
            paths.add(normalized)

    observed = _extract_labeled_fields(getattr(item, "visible_response", ""), session=session)
    file_value = observed.get("file", "")
    for match in _PATH_PATTERN.findall(file_value):
        normalized = _normalize_extracted_path(match)
        if _is_excluded_grounded_path(normalized):
            continue
        paths.add(normalized)
    return sorted(paths)


def write_stress_artifacts(
    output_dir: Path,
    result: StressRunResult,
    session: RuntimeSession | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stress_run_path = output_dir / "stress_run.json"
    breakpoints_path = output_dir / "breakpoints.json"
    report_path = output_dir / "stress_report.md"
    plan_path = output_dir / "language_refactor_plan.md"
    stress_run_path.write_text(json.dumps(result.to_dict(), indent=2))
    breakpoints_path.write_text(json.dumps([item.to_dict() for item in result.breakpoints], indent=2))
    report_path.write_text(render_stress_report(result, session=session))
    plan_path.write_text(render_language_refactor_plan(result))
    return {
        "stress_run": stress_run_path,
        "breakpoints": breakpoints_path,
        "stress_report": report_path,
        "language_refactor_plan": plan_path,
    }


def default_output_dir(base: Path | None = None) -> Path:
    root = base or (Path.cwd() / "tmp" / "stress_language")
    return root / datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_excluded_grounded_path(path: str) -> bool:
    normalized = path.strip()
    return any(fragment in normalized for fragment in EXCLUDED_GROUNDED_PATH_FRAGMENTS)
