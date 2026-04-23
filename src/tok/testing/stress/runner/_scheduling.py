from __future__ import annotations

from ..models import StressTask, ValidatedAnchor


def expected_fields_for_task(task: StressTask, anchor_history: list[ValidatedAnchor]) -> dict[str, str]:
    if task.dynamic_anchor == "oldest":
        return anchor_history[0].to_fields() if anchor_history else {}
    if task.dynamic_anchor == "latest":
        return anchor_history[-1].to_fields() if anchor_history else {}
    return {
        "file": task.expected_file,
        "verification": task.expected_verification,
    }


def task_prompt(task: StressTask, expected_fields: dict[str, str]) -> str:
    if task.dynamic_anchor:
        return task.prompt.format(
            file=expected_fields.get("file", "<missing>"),
            verification=expected_fields.get("verification", "<missing>"),
        )
    return task.prompt


def task_ready(
    task: StressTask,
    anchor_history: list[ValidatedAnchor],
    total_evidence_chars: int,
    expected_fields: dict[str, str],
    *,
    reuse_checks_run: int,
    reuse_probe_attempts: int,
    checkpoint_checks_run: int,
    min_payload_pressure_bytes: int,
) -> bool:
    if task.min_validated_anchors and len(anchor_history) < task.min_validated_anchors:
        return False
    if task.dynamic_anchor and not expected_fields:
        return False
    if task.min_reuse_checks and (reuse_checks_run + reuse_probe_attempts) < task.min_reuse_checks:
        return False
    if task.min_checkpoint_checks and checkpoint_checks_run < task.min_checkpoint_checks:
        return False
    if task.requires_memory_surfaces and ((reuse_checks_run + reuse_probe_attempts) < 1 or checkpoint_checks_run < 1):
        return False
    if task.force_payload and total_evidence_chars < min_payload_pressure_bytes // 2:
        return len(anchor_history) >= task.min_validated_anchors
    return True
