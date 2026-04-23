"""
Task catalog for stress harness.

This module contains only the TASK_CATALOG tuple with no runtime logic.
"""

from .models import StressTask

TASK_CATALOG: tuple[StressTask, ...] = (
    StressTask(
        id="anchor_seed_1",
        phase_name="anchor-seed",
        prompt=(
            "Find the exact bridge health endpoint that returns session health, including baseline-only state. "
            "Use at least one precise read-only tool before answering. When you answer, emit exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or method>"
        ),
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
        required_tool_names=("view_file", "read"),
    ),
    StressTask(
        id="anchor_seed_2",
        phase_name="anchor-seed",
        prompt=(
            "Find the exact helper in universal runtime that processes structured answer memory like file and verification anchors. "
            "Use at least one precise read-only tool before answering. When you answer, emit exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or helper>"
        ),
        expected_file="src/tok/universal_runtime.py",
        expected_verification="_process_answer_memory",
        require_fresh_evidence=True,
        require_tool_count=1,
        required_tool_names=("view_file", "read"),
    ),
    StressTask(
        id="macro_reuse_test",
        phase_name="macro-awareness",
        prompt=(
            "You have a macro @read_meta(path) available in your prompt. "
            "Use ONLY this macro to find the version of the tok-seed-lab repository in seed_lab/__init__.py. "
            "Do not call raw tools or manually implement the logic. "
            "Answer in exactly two lines:\n"
            "File=seed_lab/__init__.py\nVerification=version"
        ),
        expected_file="seed_lab/__init__.py",
        expected_verification="0.1.0",
        required_tool_names=("@read_meta",),
        requires_memory_surfaces=False,
    ),
    StressTask(
        id="reuse_probe_exact",
        phase_name="reuse-probe",
        prompt=(
            "Recover the exact oldest validated anchor from memory only. "
            "Do not use any new tools unless the session truly lost it. "
            "Answer in exactly two lines:\n"
            "File={file}\nVerification={verification}"
        ),
        dynamic_anchor="oldest",
        forbid_reacquisition=True,
        min_validated_anchors=2,
        min_checkpoint_checks=1,
    ),
    StressTask(
        id="retention_probe_early",
        phase_name="retention-probe",
        prompt=(
            "Recover the oldest validated anchor after a newer one was added. "
            "Do not substitute the most recent anchor. Use no new tools unless the session truly lost it. "
            "When you answer, emit exactly two lines:\n"
            "File={file}\nVerification={verification}"
        ),
        dynamic_anchor="oldest",
        forbid_reacquisition=True,
        min_validated_anchors=2,
        min_checkpoint_checks=1,
    ),
    StressTask(
        id="tool_contract_mixed_answer_and_tool",
        phase_name="tool-contract",
        prompt=(
            "Reconfirm the bridge health endpoint. You must use a supported read-only repo tool to verify this first. "
            "Do not mix any tool call and the final File=/Verification= answer in the same assistant turn. "
            "When you answer, stop and emit exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or method>"
        ),
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        require_fresh_evidence=True,
        require_tool_count=1,
        min_validated_anchors=2,
        min_checkpoint_checks=1,
    ),
    StressTask(
        id="tool_contract_bad_args_shape",
        phase_name="tool-contract",
        prompt=(
            "Find the response-contract helper in gateway. You must use a supported read-only repo tool and valid arguments. "
            "Do not add prose after the final answer. When you answer, emit exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or helper>"
        ),
        expected_file="src/tok/gateway.py",
        expected_verification="_response_contract_for_mode",
        require_fresh_evidence=True,
        require_tool_count=1,
        min_validated_anchors=2,
        min_checkpoint_checks=1,
    ),
    StressTask(
        id="tool_contract_toolless_fresh_answer",
        phase_name="tool-contract",
        prompt=(
            "Find where release_summary is generated. Fresh evidence is required for this answer. "
            "Do not answer from memory alone. When you answer, emit exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or symbol>"
        ),
        expected_file="src/tok/cli.py",
        expected_verification="_gate_release_summary",
        require_fresh_evidence=True,
        require_tool_count=1,
        min_validated_anchors=2,
        min_checkpoint_checks=1,
    ),
    StressTask(
        id="fresh_anchor_runtime",
        phase_name="fresh-grounding",
        prompt=(
            "Find the exact runtime method that records repeated fallback events and eventually degrades the session. "
            "Prefer a precise read or narrow search over a broad grep. Use at least one read-only tool before answering. "
            "When you answer, emit exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or method>"
        ),
        expected_file="src/tok/universal_runtime.py",
        expected_verification="record_fallback_event",
        require_fresh_evidence=True,
        require_tool_count=1,
        min_reuse_checks=1,
        min_checkpoint_checks=1,
    ),
    StressTask(
        id="reuse_probe_near_neighbor",
        phase_name="reuse-probe",
        prompt=(
            "Differentiate the bridge health endpoint from the response-contract helper in the same file family. "
            "Return only the response-contract helper. Do not reuse the old health anchor. "
            "Use no tools unless the session truly lost the distinction. When you answer, emit exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or helper>"
        ),
        expected_file="src/tok/gateway.py",
        expected_verification="_response_contract_for_mode",
        forbid_reacquisition=True,
        min_validated_anchors=2,
        min_checkpoint_checks=1,
    ),
    StressTask(
        id="retention_probe_late",
        phase_name="retention-probe",
        prompt=(
            "Return the oldest validated anchor, not the most recent runtime or fallback anchor. "
            "Do not substitute the newest validated anchor. Use no new tools unless the session truly lost it. "
            "When you answer, emit exactly two lines:\n"
            "File={file}\nVerification={verification}"
        ),
        dynamic_anchor="oldest",
        forbid_reacquisition=True,
        min_validated_anchors=3,
        min_reuse_checks=1,
        min_checkpoint_checks=1,
    ),
    StressTask(
        id="tool_contract_unsupported_tool",
        phase_name="tool-contract",
        prompt=(
            "Find the fallback recorder again using only supported read-only repo tools. "
            "Do not attempt shell, edit, write, delta, or unsupported tool names. "
            "When you answer, emit exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or method>"
        ),
        expected_file="src/tok/universal_runtime.py",
        expected_verification="record_fallback_event",
        require_fresh_evidence=True,
        require_tool_count=1,
        min_validated_anchors=2,
        min_checkpoint_checks=1,
    ),
    StressTask(
        id="fallback_anchor",
        phase_name="fresh-grounding",
        prompt=(
            "Find the exact runtime method that records repeated fallback events and eventually degrades the session. "
            "Prefer a precise read or narrow search over a broad grep. Use at least one read-only tool before answering. "
            "When you answer, emit exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or method>"
        ),
        expected_file="src/tok/universal_runtime.py",
        expected_verification="record_fallback_event",
        require_fresh_evidence=True,
        require_tool_count=1,
        requires_memory_surfaces=True,
    ),
    StressTask(
        id="reuse_check_1",
        phase_name="reuse-vs-reacquire",
        prompt=(
            "Recover the already-grounded first anchor. Do not use new tools unless the session truly lost it. "
            "Reacquiring the fact is a failure. Answer in exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or method>"
        ),
        expected_file="src/tok/gateway.py",
        expected_verification="health",
        forbid_reacquisition=True,
        min_validated_anchors=2,
        min_checkpoint_checks=1,
        requires_memory_surfaces=True,
    ),
    StressTask(
        id="prepare_request_anchor",
        phase_name="near-neighbor disambiguation",
        prompt=(
            "Differentiate request preparation from response processing. Use at least one read-only tool before answering "
            "and end with only the request-preparation path in exactly two lines:\n"
            "File=<the primary file>\nVerification=<the core function>"
        ),
        expected_file="src/tok/universal_runtime.py",
        expected_verification="prepare_request",
        require_fresh_evidence=True,
        require_tool_count=1,
        requires_memory_surfaces=True,
    ),
    StressTask(
        id="process_response_anchor",
        phase_name="near-neighbor disambiguation",
        prompt=(
            "Differentiate request preparation from response processing. Use at least one read-only tool before answering "
            "and end with only the response-processing path in exactly two lines:\n"
            "File=<the primary file>\nVerification=<the core function>"
        ),
        expected_file="src/tok/universal_runtime.py",
        expected_verification="process_response",
        require_fresh_evidence=True,
        require_tool_count=1,
        requires_memory_surfaces=True,
    ),
    StressTask(
        id="collect_behavior_anchor",
        phase_name="payload-pressure",
        prompt=(
            "Trace where repeated file/search reacquisition is detected and converted into behavior signals. "
            "Use one dense search and two targeted file reads in different files before answering. "
            "Reference a previously validated anchor if relevant. When you answer, emit exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or signal path>"
        ),
        expected_file="src/tok/universal_runtime.py",
        expected_verification="collect_behavior_signals",
        require_fresh_evidence=True,
        require_tool_count=3,
        min_validated_anchors=3,
        force_payload=True,
        requires_memory_surfaces=True,
    ),
    StressTask(
        id="invisible_pressure_anchor",
        phase_name="payload-pressure",
        prompt=(
            "Find where invisible pressure is computed. Use one dense search and two targeted file reads in different files before answering. "
            "When you answer, emit exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function>"
        ),
        expected_file="src/tok/universal_runtime.py",
        expected_verification="calculate_invisible_pressure",
        require_fresh_evidence=True,
        require_tool_count=3,
        min_validated_anchors=3,
        force_payload=True,
        requires_memory_surfaces=True,
    ),
    StressTask(
        id="retention_probe_oldest",
        phase_name="retention-probe",
        prompt=(
            "Recover the oldest validated anchor after newer anchors were added. "
            "Do not substitute the latest anchor. Use no new tools unless the session truly lost it. "
            "When you answer, emit exactly two lines:\n"
            "File={file}\nVerification={verification}"
        ),
        dynamic_anchor="oldest",
        forbid_reacquisition=True,
        min_validated_anchors=4,
        min_checkpoint_checks=1,
    ),
    StressTask(
        id="retention_probe_disambiguated",
        phase_name="retention-probe",
        prompt=(
            "Recover the oldest validated anchor, not the newest nearby runtime helper. "
            "The latest anchor is a trap here. Use no new tools unless the session truly lost it. "
            "When you answer, emit exactly two lines:\n"
            "File={file}\nVerification={verification}"
        ),
        dynamic_anchor="oldest",
        forbid_reacquisition=True,
        min_validated_anchors=5,
        min_checkpoint_checks=1,
    ),
    StressTask(
        id="late_recovery_oldest",
        phase_name="late-recovery",
        prompt=(
            "Recover the oldest validated anchor without using any new tools unless the session truly lost it. "
            "Switching to a newer anchor is wrong. When you answer, emit exactly two lines:\n"
            "File={file}\nVerification={verification}"
        ),
        dynamic_anchor="oldest",
        forbid_reacquisition=True,
        min_validated_anchors=3,
        requires_memory_surfaces=True,
    ),
    StressTask(
        id="release_summary_anchor",
        phase_name="fresh-grounding",
        prompt=(
            "Find where `release_summary` is generated. Use at least one read-only tool before answering. "
            "When you are confident, answer in exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or symbol>"
        ),
        expected_file="src/tok/cli.py",
        expected_verification="_gate_release_summary",
        require_fresh_evidence=True,
        require_tool_count=1,
        requires_memory_surfaces=True,
    ),
    StressTask(
        id="reuse_answer_memory_anchor",
        phase_name="reuse-vs-reacquire",
        prompt=(
            "Recover the already-grounded answer-memory anchor without reacquiring it. "
            "Use no new tools unless the session clearly lost the fact. Answer in exactly two lines:\n"
            "File=<the primary file>\nVerification=<the function or helper>"
        ),
        expected_file="src/tok/universal_runtime.py",
        expected_verification="_process_answer_memory",
        forbid_reacquisition=True,
        min_validated_anchors=4,
        requires_memory_surfaces=True,
    ),
    StressTask(
        id="multi_file_disambiguation",
        phase_name="near-neighbor disambiguation",
        prompt=(
            "Differentiate the fallback recorder from the bridge forwarder. Use at least one read-only tool before answering "
            "and return only the fallback recorder in exactly two lines:\n"
            "File=<the primary file>\nVerification=<the fallback function or method>"
        ),
        expected_file="src/tok/universal_runtime.py",
        expected_verification="record_fallback_event",
        require_fresh_evidence=True,
        require_tool_count=1,
        requires_memory_surfaces=True,
    ),
)
