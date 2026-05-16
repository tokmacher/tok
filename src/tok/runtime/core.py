"""Transport-agnostic universal runtime helpers for Tok."""

from __future__ import annotations

__all__ = [
    "AnswerPhaseState",
    "EvidenceSafetyState",
    "FileDeliveryState",
    "RequestPolicyState",
    "StreamingRecoveryState",
    "TOOL_COMPAT_MEMORY_PROFILE",
    "_FALLBACK_THRESHOLD",
    "NormalizedToolEvent",
    "PreparedRuntimeRequest",
    "ProcessedRuntimeResponse",
    "RuntimeRequest",
    "RuntimeSession",
    "UniversalTokRuntime",
    "_should_skip_history_rewrite",
    "apply_schema_adaptations",
    "calculate_invisible_pressure",
    "calculate_semantic_regression_score",
    "collect_transient_error_snippets",
    "compact_structured_answer_memory",
    "count_tokens",
    "evaluate_replay_gate",
    "extract_structured_answer_memory",
    "ground_structured_answer_memory",
    "reinforce_structured_answer_memory",
    "_select_resend_reason",
]

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("tok.runtime")

from ._answer_phase_state import AnswerPhaseState
from ._cache_state import CacheState
from ._fallback_state import FallbackState
from ._fidelity_state import FidelityState
from ._file_delivery_state import FileDeliveryState
from ._hot_summary_state import HotSummaryState
from ._loop_detection_state import LoopDetectionState
from ._macro_state import MacroState
from ._project_state import ProjectState
from ._request_policy_state import RequestPolicyState
from ._runtime_orchestration import (
    build_tool_compatible_resend,
    process_response_impl,
)
from ._runtime_orchestration import pressure_score as runtime_pressure_score
from ._session_observation import (
    apply_predictive_cache_warming as apply_predictive_cache_warming_impl,
)
from ._session_observation import (
    evidence_intent_advisories as evidence_intent_advisories_impl,
)
from ._session_observation import (
    hot_recent_runtime_hints as hot_recent_runtime_hints_impl,
)
from ._session_observation import (
    mine_response_paths as mine_response_paths_impl,
)
from ._session_observation import (
    prepared_prompt_tokens as prepared_prompt_tokens_impl,
)
from ._session_observation import (
    record_file_snapshot as record_file_snapshot_impl,
)
from ._session_observation import (
    record_history_snapshot as record_history_snapshot_impl,
)
from ._session_observation import (
    record_metadata_snapshot as record_metadata_snapshot_impl,
)
from ._session_observation import (
    record_search_snapshot as record_search_snapshot_impl,
)
from ._session_observation import (
    record_symbol_locations as record_symbol_locations_impl,
)
from ._session_observation import (
    record_traceback_errors as record_traceback_errors_impl,
)
from ._session_persistence import (
    bridge_memory_file,
    episode_ledger_file,
    fallback_memory_file,
    hot_summaries_file,
    initialize_session_storage,
    load_bridge_memory,
    load_episode_ledger,
    load_fallback_memory,
    load_hot_summaries,
    load_result_cache,
    result_cache_file,
    save_bridge_memory,
    save_episode_ledger,
    save_fallback_memory,
    save_hot_summaries,
    save_result_cache,
)
from ._session_persistence import record_episode as record_episode_impl
from ._smoothness_state import SmoothnessState
from ._stream_recovery_state import StreamingRecoveryState
from ._telemetry_state import TelemetryState
from ._user_prompt_state import UserPromptState
from .config import (
    _FALLBACK_THRESHOLD,
    TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS,
    TOK_REQUEST_POLICY_STICKY_TURNS,
    TOOL_COMPAT_MEMORY_PROFILE,
)
from .evidence_safety import (
    EvidenceForm,
    EvidenceLedgerEntry,
    EvidenceSafetyState,
)
from .memory.answer_memory import (
    _should_persist_to_durable,  # noqa: F401
    compact_structured_answer_memory,
    extract_structured_answer_memory,
    ground_structured_answer_memory,
    reinforce_structured_answer_memory,
)
from .memory.bridge_memory import BridgeMemoryState
from .memory.session_state import (
    calculate_reasoning_depth,
    get_adaptive_keep_turns,
    session_write_memory,
    update_session_family_mode,
)
from .memory.tok_state import (
    _build_tok_state,
    _delta_tok_state_fields,
    _prepare_tool_compatible_state,
    _select_resend_reason,
    _select_resend_strategy,
)
from .pipeline.request_preparation import (
    apply_schema_adaptations,
    collect_transient_error_snippets,
)
from .pipeline.response_handling import evaluate_replay_gate
from .pipeline.tool_processing import (
    _should_skip_history_rewrite,
    count_tokens,
)
from .policy.macro_handling import execute_jit_macro
from .policy.semantic_validation import (
    SemanticValidator,
    calculate_invisible_pressure,
    calculate_semantic_regression_score,
)
from .policy.smart_policy import (
    FamilyAdaptiveState,
    SmartZonePolicy,
    initial_state,
    policy_for_model,
)
from .smoothness.models import TokMode

# SEARCH_LIKE_TOOLS is moved to .tool_processing
from .tools import RuntimeToolExecutor
from .types import (
    EpisodeEntry,
    EpisodeLedger,
    NormalizedToolEvent,
    PreparedRuntimeRequest,
    ProcessedRuntimeResponse,
    RuntimeRequest,
    SignalPacket,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class RuntimeSession:
    """
    Session state for a Tok runtime.

    Thread Safety:
        This class provides a `lock` field for optional external synchronization.
        If concurrent access is required, callers must acquire this lock before
        accessing or modifying mutable fields (result_cache, semantic_hash_cache,
        bridge_memory, etc.). Example:

            with session.lock:
                session.result_cache[key] = value

        The lock is NOT acquired internally by any methods.
    """

    keep_turns: int = 3
    _keep_turns_explicit: bool = field(default=False, init=False, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    model: str = ""
    bridge_memory: BridgeMemoryState = field(default_factory=BridgeMemoryState)
    pending_behavior_signals: dict[str, int] = field(default_factory=dict)
    family_states: dict[str, FamilyAdaptiveState] = field(default_factory=dict)
    fallback_memory: str = ""
    memory_dir: Path | None = None
    episode_ledger: EpisodeLedger = field(default_factory=EpisodeLedger)

    # --- Grouped state sub-objects (0.1.9 architecture improvement) ---
    evidence_safety: EvidenceSafetyState = field(default_factory=EvidenceSafetyState, init=False, repr=False)
    streaming_recovery: StreamingRecoveryState = field(default_factory=StreamingRecoveryState, init=False, repr=False)
    request_policy: RequestPolicyState = field(default_factory=RequestPolicyState, init=False, repr=False)
    answer_phase: AnswerPhaseState = field(default_factory=AnswerPhaseState, init=False, repr=False)
    file_delivery: FileDeliveryState = field(default_factory=FileDeliveryState, init=False, repr=False)
    # --- New state groups (Plan 1 deepening) ---
    fallback: FallbackState = field(default_factory=FallbackState, init=False, repr=False)
    smoothness_state: SmoothnessState = field(default_factory=SmoothnessState, init=False, repr=False)
    loop_detection: LoopDetectionState = field(default_factory=LoopDetectionState, init=False, repr=False)
    cache: CacheState = field(default_factory=CacheState, init=False, repr=False)
    hot_summary: HotSummaryState = field(default_factory=HotSummaryState, init=False, repr=False)
    telemetry: TelemetryState = field(default_factory=TelemetryState, init=False, repr=False)
    macro: MacroState = field(default_factory=MacroState, init=False, repr=False)
    fidelity: FidelityState = field(default_factory=FidelityState, init=False, repr=False)
    user_prompt: UserPromptState = field(default_factory=UserPromptState, init=False, repr=False)
    project: ProjectState = field(default_factory=ProjectState, init=False, repr=False)

    def record_fallback_event(self) -> None:
        self.fallback.record_fallback_event()

    def reset_fallback_count(self) -> None:
        self.fallback.reset_fallback_count()

    def reset_session(self) -> None:
        """Reset all transient session state for a fresh start (preserves persisted data).

        LIMITATION: The Tok bridge process does not receive a stable conversation
        identifier from Claude Code.  Without an explicit reset (via
        ``TOK_RESET_SESSION=1`` or ``POST /reset-session``), first-exact-evidence
        guarantees apply only within a single continuous bridge process session, not
        across Claude Code conversation restarts that connect to the same running
        bridge.

        Users can call ``POST http://localhost:9090/reset-session`` to restore
        first-read protection at the start of a new conversation.
        """
        self.fallback.reset()
        self.smoothness_state.reset()
        self.loop_detection.reset()
        self.cache.reset()
        self.hot_summary.reset(save_before_clear=True, session=self)
        self.telemetry.reset()
        self.macro.reset()
        self.fidelity.reset()
        self.user_prompt.reset()
        self.project.reset()
        self.pending_behavior_signals.clear()
        self.family_states.clear()
        self.bridge_memory.hot.clear()
        self.bridge_memory.rolling_cmds = []
        self.evidence_safety.reset()
        self.streaming_recovery.reset()
        self.request_policy.reset()
        self.answer_phase.reset()
        self.file_delivery.reset()
        logger.info("RuntimeSession reset: all transient state cleared")

    def record_invalid_tool_history_recovery(self, *, blocked: bool) -> dict[str, int]:
        self.fallback.invalid_tool_history_recovery_count += 1
        self.note_request_policy_tool_mode_recovery()
        signals: dict[str, int] = {
            "tok_bridge_invalid_tool_history_recovery": 1,
            "tok_bridge_invalid_tool_history_blocked": 1 if blocked else 0,
        }

        if self.fallback.invalid_tool_history_recovery_count >= 2:
            self.telemetry.last_tool_compatible_state = ""
            self.telemetry.last_tool_compatible_state_fields = {}
            self.cache.observed_tool_result_ids.clear()
            self.evidence_safety.first_exact_seen.clear()
            self.evidence_safety.ledger.clear()
            self.evidence_safety.pending_exact_keys.clear()
            self.cache.result_cache.clear()
            self.cache.semantic_hash_cache.clear()
            self.project.files_read.clear()
            self.project.files_fully_delivered.clear()
            self.telemetry.suppressed_failure_markers = frozenset()
            self.streaming_recovery.reset()
            self.request_policy.reset()
            for key in ("turns", "next", "cmds", "errs", "blockers"):
                self.bridge_memory.hot.pop(key, None)
            self.bridge_memory.rolling_cmds = []
            self._save_bridge_memory()
            signals["tok_bridge_invalid_tool_history_session_reset"] = 1
            logger.warning(
                "tok_bridge_invalid_tool_history_session_reset: cleared hot bridge state after %d repeated tool-history recoveries",
                self.fallback.invalid_tool_history_recovery_count,
            )
        return signals

    def reset_invalid_tool_history_recovery(self) -> None:
        self.fallback.invalid_tool_history_recovery_count = 0

    def observe_tool_action(self, tool_name: str, tool_input_key: str) -> bool:
        return self.loop_detection.observe_tool_action(tool_name, tool_input_key)

    def consume_loop_detected(self) -> bool:
        return self.loop_detection.consume_loop_detected()

    def mark_file_edited(self, norm_path: str) -> None:
        self.project.mark_file_edited(norm_path, self.telemetry.step_count)

    def is_recently_edited(self, norm_path: str) -> bool:
        return self.project.is_recently_edited(norm_path, self.telemetry.step_count)

    def note_request_policy_stream_recovery(self, turns: int = TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS) -> None:
        self.request_policy.stream_recovery_watch_turns = max(self.request_policy.stream_recovery_watch_turns, turns)
        self.request_policy.tool_mode_sticky_turns = max(self.request_policy.tool_mode_sticky_turns, turns)

    def note_request_policy_tool_mode_recovery(self, turns: int = TOK_REQUEST_POLICY_STICKY_TURNS) -> None:
        self.request_policy.tool_recovery_watch_turns = max(self.request_policy.tool_recovery_watch_turns, turns)
        self.request_policy.tool_mode_sticky_turns = max(self.request_policy.tool_mode_sticky_turns, turns)

    def _is_edit_tool_event(self, event: NormalizedToolEvent) -> bool:
        """Check if this tool event is an edit-like tool."""
        from tok.compression import EDIT_LIKE_TOOLS

        return event.name.lower() in EDIT_LIKE_TOOLS

    def _is_unsafe_skeleton_edit(self, event: NormalizedToolEvent) -> bool:
        if not hasattr(self, "project"):
            return False

        file_path = self._extract_file_path_from_event(event)
        if not file_path:
            return False

        from .repeat_targets import normalize_path_target

        norm_path = normalize_path_target(file_path)
        return norm_path in self.project.skeleton_delivered_paths

    def _extract_file_path_from_event(self, event: NormalizedToolEvent) -> str | None:
        """Extract file path from a normalized tool event."""
        # First try the direct path field
        if event.path:
            return str(event.path)

        # Then try common path arguments
        args = event.args if isinstance(event.args, dict) else {}
        for key in ("file_path", "path", "AbsolutePath", "TargetFile", "file"):
            path = args.get(key)
            if path:
                return str(path)

        return None

    def _is_verbatim_file_read(self, event: NormalizedToolEvent) -> bool:
        """Check if this is a verbatim file read (no offset/limit parameters)."""
        if event.name.lower() not in ("read", "read_file", "fileread"):
            return False

        args = event.args if isinstance(event.args, dict) else {}
        # Check for precision read parameters (offset, limit, start, end)
        precision_params = ("offset", "limit", "start", "end")
        return not any(key in args for key in precision_params)

    def _clear_skeleton_tracking(self, event: NormalizedToolEvent) -> None:
        if not hasattr(self, "project"):
            return

        file_path = self._extract_file_path_from_event(event)
        if not file_path:
            return

        from .repeat_targets import normalize_path_target

        norm_path = normalize_path_target(file_path)
        if norm_path in self.project.skeleton_delivered_paths:
            self.project.skeleton_delivered_paths.remove(norm_path)

    def record_exact_evidence(self, key: str, digest: str = "") -> dict[str, int]:
        turn = max(1, self.bridge_memory.turn)
        signals = self.evidence_safety.record_exact(key, digest=digest, turn=turn)
        self._bump_signals(signals)
        return signals

    def record_non_exact_evidence(
        self,
        key: str,
        *,
        digest: str = "",
        form: EvidenceForm = "summary",
    ) -> dict[str, int]:
        turn = max(1, self.bridge_memory.turn)
        signals = self.evidence_safety.record_non_exact(key, digest=digest, form=form, turn=turn)
        self._bump_signals(signals)
        return signals

    def require_exact_reacquisition(self, key: str) -> dict[str, int]:
        signals = self.evidence_safety.require_exact_reacquisition(key)
        self._bump_signals(signals)
        return signals

    def evidence_requires_reacquisition(self, key: str) -> bool:
        return self.evidence_safety.requires_reacquisition(key)

    def evidence_safety_audit_summary(self) -> dict[str, int]:
        return self.evidence_safety.audit_summary()

    def adaptive_keep_turns(self) -> int:
        """Dynamically reduce history depth as the session grows."""
        return get_adaptive_keep_turns(self)

    @property
    def model_profile(self):
        """Resolve the ModelProfile for the session's model string."""
        from tok.protocol.model_profiles import get_model_profile

        return get_model_profile(self.model)

    @property
    def effective_model_profile(self):
        """Model profile with compression_aggressiveness scaled down under high labour."""
        import dataclasses

        base = self.model_profile
        labour = self.smoothness_state.current_task_labour_index
        if labour >= 40:
            scale = 0.70
        elif labour >= 20:
            scale = 0.85
        else:
            return base
        adjusted = max(0.1, base.compression_aggressiveness * scale)
        return dataclasses.replace(base, compression_aggressiveness=adjusted)

    def __post_init__(self) -> None:
        """Initialize memory directory and load persisted bridge memory."""
        explicit_memory_dir = self.memory_dir is not None
        initialize_session_storage(self, explicit_memory_dir=explicit_memory_dir)

    def _bridge_memory_file(self) -> Path:
        """Return the path to the bridge memory file."""
        return bridge_memory_file(self)

    def _load_bridge_memory(self) -> BridgeMemoryState:
        """Load bridge memory from disk."""
        return load_bridge_memory(self)

    def _save_bridge_memory(self) -> None:
        """Persist bridge memory to disk."""
        save_bridge_memory(self)
        save_hot_summaries(self)

    def _result_cache_file(self) -> Path:
        """Return the path to the result cache file."""
        return result_cache_file(self)

    def _load_result_cache(self) -> dict[str, Any]:
        """Load result cache from disk."""
        return load_result_cache(self)

    def _save_result_cache(self) -> None:
        """Persist result cache to disk."""
        save_result_cache(self)

    def _fallback_memory_file(self) -> Path:
        """Return the path to the fallback memory file."""
        return fallback_memory_file(self)

    def _load_fallback_memory(self) -> str:
        """Load fallback memory from disk."""
        return load_fallback_memory(self)

    def _save_fallback_memory(self) -> None:
        """Persist fallback memory to disk."""
        save_fallback_memory(self)

    def _episode_ledger_file(self) -> Path:
        """Return the path to the episode ledger file."""
        return episode_ledger_file(self)

    def _load_episode_ledger(self) -> EpisodeLedger:
        """Load episode ledger from disk."""
        return load_episode_ledger(self)

    def _save_episode_ledger(self) -> None:
        """Persist episode ledger to disk."""
        save_episode_ledger(self)

    def _hot_summaries_file(self) -> Path:
        """Return the path to the hot summaries file."""
        return hot_summaries_file(self)

    def _load_hot_summaries(self) -> dict[str, Any]:
        """Load hot summary records from disk."""
        return load_hot_summaries(self)

    def _save_hot_summaries(self) -> None:
        """Persist hot summary records to disk."""
        save_hot_summaries(self)

    def record_episode(self, entry: EpisodeEntry) -> None:
        record_episode_impl(self, entry)

    def reasoning_depth_per_token(self) -> float:
        """
        Dual-axis metric: reasoning diversity per token consumed.

        Combines step count, tool diversity, and tokens used.
        Higher is better — rewards rich reasoning without token bloat.
        Returns 0.0 when no tokens have been recorded.
        """
        return calculate_reasoning_depth(self)

    def policy_snapshot(self, model: str) -> tuple[str, SmartZonePolicy]:
        policy = policy_for_model(model)
        state = self.family_states.setdefault(policy.family.key, initial_state(policy))
        return state.mode, policy

    def load_memory(self, model: str = "") -> str:
        mode, policy = self.policy_snapshot(model)
        projected = self.bridge_memory.wire_state(policy.memory_profiles[mode], markers=self.project.markers)
        if projected:
            logger.debug("Using structured memory: %s", projected[:100])
            self._bump_signals({"cold_start_structured_memory": 1})
            return projected
        if self.fallback_memory:
            logger.debug("Using fallback memory")
            self._bump_signals({"cold_start_wire_fallback": 1})
            return self.fallback_memory
        logger.debug("No memory available")
        return ""

    def refresh_hot_memory(self, tok_state: str, model: str = "") -> str:
        mode, policy = self.policy_snapshot(model)
        self._bump_signals(self.bridge_memory.replace_hot_from_wire_state(tok_state))
        return self.bridge_memory.wire_state(policy.memory_profiles[mode], markers=self.project.markers)

    def write_memory(self, text: str) -> str:
        """Write memory state and return the written content."""
        return session_write_memory(self, text)

    def record_file_snapshot(self, path: str, snippet: str) -> bool:
        """Record a file snapshot in bridge memory."""
        return record_file_snapshot_impl(self, path, snippet)

    def record_search_snapshot(self, query: str, snippet: str) -> bool:
        """Record a search snapshot in bridge memory."""
        return record_search_snapshot_impl(self, query, snippet)

    def record_symbol_locations(self, raw_grep_output: str) -> int:
        """Extract symbol location facts from raw grep output."""
        return record_symbol_locations_impl(self, raw_grep_output)

    def record_traceback_errors(self, text: str) -> int:
        """Extract traceback file:line pairs and write them as errs facts."""
        return record_traceback_errors_impl(self, text)

    def mine_response_paths(self, text: str) -> int:
        """Mine file paths and line numbers from assistant response text."""
        return mine_response_paths_impl(self, text)

    def record_history_snapshot(self, path: str, revision: str, snippet: str) -> bool:
        """Record a git history snapshot in bridge memory."""
        return record_history_snapshot_impl(self, path, revision, snippet)

    def record_metadata_snapshot(self, path: str, subtype: str, snippet: str) -> bool:
        """Record a metadata snapshot in bridge memory."""
        return record_metadata_snapshot_impl(self, path, subtype, snippet)

    # ---------------------------------------------------------------------------
    # Explorer helpers - available on session for agent exploration
    # ---------------------------------------------------------------------------
    def explore_file(self, filepath: str, mode: str = "overview") -> str:
        """
        Explore a Python file and return Tok-formatted overview.

        Args:
            filepath: Path to Python file
            mode: "overview" for summary, "skeleton" for full structure

        Returns:
            Tok-formatted string with file overview

        """
        from tok.explorer import explore_file as _explore_file

        result = _explore_file(filepath, mode)
        self._bump_signals({"explore_file_invoked": 1})
        return result

    def explore_module(self, module_path: str, mode: str = "overview") -> str:
        """
        Explore a module/package and return Tok-formatted overview.

        Args:
            module_path: Path to module directory or file
            mode: "overview" for summary, "skeleton" for full structure

        Returns:
            Tok-formatted string with module overview

        """
        from tok.explorer import explore_module as _explore_module

        result = _explore_module(module_path, mode)
        self._bump_signals({"explore_module_invoked": 1})
        return result

    def get_file_overview(self, filepath: str) -> dict[str, Any]:
        """
        Get structured overview of a Python file.

        Returns:
            dict with path, line_count, classes, functions, is_large

        """
        from tok.explorer import get_file_overview as _get_file_overview

        result = _get_file_overview(filepath)
        self._bump_signals({"file_overview_invoked": 1})
        return result

    def list_large_files(self, root: str = "src/tok") -> list[dict[str, Any]]:
        """
        Find all Python files > 500 lines in a directory tree.

        Returns:
            List of dicts with file info sorted by line count

        """
        from tok.explorer import list_large_files as _list_large_files

        result = _list_large_files(root)
        self._bump_signals({"list_large_files_invoked": 1})
        return result

    def check_temp_copy_alias(self, path: str, snippet: str) -> str | None:
        from .repeat_targets import _is_temp_path, normalize_path_target

        if not _is_temp_path(path):
            return None
        normalized = normalize_path_target(path)
        existing_digests = self.bridge_memory.get_file_fact_digests()
        new_digest = self.bridge_memory._extract_file_digest(snippet, normalized)
        if not new_digest:
            new_digest = " ".join(snippet.split())[:160]
        for src_path, src_digest in existing_digests.items():
            if new_digest and src_digest and new_digest == src_digest:
                self.evidence_safety.alias_map[normalized] = src_path
                return src_path
        return None

    def prepared_prompt_tokens(self, payload: dict[str, Any]) -> int:
        """Count and cache tokens for a prepared prompt."""
        return prepared_prompt_tokens_impl(self, payload)

    def _trim_repeat_target_state(self) -> None:
        if len(self.hot_summary.recent_repeat_target_events) > 16:
            self.hot_summary.recent_repeat_target_events = self.hot_summary.recent_repeat_target_events[-16:]
        if len(self.hot_summary.records) > 64:
            ranked = sorted(
                self.hot_summary.records.items(),
                key=lambda item: (
                    item[1].stuck_promotion_turn or item[1].hot_promotion_turn,
                    item[1].last_seen_turn,
                ),
                reverse=True,
            )[:64]
            self.hot_summary.records = dict(ranked)
        if len(self.cache.observed_tool_result_ids) > 64:
            keys_to_keep = list(self.cache.observed_tool_result_ids.keys())[-64:]
            self.cache.observed_tool_result_ids = dict.fromkeys(keys_to_keep)

    def observe_repeat_target_result(
        self,
        *,
        tool_id: str,
        tool_name: str,
        path: str | None,
        query: str | None,
        command: str | None,
        raw_content: str,
        tool_args: dict[str, Any] | None = None,
        exact_evidence_key: str | None = None,
        blocker_rediscovery: bool = False,
    ) -> dict[str, int]:
        """Record a new result-bearing logical target event for repeat-target control."""
        from ._request_preparation import observe_repeat_target_result_impl

        return observe_repeat_target_result_impl(
            self,
            tool_id=tool_id,
            tool_name=tool_name,
            path=path,
            query=query,
            command=command,
            raw_content=raw_content,
            tool_args=tool_args,
            exact_evidence_key=exact_evidence_key,
            blocker_rediscovery=blocker_rediscovery,
        )

    def apply_predictive_cache_warming(self, logical_target: str) -> dict[str, int]:
        """Apply predictive cache warming for a logical target."""
        return apply_predictive_cache_warming_impl(self, logical_target)

    def hot_recent_runtime_hints(self, *, max_hints: int | None = None) -> tuple[list[str], dict[str, int]]:
        """Generate hot recent hints for eligible repeat targets."""
        return hot_recent_runtime_hints_impl(self, max_hints=max_hints)

    def evidence_intent_advisories(self) -> list[str]:
        """Generate advisories based on evidence intent patterns."""
        return evidence_intent_advisories_impl(self)

    def is_predictive_cache_hit(self, family: str, logical_target: str) -> bool:
        """Check if a target is in the predictive cache warm set."""
        return f"{family}|{logical_target}" in self.cache.predictive_cache_warm_keys

    def update_family_mode(self, model: str, signals: dict[str, int]) -> str:
        """Update the compression mode for a model family."""
        return update_session_family_mode(self, model, signals)

    def consume_behavior_signals(self) -> dict[str, int]:
        """Consume and clear pending behavior signals."""
        signals = dict(self.pending_behavior_signals)
        self.pending_behavior_signals.clear()
        return signals

    def maybe_suppress_tool_compatible_state(
        self,
        state: str,
        *,
        force_resend_on_answer_ready: bool = False,
    ) -> tuple[str, dict[str, int]]:
        """Avoid resending unchanged tool-compatible state on every turn."""
        cleaned = state.strip()
        if not cleaned:
            self.telemetry.last_tool_compatible_state = ""
            self.telemetry.last_tool_compatible_state_fields = {}
            return state, {}
        parsed, comparable, has_answer_facts = _prepare_tool_compatible_state(
            cleaned, self.telemetry.last_tool_compatible_state_fields
        )
        previous_comparable = dict(self.telemetry.last_tool_compatible_state_fields)
        if (
            self.telemetry.last_tool_compatible_state
            and not comparable
            and not previous_comparable
            and parsed
            and set(parsed.keys()) <= {"turns"}
        ):
            if force_resend_on_answer_ready:
                rendered = _build_tok_state(parsed)
                self.telemetry.last_tool_compatible_state = rendered
                self.telemetry.last_tool_compatible_state_fields = comparable
                return rendered, {
                    "state_resend_full_turn": 1,
                    "state_resend_reason_answer_ready_forced_full": 1,
                }
            return "", {"state_resend_suppressed_turn": 1}
        strategy = _select_resend_strategy(comparable, previous_comparable, has_answer_facts)
        if strategy == "suppress":
            if force_resend_on_answer_ready:
                rendered = _build_tok_state(parsed)
                self.telemetry.last_tool_compatible_state = rendered
                self.telemetry.last_tool_compatible_state_fields = comparable
                return rendered, {
                    "state_resend_full_turn": 1,
                    "state_resend_reason_answer_ready_forced_full": 1,
                }
            return "", {"state_resend_suppressed_turn": 1}
        rendered = _build_tok_state(parsed)
        self.telemetry.last_tool_compatible_state = rendered
        self.telemetry.last_tool_compatible_state_fields = comparable
        if strategy == "full":
            return rendered, {"state_resend_full_turn": 1}
        delta = _delta_tok_state_fields(previous_comparable, parsed, self.telemetry.suppressed_failure_markers)
        if delta and len(delta) < len(rendered):
            return delta, {"state_resend_delta_turn": 1}
        return rendered, {
            "state_resend_full_turn": 1,
            "state_resend_reason_delta_not_smaller": 1,
        }

    @property
    def latest_turn_smoothness_score(self) -> int:
        return self.smoothness_state.latest_turn_score

    @property
    def latest_turn_labour_index(self) -> int:
        return self.smoothness_state.latest_turn_labour_index

    @property
    def current_task_smoothness_score(self) -> int:
        return self.smoothness_state.current_task_score

    @property
    def current_task_labour_index(self) -> int:
        return self.smoothness_state.current_task_labour_index

    @property
    def current_tok_mode(self) -> TokMode:
        return self.smoothness_state.current_tok_mode

    @property
    def smoothness_event_counts(self) -> dict[str, int]:
        return dict(self.smoothness_state.event_counts)

    def update_smoothness_state(
        self,
        turn_score: int,
        labour_index: int,
        tok_mode: TokMode,
        event_counts: dict[str, int],
    ) -> None:
        self.smoothness_state.latest_turn_score = turn_score
        self.smoothness_state.latest_turn_labour_index = labour_index
        self.smoothness_state.current_tok_mode = tok_mode
        for event_type, count in event_counts.items():
            self.smoothness_state.event_counts[event_type] = (
                self.smoothness_state.event_counts.get(event_type, 0) + count
            )
        self.smoothness_state.current_task_score = turn_score
        self.smoothness_state.current_task_labour_index = labour_index

    def _bump_signals(self, signals: dict[str, int]) -> None:
        """Accumulate behavior signals for the next request."""
        for key, value in signals.items():
            self.pending_behavior_signals[key] = self.pending_behavior_signals.get(key, 0) + value
        if signals.get("tok_memory_snap_triggered"):
            self.telemetry.tok_memory_snap_triggered = 1

    # --- Backward-compat properties for migrated bare fields ---
    @property
    def result_cache(self) -> dict[str, Any]:
        return self.cache.result_cache

    @result_cache.setter
    def result_cache(self, value: dict[str, Any]) -> None:
        self.cache.result_cache = value

    @property
    def semantic_hash_cache(self) -> dict[str, str]:
        return self.cache.semantic_hash_cache

    @semantic_hash_cache.setter
    def semantic_hash_cache(self, value: dict[str, str]) -> None:
        self.cache.semantic_hash_cache = value

    @property
    def _consecutive_fallback_count(self) -> int:
        return self.fallback.consecutive_count

    @_consecutive_fallback_count.setter
    def _consecutive_fallback_count(self, value: int) -> None:
        self.fallback.consecutive_count = value

    @property
    def _baseline_only(self) -> bool:
        return self.fallback.baseline_only

    @_baseline_only.setter
    def _baseline_only(self, value: bool) -> None:
        self.fallback.baseline_only = value

    @property
    def _persistence_failures(self) -> int:
        return self.fallback.persistence_failures

    @_persistence_failures.setter
    def _persistence_failures(self, value: int) -> None:
        self.fallback.persistence_failures = value

    @property
    def _step_count(self) -> int:
        return self.telemetry.step_count

    @_step_count.setter
    def _step_count(self, value: int) -> None:
        self.telemetry.step_count = value

    @property
    def _token_count(self) -> int:
        return self.telemetry.token_count

    @_token_count.setter
    def _token_count(self, value: int) -> None:
        self.telemetry.token_count = value

    @property
    def _last_request_message_count(self) -> int:
        return self.telemetry.last_request_message_count

    @_last_request_message_count.setter
    def _last_request_message_count(self, value: int) -> None:
        self.telemetry.last_request_message_count = value

    @property
    def _tool_names_seen(self) -> set[str]:
        return self.telemetry.tool_names_seen

    @property
    def _current_tool_density(self) -> float:
        return self.telemetry.tool_density

    @_current_tool_density.setter
    def _current_tool_density(self, value: float) -> None:
        self.telemetry.tool_density = value

    @property
    def _current_context_char_count(self) -> int:
        return self.telemetry.context_char_count

    @_current_context_char_count.setter
    def _current_context_char_count(self, value: int) -> None:
        self.telemetry.context_char_count = value

    @property
    def _current_invisible_pressure(self) -> int:
        return self.telemetry.invisible_pressure

    @_current_invisible_pressure.setter
    def _current_invisible_pressure(self, value: int) -> None:
        self.telemetry.invisible_pressure = value

    @property
    def _active_tools(self) -> list[str]:
        return self.telemetry.active_tools

    @property
    def _last_tool_compatible_state(self) -> str:
        return self.telemetry.last_tool_compatible_state

    @_last_tool_compatible_state.setter
    def _last_tool_compatible_state(self, value: str) -> None:
        self.telemetry.last_tool_compatible_state = value

    @property
    def _last_tool_compatible_state_fields(self) -> dict[str, list[str]]:
        return self.telemetry.last_tool_compatible_state_fields

    @_last_tool_compatible_state_fields.setter
    def _last_tool_compatible_state_fields(self, value: dict[str, list[str]]) -> None:
        self.telemetry.last_tool_compatible_state_fields = value

    @property
    def _suppressed_failure_markers(self) -> frozenset[str]:
        return self.telemetry.suppressed_failure_markers

    @_suppressed_failure_markers.setter
    def _suppressed_failure_markers(self, value: frozenset[str]) -> None:
        self.telemetry.suppressed_failure_markers = value

    @property
    def _response_word_samples(self) -> list[int]:
        return self.telemetry.response_word_samples

    @property
    def _loop_detection_window(self) -> list[tuple[str, str]]:
        return self.loop_detection.window

    @property
    def _loop_detected(self) -> bool:
        return self.loop_detection.detected

    @_loop_detected.setter
    def _loop_detected(self, value: bool) -> None:
        self.loop_detection.detected = value

    @property
    def _recently_edited_files(self) -> dict[str, int]:
        return self.project.recently_edited_files

    @property
    def _latest_turn_smoothness_score(self) -> int:
        return self.smoothness_state.latest_turn_score

    @_latest_turn_smoothness_score.setter
    def _latest_turn_smoothness_score(self, value: int) -> None:
        self.smoothness_state.latest_turn_score = value

    @property
    def _latest_turn_labour_index(self) -> int:
        return self.smoothness_state.latest_turn_labour_index

    @_latest_turn_labour_index.setter
    def _latest_turn_labour_index(self, value: int) -> None:
        self.smoothness_state.latest_turn_labour_index = value

    @property
    def _current_task_smoothness_score(self) -> int:
        return self.smoothness_state.current_task_score

    @_current_task_smoothness_score.setter
    def _current_task_smoothness_score(self, value: int) -> None:
        self.smoothness_state.current_task_score = value

    @property
    def _current_task_labour_index(self) -> int:
        return self.smoothness_state.current_task_labour_index

    @_current_task_labour_index.setter
    def _current_task_labour_index(self, value: int) -> None:
        self.smoothness_state.current_task_labour_index = value

    @property
    def _current_tok_mode(self) -> TokMode:
        return self.smoothness_state.current_tok_mode

    @_current_tok_mode.setter
    def _current_tok_mode(self, value: TokMode) -> None:
        self.smoothness_state.current_tok_mode = value

    @property
    def _smoothness_event_counts(self) -> dict[str, int]:
        return self.smoothness_state.event_counts

    @property
    def _pending_macro_heal(self) -> str:
        return self.macro.pending_heal

    @_pending_macro_heal.setter
    def _pending_macro_heal(self, value: str) -> None:
        self.macro.pending_heal = value

    @property
    def _pending_macro_heal_turn(self) -> int:
        return self.macro.pending_heal_turn

    @_pending_macro_heal_turn.setter
    def _pending_macro_heal_turn(self, value: int) -> None:
        self.macro.pending_heal_turn = value

    @property
    def _load_global_macros(self) -> bool:
        return self.macro.load_global_macros

    @_load_global_macros.setter
    def _load_global_macros(self, value: bool) -> None:
        self.macro.load_global_macros = value

    @property
    def _recent_repeat_target_events(self) -> list[Any]:
        return self.hot_summary.recent_repeat_target_events

    @property
    def _hot_summary_records(self) -> dict[str, Any]:
        return self.hot_summary.records

    @_hot_summary_records.setter
    def _hot_summary_records(self, value: dict[str, Any]) -> None:
        self.hot_summary.records = value

    @property
    def _hot_hints_loaded_from_disk(self) -> int:
        return self.hot_summary.hints_loaded_from_disk

    @_hot_hints_loaded_from_disk.setter
    def _hot_hints_loaded_from_disk(self, value: int) -> None:
        self.hot_summary.hints_loaded_from_disk = value

    @property
    def _observed_tool_result_ids(self) -> dict[str, None]:
        return self.cache.observed_tool_result_ids

    @_observed_tool_result_ids.setter
    def _observed_tool_result_ids(self, value: dict[str, None]) -> None:
        self.cache.observed_tool_result_ids = value

    @property
    def _prepared_prompt_token_cache(self) -> dict[str, int]:
        return self.cache.prepared_prompt_token_cache

    @property
    def _predictive_cache_warm_keys(self) -> set[str]:
        return self.cache.predictive_cache_warm_keys

    @property
    def _project_markers(self) -> frozenset[str]:
        return self.project.markers

    @_project_markers.setter
    def _project_markers(self, value: frozenset[str]) -> None:
        self.project.markers = value

    @property
    def _files_read_this_session(self) -> set[str]:
        return self.project.files_read

    @property
    def _files_fully_delivered(self) -> dict[str, int]:
        return self.project.files_fully_delivered

    @property
    def _skeleton_delivered_paths(self) -> set[str]:
        return self.project.skeleton_delivered_paths

    @property
    def _last_user_prompt_text(self) -> str:
        return self.user_prompt.last_text

    @_last_user_prompt_text.setter
    def _last_user_prompt_text(self, value: str) -> None:
        self.user_prompt.last_text = value

    @property
    def _last_user_prompt_labels(self) -> tuple[str, ...]:
        return self.user_prompt.last_labels

    @_last_user_prompt_labels.setter
    def _last_user_prompt_labels(self, value: tuple[str, ...]) -> None:
        self.user_prompt.last_labels = value

    @property
    def _request_has_tools(self) -> bool:
        return self.user_prompt.request_has_tools

    @_request_has_tools.setter
    def _request_has_tools(self, value: bool) -> None:
        self.user_prompt.request_has_tools = value

    @property
    def _runtime_hint_last_turn(self) -> dict[str, int]:
        return self.user_prompt.hint_last_turn

    @_runtime_hint_last_turn.setter
    def _runtime_hint_last_turn(self, value: dict[str, int]) -> None:
        self.user_prompt.hint_last_turn = value

    @property
    def _fidelity_overrides(self) -> dict[str, int]:
        return self.fidelity.overrides

    @property
    def _file_reads_by_turn(self) -> dict[str, int]:
        return self.fidelity.file_reads_by_turn

    @property
    def _last_elevated_path(self) -> str:
        return self.fidelity.last_elevated_path

    @_last_elevated_path.setter
    def _last_elevated_path(self, value: str) -> None:
        self.fidelity.last_elevated_path = value

    @property
    def _tool_required_latch_streak(self) -> int:
        return self.fidelity.tool_required_latch_streak

    @_tool_required_latch_streak.setter
    def _tool_required_latch_streak(self, value: int) -> None:
        self.fidelity.tool_required_latch_streak = value

    # --- Backward-compat properties for old group bare fields ---
    @property
    def _answer_ready_repair_pending(self) -> bool:
        return self.answer_phase.answer_ready_repair_pending

    @_answer_ready_repair_pending.setter
    def _answer_ready_repair_pending(self, value: bool) -> None:
        self.answer_phase.answer_ready_repair_pending = value

    @property
    def _answer_ready_repair_active(self) -> bool:
        return self.answer_phase.answer_ready_repair_active

    @_answer_ready_repair_active.setter
    def _answer_ready_repair_active(self, value: bool) -> None:
        self.answer_phase.answer_ready_repair_active = value

    @property
    def _late_answer_assembly_repair_pending(self) -> bool:
        return self.answer_phase.late_assembly_repair_pending

    @_late_answer_assembly_repair_pending.setter
    def _late_answer_assembly_repair_pending(self, value: bool) -> None:
        self.answer_phase.late_assembly_repair_pending = value

    @property
    def _late_answer_assembly_repair_active(self) -> bool:
        return self.answer_phase.late_assembly_repair_active

    @_late_answer_assembly_repair_active.setter
    def _late_answer_assembly_repair_active(self, value: bool) -> None:
        self.answer_phase.late_assembly_repair_active = value

    @property
    def _late_answer_assembly_repair_mode_pending(self) -> str:
        return self.answer_phase.late_assembly_repair_mode_pending

    @_late_answer_assembly_repair_mode_pending.setter
    def _late_answer_assembly_repair_mode_pending(self, value: str) -> None:
        self.answer_phase.late_assembly_repair_mode_pending = value

    @property
    def _late_answer_assembly_repair_mode_active(self) -> str:
        return self.answer_phase.late_assembly_repair_mode_active

    @_late_answer_assembly_repair_mode_active.setter
    def _late_answer_assembly_repair_mode_active(self, value: str) -> None:
        self.answer_phase.late_assembly_repair_mode_active = value

    @property
    def _late_answer_followthrough_pending(self) -> bool:
        return self.answer_phase.late_followthrough_pending

    @_late_answer_followthrough_pending.setter
    def _late_answer_followthrough_pending(self, value: bool) -> None:
        self.answer_phase.late_followthrough_pending = value

    @property
    def _late_answer_followthrough_active(self) -> bool:
        return self.answer_phase.late_followthrough_active

    @_late_answer_followthrough_active.setter
    def _late_answer_followthrough_active(self, value: bool) -> None:
        self.answer_phase.late_followthrough_active = value

    @property
    def _answer_phase_expected_this_turn(self) -> bool:
        return self.answer_phase.answer_phase_expected_this_turn

    @_answer_phase_expected_this_turn.setter
    def _answer_phase_expected_this_turn(self, value: bool) -> None:
        self.answer_phase.answer_phase_expected_this_turn = value

    @property
    def _natural_response_acceptable_this_turn(self) -> bool:
        return self.answer_phase.natural_response_acceptable_this_turn

    @_natural_response_acceptable_this_turn.setter
    def _natural_response_acceptable_this_turn(self, value: bool) -> None:
        self.answer_phase.natural_response_acceptable_this_turn = value

    @property
    def _stream_recovery_reacquisition_budget(self) -> int:
        return self.streaming_recovery.reacquisition_budget

    @_stream_recovery_reacquisition_budget.setter
    def _stream_recovery_reacquisition_budget(self, value: int) -> None:
        self.streaming_recovery.reacquisition_budget = value

    @property
    def _stream_recovery_history_floor_budget(self) -> int:
        return self.streaming_recovery.history_floor_budget

    @_stream_recovery_history_floor_budget.setter
    def _stream_recovery_history_floor_budget(self, value: int) -> None:
        self.streaming_recovery.history_floor_budget = value

    @property
    def _stream_recovery_tool_use_only_signature(self) -> str:
        return self.streaming_recovery.tool_use_only_signature

    @_stream_recovery_tool_use_only_signature.setter
    def _stream_recovery_tool_use_only_signature(self, value: str) -> None:
        self.streaming_recovery.tool_use_only_signature = value

    @property
    def _stream_recovery_tool_use_only_repeat_count(self) -> int:
        return self.streaming_recovery.tool_use_only_repeat_count

    @_stream_recovery_tool_use_only_repeat_count.setter
    def _stream_recovery_tool_use_only_repeat_count(self, value: int) -> None:
        self.streaming_recovery.tool_use_only_repeat_count = value

    @property
    def _stream_recovery_cooldown_remaining(self) -> int:
        return self.streaming_recovery.cooldown_remaining

    @_stream_recovery_cooldown_remaining.setter
    def _stream_recovery_cooldown_remaining(self, value: int) -> None:
        self.streaming_recovery.cooldown_remaining = value

    @property
    def _stream_recovery_cooldown_suppressed(self) -> bool:
        return self.streaming_recovery.cooldown_suppressed

    @_stream_recovery_cooldown_suppressed.setter
    def _stream_recovery_cooldown_suppressed(self, value: bool) -> None:
        self.streaming_recovery.cooldown_suppressed = value

    @property
    def _stream_read_error_consecutive_count(self) -> int:
        return self.streaming_recovery.read_error_consecutive_count

    @_stream_read_error_consecutive_count.setter
    def _stream_read_error_consecutive_count(self, value: int) -> None:
        self.streaming_recovery.read_error_consecutive_count = value

    @property
    def _stream_read_error_last_stage(self) -> str:
        return self.streaming_recovery.read_error_last_stage

    @_stream_read_error_last_stage.setter
    def _stream_read_error_last_stage(self, value: str) -> None:
        self.streaming_recovery.read_error_last_stage = value

    @property
    def _request_policy_tool_mode_sticky_turns(self) -> int:
        return self.request_policy.tool_mode_sticky_turns

    @_request_policy_tool_mode_sticky_turns.setter
    def _request_policy_tool_mode_sticky_turns(self, value: int) -> None:
        self.request_policy.tool_mode_sticky_turns = value

    @property
    def _request_policy_stream_recovery_watch_turns(self) -> int:
        return self.request_policy.stream_recovery_watch_turns

    @_request_policy_stream_recovery_watch_turns.setter
    def _request_policy_stream_recovery_watch_turns(self, value: int) -> None:
        self.request_policy.stream_recovery_watch_turns = value

    @property
    def _request_policy_tool_recovery_watch_turns(self) -> int:
        return self.request_policy.tool_recovery_watch_turns

    @_request_policy_tool_recovery_watch_turns.setter
    def _request_policy_tool_recovery_watch_turns(self, value: int) -> None:
        self.request_policy.tool_recovery_watch_turns = value

    @property
    def _request_policy_last_effective_tool_compatible(self) -> bool:
        return self.request_policy.last_effective_tool_compatible

    @_request_policy_last_effective_tool_compatible.setter
    def _request_policy_last_effective_tool_compatible(self, value: bool) -> None:
        self.request_policy.last_effective_tool_compatible = value

    @property
    def _evidence_neighborhoods(self) -> dict[str, set[str]]:
        return self.evidence_safety.neighborhoods

    @property
    def _evidence_anchor_novelty_keys(self) -> dict[str, set[str]]:
        return self.evidence_safety.anchor_novelty_keys

    @property
    def _evidence_alias_map(self) -> dict[str, str]:
        return self.evidence_safety.alias_map

    @_evidence_alias_map.setter
    def _evidence_alias_map(self, value: dict[str, str]) -> None:
        self.evidence_safety.alias_map = value

    @property
    def _first_exact_evidence_seen(self) -> set[str]:
        return self.evidence_safety.first_exact_seen

    @property
    def _evidence_safety_ledger(self) -> dict[str, EvidenceLedgerEntry]:
        return self.evidence_safety.ledger

    @property
    def _pending_exact_evidence_keys(self) -> set[str]:
        return self.evidence_safety.pending_exact_keys

    @property
    def _last_mode(self) -> str:
        return self.telemetry.last_mode

    @_last_mode.setter
    def _last_mode(self, value: str) -> None:
        self.telemetry.last_mode = value

    @property
    def _drift_detected_previous_turn(self) -> bool:
        return self.request_policy.drift_detected_previous_turn

    @_drift_detected_previous_turn.setter
    def _drift_detected_previous_turn(self, value: bool) -> None:
        self.request_policy.drift_detected_previous_turn = value

    @property
    def _invalid_tool_history_recovery_count(self) -> int:
        return self.fallback.invalid_tool_history_recovery_count

    @_invalid_tool_history_recovery_count.setter
    def _invalid_tool_history_recovery_count(self, value: int) -> None:
        self.fallback.invalid_tool_history_recovery_count = value

    @property
    def _tok_memory_snap_triggered(self) -> int:
        return self.telemetry.tok_memory_snap_triggered

    @_tok_memory_snap_triggered.setter
    def _tok_memory_snap_triggered(self, value: int) -> None:
        self.telemetry.tok_memory_snap_triggered = value

    @property
    def _is_first_request(self) -> bool:
        return self.telemetry.is_first_request

    @_is_first_request.setter
    def _is_first_request(self, value: bool) -> None:
        self.telemetry.is_first_request = value


class UniversalTokRuntime:
    """Canonical transport-agnostic request/response runtime."""

    def __init__(self) -> None:
        self.tool_executor = RuntimeToolExecutor()
        self.semantic_validator = SemanticValidator()

    def execute_tool_event(
        self,
        event: NormalizedToolEvent,
        session: RuntimeSession | None = None,
    ) -> dict[str, Any]:
        """Execute a normalized tool event using the shared runtime tools."""
        if session is not None:
            if session._is_edit_tool_event(event) and session._is_unsafe_skeleton_edit(event):
                from tok.exceptions import TokSafetyError

                file_path = session._extract_file_path_from_event(event)
                raise TokSafetyError(
                    f"Cannot edit '{file_path}': Tok showed a summary instead of full content to save tokens.\n"
                    f"To edit this file:\n"
                    f"  1. Re-read with: Read path={file_path} offset=1\n"
                    f"  2. Then make your edit\n"
                    f"(The offset=1 forces Tok to return the complete file.)"
                )

            if session._is_verbatim_file_read(event):
                session._clear_skeleton_tracking(event)

        return self.tool_executor.execute_normalized_tool(event)

    def execute_tool_events_batch(
        self,
        events: list[NormalizedToolEvent],
        session: RuntimeSession | None = None,
    ) -> list[dict[str, Any]]:
        """Execute multiple tool events and return results."""
        results: list[dict[str, Any]] = []
        for event in events:
            result = self.execute_tool_event(event, session=session)
            results.append(result)
        return results

    def get_pending_deltas(self) -> list[Any]:
        """Get pending deltas from the tool executor."""
        return self.tool_executor.get_pending_deltas()

    def clear_pending_deltas(self) -> None:
        """Clear pending deltas from the tool executor."""
        self.tool_executor.clear_pending_deltas()

    def _build_tool_compatible_resend(
        self,
        request: RuntimeRequest,
        session: RuntimeSession,
        memory: str,
        skip_reason: str | None,
        behavior_signals: dict[str, Any],
        runtime_hints: list[str],
        current_pressure: int,
        hot_hint_metrics: dict[str, int],
        translated_messages: list[dict[str, Any]] | None = None,
        should_skip_history: bool = False,
        _calculate_dropped_tokens: bool = False,
        _recent_messages: list[dict[str, Any]] | None = None,
        has_answer_anchor_param: bool | None = None,
    ) -> tuple[
        str,
        list[str],
        dict[str, Any],
        dict[str, int],
        dict[str, Any],
        dict[str, Any],
        bool,
    ]:
        return build_tool_compatible_resend(
            self,
            request,
            session,
            memory,
            skip_reason,
            behavior_signals,
            runtime_hints,
            current_pressure,
            hot_hint_metrics,
            translated_messages=translated_messages,
            should_skip_history=should_skip_history,
            has_answer_anchor_param=has_answer_anchor_param,
        )

    def prepare_request(
        self,
        request: RuntimeRequest,
        session: RuntimeSession,
        *,
        result_cache: dict[str, Any] | None = None,
    ) -> PreparedRuntimeRequest:
        return self.prepare_signal_packet(
            SignalPacket.from_request(request),
            session,
            result_cache=result_cache,
        )

    def prepare_signal_packet(
        self,
        packet: SignalPacket,
        session: RuntimeSession,
        *,
        result_cache: dict[str, Any] | None = None,
    ) -> PreparedRuntimeRequest:
        from ._request_preparation import prepare_request_impl

        return prepare_request_impl(
            self,
            packet.request,
            session,
            signal_packet=packet,
            result_cache=result_cache,
        )

    def process_response(
        self,
        text: str,
        *,
        model: str,
        session: RuntimeSession,
        behavior_signals: dict[str, int] | None = None,
        tool_compatible: bool = False,
    ) -> ProcessedRuntimeResponse:
        return process_response_impl(
            self,
            text,
            model=model,
            session=session,
            behavior_signals=behavior_signals,
            tool_compatible=tool_compatible,
            jit_executor=execute_jit_macro,
        )

    def pressure_score(self, signals: dict[str, int]) -> int:
        return runtime_pressure_score(signals)
