"""Transport-agnostic universal runtime helpers for Tok."""

from __future__ import annotations

__all__ = [
    "TOOL_COMPAT_MEMORY_PROFILE",
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
]

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("tok.runtime")

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
from ._session_persistence import (
    bridge_memory_file,
    episode_ledger_file,
    fallback_memory_file,
    initialize_session_storage,
    load_bridge_memory,
    load_episode_ledger,
    load_fallback_memory,
    load_result_cache,
    result_cache_file,
    save_bridge_memory,
    save_episode_ledger,
    save_fallback_memory,
    save_result_cache,
)
from ._session_persistence import record_episode as record_episode_impl
from .config import (
    _FALLBACK_THRESHOLD,
    TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS,
    TOK_REQUEST_POLICY_STICKY_TURNS,
    TOOL_COMPAT_MEMORY_PROFILE,
)
from .memory.answer_memory import (
    _should_persist_to_durable,  # noqa: F401
    compact_structured_answer_memory,
    extract_structured_answer_memory,
    ground_structured_answer_memory,
    reinforce_structured_answer_memory,
)
from .memory.bridge_memory import BridgeMemoryState
from .memory.session_helpers import (
    calculate_reasoning_depth,
    get_adaptive_keep_turns,
    session_write_memory,
    update_session_family_mode,
)
from .memory.tok_state import (
    _build_tok_state,
    _delta_tok_state_fields,
    _prepare_tool_compatible_state,
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# SEARCH_LIKE_TOOLS is moved to .tool_processing
from .tools import RuntimeToolExecutor
from .types import (
    EpisodeEntry,
    EpisodeLedger,
    NormalizedToolEvent,
    PreparedRuntimeRequest,
    ProcessedRuntimeResponse,
    RuntimeRequest,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .repeat_targets import HotSummaryRecord, RepeatTargetEvent


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

    keep_turns: int = 2
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    result_cache: dict[str, Any] = field(default_factory=dict)
    # Maps (tool_name, args_hash) -> content_hash for dedup; see compress_tool_results.
    semantic_hash_cache: dict[str, str] = field(default_factory=dict)
    bridge_memory: BridgeMemoryState = field(default_factory=BridgeMemoryState)
    pending_behavior_signals: dict[str, int] = field(default_factory=dict)
    family_states: dict[str, FamilyAdaptiveState] = field(default_factory=dict)
    fallback_memory: str = ""
    memory_dir: Path | None = None
    episode_ledger: EpisodeLedger = field(default_factory=EpisodeLedger)
    # Accumulated telemetry for reasoning-depth computation
    _step_count: int = field(default=0, repr=False)
    _tool_names_seen: set[str] = field(default_factory=set, repr=False)
    _token_count: int = field(default=0, repr=False)
    # Memory Snap signal
    _tok_memory_snap_triggered: int = field(default=0, repr=False)
    # Structural metadata for telemetry
    _current_tool_density: float = field(default=0.0, repr=False)
    _current_context_char_count: int = field(default=0, repr=False)
    _current_invisible_pressure: int = field(default=0, repr=False)
    _active_tools: list[str] = field(default_factory=list, repr=False)
    _last_tool_compatible_state: str = field(default="", repr=False)
    _last_tool_compatible_state_fields: dict[str, list[str]] = field(default_factory=dict, repr=False)
    # Automatic session-scoped fallback tracking
    _consecutive_fallback_count: int = field(default=0, init=False, repr=False)
    _baseline_only: bool = field(default=False, init=False, repr=False)
    _answer_ready_repair_pending: bool = field(default=False, init=False, repr=False)
    _answer_ready_repair_active: bool = field(default=False, init=False, repr=False)
    _late_answer_assembly_repair_pending: bool = field(default=False, init=False, repr=False)
    _late_answer_assembly_repair_active: bool = field(default=False, init=False, repr=False)
    _late_answer_assembly_repair_mode_pending: str = field(default="", init=False, repr=False)
    _late_answer_assembly_repair_mode_active: str = field(default="", init=False, repr=False)
    _late_answer_followthrough_pending: bool = field(default=False, init=False, repr=False)
    _late_answer_followthrough_active: bool = field(default=False, init=False, repr=False)
    _last_mode: str = field(default="", init=False, repr=False)
    _drift_detected_previous_turn: bool = field(default=False, init=False, repr=False)
    _stream_recovery_reacquisition_budget: int = field(default=0, init=False, repr=False)
    _stream_recovery_history_floor_budget: int = field(default=0, init=False, repr=False)
    _stream_recovery_tool_use_only_signature: str = field(default="", init=False, repr=False)
    _stream_recovery_tool_use_only_repeat_count: int = field(default=0, init=False, repr=False)
    _stream_recovery_cooldown_remaining: int = field(default=0, init=False, repr=False)
    _stream_recovery_cooldown_suppressed: bool = field(default=False, init=False, repr=False)
    _stream_read_error_consecutive_count: int = field(default=0, init=False, repr=False)
    _stream_read_error_last_stage: str = field(default="", init=False, repr=False)
    _request_policy_tool_mode_sticky_turns: int = field(default=0, init=False, repr=False)
    _request_policy_stream_recovery_watch_turns: int = field(default=0, init=False, repr=False)
    _request_policy_tool_recovery_watch_turns: int = field(default=0, init=False, repr=False)
    # Smoothness tracking fields
    _latest_turn_smoothness_score: int = field(default=100, init=False, repr=False)
    _latest_turn_labour_index: int = field(default=0, init=False, repr=False)
    _current_task_smoothness_score: int = field(default=100, init=False, repr=False)
    _current_task_labour_index: int = field(default=0, init=False, repr=False)
    _current_tok_mode: TokMode = field(default=TokMode.FULL_TOK, init=False, repr=False)
    _smoothness_event_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _request_policy_last_effective_tool_compatible: bool = field(default=False, init=False, repr=False)
    _invalid_tool_history_recovery_count: int = field(default=0, init=False, repr=False)
    # Project-type markers discovered at session init (e.g. 'package.json', 'go.mod').
    _project_markers: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)
    _load_global_macros: bool = field(default=True, init=False, repr=False)
    # Name of a macro that was offered via JIT but whose result should be verified
    # for potential healing.  Set when a jit_offer fires; cleared after healing check.
    _pending_macro_heal: str = field(default="", init=False, repr=False)
    _pending_macro_heal_turn: int = field(default=0, init=False, repr=False)
    _recent_repeat_target_events: list[RepeatTargetEvent] = field(default_factory=list, init=False, repr=False)
    _hot_summary_records: dict[str, HotSummaryRecord] = field(default_factory=dict, init=False, repr=False)
    _observed_tool_result_ids: dict[str, None] = field(default_factory=dict, init=False, repr=False)
    _prepared_prompt_token_cache: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _predictive_cache_warm_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _evidence_neighborhoods: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    _evidence_anchor_novelty_keys: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    _evidence_alias_map: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _first_exact_evidence_seen: set[str] = field(default_factory=set, init=False, repr=False)
    _pending_exact_evidence_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _files_read_this_session: set[str] = field(default_factory=set, init=False, repr=False)
    _files_fully_delivered: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def record_fallback_event(self) -> None:
        """Increment the consecutive fail-open counter and degrade to baseline when threshold is reached."""
        self._consecutive_fallback_count += 1
        if self._consecutive_fallback_count >= _FALLBACK_THRESHOLD and not self._baseline_only:
            self._baseline_only = True
            logger.warning(
                "tok_fallback_activated: session degraded to baseline after %d consecutive fallback events",
                self._consecutive_fallback_count,
            )

    def reset_fallback_count(self) -> None:
        """Reset the consecutive fallback counter after a successful compressed request."""
        self._consecutive_fallback_count = 0

    def record_invalid_tool_history_recovery(self, *, blocked: bool) -> dict[str, int]:
        """Track recovery from broken tool history and clear hot state if it repeats."""
        self._invalid_tool_history_recovery_count += 1
        self.note_request_policy_tool_mode_recovery()
        signals: dict[str, int] = {
            "tok_bridge_invalid_tool_history_recovery": 1,
            "tok_bridge_invalid_tool_history_blocked": 1 if blocked else 0,
        }

        if self._invalid_tool_history_recovery_count >= 2:
            self._last_tool_compatible_state = ""
            self._last_tool_compatible_state_fields = {}
            self._observed_tool_result_ids.clear()
            self._first_exact_evidence_seen.clear()
            self._pending_exact_evidence_keys.clear()
            self._stream_recovery_reacquisition_budget = 0
            self._stream_recovery_history_floor_budget = 0
            self._stream_recovery_tool_use_only_signature = ""
            self._stream_recovery_tool_use_only_repeat_count = 0
            self._request_policy_tool_mode_sticky_turns = 0
            self._request_policy_stream_recovery_watch_turns = 0
            self._request_policy_tool_recovery_watch_turns = 0
            self._request_policy_last_effective_tool_compatible = False
            for key in ("turns", "next", "cmds", "errs", "blockers"):
                self.bridge_memory.hot.pop(key, None)
            self.bridge_memory.rolling_cmds = []
            self._save_bridge_memory()
            signals["tok_bridge_invalid_tool_history_session_reset"] = 1
            logger.warning(
                "tok_bridge_invalid_tool_history_session_reset: cleared hot bridge state after %d repeated tool-history recoveries",
                self._invalid_tool_history_recovery_count,
            )
        return signals

    def reset_invalid_tool_history_recovery(self) -> None:
        """Clear the repeated invalid-tool-history counter after a clean request."""
        self._invalid_tool_history_recovery_count = 0

    def note_request_policy_stream_recovery(self, turns: int = TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS) -> None:
        """Keep natural-first in tool-compatible mode briefly after stream recovery."""
        self._request_policy_stream_recovery_watch_turns = max(self._request_policy_stream_recovery_watch_turns, turns)
        self._request_policy_tool_mode_sticky_turns = max(self._request_policy_tool_mode_sticky_turns, turns)

    def note_request_policy_tool_mode_recovery(self, turns: int = TOK_REQUEST_POLICY_STICKY_TURNS) -> None:
        """Keep natural-first in tool-compatible mode briefly after tool-history recovery."""
        self._request_policy_tool_recovery_watch_turns = max(self._request_policy_tool_recovery_watch_turns, turns)
        self._request_policy_tool_mode_sticky_turns = max(self._request_policy_tool_mode_sticky_turns, turns)

    def adaptive_keep_turns(self) -> int:
        """Dynamically reduce history depth as the session grows."""
        return get_adaptive_keep_turns(self)

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
        projected = self.bridge_memory.wire_state(policy.memory_profiles[mode], markers=self._project_markers)
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
        return self.bridge_memory.wire_state(policy.memory_profiles[mode], markers=self._project_markers)

    def write_memory(self, text: str) -> str:
        """Write memory state and return the written content."""
        return session_write_memory(self, text)

    def record_file_snapshot(self, path: str, snippet: str) -> bool:
        """Record a file snapshot in bridge memory."""
        return record_file_snapshot_impl(self, path, snippet)

    def record_search_snapshot(self, query: str, snippet: str) -> bool:
        """Record a search snapshot in bridge memory."""
        return record_search_snapshot_impl(self, query, snippet)

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
                self._evidence_alias_map[normalized] = src_path
                return src_path
        return None

    def prepared_prompt_tokens(self, payload: dict[str, Any]) -> int:
        """Count and cache tokens for a prepared prompt."""
        return prepared_prompt_tokens_impl(self, payload)

    def _trim_repeat_target_state(self) -> None:
        if len(self._recent_repeat_target_events) > 16:
            self._recent_repeat_target_events = self._recent_repeat_target_events[-16:]
        if len(self._hot_summary_records) > 64:
            ranked = sorted(
                self._hot_summary_records.items(),
                key=lambda item: (
                    item[1].stuck_promotion_turn or item[1].hot_promotion_turn,
                    item[1].last_seen_turn,
                ),
                reverse=True,
            )[:64]
            self._hot_summary_records = dict(ranked)
        if len(self._observed_tool_result_ids) > 64:
            # Preserve recency by keeping the last 64 keys
            keys_to_keep = list(self._observed_tool_result_ids.keys())[-64:]
            self._observed_tool_result_ids = dict.fromkeys(keys_to_keep)

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

    def hot_recent_runtime_hints(self) -> tuple[list[str], dict[str, int]]:
        """Generate hot recent hints for eligible repeat targets."""
        return hot_recent_runtime_hints_impl(self)

    def evidence_intent_advisories(self) -> list[str]:
        """Generate advisories based on evidence intent patterns."""
        return evidence_intent_advisories_impl(self)

    def is_predictive_cache_hit(self, family: str, logical_target: str) -> bool:
        """Check if a target is in the predictive cache warm set."""
        return f"{family}|{logical_target}" in self._predictive_cache_warm_keys

    def update_family_mode(self, model: str, signals: dict[str, int]) -> str:
        """Update the compression mode for a model family."""
        return update_session_family_mode(self, model, signals)

    def consume_behavior_signals(self) -> dict[str, int]:
        """Consume and clear pending behavior signals."""
        signals = dict(self.pending_behavior_signals)
        self.pending_behavior_signals.clear()
        return signals

    def maybe_suppress_tool_compatible_state(self, state: str) -> tuple[str, dict[str, int]]:
        """Avoid resending unchanged tool-compatible state on every turn."""
        cleaned = state.strip()
        if not cleaned:
            self._last_tool_compatible_state = ""
            self._last_tool_compatible_state_fields = {}
            return state, {}
        parsed, comparable, has_answer_facts = _prepare_tool_compatible_state(
            cleaned, self._last_tool_compatible_state_fields
        )
        previous_comparable = dict(self._last_tool_compatible_state_fields)
        # Treat turn-only state as unchanged after the first emission. This keeps
        # warm tool-compatible turns from repeatedly resending a payload that carries
        # no reusable facts, files, tests, or answer anchors.
        if (
            self._last_tool_compatible_state
            and not comparable
            and not previous_comparable
            and parsed
            and set(parsed.keys()) <= {"turns"}
        ):
            return "", {"state_resend_suppressed_turn": 1}
        strategy = _select_resend_strategy(comparable, previous_comparable, has_answer_facts)
        if strategy == "suppress":
            return "", {"state_resend_suppressed_turn": 1}
        rendered = _build_tok_state(parsed)
        self._last_tool_compatible_state = rendered
        self._last_tool_compatible_state_fields = comparable
        if strategy == "full":
            return rendered, {"state_resend_full_turn": 1}
        # strategy == "delta"
        delta = _delta_tok_state_fields(previous_comparable, parsed)
        if delta and len(delta) < len(rendered):
            return delta, {"state_resend_delta_turn": 1}
        return rendered, {
            "state_resend_full_turn": 1,
            "state_resend_reason_delta_not_smaller": 1,
        }

    @property
    def latest_turn_smoothness_score(self) -> int:
        """Latest turn's smoothness score (0-100)."""
        return self._latest_turn_smoothness_score

    @property
    def latest_turn_labour_index(self) -> int:
        """Latest turn's labour index."""
        return self._latest_turn_labour_index

    @property
    def current_task_smoothness_score(self) -> int:
        """Current task's smoothness score (0-100)."""
        return self._current_task_smoothness_score

    @property
    def current_task_labour_index(self) -> int:
        """Current task's labour index."""
        return self._current_task_labour_index

    @property
    def current_tok_mode(self) -> TokMode:
        """Current Tok compression mode."""
        return self._current_tok_mode

    @property
    def smoothness_event_counts(self) -> dict[str, int]:
        """Count of smoothness events by type."""
        return dict(self._smoothness_event_counts)

    def update_smoothness_state(
        self,
        turn_score: int,
        labour_index: int,
        tok_mode: TokMode,
        event_counts: dict[str, int],
    ) -> None:
        """
        Update smoothness state after a turn completes.

        Args:
            turn_score: Smoothness score for the completed turn (0-100)
            labour_index: Labour index for the completed turn
            tok_mode: Tok mode selected for the next turn
            event_counts: Event counts for the completed turn

        """
        self._latest_turn_smoothness_score = turn_score
        self._latest_turn_labour_index = labour_index
        self._current_tok_mode = tok_mode

        for event_type, count in event_counts.items():
            self._smoothness_event_counts[event_type] = self._smoothness_event_counts.get(event_type, 0) + count

        self._current_task_smoothness_score = turn_score
        self._current_task_labour_index = labour_index

    def _bump_signals(self, signals: dict[str, int]) -> None:
        """Accumulate behavior signals for the next request."""
        for key, value in signals.items():
            self.pending_behavior_signals[key] = self.pending_behavior_signals.get(key, 0) + value
        if signals.get("tok_memory_snap_triggered"):
            self._tok_memory_snap_triggered = 1


class UniversalTokRuntime:
    """Canonical transport-agnostic request/response runtime."""

    def __init__(self) -> None:
        self.tool_executor = RuntimeToolExecutor()
        self.semantic_validator = SemanticValidator()

    def execute_tool_event(self, event: NormalizedToolEvent) -> dict[str, Any]:
        """Execute a normalized tool event using the shared runtime tools."""
        return self.tool_executor.execute_normalized_tool(event)

    def execute_tool_events_batch(self, events: list[NormalizedToolEvent]) -> list[dict[str, Any]]:
        """Execute multiple tool events and return results."""
        results: list[dict[str, Any]] = []
        for event in events:
            result = self.execute_tool_event(event)
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
        from ._request_preparation import prepare_request_impl

        return prepare_request_impl(
            self,
            request,
            session,
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
