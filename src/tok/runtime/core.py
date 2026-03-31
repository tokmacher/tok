"""Transport-agnostic universal runtime helpers for Tok."""

from __future__ import annotations

__all__ = [
    "_should_skip_history_rewrite",
    "RuntimeRequest",
    "NormalizedToolEvent",
    "PreparedRuntimeRequest",
    "ProcessedRuntimeResponse",
    "compact_structured_answer_memory",
    "extract_structured_answer_memory",
    "ground_structured_answer_memory",
    "reinforce_structured_answer_memory",
    "RuntimeSession",
    "UniversalTokRuntime",
    "apply_schema_adaptations",
    "calculate_semantic_regression_score",
    "evaluate_replay_gate",
    "calculate_invisible_pressure",
    "count_tokens",
]

import copy
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("tok.runtime")

from .memory.bridge_memory import BridgeMemoryState, clean_system_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from ..compression import (
    EDIT_LIKE_TOOLS,
    compress_history,
    compress_recent_window,
    compress_tool_results,
    inject_system_additions,
    text_of,
)

# SEARCH_LIKE_TOOLS is moved to .tool_processing
from .tools import RuntimeToolExecutor
from .policy.smart_policy import (
    FamilyAdaptiveState,
    SmartZonePolicy,
    initial_state,
    policy_for_model,
)

from ..neuro.ir import Instruction
from .config import (
    _FALLBACK_THRESHOLD,
    TOOL_DENSITY_THRESHOLD,
    TOOL_COMPAT_MEMORY_PROFILE,
    TOK_HOT_COMMAND_MAX_CHARS,
    TOK_HOT_COMMAND_MAX_LINES,
    TOK_HOT_FILE_MAX_CHARS,
    TOK_HOT_FILE_MAX_LINES,
    TOK_HOT_RECENT_MAX_HINTS,
    TOK_HOT_SEARCH_MAX_CHARS,
    TOK_HOT_SEARCH_MAX_LINES,
    TOK_NEIGHBORHOOD_THRASH_HINT,
    TOK_NEIGHBORHOOD_TRIGGER_ANCHORS,
    TOK_NEIGHBORHOOD_WINDOW_TURNS,
    TOK_NOVELTY_REQUIRED_HINT,
    TOK_PREDICTIVE_CACHE_TOP_K,
    TOK_REACQUIRE_STUCK_COUNT,
    TOK_REACQUIRE_STUCK_WINDOW_TURNS,
    TOK_REACQUIRE_TRIGGER_COUNT,
    TOK_REACQUIRE_WINDOW_TURNS,
)
from .types import (
    EpisodeEntry,
    EpisodeLedger,
    NormalizedToolEvent,
    RuntimeRequest,
    PreparedRuntimeRequest,
    ProcessedRuntimeResponse,
)
from .pipeline.request_validation import (
    validate_anthropic_request_body,
    detect_prompt_bloat,
    canonicalize_anthropic_bridge_body,
    validate_anthropic_bridge_body,
    summarize_message_structure,
)
from .pipeline.tool_processing import (
    build_tool_use_id_to_context,
    normalize_tool_events,
    collect_behavior_signals,
    _count_tool_density,
    _should_skip_history_rewrite,
    count_tokens,
    logical_target_key_from_context,
)
from .repeat_targets import (
    HotSummaryRecord,
    RepeatTargetEvent,
    build_summary_for_family,
    resolve_evidence_intent,
    stable_digest,
)
from .memory.tok_state import (
    _prepare_tool_compatible_state,
    _select_resend_strategy,
    _build_tok_state,
    _delta_tok_state_fields,
    _select_resend_reason,
)
from .policy.semantic_validation import (
    SemanticValidator,
    calculate_invisible_pressure,
    calculate_semantic_regression_score,  # noqa: F401
    pressure_score as _semantic_pressure_score,
)
from .memory.answer_memory import (
    _process_answer_memory,
    _should_persist_to_durable,
    compact_structured_answer_memory,  # noqa: F401
    extract_structured_answer_memory,  # noqa: F401
    ground_structured_answer_memory,  # noqa: F401
    reinforce_structured_answer_memory,  # noqa: F401
)
from .pipeline.response_processing import (
    heal_drift,
    response_behavior_signals,
    _is_answer_like_visible_text,
    translate_request_results,
    response_contract_for_mode,
)
from .policy.macro_handling import (
    _attribute_macro_savings,
    _jit_context_matches,
    execute_jit_macro,
)
from .pipeline.request_preparation import (
    _inject_system,
    collect_transient_error_snippets,
    _is_answer_ready_turn,
    _runtime_hints_for_turn,
    _annotate_reacquisition_diagnostics,
    _apply_tool_compatible_resend_diagnostics,
    _capture_repeat_target_snapshots,
    apply_schema_adaptations,
    mutation_signals,
)
from .pipeline.response_handling import (
    sort_cache_control_blocks,
    handle_answer_repair,
    evaluate_replay_gate,  # noqa: F401
)
from .policy.answer_repair import (
    _mark_late_answer_assembly_mode_signal,
)
from .memory.session_helpers import (
    calculate_reasoning_depth,
    update_session_family_mode,
    session_write_memory,
    get_adaptive_keep_turns,
    _discover_project_markers,
    extract_memory_items,
)
from .metrics import (
    report_protocol_drift,
)


@dataclass
class RuntimeSession:
    keep_turns: int = 2
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
    _last_tool_compatible_state_fields: dict[str, list[str]] = field(
        default_factory=dict, repr=False
    )
    # Automatic session-scoped fallback tracking
    _consecutive_fallback_count: int = field(default=0, init=False, repr=False)
    _baseline_only: bool = field(default=False, init=False, repr=False)
    _answer_ready_repair_pending: bool = field(
        default=False, init=False, repr=False
    )
    _answer_ready_repair_active: bool = field(
        default=False, init=False, repr=False
    )
    _late_answer_assembly_repair_pending: bool = field(
        default=False, init=False, repr=False
    )
    _late_answer_assembly_repair_active: bool = field(
        default=False, init=False, repr=False
    )
    _late_answer_assembly_repair_mode_pending: str = field(
        default="", init=False, repr=False
    )
    _late_answer_assembly_repair_mode_active: str = field(
        default="", init=False, repr=False
    )
    _late_answer_followthrough_pending: bool = field(
        default=False, init=False, repr=False
    )
    _late_answer_followthrough_active: bool = field(
        default=False, init=False, repr=False
    )
    _last_mode: str = field(default="", init=False, repr=False)
    _drift_detected_previous_turn: bool = field(
        default=False, init=False, repr=False
    )
    # Project-type markers discovered at session init (e.g. 'package.json', 'go.mod').
    _project_markers: frozenset[str] = field(
        default_factory=frozenset, init=False, repr=False
    )
    _load_global_macros: bool = field(default=True, init=False, repr=False)
    # Name of a macro that was offered via JIT but whose result should be verified
    # for potential healing.  Set when a jit_offer fires; cleared after healing check.
    _pending_macro_heal: str = field(default="", init=False, repr=False)
    _pending_macro_heal_turn: int = field(default=0, init=False, repr=False)
    _recent_repeat_target_events: list[RepeatTargetEvent] = field(
        default_factory=list, init=False, repr=False
    )
    _hot_summary_records: dict[str, HotSummaryRecord] = field(
        default_factory=dict, init=False, repr=False
    )
    _observed_tool_result_ids: set[str] = field(
        default_factory=set, init=False, repr=False
    )
    _prepared_prompt_token_cache: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _predictive_cache_warm_keys: set[str] = field(
        default_factory=set, init=False, repr=False
    )
    _evidence_neighborhoods: dict[str, set[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _evidence_anchor_novelty_keys: dict[str, set[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _evidence_alias_map: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    def record_fallback_event(self) -> None:
        """Increment the consecutive fail-open counter and degrade to baseline when threshold is reached."""
        self._consecutive_fallback_count += 1
        if (
            self._consecutive_fallback_count >= _FALLBACK_THRESHOLD
            and not self._baseline_only
        ):
            self._baseline_only = True
            logger.warning(
                "tok_fallback_activated: session degraded to baseline after %d consecutive fallback events",
                self._consecutive_fallback_count,
            )

    def reset_fallback_count(self) -> None:
        """Reset the consecutive fallback counter after a successful compressed request."""
        self._consecutive_fallback_count = 0

    def adaptive_keep_turns(self) -> int:
        """Dynamically reduce history depth as the session grows."""
        return get_adaptive_keep_turns(self)

    def __post_init__(self) -> None:
        """Initialize memory directory and load persisted bridge memory."""
        explicit_memory_dir = self.memory_dir is not None
        if self.memory_dir is None:
            project_dir = os.getenv("TOK_PROJECT_DIR", "")
            if project_dir:
                self.memory_dir = Path(project_dir) / ".tok"
            else:
                self.memory_dir = Path.home() / ".tok"
        # Preserve load_global_macros from passed bridge_memory if it's explicitly False
        # Otherwise, set based on explicit_memory_dir
        if not self.bridge_memory.load_global_macros:
            self._load_global_macros = False
        else:
            self._load_global_macros = not explicit_memory_dir
            self.bridge_memory = self._load_bridge_memory()
        self.result_cache = self._load_result_cache()
        self.fallback_memory = self._load_fallback_memory()
        if self.fallback_memory and not self.bridge_memory.wire_state():
            self.bridge_memory.ingest_wire_state(self.fallback_memory)
            self._save_bridge_memory()
        # Local Mesh Discovery: scan CWD for project markers and warm file heat so
        # macros whose context_requirements reference these markers get speculative
        # injection from the very first turn.
        if explicit_memory_dir:
            self._project_markers = frozenset()
        else:
            self._project_markers = _discover_project_markers()
            for marker in self._project_markers:
                self.bridge_memory.bump_file_heat(marker, weight=0.1)

    def _bridge_memory_file(self) -> Path:
        assert self.memory_dir is not None
        return self.memory_dir / "bridge_memory.tok"

    def _load_bridge_memory(self) -> BridgeMemoryState:
        """Load bridge memory from persisted file."""
        path = self._bridge_memory_file()
        if not path.exists():
            return BridgeMemoryState(
                load_global_macros=self._load_global_macros
            )
        try:
            return BridgeMemoryState.from_tok(
                path.read_text(),
                load_global_macros=self._load_global_macros,
            )
        except FileNotFoundError:
            return BridgeMemoryState(
                load_global_macros=self._load_global_macros
            )
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            logger.warning(
                "Bridge memory file corrupted at %s: %s — starting with empty memory",
                path,
                exc,
            )
            return BridgeMemoryState(
                load_global_macros=self._load_global_macros
            )
        except Exception as exc:
            logger.warning(
                "Failed to load bridge memory from %s: %s",
                path,
                exc,
            )
            return BridgeMemoryState(
                load_global_macros=self._load_global_macros
            )

    def _save_bridge_memory(self) -> None:
        """Save bridge memory to persisted file."""
        try:
            assert self.memory_dir is not None
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self._bridge_memory_file().write_text(self.bridge_memory.to_tok())
        except Exception as exc:
            logger.warning(
                "Failed to save bridge memory to %s: %s",
                self._bridge_memory_file(),
                exc,
            )

    def _result_cache_file(self) -> Path:
        assert self.memory_dir is not None
        return self.memory_dir / "result_cache.tok"

    def _load_result_cache(self) -> dict[str, Any]:
        """Load result cache from persisted file."""
        path = self._result_cache_file()
        if not path.exists():
            return {}
        try:
            result = json.loads(path.read_text())
            if isinstance(result, dict):
                return result
            logger.warning(
                "Result cache at %s is not a dict — starting empty", path
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(
                "Result cache corrupted at %s: %s — starting empty", path, exc
            )
        except Exception as exc:
            logger.warning(
                "Failed to load result cache from %s: %s", path, exc
            )
        return {}

    def _save_result_cache(self) -> None:
        """Save result cache to persisted file."""
        try:
            assert self.memory_dir is not None
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            trimmed = {
                k: (v[0], v[1][:10240])
                if isinstance(v, tuple) and len(v) == 2
                else v
                for k, v in self.result_cache.items()
            }
            self._result_cache_file().write_text(json.dumps(trimmed))
        except Exception as exc:
            logger.warning(
                "Failed to save result cache to %s: %s",
                self._result_cache_file(),
                exc,
            )

    def _fallback_memory_file(self) -> Path:
        assert self.memory_dir is not None
        return self.memory_dir / "memory.tok"

    def _load_fallback_memory(self) -> str:
        """Load persisted raw fallback memory from disk."""
        path = self._fallback_memory_file()
        if not path.exists():
            return ""
        try:
            return path.read_text().strip()
        except (UnicodeDecodeError, PermissionError) as exc:
            logger.warning(
                "Failed to load fallback memory from %s: %s", path, exc
            )
        except Exception as exc:
            logger.warning(
                "Failed to load fallback memory from %s: %s", path, exc
            )
        return ""

    def _save_fallback_memory(self) -> None:
        """Save raw fallback memory to persisted file."""
        try:
            assert self.memory_dir is not None
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self._fallback_memory_file().write_text(
                self.fallback_memory + "\n"
            )
        except Exception as exc:
            logger.warning(
                "Failed to save fallback memory to %s: %s",
                self._fallback_memory_file(),
                exc,
            )

    def _episode_ledger_file(self) -> Path:
        assert self.memory_dir is not None
        return self.memory_dir / "episode_ledger.tok"

    def _load_episode_ledger(self) -> EpisodeLedger:
        path = self._episode_ledger_file()
        if not path.exists():
            return EpisodeLedger()
        try:
            return EpisodeLedger.from_tok(path.read_text())
        except (ValueError, UnicodeDecodeError) as exc:
            logger.warning(
                "Episode ledger corrupted at %s: %s — starting empty",
                path,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load episode ledger from %s: %s", path, exc
            )
        return EpisodeLedger()

    def _save_episode_ledger(self) -> None:
        try:
            assert self.memory_dir is not None
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self._episode_ledger_file().write_text(
                self.episode_ledger.to_tok()
            )
        except Exception as exc:
            logger.warning(
                "Failed to save episode ledger to %s: %s",
                self._episode_ledger_file(),
                exc,
            )

    def record_episode(self, entry: EpisodeEntry) -> None:
        """Log a completed reasoning episode and persist the ledger."""
        self.episode_ledger.record(entry)
        self._save_episode_ledger()
        self._bump_signals({"episode_recorded": 1})

    def reasoning_depth_per_token(self) -> float:
        """Dual-axis metric: reasoning diversity per token consumed.

        Combines step count, tool diversity, and tokens used.
        Higher is better — rewards rich reasoning without token bloat.
        Returns 0.0 when no tokens have been recorded.
        """
        return calculate_reasoning_depth(self)

    def policy_snapshot(self, model: str) -> tuple[str, SmartZonePolicy]:
        policy = policy_for_model(model)
        state = self.family_states.setdefault(
            policy.family.key, initial_state(policy)
        )
        return state.mode, policy

    def load_memory(self, model: str = "") -> str:
        mode, policy = self.policy_snapshot(model)
        projected = self.bridge_memory.wire_state(
            policy.memory_profiles[mode], markers=self._project_markers
        )
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
        self._bump_signals(
            self.bridge_memory.replace_hot_from_wire_state(tok_state)
        )
        return self.bridge_memory.wire_state(
            policy.memory_profiles[mode], markers=self._project_markers
        )

    def write_memory(self, text: str) -> str:
        return session_write_memory(self, text)

    def record_file_snapshot(self, path: str, snippet: str) -> bool:
        """Persist a file snippet so the model can reuse it without rereading."""
        recorded = self.bridge_memory.record_file_snapshot(path, snippet)
        if recorded:
            self._bump_signals({"file_snapshot_recorded": 1})
            self._save_bridge_memory()
        return recorded

    def record_search_snapshot(self, query: str, snippet: str) -> bool:
        """Persist a search snippet so future turns can reuse the result."""
        recorded = self.bridge_memory.record_search_snapshot(query, snippet)
        if recorded:
            self._bump_signals({"search_snapshot_recorded": 1})
            self._save_bridge_memory()
        return recorded

    def record_history_snapshot(
        self, path: str, revision: str, snippet: str
    ) -> bool:
        recorded = self.bridge_memory.record_history_snapshot(
            path, revision, snippet
        )
        if recorded:
            self._bump_signals({"history_snapshot_recorded": 1})
            self._save_bridge_memory()
        return recorded

    def record_metadata_snapshot(
        self, path: str, subtype: str, snippet: str
    ) -> bool:
        recorded = self.bridge_memory.record_metadata_snapshot(
            path, subtype, snippet
        )
        if recorded:
            self._save_bridge_memory()
        return recorded

    def check_temp_copy_alias(self, path: str, snippet: str) -> str | None:
        from .repeat_targets import normalize_path_target, _is_temp_path

        if not _is_temp_path(path):
            return None
        normalized = normalize_path_target(path)
        existing_digests = self.bridge_memory.get_file_fact_digests()
        new_digest = self.bridge_memory._extract_file_digest(
            snippet, normalized
        )
        if not new_digest:
            new_digest = " ".join(snippet.split())[:160]
        for src_path, src_digest in existing_digests.items():
            if new_digest and src_digest and new_digest == src_digest:
                self._evidence_alias_map[normalized] = src_path
                return src_path
        return None

    def prepared_prompt_tokens(self, payload: dict[str, Any]) -> int:
        """Estimate prompt tokens for the system/messages payload with per-session caching."""
        prompt_payload = {
            "system": copy.deepcopy(payload.get("system", "")),
            "messages": copy.deepcopy(payload.get("messages", [])),
        }
        fingerprint = hashlib.sha256(
            json.dumps(prompt_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        if fingerprint in self._prepared_prompt_token_cache:
            self._bump_signals({"prepared_prompt_token_cache_hit": 1})
            return self._prepared_prompt_token_cache[fingerprint]
        token_count = count_tokens(json.dumps(prompt_payload, sort_keys=True))
        self._prepared_prompt_token_cache[fingerprint] = token_count
        if len(self._prepared_prompt_token_cache) > 32:
            oldest_key = next(iter(self._prepared_prompt_token_cache))
            self._prepared_prompt_token_cache.pop(oldest_key, None)
        return token_count

    def _trim_repeat_target_state(self) -> None:
        if len(self._recent_repeat_target_events) > 16:
            self._recent_repeat_target_events = (
                self._recent_repeat_target_events[-16:]
            )
        if len(self._hot_summary_records) > 8:
            ranked = sorted(
                self._hot_summary_records.items(),
                key=lambda item: (
                    item[1].stuck_promotion_turn or item[1].hot_promotion_turn,
                    item[1].last_seen_turn,
                ),
                reverse=True,
            )[:8]
            self._hot_summary_records = dict(ranked)
        if len(self._observed_tool_result_ids) > 64:
            self._observed_tool_result_ids = set(
                list(self._observed_tool_result_ids)[-64:]
            )

    def observe_repeat_target_result(
        self,
        *,
        tool_id: str,
        tool_name: str,
        path: str | None,
        query: str | None,
        command: str | None,
        raw_content: str,
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
            blocker_rediscovery=blocker_rediscovery,
        )

    def apply_predictive_cache_warming(
        self, logical_target: str
    ) -> dict[str, int]:
        """Warm nearby session-local cache entries using already known snapshots only."""
        candidate_keys: list[str] = []
        record = self._hot_summary_records.get(f"file_read|{logical_target}")
        if not record:
            return {}
        normalized_path = logical_target
        sibling_prefix = str(Path(normalized_path).parent)
        for key, other in sorted(
            self._hot_summary_records.items(),
            key=lambda item: item[1].last_seen_turn,
            reverse=True,
        ):
            if key == f"file_read|{logical_target}":
                continue
            if (
                other.tool_family == "file_read"
                and str(Path(other.logical_target).parent) == sibling_prefix
            ):
                candidate_keys.append(key)
            if other.tool_family == "search":
                same_turn = any(
                    event.turn_index == record.last_seen_turn
                    and event.tool_family == "search"
                    and f"{event.tool_family}|{event.logical_target}" == key
                    for event in self._recent_repeat_target_events
                )
                if same_turn:
                    candidate_keys.append(key)
        for event in sorted(
            self._recent_repeat_target_events,
            key=lambda item: item.turn_index,
            reverse=True,
        ):
            key = f"{event.tool_family}|{event.logical_target}"
            if key == f"file_read|{logical_target}":
                continue
            same_turn = any(
                other.turn_index == event.turn_index
                and other.tool_family == "file_read"
                and other.logical_target == logical_target
                for other in self._recent_repeat_target_events
            )
            if same_turn and event.tool_family == "file_read":
                candidate_keys.append(key)
        deduped: list[str] = []
        seen: set[str] = set()
        for key in candidate_keys:
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        warmed = 0
        selected = deduped[:TOK_PREDICTIVE_CACHE_TOP_K]
        for key in selected:
            if key in self._hot_summary_records:
                self._predictive_cache_warm_keys.add(key)
                warmed += 1
        if not selected:
            return {}
        return {
            "predictive_cache_warm_applied": 1,
            "predictive_cache_candidates": len(selected),
            "predictive_cache_hits": warmed,
        }

    def hot_recent_runtime_hints(self) -> tuple[list[str], dict[str, int]]:
        """Build ranked hot-target reminders for the next prepared request."""
        current_turn = max(1, self.bridge_memory.turn)
        candidates: list[HotSummaryRecord] = []
        for key, record in self._hot_summary_records.items():
            del key
            promoted_turn = max(
                record.hot_promotion_turn, record.stuck_promotion_turn
            )
            if not promoted_turn or promoted_turn <= record.last_injected_turn:
                continue
            if current_turn < promoted_turn:
                continue
            candidates.append(record)
        candidates.sort(
            key=lambda record: (
                record.stuck_window_count,
                record.last_seen_turn,
                record.token_cost,
            ),
            reverse=True,
        )
        selected = candidates[:TOK_HOT_RECENT_MAX_HINTS]
        hints: list[str] = []
        metrics = {
            "repeat_tool_collapse_applied": 0,
            "hot_recent_hint_injected": 0,
            "hot_hint_tokens_added": 0,
            "reacquisition_tokens_avoided_estimate": 0,
        }
        for record in selected:
            label = record.display_target
            if record.tool_family == "file_read":
                reminder = f"@hot_recent_file:{label} |> {record.summary}"
            elif record.tool_family == "search":
                reminder = f"@hot_recent_search:{label} |> {record.summary}"
            else:
                reminder = f"@hot_recent_command:{label} |> {record.summary}"
            guidance = (
                "This target is stuck and unchanged. Reuse the cached result and move forward without rereading it."
                if record.tool_family == "file_read"
                and record.stuck_promotion_turn
                else "Reuse this cached result unless you have a concrete reason to reacquire it."
            )
            block = reminder + "\n" + guidance
            hints.append(block)
            metrics["hot_recent_hint_injected"] += 1
            metrics["reacquisition_tokens_avoided_estimate"] += (
                record.token_cost
            )
            if (
                record.tool_family in {"search", "command"}
                and record.unchanged_result_count > 0
            ):
                metrics["repeat_tool_collapse_applied"] += 1
            record.last_injected_turn = current_turn
        if hints:
            metrics["hot_hint_tokens_added"] = count_tokens("\n\n".join(hints))
        return hints, metrics

    def evidence_intent_advisories(self) -> list[str]:
        current_turn = max(1, self.bridge_memory.turn)
        for record in self._hot_summary_records.values():
            if not record.evidence_intent:
                continue
            if not (record.hot_promotion_turn or record.stuck_promotion_turn):
                continue
            anchor = record.evidence_intent.anchor
            novelty_keys = self._evidence_anchor_novelty_keys.get(anchor)
            if novelty_keys and record.repeat_count > 1:
                return [
                    TOK_NOVELTY_REQUIRED_HINT.format(
                        anchor=record.display_target
                    )
                ]
        for (
            neighborhood,
            anchors,
        ) in self._evidence_neighborhoods.items():
            if len(anchors) < TOK_NEIGHBORHOOD_TRIGGER_ANCHORS:
                continue
            recent_count = sum(
                1
                for e in self._recent_repeat_target_events
                if e.evidence_anchor in anchors
                and current_turn - e.turn_index < TOK_NEIGHBORHOOD_WINDOW_TURNS
            )
            if recent_count >= TOK_NEIGHBORHOOD_TRIGGER_ANCHORS:
                return [
                    TOK_NEIGHBORHOOD_THRASH_HINT.format(
                        neighborhood=neighborhood
                    )
                ]
        return []

    def is_predictive_cache_hit(
        self, family: str, logical_target: str
    ) -> bool:
        return f"{family}|{logical_target}" in self._predictive_cache_warm_keys

    def update_family_mode(self, model: str, signals: dict[str, int]) -> str:
        return update_session_family_mode(self, model, signals)

    def consume_behavior_signals(self) -> dict[str, int]:
        signals = dict(self.pending_behavior_signals)
        self.pending_behavior_signals.clear()
        return signals

    def maybe_suppress_tool_compatible_state(
        self, state: str
    ) -> tuple[str, dict[str, int]]:
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
        strategy = _select_resend_strategy(
            comparable, previous_comparable, has_answer_facts
        )
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

    def _bump_signals(self, signals: dict[str, int]) -> None:
        for key, value in signals.items():
            self.pending_behavior_signals[key] = (
                self.pending_behavior_signals.get(key, 0) + value
            )
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

    def execute_tool_events_batch(
        self, events: list[NormalizedToolEvent]
    ) -> list[dict[str, Any]]:
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
        calculate_dropped_tokens: bool = False,
        recent_messages: list[dict[str, Any]] | None = None,
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
        """Build tool-compatible resend state and return processed memory, hints, signals."""
        if request.tool_compatible:
            pre_resend_memory = memory
            previous_comparable = dict(
                session._last_tool_compatible_state_fields
            )
            (
                _,
                comparable_state,
                has_answer_anchor,
            ) = _prepare_tool_compatible_state(
                pre_resend_memory, previous_comparable
            )

            # Use provided parameter or determine from signals
            if has_answer_anchor_param is not None:
                has_answer_anchor = has_answer_anchor_param

            resend_reason = _select_resend_reason(
                comparable_state, previous_comparable, has_answer_anchor
            )
            (
                processed_memory,
                resend_signals,
            ) = session.maybe_suppress_tool_compatible_state(memory)
            behavior_signals.update(
                {
                    key: behavior_signals.get(key, 0) + value
                    for key, value in resend_signals.items()
                }
            )
            _apply_tool_compatible_resend_diagnostics(
                behavior_signals,
                processed_memory,
                resend_signals,
                has_answer_anchor=has_answer_anchor,
                resend_reason=resend_reason,
                skip_reason_hint=skip_reason if should_skip_history else None,
                tok_history_compression_skipped=bool(
                    behavior_signals.get("tok_history_compression_skipped", 0)
                ),
                tool_compatible_compression=bool(
                    behavior_signals.get("tool_compatible_compression", 0)
                ),
            )
            if translated_messages is None:
                answer_ready = False
            else:
                answer_ready = _is_answer_ready_turn(
                    translated_messages,
                    tool_compatible=request.tool_compatible,
                    has_answer_anchor=behavior_signals.get(
                        "answer_anchor_present", 0
                    )
                    > 0,
                    baseline_only=session._baseline_only,
                )
            if answer_ready:
                behavior_signals["answer_ready_turn"] = 1
            if session._late_answer_followthrough_active:
                behavior_signals["late_answer_followthrough_active"] = 1
            if session._answer_ready_repair_active:
                behavior_signals["answer_ready_repair_active"] = 1
            if session._late_answer_assembly_repair_active:
                behavior_signals["late_answer_assembly_repair_active"] = 1
            _mark_late_answer_assembly_mode_signal(
                behavior_signals,
                session._late_answer_assembly_repair_mode_active,
            )
            runtime_hints.extend(
                _runtime_hints_for_turn(
                    answer_ready=answer_ready,
                    answer_ready_repair_active=session._answer_ready_repair_active,
                    late_answer_followthrough_active=session._late_answer_followthrough_active,
                    late_answer_assembly_repair_mode=session._late_answer_assembly_repair_mode_active,
                )
            )
            if not runtime_hints:
                evidence_hints = session.evidence_intent_advisories()
                if evidence_hints:
                    runtime_hints.extend(evidence_hints)
            (
                hot_recent_hints,
                hot_metrics,
            ) = session.hot_recent_runtime_hints()
            if hot_recent_hints:
                runtime_hints.extend(hot_recent_hints)
                for key, value in hot_metrics.items():
                    hot_hint_metrics[key] = (
                        hot_hint_metrics.get(key, 0) + value
                    )
            _annotate_reacquisition_diagnostics(
                behavior_signals,
                answer_ready=answer_ready,
                answer_ready_repair_active=session._answer_ready_repair_active,
            )
            logger.debug(
                "tool-compatible resend: mode=%s payload_chars=%d anchor=%d",
                next(
                    (
                        k
                        for k in resend_signals
                        if k.startswith("state_resend_")
                    ),
                    "none",
                ),
                len(processed_memory),
                behavior_signals.get("answer_anchor_present", 0),
            )
            processed_body = _inject_system(
                {},  # Will be replaced by caller
                processed_memory,
                runtime_hints,
                tool_compatible=request.tool_compatible,
                grammar=bool(request.grammar),
                todo=request.todo or "",
                deltas=bool(request.deltas),
                pressure=current_pressure,
                behavior_signals=behavior_signals,
            )

            return (
                processed_memory,
                runtime_hints,
                behavior_signals,
                hot_hint_metrics,
                processed_body,
                resend_signals,
                answer_ready,
            )

        return (
            memory,
            runtime_hints,
            behavior_signals,
            hot_hint_metrics,
            {},
            {},
            False,
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
        contract = response_contract_for_mode(
            text, tool_compatible=tool_compatible, session=session
        )
        drift_signals = (
            self.semantic_validator.validate_drift(
                text, contract.behavior_signals
            )
            if not tool_compatible
            else {}
        )
        merged_signals: dict[str, int] = {
            **session.consume_behavior_signals(),
            **(behavior_signals or {}),
            **response_behavior_signals(text, tool_compatible=tool_compatible),
            **contract.behavior_signals,
            **drift_signals,
        }

        # Apply Self-Healing if drift is detected in non-Tok response
        healed_text = heal_drift(
            text, merged_signals, tool_compatible=tool_compatible
        )
        if healed_text != text:
            merged_signals["tok_drift_healed"] = 1
            # Re-evaluating contract with healed text ensures blocks are correctly extracted
            contract = response_contract_for_mode(
                healed_text, tool_compatible=tool_compatible, session=session
            )

        visible_text = "\n".join(
            cast(str, block.get("text", ""))
            for block in contract.content_blocks
            if block.get("type") == "text"
            and str(block.get("text", "")).strip()
        ).strip()
        has_tool = any(
            block.get("type") == "tool_use"
            for block in contract.content_blocks
        )
        has_answer_text = _is_answer_like_visible_text(visible_text)
        handle_answer_repair(
            session,
            merged_signals=merged_signals,
            has_tool=has_tool,
            has_answer_text=has_answer_text,
            tool_compatible=tool_compatible,
        )
        structured_fields = _process_answer_memory(session, visible_text)
        if structured_fields:
            for field, values in structured_fields.items():
                for value in values:
                    session.bridge_memory._upsert(
                        session.bridge_memory.hot,
                        field,
                        value,
                        score_delta=3,
                    )
                    if _should_persist_to_durable(field, value):
                        session.bridge_memory._upsert(
                            session.bridge_memory.durable,
                            field,
                            value,
                            score_delta=2,
                        )
            session._save_bridge_memory()

        should_write_healed_memory = not (
            tool_compatible and merged_signals.get("tok_drift_healed")
        )
        updated_memory = (
            session.write_memory(healed_text)
            if should_write_healed_memory
            else ""
        )
        if updated_memory:
            _attribute_macro_savings(session, updated_memory)
        family_mode = session.update_family_mode(model, merged_signals)

        # Telemetry: Protocol Drift
        report_protocol_drift(
            model=model,
            merged_signals=merged_signals,
            mode=contract.mode,
            session=session,
            content_blocks=contract.content_blocks,
        )

        # Update reasoning-depth accumulators
        session._step_count += 1
        session._token_count += count_tokens(text)
        for block in contract.content_blocks:
            if block.get("type") == "tool_use" and block.get("name"):
                session._tool_names_seen.add(cast(str, block["name"]))

        # Macro JIT Execution: Intercept and execute autonomous macros
        if (
            os.getenv("TOK_NEURO_REACTOR", "0") == "1"
            and "EXECUTE_JIT(@" in text
        ):
            import re

            jit_match = re.search(r"EXECUTE_JIT\(@(\w+)\((.*?)\)\)", text)
            if jit_match:
                m_name = jit_match.group(1)
                m_args_raw = jit_match.group(2)

                # Execute symbolically
                jit_result = execute_jit_macro(session, m_name, m_args_raw)

                # Append result to content_blocks
                contract.content_blocks.append(
                    {
                        "type": "text",
                        "text": f"\n\n[JIT Execution Result for @{m_name}]:\n{jit_result}",
                    }
                )

                # Add telemetry signals
                merged_signals["jit_executed"] = 1
                merged_signals[f"jit_macro_executed_{m_name}"] = 1

        session._drift_detected_previous_turn = bool(
            merged_signals.get("semantic_drift_detected")
            or merged_signals.get("non_tok_response")
        )

        return ProcessedRuntimeResponse(
            content_blocks=contract.content_blocks,
            output_saved_tokens=contract.output_saved_tokens,
            behavior_signals=merged_signals,
            mode=contract.mode,
            family_mode=family_mode,
            updated_memory=updated_memory,
        )

    def pressure_score(self, signals: dict[str, int]) -> int:
        return _semantic_pressure_score(signals)
