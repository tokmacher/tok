"""Model-family smart-zone policy for adaptive Tok compression."""

from __future__ import annotations

from dataclasses import dataclass

COMPRESSION_MODES = ("aggressive", "balanced", "recovery")

# Task type classification
TASK_TYPES = ("coding", "research", "mixed")

UNIVERSAL_MODE = "tok-universal"


@dataclass(frozen=True)
class ModelFamily:
    provider: str
    family: str

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.family}"


CANONICAL_WIRE_FIELD_ORDER: tuple[str, ...] = (
    "turns",
    "goal",
    "blockers",
    "files",
    "facts",
    "cmds",
    "tests",
    "errs",
    "constraints",
    "questions",
    "next",
)


@dataclass(frozen=True)
class MemoryProjectionProfile:
    field_limits: dict[str, int]
    question_limit: int = 0
    fact_limit: int = 0
    field_order: tuple[str, ...] = CANONICAL_WIRE_FIELD_ORDER


@dataclass(frozen=True)
class SmartZonePolicy:
    family: ModelFamily
    default_mode: str
    balanced_threshold: int
    recovery_threshold: int
    relax_threshold: int
    memory_profiles: dict[str, MemoryProjectionProfile]
    history_profiles: dict[str, dict[str, int | list[str]]]
    tool_levels: dict[str, str]


@dataclass
class FamilyAdaptiveState:
    mode: str
    recent_pressure: int = 0
    clean_streak: int = 0
    task_type: str = "mixed"  # coding, research, or mixed
    task_confidence: float = 0.0  # confidence score for task detection


def identify_model_family(model: str) -> ModelFamily:
    """Identify the model family from a model identifier string."""
    lowered = model.strip().lower()
    if lowered.startswith("google/") or "gemini" in lowered:
        return ModelFamily(provider="google", family="gemini")
    if lowered.startswith(("openai/", "gpt-")) or "/gpt-" in lowered:
        return ModelFamily(provider="openai", family="gpt")
    if lowered.startswith(("anthropic/", "claude-")) or "/claude-" in lowered:
        return ModelFamily(provider="anthropic", family="claude")
    if "deepseek" in lowered:
        return ModelFamily(provider="deepseek", family="deepseek")
    if "/" in lowered:
        provider, rest = lowered.split("/", 1)
        family = rest.split("-", 1)[0] or "unknown"
        return ModelFamily(provider=provider or "unknown", family=family)
    return ModelFamily(provider="unknown", family="unknown")


def policy_for_model(_model: str) -> SmartZonePolicy:
    """Return the smart zone policy for a given model."""
    family = ModelFamily(provider="universal", family="universal")
    return _make_policy(
        family,
        default_mode=UNIVERSAL_MODE,
        balanced_threshold=2,
        recovery_threshold=5,
    )


def initial_state(policy: SmartZonePolicy) -> FamilyAdaptiveState:
    """Create initial adaptive state for a policy."""
    return FamilyAdaptiveState(mode=policy.default_mode)


def pressure_score(signals: dict[str, int]) -> int:
    """Calculate pressure score from behavior signals."""
    weights = {
        "repeat_file_read": 2,
        "repeat_search": 2,
        "python_c_workaround": 1,
        "stderr_workaround": 1,
        "non_tok_response": 2,
        "fail_open_compat_response": 2,
        "malformed_tok_response": 2,
        "tool_contract_failure": 2,
        "cold_start_wire_fallback": 1,
        "blocker_rediscovery": 2,
    }
    return sum(signals.get(key, 0) * weight for key, weight in weights.items())


def detect_task_type(
    tool_names: list[str],
    content_patterns: dict[str, int] | None = None,
) -> tuple[str, float]:
    """
    Detect task type (coding/research/mixed) from tool usage patterns.

    Returns (task_type, confidence) tuple.

    Coding indicators: file writes, edits, test runs, error handling
    Research indicators: searches, reads, exploration without modification
    """
    if not tool_names:
        return "mixed", 0.0

    coding_tools = {
        "write_file",
        "edit_file",
        "replace",
        "create_file",
        "run_shell",
        "execute",
        "pytest",
        "test",
        "apply_patch",
    }
    research_tools = {
        "grep_search",
        "search",
        "view_file",
        "read_file",
        "list_dir",
        "find_by_name",
        "code_search",
    }

    tool_set = {t.lower().replace("_", "").replace("-", "") for t in tool_names}
    coding_set = {t.lower().replace("_", "").replace("-", "") for t in coding_tools}
    research_set = {t.lower().replace("_", "").replace("-", "") for t in research_tools}

    coding_matches = len(tool_set & coding_set)
    research_matches = len(tool_set & research_set)
    total = len(tool_names)

    # Check content patterns for additional signals
    if content_patterns:
        if content_patterns.get("error_messages", 0) > 0:
            coding_matches += 2  # Errors suggest debugging/coding
        if content_patterns.get("questions_asked", 0) > 0:
            research_matches += 1  # Questions suggest research

    if coding_matches > research_matches * 1.5:
        confidence = min(1.0, coding_matches / max(total, 1))
        return "coding", confidence
    if research_matches > coding_matches * 1.5:
        confidence = min(1.0, research_matches / max(total, 1))
        return "research", confidence

    return "mixed", 0.5


def select_optimal_mode(model: str, task_type: str) -> str:
    """Select the single universal compression mode."""
    del model, task_type
    return UNIVERSAL_MODE


def advance_state(
    policy: SmartZonePolicy,
    state: FamilyAdaptiveState,
    signals: dict[str, int],
    tool_names: list[str] | None = None,
    content_patterns: dict[str, int] | None = None,
) -> FamilyAdaptiveState:
    """Advance state while keeping the runtime on the universal profile."""
    pressure = pressure_score(signals)
    clean_streak = state.clean_streak + 1 if pressure <= policy.relax_threshold else 0

    # Update task type detection if tool names provided
    task_type = state.task_type
    task_confidence = state.task_confidence
    if tool_names:
        detected_type, detected_conf = detect_task_type(tool_names, content_patterns)
        # Only update if confidence is higher than current
        if detected_conf > task_confidence:
            task_type = detected_type
            task_confidence = detected_conf

    return FamilyAdaptiveState(
        mode=policy.default_mode,
        recent_pressure=pressure,
        clean_streak=clean_streak,
        task_type=task_type,
        task_confidence=task_confidence,
    )


def _make_policy(
    family: ModelFamily,
    *,
    default_mode: str,
    balanced_threshold: int,
    recovery_threshold: int,
) -> SmartZonePolicy:
    """Create a smart zone policy for a model family."""
    universal_memory_profile = MemoryProjectionProfile(
        field_limits={
            "files": 3,
            "cmds": 4,
            "tests": 2,
            "errs": 2,
            "constraints": 2,
        },
        question_limit=2,
        fact_limit=4,
        field_order=CANONICAL_WIRE_FIELD_ORDER,
    )
    universal_history_profile: dict[str, int | list[str]] = {
        "files": 3,
        "cmds": 4,
        "tests": 2,
        "errs": 2,
        "constraints": 2,
        "questions": 2,
        "facts": 4,
        "_max_chars": 640,
        "_drop_priority": [
            "facts",
            "questions",
            "cmds",
            "tests",
            "files",
            "constraints",
            "goal",
            "next",
        ],
    }
    memory_profiles = {
        UNIVERSAL_MODE: universal_memory_profile,
        "aggressive": universal_memory_profile,
        "balanced": universal_memory_profile,
        "recovery": universal_memory_profile,
    }
    history_profiles: dict[str, dict[str, int | list[str]]] = {
        UNIVERSAL_MODE: universal_history_profile,
        "aggressive": universal_history_profile,
        "balanced": universal_history_profile,
        "recovery": universal_history_profile,
    }
    tool_levels = {
        UNIVERSAL_MODE: "balanced",
        "aggressive": "balanced",
        "balanced": "balanced",
        "recovery": "balanced",
    }
    return SmartZonePolicy(
        family=family,
        default_mode=default_mode,
        balanced_threshold=balanced_threshold,
        recovery_threshold=recovery_threshold,
        relax_threshold=0,
        memory_profiles=memory_profiles,
        history_profiles=history_profiles,
        tool_levels=tool_levels,
    )
