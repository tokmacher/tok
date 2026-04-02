"""Model-family smart-zone policy for adaptive Tok compression."""

from __future__ import annotations

from dataclasses import dataclass

COMPRESSION_MODES = ("aggressive", "balanced", "recovery")


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


def identify_model_family(model: str) -> ModelFamily:
    lowered = model.strip().lower()
    if lowered.startswith("google/") or "gemini" in lowered:
        return ModelFamily(provider="google", family="gemini")
    if (
        lowered.startswith("openai/")
        or lowered.startswith("gpt-")
        or "/gpt-" in lowered
    ):
        return ModelFamily(provider="openai", family="gpt")
    if (
        lowered.startswith("anthropic/")
        or lowered.startswith("claude-")
        or "/claude-" in lowered
    ):
        return ModelFamily(provider="anthropic", family="claude")
    if "/" in lowered:
        provider, rest = lowered.split("/", 1)
        family = rest.split("-", 1)[0] or "unknown"
        return ModelFamily(provider=provider or "unknown", family=family)
    return ModelFamily(provider="unknown", family="unknown")


def policy_for_model(model: str) -> SmartZonePolicy:
    family = identify_model_family(model)
    if family.key == "anthropic:claude":
        return _make_policy(
            family,
            default_mode="aggressive",
            balanced_threshold=2,
            recovery_threshold=5,
        )
    if family.key == "openai:gpt":
        return _make_policy(
            family,
            default_mode="aggressive",
            balanced_threshold=2,
            recovery_threshold=5,
        )
    if family.key == "google:gemini":
        return _make_policy(
            family,
            default_mode="balanced",
            balanced_threshold=2,
            recovery_threshold=5,
        )
    return _make_policy(
        family,
        default_mode="balanced",
        balanced_threshold=2,
        recovery_threshold=5,
    )


def initial_state(policy: SmartZonePolicy) -> FamilyAdaptiveState:
    return FamilyAdaptiveState(mode=policy.default_mode)


def pressure_score(signals: dict[str, int]) -> int:
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


def advance_state(
    policy: SmartZonePolicy,
    state: FamilyAdaptiveState,
    signals: dict[str, int],
) -> FamilyAdaptiveState:
    pressure = pressure_score(signals)
    clean_streak = (
        state.clean_streak + 1 if pressure <= policy.relax_threshold else 0
    )

    if pressure >= policy.recovery_threshold:
        return FamilyAdaptiveState(
            mode="recovery", recent_pressure=pressure, clean_streak=0
        )
    if pressure >= policy.balanced_threshold:
        return FamilyAdaptiveState(
            mode="balanced", recent_pressure=pressure, clean_streak=0
        )
    if clean_streak >= 2:
        target = policy.default_mode
        if state.mode != target:
            return FamilyAdaptiveState(
                mode=target,
                recent_pressure=pressure,
                clean_streak=clean_streak,
            )
    return FamilyAdaptiveState(
        mode=state.mode, recent_pressure=pressure, clean_streak=clean_streak
    )


def _make_policy(
    family: ModelFamily,
    *,
    default_mode: str,
    balanced_threshold: int,
    recovery_threshold: int,
) -> SmartZonePolicy:
    memory_profiles = {
        "aggressive": MemoryProjectionProfile(
            field_limits={
                "files": 2,
                "cmds": 1,
                "tests": 1,
                "errs": 2,
                "constraints": 1,
            },
            question_limit=1,
            fact_limit=1,
            field_order=CANONICAL_WIRE_FIELD_ORDER,
        ),
        # tok-minimal mode: preserve more context than aggressive to improve task success
        "minimal": MemoryProjectionProfile(
            field_limits={
                "files": 3,
                "cmds": 2,
                "tests": 2,
                "errs": 2,
                "constraints": 2,
            },
            question_limit=2,
            fact_limit=3,
            field_order=CANONICAL_WIRE_FIELD_ORDER,
        ),
        "balanced": MemoryProjectionProfile(
            field_limits={
                "files": 3,
                "cmds": 4,
                "tests": 2,
                "errs": 2,
                "constraints": 2,
            },
            question_limit=2,
            fact_limit=2,
            field_order=CANONICAL_WIRE_FIELD_ORDER,
        ),
        "recovery": MemoryProjectionProfile(
            field_limits={
                "files": 4,
                "cmds": 3,
                "tests": 3,
                "errs": 3,
                "constraints": 2,
            },
            question_limit=2,
            fact_limit=3,
            field_order=CANONICAL_WIRE_FIELD_ORDER,
        ),
    }
    history_profiles: dict[str, dict[str, int | list[str]]] = {
        "aggressive": {
            "files": 2,
            "cmds": 1,
            "tests": 1,
            "errs": 2,
            "constraints": 1,
            "questions": 1,
            "facts": 2,
            "_max_chars": 520,
            "_drop_priority": [
                "facts",
                "questions",
                "constraints",
                "tests",
                "cmds",
                "files",
                "goal",
                "next",
            ],
        },
        # tok-minimal mode: preserve more context, especially goal and files
        "minimal": {
            "files": 3,
            "cmds": 2,
            "tests": 2,
            "errs": 2,
            "constraints": 2,
            "questions": 2,
            "facts": 3,
            "_max_chars": 600,
            "_drop_priority": [
                "questions",
                "constraints",
                "tests",
                "cmds",
                "facts",
                "errs",
                "files",
                "goal",
                "next",
            ],
        },
        "balanced": {
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
        },
        "recovery": {
            "files": 4,
            "cmds": 3,
            "tests": 3,
            "errs": 3,
            "constraints": 2,
            "questions": 2,
            "facts": 4,
            "_max_chars": 760,
            "_drop_priority": [
                "facts",
                "cmds",
                "questions",
                "tests",
                "constraints",
                "files",
                "goal",
                "next",
            ],
        },
    }
    tool_levels = {
        "aggressive": "aggressive",
        "balanced": "balanced",
        "recovery": "recovery",
        "minimal": "aggressive",  # Use aggressive tool compression but keep more memory
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
