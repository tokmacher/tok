"""Configuration and constants for the Tok runtime."""

import logging
import os

from .policy.smart_policy import MemoryProjectionProfile

logger = logging.getLogger("tok.runtime.config")


def _env_int(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer config %s=%r; using fallback %d", name, raw, fallback)
        return fallback


TTL_SECONDS = {"1h": 3600, "30m": 1800, "15m": 900, "5m": 300, "1m": 60}

RESULT_CACHE_TTL_SECONDS: int = _env_int("TOK_RESULT_CACHE_TTL", 1800)

# Consecutive fail-open events in a session that trigger automatic baseline degradation.
_FALLBACK_THRESHOLD: int = _env_int("TOK_FALLBACK_THRESHOLD", 3)

# Short session threshold: sessions with fewer turns use baseline mode to avoid overhead.
_SHORT_SESSION_THRESHOLD: int = _env_int("TOK_SHORT_SESSION_THRESHOLD", 8)

_HOT_HINT_MIN_TURN: int = _env_int("TOK_HOT_HINT_MIN_TURN", 10)

_SHORT_MEMORY_TURN_CEILING: int = _env_int("TOK_SHORT_MEMORY_TURN_CEILING", 14)

# Known project-type marker filenames used for Local Mesh Discovery.
_PROJECT_MARKER_FILES: frozenset[str] = frozenset(
    {
        "package.json",
        "go.mod",
        "Cargo.toml",
        "requirements.txt",
        "pyproject.toml",
        "pom.xml",
        "build.gradle",
        "setup.py",
        ".git",
        "composer.json",
        "Gemfile",
        "mix.exs",
    }
)

# Deterministic thresholds for tool-density/volume-based compression decisions.
TOOL_DENSITY_THRESHOLD: float = 0.35
TOOL_VOLUME_HEAVY_BYTES: int = 4_000

TOOL_COMPAT_MEMORY_PROFILE = MemoryProjectionProfile(
    field_limits={
        "turns": 1,
        "goal": 1,
        "next": 0,
        "files": 2,
        "cmds": 0,
        "tests": 1,
        "errs": 1,
        "constraints": 0,
        "blockers": 0,
    },
    question_limit=0,
    fact_limit=2,
)

TOOL_COMPAT_MEMORY_PROFILE_SHORT = MemoryProjectionProfile(
    field_limits={
        "turns": 1,
        "goal": 1,
    },
    question_limit=0,
    fact_limit=0,
)
TOOL_COMPAT_DELTA_KEYS = ("turns", "goal", "files", "tests", "errs", "facts")
TOOL_COMPAT_STICKY_KEYS = ("files", "tests")
TOOL_COMPAT_MAX_FILES = 2

TOK_REACQUIRE_TRIGGER_COUNT: int = _env_int("TOK_REACQUIRE_TRIGGER_COUNT", 2)
TOK_REACQUIRE_STUCK_COUNT: int = _env_int("TOK_REACQUIRE_STUCK_COUNT", 3)
TOK_REACQUIRE_WINDOW_TURNS: int = _env_int("TOK_REACQUIRE_WINDOW_TURNS", 6)
TOK_REACQUIRE_STUCK_WINDOW_TURNS: int = _env_int("TOK_REACQUIRE_STUCK_WINDOW_TURNS", 8)
TOK_HOT_RECENT_MAX_HINTS: int = _env_int("TOK_HOT_RECENT_MAX_HINTS", 2)
TOK_HOT_FILE_MAX_LINES: int = _env_int("TOK_HOT_FILE_MAX_LINES", 12)
TOK_HOT_FILE_MAX_CHARS: int = _env_int("TOK_HOT_FILE_MAX_CHARS", 400)
TOK_HOT_SEARCH_MAX_LINES: int = _env_int("TOK_HOT_SEARCH_MAX_LINES", 6)
TOK_HOT_SEARCH_MAX_CHARS: int = _env_int("TOK_HOT_SEARCH_MAX_CHARS", 300)
TOK_HOT_COMMAND_MAX_LINES: int = _env_int("TOK_HOT_COMMAND_MAX_LINES", 6)
TOK_HOT_COMMAND_MAX_CHARS: int = _env_int("TOK_HOT_COMMAND_MAX_CHARS", 300)
TOK_PREDICTIVE_CACHE_TOP_K: int = _env_int("TOK_PREDICTIVE_CACHE_TOP_K", 3)
TOK_REQUEST_POLICY_STICKY_TURNS: int = _env_int("TOK_REQUEST_POLICY_STICKY_TURNS", 3)
TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS: int = _env_int("TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS", 2)
TOK_REQUEST_POLICY_TOOL_DENSE_ASSISTANT_TURNS: int = _env_int("TOK_REQUEST_POLICY_TOOL_DENSE_ASSISTANT_TURNS", 2)

# Runtime repair and followthrough hints
ANSWER_READY_RUNTIME_HINT = (
    "Answer now using the existing File=/Verification= evidence. Do not call tools in this turn."
)
ANSWER_READY_REPAIR_HINT = (
    "Previous turn failed answer assembly. You already read the target files —"
    " their content is in your context or was replaced by @stable_result (meaning unchanged)."
    " Reply now with only File=... and Verification=..."
    " If you cannot locate the file content in context, state which files you need re-sent."
    " Do not call tools in this turn."
)
LATE_ANSWER_ASSEMBLY_TOOL_ONLY_REPAIR_HINT = (
    "Previous turn tried to answer before satisfying the late fresh-evidence "
    "contract. In this turn, use only one supported read-only tool and do not answer."
)
LATE_ANSWER_ASSEMBLY_ANSWER_ONLY_REPAIR_HINT = (
    "Previous turn failed final answer assembly after evidence was available."
    " The evidence was already read — check your context or @stable_result hashes (meaning unchanged)."
    " In this turn, do not call tools. Reply only with File=... and Verification=..."
    " If content is missing from context, say so explicitly."
)
LATE_ANSWER_FOLLOWTHROUGH_HINT = (
    "Previous turn gathered the required evidence. In this turn, do not call "
    "tools. Reply only with File=... and Verification=..."
)

TOK_NOVELTY_REQUIRED_HINT = (
    "You already have evidence for {anchor}. Reuse it unless you need a different revision, line range, or query scope."
)
TOK_NEIGHBORHOOD_THRASH_HINT = (
    "You've explored multiple files in {neighborhood} without new findings. "
    "Synthesize what you have or choose one genuinely new target."
)
TOK_LARGE_FILE_HINT = (
    "For files >500 lines, use session.explore_file(path) or session.get_file_overview(path) "
    "to get a Tok-formatted overview before reading specific sections with Read(offset=X, limit=Y)."
)
TOK_READ_PLAN_HINT = (
    "If a turn needs many file reads, do not fan out raw reads. Emit a compact ReadPlan first: "
    "targets, chunk order, and one-sentence synthesis goal. Keep parallel file opens small and "
    "summarize before the next batch."
)
TOK_STABLE_RESULT_INFO_HINT = "@stable_result(hash:...) means a previously seen tool result is unchanged."
TOK_REPEAT_COMMAND_SUPPRESSION_HINT = (
    "You just ran an identical command and got the same successful result. "
    "Do not rerun it unless files or constraints changed; proceed to synthesis or next distinct step."
)
TOK_NEIGHBORHOOD_TRIGGER_ANCHORS: int = _env_int("TOK_NEIGHBORHOOD_TRIGGER_ANCHORS", 3)
TOK_NEIGHBORHOOD_WINDOW_TURNS: int = _env_int("TOK_NEIGHBORHOOD_WINDOW_TURNS", 6)
TOK_TOOL_REQUIRED_LATCH_THRESHOLD: int = _env_int("TOK_TOOL_REQUIRED_LATCH_THRESHOLD", 2)
TOK_RUNTIME_HINT_COOLDOWN_TURNS: int = _env_int("TOK_RUNTIME_HINT_COOLDOWN_TURNS", 2)

_TOOL_REQUIRED_PROMPT_PATTERNS = (
    "fresh evidence",
    "use the read-only tools first",
    "gather it before answering",
    "do one narrow search",
    "do one direct file read",
    "required before finalizing",
    "continue using the allowed tools",
    "read/edit/test steps",
    "run the necessary read/edit/test steps",
    "before finalizing",
)

# Truncation limits for compact state fields and digests
SNIPPET_TRUNCATION_LIMIT: int = 96
DIGEST_TRUNCATION_LIMIT: int = 160
HYPOTHESIS_TRUNCATION_LIMIT: int = 100
BLOCKER_TRUNCATION_LIMIT: int = 150

# Tool density thresholds for history rewrite decisions
TOOL_RESULT_DENSITY_THRESHOLD: int = 8
TOOL_USE_DENSITY_THRESHOLD: int = 10
FILE_READ_DENSITY_THRESHOLD: int = 6
COMMAND_DENSITY_THRESHOLD: int = 5
HEAVY_RESULT_THRESHOLD: int = 6
HEAVY_RESULT_SKIP_THRESHOLD: int = 4

RUNTIME_HINTS_MAX_PER_TURN: int = _env_int("TOK_RUNTIME_HINTS_MAX_PER_TURN", 3)

# Word-count thresholds for drift detection
DRIFT_SHORT_TEXT_WORDS: int = 10
DRIFT_LONG_TEXT_WORDS: int = 40
DRIFT_BULLET_LINE_THRESHOLD: int = 2

# Diff ratio threshold for result cache
DIFF_RATIO_THRESHOLD: float = 0.7

TOK_FILE_DELIVERY_STALE_TURNS: int = _env_int("TOK_FILE_DELIVERY_STALE_TURNS", 2)

# Force file codec: when enabled (1), treat all file-read tool results as cacheable
# by the file codec, producing @stable_result/@stable_summary stubs aggressively.
TOK_FORCE_FILE_CODEC: bool = os.getenv("TOK_FORCE_FILE_CODEC", "0") == "1"

# Repair loop detection: when enabled (1, default), detect and break loop patterns.
TOK_LOOP_DETECTION_ENABLED: bool = os.getenv("TOK_LOOP_DETECTION_ENABLED", "1") == "1"
TOK_LOOP_DETECTION_THRESHOLD: int = _env_int("TOK_LOOP_DETECTION_THRESHOLD", 3)

# Compression feature flags (off by default for conservative rollout).
TOK_ENABLE_PYTEST_FAIL_COMPRESSION: bool = os.getenv("TOK_ENABLE_PYTEST_FAIL_COMPRESSION", "0") == "1"
TOK_ENABLE_JSON_NONEXPANSION_GUARD: bool = os.getenv("TOK_ENABLE_JSON_NONEXPANSION_GUARD", "0") == "1"
TOK_ENABLE_FILE_OVERLAP_DELTA: bool = os.getenv("TOK_ENABLE_FILE_OVERLAP_DELTA", "0") == "1"
TOK_ENABLE_FILE_REREAD_DIFF: bool = os.getenv("TOK_ENABLE_FILE_REREAD_DIFF", "0") == "1"
TOK_ENABLE_SEARCH_OVERLAP_DELTA: bool = os.getenv("TOK_ENABLE_SEARCH_OVERLAP_DELTA", "0") == "1"
TOK_ENABLE_STACK_REPEAT_DELTA: bool = os.getenv("TOK_ENABLE_STACK_REPEAT_DELTA", "0") == "1"

__all__ = [
    "ANSWER_READY_REPAIR_HINT",
    "ANSWER_READY_RUNTIME_HINT",
    "BLOCKER_TRUNCATION_LIMIT",
    "COMMAND_DENSITY_THRESHOLD",
    "DIFF_RATIO_THRESHOLD",
    "DIGEST_TRUNCATION_LIMIT",
    "DRIFT_BULLET_LINE_THRESHOLD",
    "DRIFT_LONG_TEXT_WORDS",
    "DRIFT_SHORT_TEXT_WORDS",
    "FILE_READ_DENSITY_THRESHOLD",
    "HEAVY_RESULT_SKIP_THRESHOLD",
    "HEAVY_RESULT_THRESHOLD",
    "HYPOTHESIS_TRUNCATION_LIMIT",
    "LATE_ANSWER_ASSEMBLY_ANSWER_ONLY_REPAIR_HINT",
    "LATE_ANSWER_ASSEMBLY_TOOL_ONLY_REPAIR_HINT",
    "LATE_ANSWER_FOLLOWTHROUGH_HINT",
    "RESULT_CACHE_TTL_SECONDS",
    "RUNTIME_HINTS_MAX_PER_TURN",
    "SNIPPET_TRUNCATION_LIMIT",
    "TOK_FILE_DELIVERY_STALE_TURNS",
    "TOK_FORCE_FILE_CODEC",
    "TOK_HOT_COMMAND_MAX_CHARS",
    "TOK_HOT_COMMAND_MAX_LINES",
    "TOK_HOT_FILE_MAX_CHARS",
    "TOK_HOT_FILE_MAX_LINES",
    "TOK_HOT_RECENT_MAX_HINTS",
    "TOK_LOOP_DETECTION_ENABLED",
    "TOK_LOOP_DETECTION_THRESHOLD",
    "TOK_HOT_SEARCH_MAX_CHARS",
    "TOK_HOT_SEARCH_MAX_LINES",
    "TOK_LARGE_FILE_HINT",
    "TOK_NEIGHBORHOOD_THRASH_HINT",
    "TOK_NEIGHBORHOOD_TRIGGER_ANCHORS",
    "TOK_NEIGHBORHOOD_WINDOW_TURNS",
    "TOK_NOVELTY_REQUIRED_HINT",
    "TOK_PREDICTIVE_CACHE_TOP_K",
    "TOK_REACQUIRE_STUCK_COUNT",
    "TOK_REACQUIRE_STUCK_WINDOW_TURNS",
    "TOK_REACQUIRE_TRIGGER_COUNT",
    "TOK_REACQUIRE_WINDOW_TURNS",
    "TOK_READ_PLAN_HINT",
    "TOK_REPEAT_COMMAND_SUPPRESSION_HINT",
    "TOK_RUNTIME_HINT_COOLDOWN_TURNS",
    "TOK_TOOL_REQUIRED_LATCH_THRESHOLD",
    "TOK_REQUEST_POLICY_RECOVERY_WATCH_TURNS",
    "TOK_REQUEST_POLICY_STICKY_TURNS",
    "TOK_REQUEST_POLICY_TOOL_DENSE_ASSISTANT_TURNS",
    "TOK_ENABLE_FILE_OVERLAP_DELTA",
    "TOK_ENABLE_FILE_REREAD_DIFF",
    "TOK_ENABLE_JSON_NONEXPANSION_GUARD",
    "TOK_ENABLE_PYTEST_FAIL_COMPRESSION",
    "TOK_ENABLE_SEARCH_OVERLAP_DELTA",
    "TOK_ENABLE_STACK_REPEAT_DELTA",
    "TOK_STABLE_RESULT_INFO_HINT",
    "TOOL_COMPAT_DELTA_KEYS",
    "TOOL_COMPAT_MAX_FILES",
    "TOOL_COMPAT_MEMORY_PROFILE",
    "TOOL_COMPAT_MEMORY_PROFILE_SHORT",
    "TOOL_COMPAT_STICKY_KEYS",
    "TOOL_DENSITY_THRESHOLD",
    "TOOL_RESULT_DENSITY_THRESHOLD",
    "TOOL_USE_DENSITY_THRESHOLD",
    "TOOL_VOLUME_HEAVY_BYTES",
    "TTL_SECONDS",
    "_FALLBACK_THRESHOLD",
    "_PROJECT_MARKER_FILES",
    "_HOT_HINT_MIN_TURN",
    "_SHORT_MEMORY_TURN_CEILING",
    "_SHORT_SESSION_THRESHOLD",
    "_TOOL_REQUIRED_PROMPT_PATTERNS",
]
