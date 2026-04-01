"""Configuration and constants for the Tok runtime."""

import os

from .policy.smart_policy import MemoryProjectionProfile

TTL_SECONDS = {"1h": 3600, "30m": 1800, "15m": 900, "5m": 300, "1m": 60}

RESULT_CACHE_TTL_SECONDS: int = int(os.getenv("TOK_RESULT_CACHE_TTL", "1800"))

# Consecutive fail-open events in a session that trigger automatic baseline degradation.
_FALLBACK_THRESHOLD: int = int(os.getenv("TOK_FALLBACK_THRESHOLD", "3"))

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
TOOL_COMPAT_DELTA_KEYS = ("turns", "goal", "files", "tests", "errs", "facts")
TOOL_COMPAT_STICKY_KEYS = ("files", "tests")
TOOL_COMPAT_MAX_FILES = 2

TOK_REACQUIRE_TRIGGER_COUNT: int = int(
    os.getenv("TOK_REACQUIRE_TRIGGER_COUNT", "2")
)
TOK_REACQUIRE_STUCK_COUNT: int = int(
    os.getenv("TOK_REACQUIRE_STUCK_COUNT", "3")
)
TOK_REACQUIRE_WINDOW_TURNS: int = int(
    os.getenv("TOK_REACQUIRE_WINDOW_TURNS", "6")
)
TOK_REACQUIRE_STUCK_WINDOW_TURNS: int = int(
    os.getenv("TOK_REACQUIRE_STUCK_WINDOW_TURNS", "8")
)
TOK_HOT_RECENT_MAX_HINTS: int = int(os.getenv("TOK_HOT_RECENT_MAX_HINTS", "2"))
TOK_HOT_FILE_MAX_LINES: int = int(os.getenv("TOK_HOT_FILE_MAX_LINES", "12"))
TOK_HOT_FILE_MAX_CHARS: int = int(os.getenv("TOK_HOT_FILE_MAX_CHARS", "400"))
TOK_HOT_SEARCH_MAX_LINES: int = int(os.getenv("TOK_HOT_SEARCH_MAX_LINES", "6"))
TOK_HOT_SEARCH_MAX_CHARS: int = int(
    os.getenv("TOK_HOT_SEARCH_MAX_CHARS", "300")
)
TOK_HOT_COMMAND_MAX_LINES: int = int(
    os.getenv("TOK_HOT_COMMAND_MAX_LINES", "6")
)
TOK_HOT_COMMAND_MAX_CHARS: int = int(
    os.getenv("TOK_HOT_COMMAND_MAX_CHARS", "300")
)
TOK_PREDICTIVE_CACHE_TOP_K: int = int(
    os.getenv("TOK_PREDICTIVE_CACHE_TOP_K", "3")
)

# Runtime repair and followthrough hints
ANSWER_READY_RUNTIME_HINT = (
    "Answer now using the existing File=/Verification= evidence. "
    "Do not call tools in this turn."
)
ANSWER_READY_REPAIR_HINT = (
    "Previous turn failed answer assembly. You already read the target files —"
    " their content is in your context or was replaced by @stable_result (meaning unchanged)."
    " Reply now with only File=... and Verification=..."
    " If you cannot locate the file content in context, state which files you need re-sent."
    " If verbatim bytes are truly required in a later turn, emit @tok_bypass_next_read"
    " immediately before the next supported read tool call."
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
    " If verbatim bytes are truly required in a later turn, emit @tok_bypass_next_read"
    " immediately before the next supported read tool call."
)
LATE_ANSWER_FOLLOWTHROUGH_HINT = (
    "Previous turn gathered the required evidence. In this turn, do not call "
    "tools. Reply only with File=... and Verification=..."
)

TOK_NOVELTY_REQUIRED_HINT = (
    "You already have evidence for {anchor}. Reuse it unless you need "
    "a different revision, line range, or query scope."
)
TOK_NEIGHBORHOOD_THRASH_HINT = (
    "You've explored multiple files in {neighborhood} without new findings. "
    "Synthesize what you have or choose one genuinely new target."
)
TOK_LARGE_FILE_HINT = (
    "For files >500 lines, use session.explore_file(path) or session.get_file_overview(path) "
    "to get a Tok-formatted overview before reading specific sections with Read(offset=X, limit=Y)."
)
TOK_NEIGHBORHOOD_TRIGGER_ANCHORS: int = int(
    os.getenv("TOK_NEIGHBORHOOD_TRIGGER_ANCHORS", "3")
)
TOK_NEIGHBORHOOD_WINDOW_TURNS: int = int(
    os.getenv("TOK_NEIGHBORHOOD_WINDOW_TURNS", "6")
)

_TOOL_REQUIRED_PROMPT_PATTERNS = (
    "fresh evidence",
    "use the read-only tools first",
    "gather it before answering",
    "do one narrow search",
    "do one direct file read",
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

# Word-count thresholds for drift detection
DRIFT_SHORT_TEXT_WORDS: int = 10
DRIFT_LONG_TEXT_WORDS: int = 40
DRIFT_BULLET_LINE_THRESHOLD: int = 2

# Diff ratio threshold for result cache
DIFF_RATIO_THRESHOLD: float = 0.7

__all__ = [
    "TTL_SECONDS",
    "RESULT_CACHE_TTL_SECONDS",
    "_FALLBACK_THRESHOLD",
    "_PROJECT_MARKER_FILES",
    "TOOL_DENSITY_THRESHOLD",
    "TOOL_VOLUME_HEAVY_BYTES",
    "TOOL_COMPAT_MEMORY_PROFILE",
    "TOOL_COMPAT_DELTA_KEYS",
    "TOOL_COMPAT_STICKY_KEYS",
    "TOOL_COMPAT_MAX_FILES",
    "TOK_REACQUIRE_TRIGGER_COUNT",
    "TOK_REACQUIRE_STUCK_COUNT",
    "TOK_REACQUIRE_WINDOW_TURNS",
    "TOK_REACQUIRE_STUCK_WINDOW_TURNS",
    "TOK_HOT_RECENT_MAX_HINTS",
    "TOK_HOT_FILE_MAX_LINES",
    "TOK_HOT_FILE_MAX_CHARS",
    "TOK_HOT_SEARCH_MAX_LINES",
    "TOK_HOT_SEARCH_MAX_CHARS",
    "TOK_HOT_COMMAND_MAX_LINES",
    "TOK_HOT_COMMAND_MAX_CHARS",
    "TOK_PREDICTIVE_CACHE_TOP_K",
    "ANSWER_READY_RUNTIME_HINT",
    "ANSWER_READY_REPAIR_HINT",
    "LATE_ANSWER_ASSEMBLY_TOOL_ONLY_REPAIR_HINT",
    "LATE_ANSWER_ASSEMBLY_ANSWER_ONLY_REPAIR_HINT",
    "LATE_ANSWER_FOLLOWTHROUGH_HINT",
    "TOK_NOVELTY_REQUIRED_HINT",
    "TOK_NEIGHBORHOOD_THRASH_HINT",
    "TOK_LARGE_FILE_HINT",
    "TOK_NEIGHBORHOOD_TRIGGER_ANCHORS",
    "TOK_NEIGHBORHOOD_WINDOW_TURNS",
    "_TOOL_REQUIRED_PROMPT_PATTERNS",
    "SNIPPET_TRUNCATION_LIMIT",
    "DIGEST_TRUNCATION_LIMIT",
    "HYPOTHESIS_TRUNCATION_LIMIT",
    "BLOCKER_TRUNCATION_LIMIT",
    "TOOL_RESULT_DENSITY_THRESHOLD",
    "TOOL_USE_DENSITY_THRESHOLD",
    "FILE_READ_DENSITY_THRESHOLD",
    "COMMAND_DENSITY_THRESHOLD",
    "HEAVY_RESULT_THRESHOLD",
    "HEAVY_RESULT_SKIP_THRESHOLD",
    "DRIFT_SHORT_TEXT_WORDS",
    "DRIFT_LONG_TEXT_WORDS",
    "DRIFT_BULLET_LINE_THRESHOLD",
    "DIFF_RATIO_THRESHOLD",
]
