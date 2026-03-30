from __future__ import annotations

import os
import re

FILE_LIKE_TOOLS = frozenset(
    {
        "view",
        "view_file",
        "read",
        "read_file",
        "cat",
        "open_file",
        "get_file",
    }
)

EDIT_LIKE_TOOLS = frozenset(
    {
        "edit",
        "write",
        "edit_file",
        "write_file",
        "apply_patch",
        "str_replace_based_edit_tool",
    }
)

_ERROR_EQUIVALENCE_PATTERNS = [
    (
        re.compile(r"no such file|file not found|does not exist|enoent", re.I),
        "enoent",
    ),
    (re.compile(r"permission denied|access denied|eacces", re.I), "eacces"),
    (re.compile(r"not found|cannot find|could not find", re.I), "not_found"),
    (
        re.compile(r"regex.*error|error.*regex|invalid regex|bad regex", re.I),
        "regex_error",
    ),
    (
        re.compile(
            r"importerror|modulenotfounderror|no module named|import error",
            re.I,
        ),
        "import_error",
    ),
    (
        re.compile(r"syntaxerror|parse.?error|parse error", re.I),
        "syntax_error",
    ),
    (
        re.compile(
            r"attributeerror|has no attribute|no attribute|attribute error",
            re.I,
        ),
        "attr_error",
    ),
    (
        re.compile(r"typeerror|incompatible type|type error", re.I),
        "type_error",
    ),
    (re.compile(r"valueerror|invalid value|value error", re.I), "value_error"),
    (re.compile(r"timeout|timed out|deadline exceeded", re.I), "timeout"),
    (
        re.compile(
            r"connection.*refused|connection.*reset|network.*error", re.I
        ),
        "network_error",
    ),
    (
        re.compile(r"command failed|exit code [0-9]+|non-zero exit", re.I),
        "command_failed",
    ),
    (re.compile(r"already exists|file exists", re.I), "already_exists"),
    (re.compile(r"empty|zero bytes|no such", re.I), "empty_or_missing"),
]


CANONICAL_MEMORY_FIELDS = (
    "turns",
    "goal",
    "files",
    "edited",
    "cmds",
    "tests",
    "errs",
    "constraints",
    "next",
)

TOK_FIELD_ALIAS = {
    "turns": "t",
    "goal": "g",
    "next": "n",
    "files": "f",
    "cmds": "c",
    "errs": "e",
    "tests": "s",
    "blockers": "b",
    "facts": "x",
    "questions": "q",
    "constraints": "k",
    "episodes": "p",
    "edited": "i",
}

TOK_REVERSE_ALIAS = {v: k for k, v in TOK_FIELD_ALIAS.items()}

STOP_WORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "this",
        "that",
        "with",
        "have",
        "from",
        "are",
        "was",
        "not",
        "but",
        "they",
        "you",
        "what",
        "can",
        "will",
        "your",
        "just",
        "use",
        "also",
        "more",
        "its",
        "their",
    ]
)

QUESTION_PREFIXES = (
    "why ",
    "what ",
    "which ",
    "who ",
    "where ",
    "when ",
    "how ",
    "is ",
    "are ",
    "should ",
    "can ",
    "could ",
    "would ",
    "do ",
    "does ",
    "did ",
)

TOK_PROTOCOL_LAW = (
    "[Tok law] No JSON, no prose. Use @Tool name= |> body. SNAP: |> SNAP\n"
)

TOK_OUTPUT_DIRECTIVE = """\
[Tok Mode] >>> t:N|usr:X|agt:Y|state:Z
@msg role:assistant |> Reply
@Tool name=... |> args...
Turn 2+: Omit header and @msg if mode unchanged. No prose.
"""

TOK_TOOL_COMPAT_DIRECTIVE = (
    "Native tools only. Plain text. Omit all headers.\n"
)

TOK_OUTPUT_DIRECTIVE_MINIMAL = (
    ">>> t:N|usr:X|agt:Y|state:Z\n@msg role:assistant |> Reply\n"
)

TOK_OUTPUT_DIRECTIVE_REINFORCED = """\
[Tok — PROTOCOL REINFORCEMENT]
Recent responses show protocol drift. Strict compliance required:
1. Start every response with the >>> line.
2. Use @msg role:assistant |> for text.
3. Use @Tool name=... |> for tool args.
4. No raw markdown headers. No raw JSON tool blocks.
"""

# Appended to the system prompt when @stable_result tokens appear in history.
_STABLE_RESULT_EXPLANATION = (
    "@stable_result(hash:...) means the tool output is identical to a previous turn."
    " Treat it as: the state is unchanged, no new information."
)

# Minimum content length to be eligible for semantic hash deduplication.
_SEMANTIC_HASH_MIN_CHARS = int(os.getenv("TOK_SEMANTIC_HASH_MIN_CHARS", "200"))
