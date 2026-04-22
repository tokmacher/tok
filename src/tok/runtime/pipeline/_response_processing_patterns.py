"""Regex patterns and stopwords used by response processing."""

from __future__ import annotations

import re

STRUCTURED_LABEL_RE = re.compile(r"(?<![\w-])(file|verification|related)(?![\w-])\s*[:=]\s*([^\n|]+)", re.IGNORECASE)
STRUCTURED_FIELD_NAMES = ("file", "verification", "related")
IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
PATH_RE = re.compile(r"(?:^|[\s`'\"])((?:src/)?[A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|go|rs|rb))(?:[:#]L?\d+)?")
TOOL_INTENT_TEXT_RE = re.compile(
    r"(@tool\b|tool_use\b|\"type\"\s*:\s*\"tool_use\"|'type'\s*:\s*'tool_use'|\bcall(?:ing)?\s+(?:the\s+)?tool\b)",
    re.IGNORECASE,
)

VERIFICATION_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "class",
    "for",
    "from",
    "function",
    "in",
    "is",
    "line",
    "method",
    "of",
    "on",
    "or",
    "result",
    "the",
    "to",
    "via",
    "with",
}
