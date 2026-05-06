"""Compression-layer feature flags — self-contained, no runtime imports."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("tok.compression.flags")


def _env_int(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer config %s=%r; using fallback %d", name, raw, fallback)
        return fallback


TOK_FORCE_FILE_CODEC: bool = os.getenv("TOK_FORCE_FILE_CODEC", "0") == "1"

TOK_ENABLE_PYTEST_FAIL_COMPRESSION: bool = os.getenv("TOK_ENABLE_PYTEST_FAIL_COMPRESSION", "0") == "1"
TOK_ENABLE_JSON_NONEXPANSION_GUARD: bool = os.getenv("TOK_ENABLE_JSON_NONEXPANSION_GUARD", "0") == "1"
TOK_ENABLE_FILE_OVERLAP_DELTA: bool = os.getenv("TOK_ENABLE_FILE_OVERLAP_DELTA", "0") == "1"
TOK_ENABLE_FILE_REREAD_DIFF: bool = os.getenv("TOK_ENABLE_FILE_REREAD_DIFF", "0") == "1"
TOK_ENABLE_SEARCH_OVERLAP_DELTA: bool = os.getenv("TOK_ENABLE_SEARCH_OVERLAP_DELTA", "1") == "1"
TOK_ENABLE_STACK_REPEAT_DELTA: bool = os.getenv("TOK_ENABLE_STACK_REPEAT_DELTA", "0") == "1"

RESULT_CACHE_TTL_SECONDS: int = _env_int("TOK_RESULT_CACHE_TTL", 1800)
