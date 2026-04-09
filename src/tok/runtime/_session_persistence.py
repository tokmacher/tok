"""Persistence helpers for runtime session state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import RESULT_CACHE_TTL_SECONDS
from .memory.bridge_memory import BridgeMemoryState
from .memory.session_helpers import _discover_project_markers
from .types import EpisodeEntry, EpisodeLedger

if TYPE_CHECKING:
    import logging

    from .core import RuntimeSession


def initialize_session_storage(session: RuntimeSession, *, explicit_memory_dir: bool) -> None:
    """Initialize session storage directories and load persisted state."""
    if session.memory_dir is None:
        project_dir = os.getenv("TOK_PROJECT_DIR", "")
        if project_dir:
            session.memory_dir = Path(project_dir) / ".tok"
        else:
            session.memory_dir = Path.home() / ".tok"
    # Save the load_global_macros flag BEFORE overwriting bridge_memory
    load_global_macros_flag = session.bridge_memory.load_global_macros
    if not load_global_macros_flag:
        session._load_global_macros = False
    else:
        session._load_global_macros = not explicit_memory_dir
    session.bridge_memory = load_bridge_memory(session)
    session.result_cache = load_result_cache(session)
    session.fallback_memory = load_fallback_memory(session)
    if session.fallback_memory and not session.bridge_memory.wire_state():
        session.bridge_memory.ingest_wire_state(session.fallback_memory)
        save_bridge_memory(session)
    if explicit_memory_dir:
        session._project_markers = frozenset()
    else:
        session._project_markers = _discover_project_markers()
        for marker in session._project_markers:
            session.bridge_memory.bump_file_heat(marker, weight=0.1)


def bridge_memory_file(session: RuntimeSession) -> Path:
    """Return the path to the bridge memory file for this session."""
    assert session.memory_dir is not None
    return session.memory_dir / "bridge_memory.tok"


def load_bridge_memory(session: RuntimeSession) -> BridgeMemoryState:
    """Load bridge memory state from disk, or return empty state if not found."""
    path = bridge_memory_file(session)
    if not path.exists():
        return BridgeMemoryState(load_global_macros=session._load_global_macros)
    try:
        return BridgeMemoryState.from_tok(
            path.read_text(),
            load_global_macros=session._load_global_macros,
        )
    except FileNotFoundError:
        return BridgeMemoryState(load_global_macros=session._load_global_macros)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        session_logger = session_logger_for(session)
        session_logger.warning(
            "Bridge memory file corrupted at %s: %s — starting with empty memory",
            path,
            exc,
        )
    except Exception as exc:
        session_logger = session_logger_for(session)
        session_logger.warning(
            "Failed to load bridge memory from %s: %s",
            path,
            exc,
        )
    return BridgeMemoryState(load_global_macros=session._load_global_macros)


def save_bridge_memory(session: RuntimeSession) -> None:
    """Persist bridge memory state to disk."""
    try:
        assert session.memory_dir is not None
        session.memory_dir.mkdir(parents=True, exist_ok=True)
        bridge_memory_file(session).write_text(session.bridge_memory.to_tok())
    except Exception as exc:
        session_logger = session_logger_for(session)
        session_logger.warning(
            "Failed to save bridge memory to %s: %s",
            bridge_memory_file(session),
            exc,
        )


def result_cache_file(session: RuntimeSession) -> Path:
    """Return the path to the result cache file for this session."""
    assert session.memory_dir is not None
    return session.memory_dir / "result_cache.tok"


def load_result_cache(session: RuntimeSession) -> dict[str, Any]:
    """Load result cache from disk, filtering expired entries."""
    import time as time_module

    path = result_cache_file(session)
    if not path.exists():
        return {}
    try:
        result = json.loads(path.read_text())
        if isinstance(result, dict):
            cleaned = {}
            current_time = time_module.time()
            for key, value in result.items():
                if isinstance(value, list) and len(value) == 3:
                    _cached_hash, _raw, timestamp = value
                    if current_time - timestamp < RESULT_CACHE_TTL_SECONDS:
                        cleaned[key] = value
                elif isinstance(value, list) and len(value) == 2:
                    cleaned[key] = value
                else:
                    cleaned[key] = value
            return cleaned
        session_logger_for(session).warning("Result cache at %s is not a dict — starting empty", path)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        session_logger_for(session).warning("Result cache corrupted at %s: %s — starting empty", path, exc)
    except Exception as exc:
        session_logger_for(session).warning("Failed to load result cache from %s: %s", path, exc)
    return {}


def save_result_cache(session: RuntimeSession) -> None:
    """Persist result cache to disk with size trimming."""
    try:
        assert session.memory_dir is not None
        session.memory_dir.mkdir(parents=True, exist_ok=True)
        trimmed = {
            key: (value[0], value[1][:10240], value[2])
            if isinstance(value, tuple) and len(value) == 3
            else (value[0], value[1][:10240])
            if isinstance(value, tuple) and len(value) == 2
            else value
            for key, value in session.result_cache.items()
        }
        result_cache_file(session).write_text(json.dumps(trimmed))
    except Exception as exc:
        session_logger_for(session).warning(
            "Failed to save result cache to %s: %s",
            result_cache_file(session),
            exc,
        )


def fallback_memory_file(session: RuntimeSession) -> Path:
    """Return the path to the fallback memory file for this session."""
    assert session.memory_dir is not None
    return session.memory_dir / "memory.tok"


def load_fallback_memory(session: RuntimeSession) -> str:
    """Load fallback memory content from disk, or return empty string if not found."""
    path = fallback_memory_file(session)
    if not path.exists():
        return ""
    try:
        return path.read_text().strip()
    except (UnicodeDecodeError, PermissionError) as exc:
        session_logger_for(session).warning("Failed to load fallback memory from %s: %s", path, exc)
    except Exception as exc:
        session_logger_for(session).warning("Failed to load fallback memory from %s: %s", path, exc)
    return ""


def save_fallback_memory(session: RuntimeSession) -> None:
    """Persist fallback memory content to disk."""
    try:
        assert session.memory_dir is not None
        session.memory_dir.mkdir(parents=True, exist_ok=True)
        fallback_memory_file(session).write_text(session.fallback_memory + "\n")
    except Exception as exc:
        session_logger_for(session).warning(
            "Failed to save fallback memory to %s: %s",
            fallback_memory_file(session),
            exc,
        )


def episode_ledger_file(session: RuntimeSession) -> Path:
    """Return the path to the episode ledger file for this session."""
    assert session.memory_dir is not None
    return session.memory_dir / "episode_ledger.tok"


def load_episode_ledger(session: RuntimeSession) -> EpisodeLedger:
    """Load episode ledger from disk, or return empty ledger if not found."""
    path = episode_ledger_file(session)
    if not path.exists():
        return EpisodeLedger()
    try:
        return EpisodeLedger.from_tok(path.read_text())
    except (ValueError, UnicodeDecodeError) as exc:
        session_logger_for(session).warning(
            "Episode ledger corrupted at %s: %s — starting empty",
            path,
            exc,
        )
    except Exception as exc:
        session_logger_for(session).warning("Failed to load episode ledger from %s: %s", path, exc)
    return EpisodeLedger()


def save_episode_ledger(session: RuntimeSession) -> None:
    """Persist episode ledger to disk."""
    try:
        assert session.memory_dir is not None
        session.memory_dir.mkdir(parents=True, exist_ok=True)
        episode_ledger_file(session).write_text(session.episode_ledger.to_tok())
    except Exception as exc:
        session_logger_for(session).warning(
            "Failed to save episode ledger to %s: %s",
            episode_ledger_file(session),
            exc,
        )


def record_episode(session: RuntimeSession, entry: EpisodeEntry) -> None:
    """Record an episode entry and persist the updated ledger."""
    session.episode_ledger.record(entry)
    save_episode_ledger(session)
    session._bump_signals({"episode_recorded": 1})


def session_logger_for(_session: RuntimeSession) -> logging.Logger:
    """Return the logger instance for the session module."""
    from .core import logger

    return logger
