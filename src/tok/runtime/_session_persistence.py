"""Persistence helpers for runtime session state."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time as time_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import RESULT_CACHE_TTL_SECONDS
from .memory.bridge_memory import BridgeMemoryState
from .memory.session_state import _discover_project_markers
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
    provided_fallback_memory = session.fallback_memory
    # Save the load_global_macros flag BEFORE overwriting bridge_memory
    load_global_macros_flag = session.bridge_memory.load_global_macros
    if not load_global_macros_flag:
        session._load_global_macros = False
    else:
        session._load_global_macros = not explicit_memory_dir
    session.bridge_memory = load_bridge_memory(session)
    session.result_cache = load_result_cache(session)
    session.episode_ledger = load_episode_ledger(session)
    warm_records = load_hot_summaries(session)
    if warm_records:
        session._hot_summary_records.update(warm_records)
        session._hot_hints_loaded_from_disk = len(warm_records)
    loaded_fallback_memory = load_fallback_memory(session)
    session.fallback_memory = provided_fallback_memory or loaded_fallback_memory
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
        session._persistence_failures += 1
        session_logger = session_logger_for(session)
        session_logger.warning(
            "Bridge memory file corrupted at %s: %s — starting with empty memory",
            path,
            exc,
        )
    except Exception as exc:
        session._persistence_failures += 1
        session_logger = session_logger_for(session)
        session_logger.warning(
            "Failed to load bridge memory from %s: %s",
            path,
            exc,
        )
    return BridgeMemoryState(load_global_macros=session._load_global_macros)


def save_bridge_memory(session: RuntimeSession) -> None:
    """Persist bridge memory state to disk atomically."""
    tmp_path = None
    try:
        assert session.memory_dir is not None
        session.memory_dir.mkdir(parents=True, exist_ok=True)
        target_path = bridge_memory_file(session)
        content = session.bridge_memory.to_tok()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_path.parent,
            prefix=".tmp_",
            suffix=".tok",
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
        os.replace(tmp_path, target_path)
    except Exception as exc:
        session._persistence_failures += 1
        session_logger = session_logger_for(session)
        session_logger.warning(
            "Failed to save bridge memory to %s: %s",
            bridge_memory_file(session),
            exc,
        )
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


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
            cleaned: dict[str, Any] = {}
            current_time = time_module.time()
            for key, value in result.items():
                if isinstance(value, list) and len(value) == 3:
                    _cached_hash, _raw, timestamp = value
                    if current_time - timestamp < RESULT_CACHE_TTL_SECONDS:
                        cleaned[key] = {
                            "hash": str(_cached_hash) if _cached_hash else "",
                            "raw": str(_raw) if _raw else "",
                            "timestamp": timestamp,
                            "first_read_complete": False,
                        }
                elif isinstance(value, list) and len(value) == 2:
                    upgraded: dict[str, Any] = {
                        "hash": str(value[0]) if value[0] else "",
                        "raw": str(value[1]) if len(value) > 1 else "",
                        "timestamp": current_time,
                        "first_read_complete": False,
                    }
                    cleaned[key] = upgraded
                elif isinstance(value, dict):
                    # Reset first_read_complete for every entry loaded from a prior
                    # session.  The persistent cache exists to avoid re-hashing; it
                    # does NOT grant compression rights at the start of a new session.
                    # Without this reset, cross-session cache hits skip verbatim
                    # delivery and return a skeleton stub on the very first Read.
                    entry: dict[str, Any] = dict(value)
                    entry["first_read_complete"] = False
                    cleaned[key] = entry
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

        def _trim_raw(raw_val: str) -> str:
            if isinstance(raw_val, str) and len(raw_val) > 10240:
                session_logger_for(session).warning(
                    "Result cache entry truncated from %d to 10240 chars during save",
                    len(raw_val),
                )
                return raw_val[:10240]
            return raw_val

        trimmed: dict[str, Any] = {}
        for key, value in session.result_cache.items():
            if isinstance(value, dict):
                trimmed[key] = {
                    **value,
                    "raw": _trim_raw(value.get("raw", "")),
                    "timestamp": value.get("timestamp"),
                }
            elif isinstance(value, tuple | list) and len(value) >= 2:
                trimmed[key] = [
                    value[0],
                    _trim_raw(value[1]),
                    *value[2:],
                ]
            else:
                trimmed[key] = value
        result_cache_file(session).write_text(json.dumps(trimmed))
    except Exception as exc:
        session._persistence_failures += 1
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
        content = session.fallback_memory
        if len(content) > 100_000:
            content = content[-100_000:]
        fallback_memory_file(session).write_text(content + "\n")
    except Exception as exc:
        session._persistence_failures += 1
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
        session._persistence_failures += 1
        session_logger_for(session).warning(
            "Failed to save episode ledger to %s: %s",
            episode_ledger_file(session),
            exc,
        )


_HOT_SUMMARIES_TTL_SECONDS: int = 3 * 3600  # 3 hours


def hot_summaries_file(session: RuntimeSession) -> Path:
    """Return the path to the hot summaries file for this session."""
    assert session.memory_dir is not None
    return session.memory_dir / "hot_summaries.tok"


def load_hot_summaries(session: RuntimeSession) -> dict[str, Any]:
    """Load hot summary records from disk, normalising turn counters for the new session."""
    from .repeat_targets import EvidenceIntent, HotSummaryRecord

    path = hot_summaries_file(session)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return {}
        current_time = time_module.time()
        records: dict[str, HotSummaryRecord] = {}
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            timestamp = value.get("_timestamp", 0)
            if current_time - timestamp > _HOT_SUMMARIES_TTL_SECONDS:
                continue
            evidence_intent: EvidenceIntent | None = None
            ei_data = value.get("evidence_intent")
            if isinstance(ei_data, dict):
                try:
                    evidence_intent = EvidenceIntent(**{k: v for k, v in ei_data.items() if k != "_timestamp"})
                except Exception:
                    pass
            record = HotSummaryRecord(
                tool_family=str(value.get("tool_family", "")),
                logical_target=str(value.get("logical_target", "")),
                display_target=str(value.get("display_target", "")),
                summary=str(value.get("summary", "")),
                token_cost=int(value.get("token_cost", 0)),
                result_digest=str(value.get("result_digest", "")),
                last_seen_turn=int(value.get("last_seen_turn", 0)),
                exact_evidence_key=str(value.get("exact_evidence_key", "")),
                # Normalise promotion turns: eligible from turn 1 in the new session
                hot_promotion_turn=1 if int(value.get("hot_promotion_turn", 0)) > 0 else 0,
                stuck_promotion_turn=1 if int(value.get("stuck_promotion_turn", 0)) > 0 else 0,
                last_injected_turn=0,  # must re-fire in the new session
                repeat_count=int(value.get("repeat_count", 0)),
                recent_window_count=int(value.get("recent_window_count", 0)),
                stuck_window_count=int(value.get("stuck_window_count", 0)),
                unchanged_result_count=int(value.get("unchanged_result_count", 0)),
                evidence_intent=evidence_intent,
                skeleton=str(value.get("skeleton", "")),
                tokens_saved=int(value.get("tokens_saved", 0)),
            )
            records[key] = record
        return records
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        session_logger_for(session).warning("Hot summaries corrupted at %s: %s — starting empty", path, exc)
    except Exception as exc:
        session_logger_for(session).warning("Failed to load hot summaries from %s: %s", path, exc)
    return {}


def save_hot_summaries(session: RuntimeSession) -> None:
    """Persist hot summary records to disk."""
    try:
        assert session.memory_dir is not None
        session.memory_dir.mkdir(parents=True, exist_ok=True)
        current_time = time_module.time()
        records: dict[str, Any] = {}
        for key, record in session._hot_summary_records.items():
            d = dataclasses.asdict(record)
            d["_timestamp"] = current_time
            records[key] = d
        path = hot_summaries_file(session)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records))
        tmp.replace(path)
    except Exception as exc:
        session._persistence_failures += 1
        session_logger_for(session).warning(
            "Failed to save hot summaries to %s: %s",
            hot_summaries_file(session),
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
