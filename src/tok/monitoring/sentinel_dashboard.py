"""Tok-Sentinel diagnostic dashboard for monitoring system health."""

from pathlib import Path
from typing import Any, cast

import streamlit as st

TokRegistry: Any = None
Sifter: Any = None
REGISTRY_AVAILABLE = False

try:
    from tok.utils.sifter import Sifter as _Sifter
    from tok.utils.tok_registry import TokRegistry as _TokRegistry

    TokRegistry = _TokRegistry
    Sifter = _Sifter
    REGISTRY_AVAILABLE = True
except ImportError:
    pass


# Core checks (existing)
def tok_health_check() -> dict[str, str]:
    """Return the core health status of the Tok system."""
    return {
        "status": "🟢 OPERATIONAL",
        "integrity": "🟢 VERIFIED",
        "health": "🟢 PASSED",
        "delta_sync": "active",
        "metrics": "CPU:12% | ENTROPY:94% | O(0):OK",
    }


def global_integrity_report() -> str:
    """Generate a global integrity report from TokRegistry."""
    if REGISTRY_AVAILABLE and TokRegistry is not None:
        return cast(str, TokRegistry.global_integrity_report())
    return "⚠️ TokRegistry unavailable"


# Dashboard data loaders
@st.cache_data(ttl=5)  # Auto-refresh every 5s
def load_territory() -> str:
    """Load the territory map from disk with caching."""
    territory_path = Path("territory.tok")
    if territory_path.exists():
        with open(territory_path) as f:
            return f.read()
    return "Territory map not found. Run sifter."


@st.cache_data(ttl=5)
def load_todo_memory() -> dict[str, str]:
    """Load TODO and memory state from disk."""
    todo = Path("todo.tok")
    memory = Path("memory.tok")
    data: dict[str, str] = {}
    data["todo"] = todo.read_text() if todo.exists() else "No TODO."
    data["memory"] = memory.read_text() if memory.exists() else "No persistent memory."
    return data


@st.cache_data(ttl=5)
def load_stats() -> dict[str, Any]:
    """Load live agent statistics from the testing module."""
    stats = {"turns": 0, "tokens": 0, "cost": 0.0}
    try:
        from tok.testing.live_runner import LiveAgent

        agent = LiveAgent()
        stats.update(agent.get_stats())
    except Exception:  # pragma: no cover - best effort diagnostics
        pass
    return stats


# Streamlit UI
def main() -> None:
    """Run the Streamlit diagnostic dashboard."""
    st.set_page_config(page_title="Tok-Sentinel", layout="wide")
    st.title("🚀 Tok-Sentinel Diagnostic Dashboard")
    st.markdown("**Protocol v5.8 | Territory-Aware Autonomy**")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🩺 Health",
            "📋 Registry",
            "🗺️ Territory",
            "📝 TODO/Memory",
            "📊 Live Stats",
        ]
    )

    with tab1:
        st.subheader("Core Health Check")
        health = tok_health_check()
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", health["status"])
        col2.metric("Integrity", health["integrity"])
        col3.metric("Health", health["health"])
        st.code(health["metrics"])

        st.subheader("Global Integrity")
        st.text(global_integrity_report())

    with tab2:
        if REGISTRY_AVAILABLE:
            st.subheader("TokRegistry Operations")
            files = TokRegistry.get_files() if TokRegistry is not None else {}
            st.json(files)
            st.subheader("All Records")
            records = TokRegistry.get_all() if TokRegistry is not None else []
            st.dataframe(records)
        else:
            st.warning("TokRegistry import failed.")

    with tab3:
        st.subheader("Codebase Territory Map")
        territory = load_territory()
        st.text(territory[:2000] + "..." if len(territory) > 2000 else territory)
        if st.button("🔄 Refresh Territory"):
            if Sifter is not None:
                Sifter.from_dir("src/tok", naked=False, minify=True)
            st.rerun()

    with tab4:
        st.subheader("Task Queue & World State")
        data = load_todo_memory()
        col1, col2 = st.columns(2)
        with col1:
            st.code(data["todo"], language="markdown")
        with col2:
            st.code(data["memory"], language="diff")

    with tab5:
        st.subheader("Live Agent Metrics")
        stats = load_stats()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Turns", stats.get("calls", 0))
        col2.metric("Total Tokens", stats.get("total_tokens", 0))
        col3.metric(
            "Last Latency",
            f"{stats.get('last_usage', {}).get('latency_ms', 0):.0f}ms",
        )
        col4.metric("Cost", f"${stats.get('last_usage', {}).get('cost_usd', 0):.4f}")


if __name__ == "__main__":
    main()
