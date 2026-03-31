"""Phase 6 verification: Macro ROI tracking and hypothesis promotion."""

from __future__ import annotations

import datetime
import pytest

from tok.neuro.ir import Instruction, Macro, MacroRegistry
from tok.bridge_memory import BridgeMemoryState, MemoryEntry


@pytest.fixture(autouse=True)
def isolate_macro_registry(monkeypatch):
    monkeypatch.setattr(
        MacroRegistry, "load_global", lambda self, *a, **kw: None
    )
    monkeypatch.setattr(
        MacroRegistry, "save_global", lambda self, *a, **kw: None
    )


# ---------------------------------------------------------------------------
# ROI field serialization
# ---------------------------------------------------------------------------


def test_macro_roi_fields_round_trip():
    """lifetime_savings and avg_tokens_per_use survive to_dict / from_dict."""
    macro = Macro(
        name="m0",
        instructions=(Instruction(op="view", args=("src/foo.py",)),),
        inputs=("p0",),
        lifetime_savings=120,
        avg_tokens_per_use=40.0,
    )
    restored = Macro.from_dict(macro.to_dict())
    assert restored.lifetime_savings == 120
    assert restored.avg_tokens_per_use == 40.0


def test_macro_roi_defaults_to_zero():
    """Macros loaded without ROI data default to zero without error."""
    data = {
        "name": "m1",
        "instructions": [{"op": "cat", "args": [], "target": None}],
        "inputs": [],
        "hit_count": 1,
        "last_seen": None,
        "is_durable": False,
    }
    macro = Macro.from_dict(data)
    assert macro.lifetime_savings == 0
    assert macro.avg_tokens_per_use == 0.0


# ---------------------------------------------------------------------------
# record_savings
# ---------------------------------------------------------------------------


def test_record_savings_accumulates():
    registry = MacroRegistry()
    macro = Macro(
        name="m0",
        instructions=(Instruction(op="pytest", args=("-v",)),),
        inputs=(),
        hit_count=3,
    )
    registry.macros["m0"] = macro

    registry.record_savings("m0", 30)
    assert macro.lifetime_savings == 30
    assert macro.avg_tokens_per_use == 10.0  # 30 / 3

    registry.record_savings("m0", 60)
    assert macro.lifetime_savings == 90
    assert macro.avg_tokens_per_use == 30.0  # 90 / 3


def test_record_savings_ignores_unknown_macro():
    registry = MacroRegistry()
    registry.record_savings("nonexistent", 50)  # must not raise


def test_record_savings_ignores_non_positive():
    registry = MacroRegistry()
    macro = Macro(
        name="m0",
        instructions=(Instruction(op="ls", args=()),),
        inputs=(),
    )
    registry.macros["m0"] = macro
    registry.record_savings("m0", 0)
    registry.record_savings("m0", -5)
    assert macro.lifetime_savings == 0


# ---------------------------------------------------------------------------
# apply_decay: ROI protection
# ---------------------------------------------------------------------------


def _old_macro(name: str, hits: int, savings: int = 0) -> Macro:
    """Helper: create a macro that looks stale (last_seen 30 days ago)."""
    stale_date = (
        datetime.datetime.now() - datetime.timedelta(days=30)
    ).isoformat()
    return Macro(
        name=name,
        instructions=(Instruction(op="view", args=()),),
        inputs=(),
        hit_count=hits,
        last_seen=stale_date,
        lifetime_savings=savings,
    )


def test_apply_decay_prunes_low_roi_stale_macro():
    registry = MacroRegistry()
    registry.macros["stale"] = _old_macro("stale", hits=1, savings=0)
    registry.apply_decay(max_age_days=7, min_hits=3)
    assert "stale" not in registry.macros


def test_apply_decay_spares_high_roi_stale_macro():
    """A macro that saved >= ROI_PROTECTION_THRESHOLD tokens must survive decay."""
    registry = MacroRegistry()
    registry.macros["valuable"] = _old_macro(
        "valuable", hits=1, savings=MacroRegistry.ROI_PROTECTION_THRESHOLD
    )
    registry.apply_decay(max_age_days=7, min_hits=3)
    assert "valuable" in registry.macros


def test_apply_decay_spares_durable_macro_regardless_of_roi():
    registry = MacroRegistry()
    stale_date = (
        datetime.datetime.now() - datetime.timedelta(days=30)
    ).isoformat()
    macro = Macro(
        name="durable_m",
        instructions=(Instruction(op="cat", args=()),),
        inputs=(),
        hit_count=0,
        last_seen=stale_date,
        is_durable=True,
        lifetime_savings=0,
    )
    registry.macros["durable_m"] = macro
    registry.apply_decay()
    assert "durable_m" in registry.macros


# ---------------------------------------------------------------------------
# Hypothesis promotion (bridge_memory.py)
# ---------------------------------------------------------------------------


def test_hypothesis_promotion_clears_answered_question():
    """A fact with high word-overlap to a question should promote fact to durable
    and remove the question from hot memory."""
    state = BridgeMemoryState()

    # Add a question about the release_summary export path
    state.hot["questions"] = [
        MemoryEntry(
            value="where does release_summary get written to the export path?",
            score=2,
            last_seen_turn=1,
        )
    ]
    # Add a fact that clearly answers it
    state.hot["facts"] = [
        MemoryEntry(
            value="release_summary export path is export/release_summary.json",
            score=3,
            last_seen_turn=2,
        )
    ]

    metrics = state._promote_facts_for_questions()

    # Question should be cleared from hot
    assert "questions" not in state.hot or not state.hot["questions"]
    # Fact should be promoted to durable
    assert any(
        "release_summary" in e.value for e in state.durable.get("facts", [])
    )
    assert metrics.get("hypothesis_promotions", 0) >= 1


def test_hypothesis_promotion_preserves_unanswered_questions():
    """Questions with no matching fact must remain in hot."""
    state = BridgeMemoryState()
    state.hot["questions"] = [
        MemoryEntry(
            value="what is the retry policy for the API?",
            score=1,
            last_seen_turn=1,
        )
    ]
    state.hot["facts"] = [
        MemoryEntry(
            value="release_summary is gated by savings_pct threshold",
            score=2,
            last_seen_turn=2,
        )
    ]

    state._promote_facts_for_questions()

    assert "questions" in state.hot
    assert len(state.hot["questions"]) == 1


def test_hypothesis_promotion_via_ingest_wire_state():
    """End-to-end: ingest_wire_state triggers promotion when a fact answers a question."""
    state = BridgeMemoryState()
    # Pre-load a question
    state.hot["questions"] = [
        MemoryEntry(
            value="where is the gateway compression logic?",
            score=2,
            last_seen_turn=1,
        )
    ]

    # Ingest a wire state that includes a fact answering the question
    state.ingest_wire_state(
        ">>> t:2|g:refactor|"
        "facts:gateway compression logic lives in src/tok/compression.py"
    )

    # Question should be resolved
    assert not state.hot.get("questions"), (
        "Question should be cleared after fact answers it"
    )
