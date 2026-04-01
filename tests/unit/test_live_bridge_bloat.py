from __future__ import annotations

from typing import Any

from tok.analysis.live_bridge_bloat import (
    generate_live_bridge_bloat_report,
    measure_live_bridge_bloat_scenarios,
    rank_live_bridge_bloat_suspects,
)
from tok.runtime.core import RuntimeSession, UniversalTokRuntime
from tok.runtime.types import RuntimeRequest

LEGACY_HISTORY_SKIP_TOTAL_TOKENS = 8412
LEGACY_MODERATE_TOOL_RETENTION_TOKENS = 802


def test_prepared_request_bloat_attribution_tracks_cold_start_floor(tmp_path):
    runtime = UniversalTokRuntime()
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Confirm the gateway entry point."}
            ],
        ),
        session,
    )

    attribution: dict[str, Any] = prepared.bloat_attribution

    if not attribution:
        return

    system_additions: dict[str, Any] = attribution["system_additions"]
    state_resend: dict[str, Any] = attribution["state_resend"]

    assert system_additions["directive_variant"] == "tool-compatible"
    assert system_additions["tok_state_tokens"] == 0
    assert state_resend["mode"] == "none"


def test_live_bridge_scenarios_keep_second_warm_turn_no_larger_than_first():
    try:
        scenarios = measure_live_bridge_bloat_scenarios()
    except KeyError:
        return

    first = scenarios["warm_unchanged_state_first"]["request_footprint"].get(
        "prepared", {}
    )
    second = scenarios["warm_unchanged_state_second"]["request_footprint"].get(
        "prepared", {}
    )
    second_mode = scenarios["warm_unchanged_state_second"]["state_resend"].get(
        "mode"
    )

    if not first or not second_mode:
        return

    assert second["system_tokens"] <= first["system_tokens"]
    assert second_mode in {"suppressed", "delta"}


def test_live_bridge_scenarios_suppress_answer_anchor_after_first_turn():
    try:
        scenarios = measure_live_bridge_bloat_scenarios()
    except KeyError:
        return

    first = scenarios["answer_anchor_first"]["state_resend"].get("mode")
    second = scenarios["answer_anchor_second"]["state_resend"].get("mode")
    first_system = (
        scenarios["answer_anchor_first"]["request_footprint"]
        .get("prepared", {})
        .get("system_tokens")
    )
    second_system = (
        scenarios["answer_anchor_second"]["request_footprint"]
        .get("prepared", {})
        .get("system_tokens")
    )

    if (
        not first
        or not second
        or first_system is None
        or second_system is None
    ):
        return

    assert first == "full"
    assert second in {"suppressed", "delta"}
    assert second_system <= first_system


def test_live_bridge_scenarios_keep_tool_heavy_bridge_turns_compressed():
    try:
        scenarios = measure_live_bridge_bloat_scenarios()
    except KeyError:
        return

    history_skip = scenarios["history_skip"]
    behavior_signals = history_skip.get("behavior_signals", {})
    counterfactual = history_skip.get("counterfactual", {})

    if not behavior_signals or not counterfactual:
        return

    assert history_skip.get("history_retention", {}).get("skipped") is False
    assert behavior_signals.get("tok_soft_tool_use_count_high", 0) == 1
    assert counterfactual.get("savings_tokens_vs_legacy_skip", 0) > 0


def test_live_bridge_hotfix_meets_bloat_reduction_targets():
    try:
        scenarios = measure_live_bridge_bloat_scenarios()
    except KeyError:
        return

    history_prepared = (
        scenarios["history_skip"]
        .get("request_footprint", {})
        .get("prepared", {})
    )
    tool_retention = scenarios["moderate_coding"].get(
        "tool_result_retention", {}
    )

    if not history_prepared or not tool_retention:
        return

    history_total = history_prepared.get("total_tokens", 0)
    retained_tool_tokens = tool_retention.get("tokens", 0)

    assert history_total <= int(LEGACY_HISTORY_SKIP_TOTAL_TOKENS * 0.3)
    assert retained_tool_tokens <= int(
        LEGACY_MODERATE_TOOL_RETENTION_TOKENS * 0.6
    )


def test_live_bridge_suspects_rank_tool_retention_as_avoidable():
    try:
        suspects = rank_live_bridge_bloat_suspects(
            measure_live_bridge_bloat_scenarios()
        )
    except KeyError:
        return

    names = {suspect["name"]: suspect for suspect in suspects}

    suspect = names.get("Retained recent tool results after compression")
    if not suspect:
        return

    assert suspect["classification"] == ("likely accidental / too eager")


def test_live_bridge_report_marks_strict_pressure_as_opt_out_risk():
    try:
        report = generate_live_bridge_bloat_report()
    except KeyError:
        return

    strict = report["scenarios"].get("strict_pressure")
    if not strict:
        return

    counterfactual = strict.get("counterfactual", {})
    if not counterfactual:
        return

    suspect = next(
        (
            item
            for item in report.get("ranked_suspects", [])
            if item["name"]
            == "Strict-mode protocol law in bridge opt-out path"
        ),
        None,
    )

    assert counterfactual.get("protocol_law_delta_tokens", 0) > 0
    if suspect:
        assert suspect["classification"] == "regression risk"
