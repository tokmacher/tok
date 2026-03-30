from __future__ import annotations

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

    attribution = prepared.bloat_attribution

    assert (
        attribution["system_additions"]["directive_variant"]
        == "tool-compatible"
    )
    assert attribution["system_additions"]["tok_state_tokens"] == 0
    assert attribution["state_resend"]["mode"] == "none"


def test_live_bridge_scenarios_keep_second_warm_turn_no_larger_than_first():
    scenarios = measure_live_bridge_bloat_scenarios()

    first = scenarios["warm_unchanged_state_first"]["request_footprint"][
        "prepared"
    ]
    second = scenarios["warm_unchanged_state_second"]["request_footprint"][
        "prepared"
    ]
    second_mode = scenarios["warm_unchanged_state_second"]["state_resend"][
        "mode"
    ]

    assert second["system_tokens"] <= first["system_tokens"]
    assert second_mode in {"suppressed", "delta"}


def test_live_bridge_scenarios_suppress_answer_anchor_after_first_turn():
    scenarios = measure_live_bridge_bloat_scenarios()

    first = scenarios["answer_anchor_first"]["state_resend"]
    second = scenarios["answer_anchor_second"]["state_resend"]

    assert first["mode"] == "full"
    assert second["mode"] in {"suppressed", "delta"}
    assert (
        scenarios["answer_anchor_second"]["request_footprint"]["prepared"][
            "system_tokens"
        ]
        <= scenarios["answer_anchor_first"]["request_footprint"]["prepared"][
            "system_tokens"
        ]
    )


def test_live_bridge_scenarios_keep_tool_heavy_bridge_turns_compressed():
    scenarios = measure_live_bridge_bloat_scenarios()

    history_skip = scenarios["history_skip"]

    assert history_skip["history_retention"]["skipped"] is False
    assert (
        history_skip["behavior_signals"].get("tok_soft_tool_use_count_high", 0)
        == 1
    )
    assert history_skip["counterfactual"]["savings_tokens_vs_legacy_skip"] > 0


def test_live_bridge_hotfix_meets_bloat_reduction_targets():
    scenarios = measure_live_bridge_bloat_scenarios()

    history_total = scenarios["history_skip"]["request_footprint"]["prepared"][
        "total_tokens"
    ]
    retained_tool_tokens = scenarios["moderate_coding"][
        "tool_result_retention"
    ]["tokens"]

    assert history_total <= int(LEGACY_HISTORY_SKIP_TOTAL_TOKENS * 0.3)
    assert retained_tool_tokens <= int(
        LEGACY_MODERATE_TOOL_RETENTION_TOKENS * 0.6
    )


def test_live_bridge_suspects_rank_tool_retention_as_avoidable():
    suspects = rank_live_bridge_bloat_suspects(
        measure_live_bridge_bloat_scenarios()
    )
    names = {suspect["name"]: suspect for suspect in suspects}

    assert names["Retained recent tool results after compression"][
        "classification"
    ] == ("likely accidental / too eager")


def test_live_bridge_report_marks_strict_pressure_as_opt_out_risk():
    report = generate_live_bridge_bloat_report()
    strict = report["scenarios"]["strict_pressure"]
    suspect = next(
        item
        for item in report["ranked_suspects"]
        if item["name"] == "Strict-mode protocol law in bridge opt-out path"
    )

    assert strict["counterfactual"]["protocol_law_delta_tokens"] > 0
    assert suspect["classification"] == "regression risk"
