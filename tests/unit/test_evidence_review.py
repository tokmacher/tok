import json

from tok.evidence_review import (
    build_coverage_report,
    load_stress_evidence,
    rank_candidates,
    review_capture_dir,
    summarize_capture_file,
)


def test_summarize_capture_file_reads_response_quality_and_savings(tmp_path):
    session = tmp_path / "session.jsonl"
    session.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "request",
                        "model": "claude-sonnet-4",
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                ),
                json.dumps(
                    {
                        "event": "response",
                        "model": "claude-sonnet-4",
                        "baseline_only": False,
                        "fallback_count": 1,
                        "session_quality": "watch",
                        "session_savings_pct": 42.0,
                        "last_degradation_reason": "context reacquisition",
                    }
                ),
            ]
        )
        + "\n"
    )

    summary = summarize_capture_file(session)

    assert summary["dominant_model"] == "claude-sonnet-4"
    assert summary["verdict"] == "watch"
    assert summary["positive_savings"] is True
    assert summary["candidate_label"] == "context reacquisition"


def test_review_capture_dir_filters_and_aggregates(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text(
        json.dumps(
            {
                "event": "response",
                "model": "claude-sonnet-4",
                "baseline_only": False,
                "fallback_count": 0,
                "session_quality": "clean",
                "last_degradation_reason": "",
            }
        )
        + "\n"
    )
    second.write_text(
        json.dumps(
            {
                "event": "response",
                "model": "gpt-4.1-mini",
                "baseline_only": True,
                "fallback_count": 2,
                "session_quality": "degraded",
                "last_degradation_reason": "baseline fallback",
            }
        )
        + "\n"
    )

    review = review_capture_dir(tmp_path, verdict="investigate")

    assert review["aggregate"]["total_sessions"] == 1
    assert review["aggregate"]["verdict_counts"]["investigate"] == 1
    assert review["sessions"][0]["name"] == "b.jsonl"


def test_rank_candidates_boosts_cross_model_and_investigate_sessions():
    sessions = [
        {
            "name": "a.jsonl",
            "models": ["claude-sonnet-4"],
            "verdict": "watch",
            "positive_savings": True,
            "fallback_count": 1,
            "repeat_search_count": 0,
            "repeat_file_read_count": 0,
            "candidate_label": "baseline fallback",
        },
        {
            "name": "b.jsonl",
            "models": ["gpt-4.1-mini"],
            "verdict": "investigate",
            "positive_savings": True,
            "fallback_count": 1,
            "repeat_search_count": 0,
            "repeat_file_read_count": 0,
            "candidate_label": "baseline fallback",
        },
    ]

    ranked = rank_candidates(sessions)

    assert ranked[0]["candidate"] == "baseline fallback"
    assert ranked[0]["score"] >= 6
    assert "gpt-4.1-mini" in ranked[0]["affected_models"]


def test_load_stress_evidence_maps_breakpoint_classes(tmp_path):
    stress_dir = tmp_path / "stress"
    stress_dir.mkdir()
    (stress_dir / "breakpoints.json").write_text(
        json.dumps(
            [
                {"breakpoint_class": "protocol_drift"},
                {"breakpoint_class": "reacquisition_loop"},
                {"breakpoint_class": "reacquisition_loop"},
            ]
        )
    )

    loaded = load_stress_evidence(stress_dir)

    assert loaded["candidate_counts"]["response contract drift"] == 1
    assert loaded["candidate_counts"]["context reacquisition"] == 2


def test_build_coverage_report_marks_required_exploratory_and_uncovered(
    tmp_path,
):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for name in (
        "runtime_conformance",
        "release_reacquisition",
        "cache_stable_research_turns",
    ):
        (replay_dir / f"{name}.jsonl.meta.json").write_text("{}")
    gate_config = tmp_path / "gate-config.json"
    gate_config.write_text(
        json.dumps(
            {
                "required_fixtures": [
                    "runtime_conformance",
                    "alternating_adapters",
                    "release_reacquisition",
                ]
            }
        )
    )

    sessions = [
        {"candidate_label": "response contract drift"},
        {"candidate_label": "context reacquisition"},
        {"candidate_label": "baseline fallback"},
    ]
    stress = {"candidate_counts": {"answer anchor retention": 1}}

    coverage = build_coverage_report(
        sessions,
        stress_evidence=stress,
        replay_dir=replay_dir,
        gate_config_path=gate_config,
    )
    by_name = {item["candidate"]: item for item in coverage}

    assert (
        by_name["response contract drift"]["coverage_status"]
        == "already covered by release fixture"
    )
    assert (
        by_name["context reacquisition"]["coverage_status"]
        == "already covered by release fixture"
    )
    assert (
        by_name["answer anchor retention"]["coverage_status"]
        == "already covered by exploratory fixture"
    )
    assert by_name["baseline fallback"]["coverage_status"] == "not yet covered"
