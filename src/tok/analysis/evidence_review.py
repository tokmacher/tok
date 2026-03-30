from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..runtime.pipeline.tool_processing import (
    build_tool_use_id_to_context,
    collect_behavior_signals,
)


CANDIDATE_LABELS = {
    "baseline fallback": "baseline fallback",
    "fail-open compatibility": "fail-open compatibility",
    "response contract drift": "response contract drift",
    "context reacquisition": "context reacquisition",
    "answer anchor retention": "answer anchor retention",
    "tool contract failure": "tool contract failure",
    "macro redundancy": "macro redundancy",
}

STRESS_BREAKPOINT_TO_CANDIDATE = {
    "baseline_fallback": "baseline fallback",
    "protocol_drift": "response contract drift",
    "reacquisition_loop": "context reacquisition",
    "retention_loss": "answer anchor retention",
    "compaction_loss": "answer anchor retention",
    "tool_contract_failure": "tool contract failure",
}

CANDIDATE_FIXTURE_MAP: dict[str, dict[str, list[str]]] = {
    "response contract drift": {
        "required": ["runtime_conformance"],
        "exploratory": [],
    },
    "context reacquisition": {
        "required": ["release_reacquisition"],
        "exploratory": ["refined_search_recovery"],
    },
    "answer anchor retention": {
        "required": [],
        "exploratory": ["cache_stable_research_turns"],
    },
    "fail-open compatibility": {
        "required": ["runtime_conformance"],
        "exploratory": [],
    },
    "baseline fallback": {
        "required": [],
        "exploratory": [],
    },
    "tool contract failure": {
        "required": [],
        "exploratory": [],
    },
    "macro redundancy": {
        "required": [],
        "exploratory": ["macro_reuse_rehearsal"],
    },
}


def _candidate_label_for_reason(reason: str) -> str:
    cleaned = reason.strip().lower()
    for known in CANDIDATE_LABELS:
        if cleaned == known:
            return CANDIDATE_LABELS[known]
    return reason.strip()


def _dominant_model(model_counts: Counter[str]) -> str:
    if not model_counts:
        return "unknown"
    return max(model_counts.items(), key=lambda item: (item[1], item[0]))[0]


def summarize_capture_file(path: Path) -> dict[str, Any]:
    request_count = 0
    model_counts: Counter[str] = Counter()
    repeat_search = 0
    repeat_file_read = 0
    fallback_events = 0
    baseline_only = False
    degradation_reason = ""
    session_quality = "clean"
    savings_pct = 0.0

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        event = str(record.get("event", "request"))
        if event == "request" or "messages" in record:
            request_count += 1
            model = str(record.get("model", "")).strip()
            if model:
                model_counts[model] += 1
            messages = record.get("messages", [])
            if isinstance(messages, list):
                context = build_tool_use_id_to_context(messages)
                signals = collect_behavior_signals(messages, context)
                repeat_search += int(signals.get("repeat_search", 0))
                repeat_file_read += int(signals.get("repeat_file_read", 0))
        elif event == "response":
            model = str(record.get("model", "")).strip()
            if model:
                model_counts[model] += 1
            baseline_only = baseline_only or bool(record.get("baseline_only"))
            fallback_events = max(
                fallback_events, int(record.get("fallback_count", 0))
            )
            response_signals = record.get("behavior_signals", {})
            if isinstance(response_signals, dict):
                fallback_events = max(
                    fallback_events,
                    int(response_signals.get("tok_fallback_activated", 0)),
                )
            reason = str(record.get("last_degradation_reason", "")).strip()
            if reason:
                degradation_reason = reason
            quality = str(record.get("session_quality", "")).strip()
            if quality:
                session_quality = quality
            savings_pct = max(
                savings_pct, float(record.get("session_savings_pct", 0.0))
            )

    verdict = "clean"
    if baseline_only or session_quality == "degraded":
        verdict = "investigate"
    elif (
        session_quality == "watch"
        or fallback_events > 0
        or repeat_search > 0
        or repeat_file_read > 0
        or degradation_reason
    ):
        verdict = "watch"

    candidate_label = (
        _candidate_label_for_reason(degradation_reason)
        if degradation_reason
        else ""
    )
    return {
        "path": str(path),
        "name": path.name,
        "dominant_model": _dominant_model(model_counts),
        "models": sorted(model_counts),
        "request_count": request_count,
        "verdict": verdict,
        "fallback_count": fallback_events,
        "repeat_search_count": repeat_search,
        "repeat_file_read_count": repeat_file_read,
        "degradation_reason": degradation_reason,
        "session_quality": session_quality,
        "savings_pct": round(savings_pct, 1),
        "positive_savings": savings_pct > 0,
        "candidate_label": candidate_label,
    }


def review_capture_dir(
    capture_dir: Path,
    *,
    verdict: str | None = None,
    reason_substring: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    summaries = [
        summarize_capture_file(path)
        for path in sorted(capture_dir.glob("*.jsonl"))
    ]
    if verdict:
        summaries = [item for item in summaries if item["verdict"] == verdict]
    if reason_substring:
        needle = reason_substring.lower()
        summaries = [
            item
            for item in summaries
            if needle in str(item["degradation_reason"]).lower()
        ]
    if limit is not None:
        summaries = summaries[: max(limit, 0)]

    verdict_counts = Counter(str(item["verdict"]) for item in summaries)
    reason_counts = Counter(
        str(item["degradation_reason"])
        for item in summaries
        if str(item["degradation_reason"]).strip()
    )
    aggregate = {
        "total_sessions": len(summaries),
        "verdict_counts": {
            "clean": verdict_counts.get("clean", 0),
            "watch": verdict_counts.get("watch", 0),
            "investigate": verdict_counts.get("investigate", 0),
        },
        "top_degradation_reasons": [
            {"reason": reason, "count": count}
            for reason, count in reason_counts.most_common()
        ],
        "sessions_with_fallback_activity": sum(
            1 for item in summaries if int(item["fallback_count"]) > 0
        ),
        "sessions_with_reacquisition_pressure": sum(
            1
            for item in summaries
            if int(item["repeat_search_count"]) > 0
            or int(item["repeat_file_read_count"]) > 0
        ),
    }
    return {"sessions": summaries, "aggregate": aggregate}


def rank_candidates(
    session_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in session_summaries:
        label = str(item.get("candidate_label", "")).strip()
        if label:
            grouped[label].append(item)

    candidates: list[dict[str, Any]] = []
    for label, sessions in grouped.items():
        models = sorted(
            {model for item in sessions for model in item.get("models", [])}
        )
        investigate_count = sum(
            1 for item in sessions if item["verdict"] == "investigate"
        )
        watch_count = sum(1 for item in sessions if item["verdict"] == "watch")
        positive_savings_count = sum(
            1
            for item in sessions
            if item["positive_savings"]
            and (
                int(item["fallback_count"]) > 0
                or int(item["repeat_search_count"]) > 0
                or int(item["repeat_file_read_count"]) > 0
            )
        )
        signal_counts: Counter[str] = Counter()
        for item in sessions:
            if int(item["fallback_count"]) > 0:
                signal_counts["fallback"] += 1
            if (
                int(item["repeat_search_count"]) > 0
                or int(item["repeat_file_read_count"]) > 0
            ):
                signal_counts["reacquisition"] += 1
            if label == "response contract drift":
                signal_counts["drift"] += 1

        score = (
            len(sessions) * 2
            + max(len(models) - 1, 0)
            + investigate_count
            + positive_savings_count
        )
        if len(sessions) < 2:
            next_action = "capture more evidence"
        elif not CANDIDATE_FIXTURE_MAP.get(label, {}).get(
            "required"
        ) and not CANDIDATE_FIXTURE_MAP.get(label, {}).get("exploratory"):
            next_action = "convert to exploratory replay fixture"
        else:
            next_action = "consider promotion after repeated clean rehearsals"

        if (
            watch_count > 0
            and next_action
            == "consider promotion after repeated clean rehearsals"
        ):
            why = "Repeated operator-visible friction with existing replay coverage suggests promotion review."
        elif next_action == "convert to exploratory replay fixture":
            why = "Repeated evidence exists without clear replay coverage."
        else:
            why = "Evidence is still too sparse for replay promotion work."

        candidates.append(
            {
                "candidate": label,
                "score": score,
                "evidence_count": len(sessions),
                "affected_models": models,
                "dominant_signals": [
                    name for name, _count in signal_counts.most_common()
                ],
                "supporting_sessions": [item["name"] for item in sessions],
                "recommended_next_action": next_action,
                "why_it_matters": why,
                "investigate_sessions": investigate_count,
                "watch_sessions": watch_count,
            }
        )

    candidates.sort(
        key=lambda item: (
            -int(item["score"]),
            -int(item["evidence_count"]),
            str(item["candidate"]),
        )
    )
    return candidates


def load_stress_evidence(stress_dir: Path) -> dict[str, Any]:
    breakpoints_path = stress_dir / "breakpoints.json"
    if not breakpoints_path.exists():
        raise FileNotFoundError(
            f"Stress breakpoints not found: {breakpoints_path}"
        )
    raw = json.loads(breakpoints_path.read_text())
    seen_counts: Counter[str] = Counter()
    seen_breakpoints: dict[str, list[str]] = defaultdict(list)
    for item in raw:
        breakpoint_class = str(item.get("breakpoint_class", "")).strip()
        candidate = STRESS_BREAKPOINT_TO_CANDIDATE.get(
            breakpoint_class, breakpoint_class
        )
        seen_counts[candidate] += 1
        seen_breakpoints[candidate].append(breakpoint_class)
    return {
        "path": str(stress_dir),
        "candidate_counts": dict(seen_counts),
        "breakpoint_classes": {
            key: sorted(set(values))
            for key, values in seen_breakpoints.items()
        },
    }


def _load_required_fixtures(gate_config_path: Path) -> set[str]:
    if not gate_config_path.exists():
        return set()
    data = json.loads(gate_config_path.read_text())
    return {str(item) for item in data.get("required_fixtures", [])}


def _load_available_fixtures(replay_dir: Path) -> set[str]:
    return {
        path.name.replace(".jsonl.meta.json", "")
        for path in replay_dir.glob("*.jsonl.meta.json")
    }


def build_coverage_report(
    session_summaries: list[dict[str, Any]],
    *,
    stress_evidence: dict[str, Any] | None = None,
    replay_dir: Path,
    gate_config_path: Path,
) -> list[dict[str, Any]]:
    required_fixtures = _load_required_fixtures(gate_config_path)
    available_fixtures = _load_available_fixtures(replay_dir)
    session_candidates = {
        str(item["candidate_label"])
        for item in session_summaries
        if str(item.get("candidate_label", "")).strip()
    }
    stress_candidates = set()
    if stress_evidence:
        stress_candidates = set(
            stress_evidence.get("candidate_counts", {}).keys()
        )

    coverage: list[dict[str, Any]] = []
    for candidate in sorted(session_candidates | stress_candidates):
        mapped = CANDIDATE_FIXTURE_MAP.get(
            candidate, {"required": [], "exploratory": []}
        )
        required = [
            name
            for name in mapped.get("required", [])
            if name in available_fixtures and name in required_fixtures
        ]
        exploratory = [
            name
            for name in mapped.get("exploratory", [])
            if name in available_fixtures and name not in required_fixtures
        ]
        if required:
            status = "already covered by release fixture"
            next_action = "consider promotion after repeated clean rehearsals"
        elif exploratory:
            status = "already covered by exploratory fixture"
            next_action = "consider promotion after repeated clean rehearsals"
        else:
            status = "not yet covered"
            next_action = "convert to exploratory replay fixture"
        coverage.append(
            {
                "candidate": candidate,
                "seen_in_real_sessions": candidate in session_candidates,
                "seen_in_stress_harness": candidate in stress_candidates,
                "coverage_status": status,
                "required_fixtures": required,
                "exploratory_fixtures": exploratory,
                "recommended_next_action": (
                    next_action
                    if candidate in session_candidates
                    else "capture more evidence"
                ),
            }
        )
    return coverage
