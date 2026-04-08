import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_pr_comment_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "pr-comment.py"
    spec = importlib.util.spec_from_file_location("pr_comment_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pr_comment_load_results_accepts_structured_export(tmp_path) -> None:
    mod = _load_pr_comment_module()
    payload = {
        "results": [
            {
                "fixture": "demo",
                "passed": True,
                "savings_pct": 42.0,
                "pressure": 1,
            }
        ],
        "release_summary": {
            "avg_savings_pct": 42.0,
            "avg_invisible_pressure": 1.0,
            "fallback_fixture_rate": 0.0,
            "billing_delta_usd": 0.12,
            "billing_delta_pct": 4.2,
        },
    }
    results_file = tmp_path / "results.json"
    results_file.write_text(json.dumps(payload))

    loaded = mod.load_results(str(results_file))

    assert loaded["results"][0]["fixture"] == "demo"
    assert loaded["release_summary"]["billing_delta_usd"] == 0.12


def test_pr_comment_generate_comment_surfaces_release_summary() -> None:
    mod = _load_pr_comment_module()
    payload = {
        "results": [
            {
                "fixture": "demo",
                "passed": True,
                "savings_pct": 42.0,
                "pressure": 1,
            }
        ],
        "release_summary": {
            "avg_savings_pct": 42.0,
            "avg_invisible_pressure": 1.0,
            "fallback_fixture_rate": 12.5,
            "billing_delta_usd": 0.67,
            "billing_delta_pct": 3.5,
        },
    }

    comment = mod.generate_comment(payload, "feature")

    assert "Fallback Fixture Rate" in comment
    assert "Billing Delta" in comment
    assert "$0.6700 (3.5%)" in comment


def test_pr_comment_accepts_structured_export_with_stability_check() -> None:
    mod = _load_pr_comment_module()
    payload = {
        "results": [
            {
                "fixture": "demo",
                "passed": True,
                "savings_pct": 42.0,
                "pressure": 1,
            }
        ],
        "release_summary": {
            "avg_savings_pct": 42.0,
            "avg_invisible_pressure": 1.0,
            "fallback_fixture_rate": 12.5,
            "billing_delta_usd": 0.67,
            "billing_delta_pct": 3.5,
        },
        "stability_check": {
            "coding-loop-5": {"passed": True},
            "research-loop-5": {"passed": True},
        },
    }

    comment = mod.generate_comment(payload, "full")

    assert "## 🚪 Tok Gate Check Results" in comment
    assert "Fixture Set" in comment
    assert "Fallback Fixture Rate" in comment
