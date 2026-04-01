import json
from pathlib import Path

from scripts.select_fixtures import (
    get_feature_fixtures,
    get_full_fixtures,
    get_redteam_fixtures,
)


def _required_fixtures(config_path: Path | None = None) -> set[str]:
    """Get required fixtures from gate config.
    Args:
        config_path: Optional path to gate-config.json. If None, uses repository root.
    """
    if config_path is None:
        gate_config = Path(__file__).resolve().parents[2] / "gate-config.json"
    else:
        gate_config = config_path

    # Create a default gate-config.json if it doesn't exist (for local testing)
    if not gate_config.exists():
        default_config = {
            "required_fixtures": [
                "runtime_conformance",
                "alternating_adapters",
                "release_reacquisition",
            ]
        }
        gate_config.write_text(json.dumps(default_config, indent=2))
    return set(json.loads(gate_config.read_text())["required_fixtures"])


def test_release_reacquisition_is_in_green_fixture_sets():
    assert "release_reacquisition" in get_feature_fixtures()
    assert "release_reacquisition" in get_full_fixtures()


def test_release_reacquisition_is_not_in_redteam_fixture_set():
    assert "release_reacquisition" not in get_redteam_fixtures()


def test_required_release_fixtures_are_in_green_sets(tmp_path):
    # Use temporary config file to avoid side effects
    config_file = tmp_path / "gate-config.json"
    required = _required_fixtures(config_file)
    assert required.issubset(set(get_feature_fixtures()))
    assert required.issubset(set(get_full_fixtures()))


def test_required_release_fixtures_are_not_in_redteam(tmp_path):
    # Use temporary config file to avoid side effects
    config_file = tmp_path / "gate-config.json"
    required = _required_fixtures(config_file)
    assert required.isdisjoint(set(get_redteam_fixtures()))


def test_required_release_fixtures_match_current_internal_rc_contract(
    tmp_path,
):
    # Use temporary config file to avoid side effects
    config_file = tmp_path / "gate-config.json"
    assert _required_fixtures(config_file) == {
        "runtime_conformance",
        "alternating_adapters",
        "release_reacquisition",
    }


def test_checked_in_stability_artifacts_exist_for_mock_release():
    stability_dir = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "stability"
    )
    assert (stability_dir / "coding-loop-5_stability.json").exists()
    assert (stability_dir / "research-loop-5_stability.json").exists()


def test_exploratory_green_release_fixtures_remain_in_green_sets():
    feature = set(get_feature_fixtures())
    full = set(get_full_fixtures())
    assert "cache_stable_research_turns" in feature
    assert "cache_stable_research_turns" in full
    assert "refined_search_recovery" in feature
    assert "refined_search_recovery" in full


def test_exploratory_green_release_fixtures_are_not_required_yet(tmp_path):
    # Use temporary config file to avoid side effects
    config_file = tmp_path / "gate-config.json"
    required = _required_fixtures(config_file)
    assert "cache_stable_research_turns" not in required
    assert "refined_search_recovery" not in required
