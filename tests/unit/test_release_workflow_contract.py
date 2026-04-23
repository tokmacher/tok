from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _workflow_triggers(workflow: dict) -> dict:
    return workflow.get("on") or workflow.get(True) or {}


def _job_needs(job: dict) -> list[str]:
    needs = job.get("needs", [])
    if isinstance(needs, str):
        return [needs]
    return list(needs)


def test_release_workflow_is_tag_only_and_publishes_only_from_tag_refs() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    triggers = _workflow_triggers(workflow)
    jobs = workflow["jobs"]

    assert "workflow_dispatch" not in triggers
    assert triggers["push"]["tags"] == ["v*"]
    assert _job_needs(jobs["build"]) == ["validate"]
    assert _job_needs(jobs["publish"]) == ["build"]
    assert _job_needs(jobs["github-release"]) == ["build"]
    assert jobs["publish"]["if"] == "startsWith(github.ref, 'refs/tags/v')"
    assert jobs["github-release"]["if"] == "startsWith(github.ref, 'refs/tags/v')"


def test_release_workflow_revalidates_before_publish_and_checks_version_consistency() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    validate_steps = workflow["jobs"]["validate"]["steps"]
    validate_runs = "\n".join(step.get("run", "") for step in validate_steps)

    assert "pyproject.toml" in validate_runs
    assert "src/tok/__init__.py" in validate_runs
    assert "uv sync --frozen --extra dev" in validate_runs
    assert "uv run pre-commit run --all-files" in validate_runs
    assert "uv run ruff check src/tok tests" in validate_runs
    assert "uv run mypy src/tok" in validate_runs
    assert (
        "uv run pytest tests/unit tests/integration -v --cov=src/tok --cov-report=term-missing --cov-fail-under=80"
        in validate_runs
    )
    assert "uv run python scripts/check_repo_hygiene.py" in validate_runs
    assert "uv run python scripts/run_security_audit.py" in validate_runs
    assert "uv run python scripts/run_release_smoke.py" in validate_runs


def test_release_workflow_checks_built_artifact_metadata_before_upload() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    build_steps = workflow["jobs"]["build"]["steps"]
    build_runs = "\n".join(step.get("run", "") for step in build_steps)

    assert "uv build" in build_runs
    assert "twine check dist/*.whl dist/*.tar.gz" in build_runs
