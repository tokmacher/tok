from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

SAMPLE_UV_LOCK = """version = 1
revision = 3
requires-python = ">=3.10"

[[package]]
name = "alpha"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "beta" },
    { name = "gamma", marker = "python_full_version < '3.12'" },
]
sdist = { url = "https://files.pythonhosted.org/packages/source/alpha-1.0.0.tar.gz", hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", size = 101, upload-time = "2025-01-01T00:00:00Z" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/wheel/alpha-1.0.0-py3-none-any.whl", hash = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", size = 102, upload-time = "2025-01-01T00:00:01Z" },
]

[[package]]
name = "beta"
version = "2.0.0"
source = { registry = "https://pypi.org/simple" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/wheel/beta-2.0.0-py3-none-any.whl", hash = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", size = 103, upload-time = "2024-05-01T00:00:00Z" },
]

[[package]]
name = "gamma"
version = "0.5.0"
source = { editable = "." }
dependencies = [
    { name = "beta" },
]
"""


def _load_module(name: str, relative_path: str) -> ModuleType:
    sys.path.insert(0, str(REPO_ROOT))
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_uv_lock_loader_normalizes_packages(tmp_path, monkeypatch) -> None:
    module = _load_module("test_uv_lock_helper", "scripts/_uv_lock.py")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").write_text(SAMPLE_UV_LOCK, encoding="utf-8")

    packages = module.load_uv_lock()
    by_name = {package["name"]: package for package in packages}

    alpha = by_name["alpha"]
    beta = by_name["beta"]
    gamma = by_name["gamma"]

    assert alpha["dependencies"] == ["beta", "gamma"]
    assert alpha["artifact_url"].endswith("alpha-1.0.0.tar.gz")
    assert alpha["artifact_hash"].startswith("sha256:")
    assert len(alpha["hashes"]) == 2
    assert alpha["upload_time"] is not None

    assert beta["dependencies"] == []
    assert beta["artifact_url"].endswith("beta-2.0.0-py3-none-any.whl")
    assert beta["artifact_hash"].startswith("sha256:")

    assert gamma["editable"] == "."
    assert gamma["dependencies"] == ["beta"]
    assert gamma["artifact_url"] == ""


def test_verify_dependency_integrity_parse_uses_normalized_lock_records(tmp_path, monkeypatch) -> None:
    module = _load_module("test_verify_dependency_integrity", "scripts/verify_dependency_integrity.py")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").write_text(SAMPLE_UV_LOCK, encoding="utf-8")

    packages = module.parse_uv_lock()
    by_name = {package["name"]: package for package in packages}

    assert by_name["alpha"]["artifact_url"].endswith("alpha-1.0.0.tar.gz")
    assert by_name["alpha"]["hashes"] == [
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ]
    assert by_name["gamma"]["source"] == {"editable": "."}


def test_verify_dependency_integrity_logs_detailed_findings(monkeypatch) -> None:
    module = _load_module("test_verify_dependency_integrity_logging", "scripts/verify_dependency_integrity.py")
    warnings: list[str] = []
    errors: list[str] = []
    infos: list[str] = []

    monkeypatch.setattr(
        module,
        "parse_uv_lock",
        lambda: [
            {
                "name": "pycrypto",
                "version": "2.6.1",
                "source": {"registry": "https://pypi.org/simple"},
                "artifact_url": "https://files.pythonhosted.org/packages/pycrypto.whl",
                "hashes": ["sha256:" + ("a" * 64)],
                "upload_time": None,
            },
            {
                "name": "alpha",
                "version": "1.0.0",
                "source": {"registry": "https://pypi.org/simple"},
                "artifact_url": "https://files.pythonhosted.org/packages/alpha.whl",
                "hashes": [],
                "upload_time": None,
            },
        ],
    )
    monkeypatch.setattr(
        module.logger,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )
    monkeypatch.setattr(
        module.logger,
        "error",
        lambda message, *args: errors.append(message % args if args else message),
    )
    monkeypatch.setattr(
        module.logger,
        "info",
        lambda message, *args: infos.append(message % args if args else message),
    )

    result = module.main()

    assert result == 1
    assert any("Blocked package found: pycrypto" in entry for entry in errors)
    assert any("No upload timestamp available for alpha@1.0.0" in entry for entry in warnings)
    assert any("No artifact hashes recorded for alpha@1.0.0" in entry for entry in warnings)


def test_security_dashboard_main_reuses_printed_dashboard(tmp_path, monkeypatch) -> None:
    module = _load_module("test_security_dashboard_main", "scripts/security_dashboard.py")
    monkeypatch.chdir(tmp_path)

    dashboard = {
        "timestamp": "2026-04-09T00:00:00Z",
        "security_score": {"score": 100, "grade": "A", "issues": [], "recommendations": []},
        "summary": {},
        "security_metrics": {},
        "dependency_analysis": {},
        "vulnerabilities": [],
        "alerts": [],
        "trends": {},
    }
    calls: list[str] = []

    class FakeMonitor:
        def print_dashboard(self) -> dict[str, object]:
            calls.append("print_dashboard")
            return dashboard

    monkeypatch.setattr(module, "SecurityMonitor", FakeMonitor)

    module.main()

    assert calls == ["print_dashboard"]
    assert json.loads((tmp_path / "security-dashboard.json").read_text(encoding="utf-8")) == dashboard


def test_generate_spdx_uses_primary_artifact_download_location_and_checksum(tmp_path, monkeypatch) -> None:
    module = _load_module("test_generate_spdx_sbom", "scripts/generate_spdx_sbom.py")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").write_text(SAMPLE_UV_LOCK, encoding="utf-8")

    module.main()

    sbom = json.loads((tmp_path / "sbom.spdx").read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in sbom["packages"]}
    project_metadata = module.read_project_metadata()

    main = packages["tok-protocol"]
    alpha = packages["alpha"]
    beta = packages["beta"]

    assert main["versionInfo"] == project_metadata["version"]
    assert main["externalRefs"][0]["referenceLocator"] == f"pkg:pypi/tok-protocol@{project_metadata['version']}"
    assert alpha["downloadLocation"].endswith("alpha-1.0.0.tar.gz")
    assert alpha["checksums"] == [
        {
            "algorithm": "SHA256",
            "checksumValue": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        }
    ]
    assert beta["downloadLocation"].endswith("beta-2.0.0-py3-none-any.whl")
    assert beta["checksums"] == [
        {
            "algorithm": "SHA256",
            "checksumValue": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        }
    ]


def test_checked_in_sbom_matches_project_metadata_and_has_single_root_package() -> None:
    module = _load_module("test_generate_spdx_sbom_current", "scripts/generate_spdx_sbom.py")
    metadata = module.read_project_metadata()
    sbom = json.loads((REPO_ROOT / "sbom.spdx").read_text(encoding="utf-8"))
    project_packages = [package for package in sbom["packages"] if package["name"] == metadata["name"]]

    assert len(project_packages) == 1
    package = project_packages[0]
    assert package["versionInfo"] == metadata["version"]
    assert package["externalRefs"][0]["referenceLocator"] == f"pkg:pypi/{metadata['name']}@{metadata['version']}"
    assert not any(
        relationship["relatedSpdxElement"] == "SPDXRef-tok_protocol" for relationship in sbom["relationships"]
    )


def test_security_utility_user_agents_follow_project_version() -> None:
    metadata_module = _load_module("test_project_metadata", "scripts/_project_metadata.py")
    dashboard = _load_module("test_security_dashboard_version", "scripts/security_dashboard.py")
    analyzer = _load_module("test_analyze_dependency_tree_version", "scripts/analyze_dependency_tree.py")
    version = metadata_module.read_project_metadata()["version"]

    assert dashboard.SECURITY_CONFIG["user_agent"] == f"tok-security-dashboard/{version}"
    assert analyzer.SECURITY_CONFIG["user_agent"] == f"tok-dependency-analyzer/{version}"


def test_dependency_analysis_uses_normalized_dependencies_and_skips_editable_packages(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_module("test_analyze_dependency_tree", "scripts/analyze_dependency_tree.py")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").write_text(SAMPLE_UV_LOCK, encoding="utf-8")

    packages = module.parse_uv_lock()
    calls: list[tuple[str, str]] = []

    def fake_security_data(package_name: str, version: str) -> dict[str, object]:
        calls.append((package_name, version))
        return {
            "upload_time": "2025-01-02T00:00:00Z",
            "package_size": 2048,
            "has_wheel": True,
            "has_source": package_name == "alpha",
        }

    monkeypatch.setattr(module, "get_package_security_data", fake_security_data)

    analysis = module.analyze_dependency_tree(packages)

    assert analysis["summary"]["total_dependencies"] == 3
    assert analysis["dependency_analysis"]["max_dependency_depth"] == 2
    assert analysis["dependency_analysis"]["packages_with_no_dependencies"] == 1
    assert ("gamma", "0.5.0") not in calls
    assert analysis["package_details"]["alpha"]["has_source"] is True
    assert analysis["package_details"]["beta"]["has_wheel"] is True


def test_get_package_security_data_uses_urls_shape_and_handles_404(monkeypatch) -> None:
    module = _load_module("test_analyze_dependency_tree_fetch", "scripts/analyze_dependency_tree.py")

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object] | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                error = module.requests.exceptions.HTTPError("boom")
                error.response = self
                raise error

        def json(self) -> dict[str, object]:
            return self._payload

    class FakeSession:
        def __init__(self, response: FakeResponse) -> None:
            self.response = response

        def get(self, *args, **kwargs) -> FakeResponse:
            return self.response

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        module,
        "create_secure_session",
        lambda: FakeSession(
            FakeResponse(
                200,
                {
                    "urls": [
                        {
                            "packagetype": "bdist_wheel",
                            "size": 120,
                            "upload_time_iso_8601": "2025-01-03T00:00:00.000Z",
                        },
                        {
                            "packagetype": "sdist",
                            "size": 80,
                            "upload_time": "2025-01-02T00:00:00",
                        },
                    ]
                },
            )
        ),
    )
    security_data = module.get_package_security_data("alpha", "1.0.0")
    assert security_data == {
        "upload_time": "2025-01-02T00:00:00Z",
        "package_size": 200,
        "has_wheel": True,
        "has_source": True,
    }

    monkeypatch.setattr(module, "create_secure_session", lambda: FakeSession(FakeResponse(404)))
    assert module.get_package_security_data("missing", "9.9.9") == {}


def test_calculate_max_depth_returns_zero_for_empty_graph() -> None:
    module = _load_module("test_analyze_dependency_tree_empty", "scripts/analyze_dependency_tree.py")
    assert module.calculate_max_depth({}) == 0


def test_dependency_review_workflow_requires_collaborator_permission() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "dependency-security-review.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    permissions = workflow["permissions"]
    script = workflow["jobs"]["block-merge"]["steps"][0]["with"]["script"]

    assert permissions["issues"] == "write"
    assert "comment.user.type === 'User'" not in script
    assert "getCollaboratorPermissionLevel" in script
    assert "new Set(['write', 'maintain', 'admin'])" in script
    assert "Security review completed - approved" in script
