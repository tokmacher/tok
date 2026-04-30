"""Architecture sanity tests for Tok module dependencies."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "orchestrator": ROOT / "src" / "tok" / "adapters" / "orchestrator.py",
    "parser": ROOT / "src" / "tok" / "protocol" / "parser.py",
    "bridge": ROOT / "src" / "tok" / "protocol" / "format_bridge.py",
}
SRC_TOK = ROOT / "src" / "tok"


def _collect_import_statements(
    path: Path,
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    tree = ast.parse(path.read_text())
    from_imports: list[tuple[str, list[str]]] = []
    direct_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = ("." * node.level) + (node.module or "")
            from_imports.append((module, [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            direct_imports.extend(alias.name for alias in node.names)
    return from_imports, direct_imports


def _iter_python_files(package: str) -> list[Path]:
    return sorted((SRC_TOK / package).rglob("*.py"))


def _absolute_imports(path: Path) -> set[str]:
    from_imports, direct_imports = _collect_import_statements(path)
    imports = set(direct_imports)
    imports.update(module for module, _names in from_imports if not module.startswith("."))
    return imports


def _assert_no_import_prefixes(package: str, forbidden_prefixes: tuple[str, ...]) -> None:
    violations: list[str] = []
    for path in _iter_python_files(package):
        for module in _absolute_imports(path):
            if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden_prefixes):
                rel = path.relative_to(ROOT)
                violations.append(f"{rel}: {module}")
    assert violations == [], "forbidden imports:\n" + "\n".join(violations)


def test_orchestrator_depends_on_parser_and_bridge() -> None:
    from_imports, _ = _collect_import_statements(SOURCES["orchestrator"])
    assert any("protocol" in module and "TokParser" in names for module, names in from_imports), (
        "Orchestrator must import the parser"
    )
    assert any("protocol" in module and "Bridge" in names for module, names in from_imports), (
        "Orchestrator must import the bridge"
    )


def test_protocol_components_do_not_import_orchestrator() -> None:
    for name in ("parser", "bridge"):
        _, modules = _collect_import_statements(SOURCES[name])
        assert all("adapters.orchestrator" not in mod for mod in modules), (
            f"{name} must not import adapters.orchestrator"
        )


def test_runtime_modules_do_not_import_gateway_or_cli_layers() -> None:
    _assert_no_import_prefixes("runtime", ("tok.gateway", "tok.cli"))


def test_compression_modules_do_not_import_gateway_or_cli_layers() -> None:
    _assert_no_import_prefixes("compression", ("tok.gateway", "tok.cli"))


def test_architecture_0_2_roadmap_documents_future_contracts() -> None:
    roadmap = ROOT / "docs" / "architecture-0.2.md"

    text = roadmap.read_text()

    assert "This document is a roadmap, not the current runtime contract" in text
    assert "Request Lifecycle Contract" in text
    assert "Behavior-Signal Registry" in text
    assert "Config Strictness Table" in text
    assert "Runtime State Grouping" in text
    assert "Evidence-Safety Contract" in text
    assert "runtime` and `compression` must not import `gateway` or `cli`" in text
