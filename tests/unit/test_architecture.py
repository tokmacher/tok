"""Architecture sanity tests for Tok module dependencies."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "orchestrator": ROOT / "src" / "tok" / "adapters" / "orchestrator.py",
    "parser": ROOT / "src" / "tok" / "protocol" / "parser.py",
    "bridge": ROOT / "src" / "tok" / "protocol" / "format_bridge.py",
}


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
