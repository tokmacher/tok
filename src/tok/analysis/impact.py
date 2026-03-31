import ast
import os
from pathlib import Path

from ..utils.delta import TokDelta
from ..utils.sifter import DirectoryWalker


class ImpactEngine:
    """
    Analyzes the impact of changes (deltas) across a project.
    Detects affected callers and identifies broken call sites.
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.walker = DirectoryWalker()
        self.index: dict[
            str, set[str]
        ] = {}  # {qualified_name: {files_that_use_it}}
        self.signatures: dict[
            str, list[str]
        ] = {}  # {qualified_name: [arg_names]}

    def build_project_index(self):
        """Scan all python files and build a usage index."""
        py_files = self.walker.walk(str(self.project_root))

        for path in py_files:
            try:
                with open(path, encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source)

                # Relativize path for module name
                rel_parts = (
                    path.relative_to(self.project_root).with_suffix("").parts
                )
                module_name = ".".join(rel_parts)

                self._index_file_contents(module_name, tree, str(path))
            except Exception:
                continue

    def _index_file_contents(
        self, _module_name: str, tree: ast.AST, file_path: str
    ) -> None:
        """Extract imports and calls to build the index."""
        # This is a simplified version for demonstration
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Try to resolve call name
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr

                if name:
                    # In a real engine, we'd do proper name resolution (imports, etc)
                    # For this test, we'll index by name
                    self.index.setdefault(name, set()).add(file_path)

    def analyze_impact(self, deltas: list[TokDelta]) -> list[dict[str, str]]:
        """Identify callers affected by the changes."""
        impacted = []

        for delta in deltas:
            if delta.op != "update" or delta.target_type != "func":
                continue

            name = delta.target_label
            if name in self.index:
                for caller_file in self.index[name]:
                    # Check if actually 'broken'
                    status = self._check_if_broken(
                        caller_file, name, delta.new_attrs.get("params", "")
                    )
                    impacted.append(
                        {
                            "file": os.path.relpath(
                                caller_file, self.project_root
                            ),
                            "target": name,
                            "status": status,
                        }
                    )

        return impacted

    def _check_if_broken(
        self, file_path: str, func_name: str, new_params_str: str
    ) -> str:
        """Heuristic check if a call site is broken by new signature."""
        try:
            with open(file_path) as f:
                source = f.read()
            tree = ast.parse(source)

            # Simple check: parse params count
            new_params = [
                p.strip() for p in new_params_str.split(",") if p.strip()
            ]
            required_count = len([p for p in new_params if "=" not in p])

            broken = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = None
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr

                    if name == func_name:
                        # Check arg counts
                        total_args = len(node.args) + len(node.keywords)
                        if total_args < required_count:
                            broken = True
                            break

            return "broken" if broken else "stable"
        except Exception:
            return "unknown"


def format_impact_tok(impacted: list[dict[str, str]]) -> str:
    """Format impact results as a compressed Tok @impact_context block."""
    if not impacted:
        return ""

    if len(impacted) > 5:
        # High Impact Event Summarization
        targets = set(i["target"] for i in impacted)
        files = set(i["file"] for i in impacted)
        return (
            f"\n@impact_context\n"
            f"  @alert HighImpactEvent: {len(files)}_files_affected\n"
            f"  !targets: {','.join(targets)}\n"
            f"  !summary: Wide-reaching_structural_change."
        )

    lines = ["", "@impact_context"]
    for i in impacted:
        lines.append(f"  !impact: {i['file']} ({i['status']})")

    return "\n".join(lines)
