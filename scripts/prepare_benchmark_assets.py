#!/usr/bin/env python3

"""Build and verify checked-in benchmark asset snapshots."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tok.testing.benchmark_suite import BenchmarkTaskManifest, load_benchmark_catalog

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_ROOT = ROOT / "benchmarks"
ASSET_LOCK_FILENAME = "asset.lock.json"
ASSET_BUILDER_VERSION = "1"
PROJECT_METADATA_FILES = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "pytest.ini",
    "tox.ini",
    "MANIFEST.in",
    "README.md",
    "README.rst",
    "LICENSE",
    "LICENSE.txt",
)
SKIP_NAMES = {
    ".git",
    ".github",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    ".ruff_cache",
    ".venv",
    "docs",
    "doc",
    "examples",
    "example",
    "site",
    "tmp",
}


def _task_iter(root: Path, *, task_ids: tuple[str, ...], public_only: bool) -> tuple[BenchmarkTaskManifest, ...]:
    catalog = load_benchmark_catalog(root)
    selected: list[BenchmarkTaskManifest] = []
    requested = set(task_ids)
    for task in catalog.tasks:
        if requested and task.id not in requested:
            continue
        if public_only and not task.public_release:
            continue
        selected.append(task)
    return tuple(selected)


def _run(
    command: list[str], *, cwd: Path | None = None, input_bytes: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        cwd=cwd,
        input=input_bytes,
        check=True,
        capture_output=True,
    )


def _git_stdout(command: list[str], *, cwd: Path, strip: bool = True) -> str:
    text = _run(["git", *command], cwd=cwd).stdout.decode("utf-8")
    return text.strip() if strip else text


def _workspace_sha256(workspace_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in workspace_root.rglob("*") if item.is_file()):
        rel_path = path.relative_to(workspace_root).as_posix()
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=lambda _root, names: [name for name in names if name in SKIP_NAMES],
    )


def _path_without_selector(raw_path: str) -> Path:
    return Path(str(raw_path).split("::", 1)[0])


def _source_root_for_path(raw_path: str) -> Path:
    path = _path_without_selector(raw_path)
    parts = path.parts
    if not parts:
        return path
    if parts[0] == "src" and len(parts) >= 2:
        return Path(parts[0]) / parts[1]
    return Path(parts[0])


def _repo_root_for_task(task: BenchmarkTaskManifest) -> Path:
    return ROOT if task.repo == "tok" else ROOT


def _export_archive(task: BenchmarkTaskManifest, destination: Path) -> str:
    if task.repo == "tok":
        resolved = _git_stdout(["rev-parse", task.ref], cwd=ROOT)
        archive_bytes = _run(["git", "archive", "--format=tar", resolved], cwd=ROOT).stdout
    else:
        with tempfile.TemporaryDirectory(prefix="tok-benchmark-fetch-") as td:
            repo_dir = Path(td) / "repo"
            repo_dir.mkdir(parents=True, exist_ok=True)
            _run(["git", "init", "-q"], cwd=repo_dir)
            _run(["git", "remote", "add", "origin", f"https://github.com/{task.repo}.git"], cwd=repo_dir)
            fetch_refs = [task.ref]
            if not task.ref.startswith("v"):
                fetch_refs.append(f"v{task.ref}")
            last_error: subprocess.CalledProcessError | None = None
            for fetch_ref in fetch_refs:
                try:
                    _run(["git", "fetch", "--depth", "1", "origin", fetch_ref], cwd=repo_dir)
                except subprocess.CalledProcessError as exc:
                    last_error = exc
                    continue
                resolved = _git_stdout(["rev-parse", "FETCH_HEAD"], cwd=repo_dir)
                archive_bytes = _run(["git", "archive", "--format=tar", "FETCH_HEAD"], cwd=repo_dir).stdout
                break
            else:
                assert last_error is not None
                raise last_error
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive.getmembers():
            if member.islnk() or member.issym():
                continue
            member_path = destination / member.name
            try:
                member_path.resolve().relative_to(destination.resolve())
            except ValueError:
                continue
        archive.extractall(destination, members=[m for m in archive.getmembers() if not (m.islnk() or m.issym())])
    return resolved


def _selected_execution_paths(source_root: Path, task: BenchmarkTaskManifest) -> set[Path]:
    selected: set[Path] = set()
    for metadata_name in PROJECT_METADATA_FILES:
        metadata_path = source_root / metadata_name
        if metadata_path.exists():
            selected.add(Path(metadata_name))
    for raw_path in task.allowed_paths:
        root = _source_root_for_path(raw_path)
        if (source_root / root).exists():
            selected.add(root)
    test_roots = {_source_root_for_path(test_name) for test_name in (*task.visible_tests, *task.hidden_tests)}
    if not test_roots:
        for default_test_root in ("tests", "testing"):
            candidate = source_root / default_test_root
            if candidate.exists():
                test_roots.add(Path(default_test_root))
    for root in sorted(test_roots):
        if (source_root / root).exists():
            selected.add(root)
    return selected


def _selected_grounding_paths(source_root: Path, task: BenchmarkTaskManifest) -> set[Path]:
    selected: set[Path] = set()
    for metadata_name in PROJECT_METADATA_FILES:
        metadata_path = source_root / metadata_name
        if metadata_path.exists():
            selected.add(Path(metadata_name))
    for raw_path in (*task.required_files, *(span["file"] for span in task.supporting_spans)):
        path = Path(raw_path)
        if (source_root / path).exists():
            selected.add(path)
    return selected


def _copy_selected_paths(source_root: Path, workspace_root: Path, selected_paths: set[Path]) -> None:
    workspace_root.mkdir(parents=True, exist_ok=True)
    for relative_path in sorted(selected_paths):
        source_path = source_root / relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"asset build path not found: {source_path}")
        destination_path = workspace_root / relative_path
        if source_path.is_dir():
            _copy_tree(source_path, destination_path)
        else:
            _copy_file(source_path, destination_path)


def _replace_once(path: Path, *, old_string: str, new_string: str) -> None:
    original = path.read_text()
    if old_string not in original:
        raise ValueError(f"seed patch source text not found in {path}")
    path.write_text(original.replace(old_string, new_string, 1))


def _build_seed_patch(workspace_root: Path, task: BenchmarkTaskManifest) -> str:
    edits = task.family_payload.get("seed_patch_edits") or []
    if not isinstance(edits, list) or not edits:
        raise ValueError(f"execution_patch task {task.id} requires family_payload.seed_patch_edits")
    _run(["git", "init", "-q"], cwd=workspace_root)
    _run(["git", "config", "user.email", "benchmark@example.test"], cwd=workspace_root)
    _run(["git", "config", "user.name", "Tok Benchmark"], cwd=workspace_root)
    _run(["git", "add", "-A"], cwd=workspace_root)
    _run(["git", "commit", "-qm", "clean workspace"], cwd=workspace_root)
    for edit in edits:
        file_path = workspace_root / str(edit["path"])
        _replace_once(
            file_path,
            old_string=str(edit["old_string"]),
            new_string=str(edit["new_string"]),
        )
    patch = _git_stdout(["diff", "--binary"], cwd=workspace_root, strip=False)
    if not patch.strip():
        raise ValueError(f"execution_patch task {task.id} did not produce a seed.patch diff")
    _run(["git", "checkout", "--", "."], cwd=workspace_root)
    shutil.rmtree(workspace_root / ".git")
    return patch + "\n"


def _resolve_supporting_span(workspace_root: Path, span: dict[str, Any]) -> dict[str, Any]:
    file_path = workspace_root / str(span["file"])
    contents = file_path.read_text().splitlines()
    anchor = str(span["anchor"])
    start_line = 0
    for index, line in enumerate(contents, start=1):
        if anchor in line:
            start_line = index
            break
    if start_line == 0:
        raise ValueError(f"supporting span anchor not found: {span['file']} :: {anchor}")
    return {
        "file": str(span["file"]),
        "anchor": anchor,
        "why": str(span.get("why", "")),
        "start_line": start_line,
        "end_line": start_line,
    }


def _build_gold_answer(task: BenchmarkTaskManifest, workspace_root: Path) -> dict[str, Any]:
    return {
        "required_files": list(task.required_files),
        "required_symbols": list(task.required_symbols),
        "answer_contract": task.answer_contract,
        "accepted_facts": [
            {
                "label": "required_files",
                "match": "all",
                "terms": list(task.required_files),
            },
            {
                "label": "required_symbols",
                "match": "all",
                "terms": list(task.required_symbols),
            },
        ],
        "alternates": [],
        "supporting_spans": [_resolve_supporting_span(workspace_root, span) for span in task.supporting_spans],
    }


def _write_asset_lock(task: BenchmarkTaskManifest, asset_root: Path, *, resolved_commit: str) -> None:
    workspace_root = asset_root / "workspace"
    payload = {
        "schema_version": 1,
        "task_id": task.id,
        "repo": task.repo,
        "upstream_ref": task.ref,
        "resolved_commit": resolved_commit,
        "asset_builder_version": ASSET_BUILDER_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "workspace_sha256": _workspace_sha256(workspace_root),
    }
    (asset_root / ASSET_LOCK_FILENAME).write_text(json.dumps(payload, indent=2) + "\n")


def build_task_asset(task: BenchmarkTaskManifest, *, root: Path, force: bool) -> None:
    asset_root = root / task.asset_dir
    if asset_root.exists() and force:
        shutil.rmtree(asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)
    workspace_root = asset_root / "workspace"
    if workspace_root.exists():
        shutil.rmtree(workspace_root)

    with tempfile.TemporaryDirectory(prefix=f"tok-asset-{task.id}-") as td:
        source_root = Path(td) / "source"
        source_root.mkdir(parents=True, exist_ok=True)
        resolved_commit = _export_archive(task, source_root)
        if task.family == "execution_patch":
            selected_paths = _selected_execution_paths(source_root, task)
        elif task.family == "repo_grounding":
            selected_paths = _selected_grounding_paths(source_root, task)
        else:
            selected_paths = _selected_grounding_paths(source_root, task)
        _copy_selected_paths(source_root, workspace_root, selected_paths)

    if task.family == "execution_patch":
        seed_patch = _build_seed_patch(workspace_root, task)
        seed_patch_path = asset_root / task.seed_patch
        seed_patch_path.parent.mkdir(parents=True, exist_ok=True)
        seed_patch_path.write_text(seed_patch)
    elif task.family == "repo_grounding":
        gold_answer_path = asset_root / str(task.family_payload["gold_answer_path"])
        gold_answer_path.parent.mkdir(parents=True, exist_ok=True)
        gold_answer_path.write_text(json.dumps(_build_gold_answer(task, workspace_root), indent=2) + "\n")
    _write_asset_lock(task, asset_root, resolved_commit=resolved_commit)


def _verify_task_asset(task: BenchmarkTaskManifest, *, root: Path) -> list[str]:
    errors: list[str] = []
    asset_root = root / task.asset_dir
    workspace_root = asset_root / "workspace"
    lock_path = asset_root / ASSET_LOCK_FILENAME
    if not asset_root.exists():
        return [f"{task.id}: asset directory missing ({asset_root})"]
    if task.public_release and task.workspace_source_kind() != "asset_snapshot":
        errors.append(f"{task.id}: public task must use asset_snapshot workspace_source")
    if not workspace_root.exists():
        errors.append(f"{task.id}: workspace missing ({workspace_root})")
    if not lock_path.exists():
        errors.append(f"{task.id}: asset.lock.json missing")
    else:
        lock_payload = json.loads(lock_path.read_text())
        recorded_hash = str(lock_payload.get("workspace_sha256", "")).strip()
        if not recorded_hash:
            errors.append(f"{task.id}: asset.lock.json missing workspace_sha256")
        elif workspace_root.exists() and _workspace_sha256(workspace_root) != recorded_hash:
            errors.append(f"{task.id}: asset.lock.json workspace hash mismatch")
    if task.family == "execution_patch":
        seed_patch = asset_root / task.seed_patch
        if not seed_patch.exists():
            errors.append(f"{task.id}: seed.patch missing")
        if task.public_release and not task.evaluator_spec_ref():
            errors.append(f"{task.id}: public execution task missing success_evaluator.evaluator_spec")
        for allowed_path in task.allowed_paths:
            if not (workspace_root / allowed_path).exists():
                errors.append(f"{task.id}: allowed path missing from workspace ({allowed_path})")
    if task.family == "repo_grounding":
        gold_answer_path = asset_root / str(task.family_payload.get("gold_answer_path", ""))
        if not gold_answer_path.exists():
            errors.append(f"{task.id}: gold answer missing")
        for required_file in task.required_files:
            if not (workspace_root / required_file).exists():
                errors.append(f"{task.id}: required file missing from workspace ({required_file})")
    return errors


def verify_assets(
    root: Path,
    *,
    task_ids: tuple[str, ...],
    public_only: bool,
    tasks: tuple[BenchmarkTaskManifest, ...] = (),
) -> list[str]:
    errors: list[str] = []
    task_list = tasks or _task_iter(root, task_ids=task_ids, public_only=public_only)
    for task in task_list:
        errors.extend(_verify_task_asset(task, root=root))
    return errors


def refresh_locks(
    root: Path,
    *,
    task_ids: tuple[str, ...],
    public_only: bool,
    tasks: tuple[BenchmarkTaskManifest, ...] = (),
) -> None:
    task_list = tasks or _task_iter(root, task_ids=task_ids, public_only=public_only)
    for task in task_list:
        asset_root = root / task.asset_dir
        workspace_root = asset_root / "workspace"
        lock_path = asset_root / ASSET_LOCK_FILENAME
        if not workspace_root.exists() or not lock_path.exists():
            raise FileNotFoundError(f"cannot refresh lock for {task.id}; asset or lock missing")
        payload = json.loads(lock_path.read_text())
        payload["asset_builder_version"] = ASSET_BUILDER_VERSION
        payload["created_at"] = datetime.now(UTC).isoformat()
        payload["workspace_sha256"] = _workspace_sha256(workspace_root)
        lock_path.write_text(json.dumps(payload, indent=2) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_BENCHMARK_ROOT, help="Benchmark root")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify", "refresh-locks"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--task", action="append", default=[], help="Task id to operate on")
        sub.add_argument("--all", action="store_true", help="Include non-public advisory/internal tasks")
        if name == "build":
            sub.add_argument("--force", action="store_true", help="Replace existing asset directories")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    task_ids = tuple(str(task_id) for task_id in args.task)
    public_only = not bool(args.all)

    if args.command == "build":
        for task in _task_iter(root, task_ids=task_ids, public_only=public_only):
            build_task_asset(task, root=root, force=bool(args.force))
            print(f"built {task.id} -> {root / task.asset_dir}")
        return 0

    if args.command == "verify":
        errors = verify_assets(root, task_ids=task_ids, public_only=public_only)
        if errors:
            for error in errors:
                print(error)
            return 1
        print("benchmark assets verified")
        return 0

    refresh_locks(root, task_ids=task_ids, public_only=public_only)
    print("benchmark asset locks refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
