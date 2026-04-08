"""Tok Delta - Semantic diffing for Tok representations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokDelta:
    """Represents a single change between two Tok representations."""

    op: str
    target_type: str
    target_label: str
    old_attrs: dict[str, Any] = field(default_factory=dict)
    new_attrs: dict[str, Any] = field(default_factory=dict)
    file: str | None = None
    line: int | None = None
    ref: str | None = None

    def changed_fields(self) -> list[str]:
        """Return list of attribute names that changed."""
        old_keys = set(self.old_attrs.keys())
        new_keys = set(self.new_attrs.keys())
        all_keys = old_keys | new_keys
        changed = []
        for k in all_keys:
            if k in old_keys and k in new_keys:
                if self.old_attrs[k] != self.new_attrs[k]:
                    changed.append(k)
            elif (k in new_keys and k not in old_keys) or (k in old_keys and k not in new_keys):
                changed.append(k)
        return changed


def _parse_func_entry(
    stripped: str,
    current_module: str | None,
    current_file: str | None,
    include_refs: bool,
) -> tuple[tuple[str, str, str], dict[str, Any]] | None:
    func_match = re.search(r"@func\s+(\w+)\s*\(([^)]*)\)", stripped)
    if not func_match:
        return None
    name = func_match.group(1)
    args = func_match.group(2)
    ref_match = re.search(r"ref:\*(\w+)", stripped)
    ref = ref_match.group(1) if ref_match else None
    attrs = {"params": args} if args else {}
    if include_refs and ref:
        attrs["ref"] = ref
    key = (
        "func",
        name,
        current_module or current_file or "unknown",
    )
    return key, attrs


def _parse_class_entry(
    stripped: str,
    current_module: str | None,
    current_file: str | None,
    include_refs: bool,
) -> tuple[tuple[str, str, str], dict[str, Any]] | None:
    class_match = re.search(r"@class\s+(\w+)", stripped)
    if not class_match:
        return None
    name = class_match.group(1)
    bases_match = re.search(r"bases:(\S+)", stripped)
    attrs = {"bases": bases_match.group(1)} if bases_match else {}
    ref_match = re.search(r"ref:\*(\w+)", stripped)
    if include_refs and ref_match:
        attrs["ref"] = ref_match.group(1)
    key = (
        "class",
        name,
        current_module or current_file or "unknown",
    )
    return key, attrs


_SKIP_PREFIXES = ("@chunk", "@corpus", "@deps", "@func", "@class")


def _is_module_directive(stripped: str) -> bool:
    return stripped.startswith("@") and not any(stripped.startswith(p) for p in _SKIP_PREFIXES)


def _extract_module_name(stripped: str, current_module: str | None) -> str | None:
    if current_module:
        return current_module
    potential = stripped.lstrip("@")
    return potential.split()[0] if potential else None


def parse_skeleton(skeleton_tok: str, include_refs: bool = True) -> dict[tuple[str, str, str], dict[str, Any]]:
    """
    Parse a Tok skeleton into an index.

    Args:
        skeleton_tok: The Tok skeleton string
        include_refs: If False, ignore ref pointers in comparison (useful when refs are non-deterministic)

    """
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    current_file = None
    current_module = None

    for line in skeleton_tok.split("\n"):
        stripped = line.strip()

        if stripped.startswith("@repo"):
            match = re.search(r"@repo\s+(\S+)", stripped)
            if match:
                current_file = match.group(1)
                current_module = current_file

        if _is_module_directive(stripped):
            current_module = _extract_module_name(stripped, current_module)

        elif stripped.startswith("@func"):
            result = _parse_func_entry(stripped, current_module, current_file, include_refs)
            if result:
                index[result[0]] = result[1]

        elif stripped.startswith("@class"):
            result = _parse_class_entry(stripped, current_module, current_file, include_refs)
            if result:
                index[result[0]] = result[1]

    return index


def compute_attr_diff(old_attrs: dict[str, Any], new_attrs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compute meaningful attribute differences."""
    diff = {}

    old_keys = set(old_attrs.keys())
    new_keys = set(new_attrs.keys())
    all_keys = old_keys | new_keys

    for k in all_keys:
        old_val = old_attrs.get(k, "<missing>")
        new_val = new_attrs.get(k, "<missing>")

        if old_val != new_val:
            diff[k] = {"from": old_val, "to": new_val}

    return diff


def diff_tok(old_tok: str, new_tok: str, include_refs: bool = False) -> list[TokDelta]:
    """
    Compute semantic diff between two Tok skeleton representations.

    Args:
        old_tok: Tok skeleton before change
        new_tok: Tok skeleton after change
        include_refs: If True, include ref pointers in comparison (default False for stability)

    Returns:
        List of TokDelta objects representing changes

    """
    old_index = parse_skeleton(old_tok, include_refs=include_refs)
    new_index = parse_skeleton(new_tok, include_refs=include_refs)

    deltas: list[TokDelta] = []

    old_keys = set(old_index.keys())
    new_keys = set(new_index.keys())

    added_keys = new_keys - old_keys
    removed_keys = old_keys - new_keys
    common_keys = old_keys & new_keys

    for key in added_keys:
        target_type, target_label, file = key
        attrs = new_index[key]
        deltas.append(
            TokDelta(
                op="add",
                target_type=target_type,
                target_label=target_label,
                new_attrs=attrs,
                file=file,
                ref=attrs.get("ref"),
            )
        )

    for key in removed_keys:
        target_type, target_label, file = key
        attrs = old_index[key]
        deltas.append(
            TokDelta(
                op="remove",
                target_type=target_type,
                target_label=target_label,
                old_attrs=attrs,
                file=file,
                ref=attrs.get("ref"),
            )
        )

    for key in common_keys:
        target_type, target_label, file = key
        old_attrs = old_index[key]
        new_attrs = new_index[key]

        attr_diff = compute_attr_diff(old_attrs, new_attrs)

        if attr_diff:
            deltas.append(
                TokDelta(
                    op="update",
                    target_type=target_type,
                    target_label=target_label,
                    old_attrs=old_attrs,
                    new_attrs=new_attrs,
                    file=file,
                    ref=new_attrs.get("ref") or old_attrs.get("ref"),
                )
            )

    return deltas


def _format_delta_attrs(delta: TokDelta) -> list[str]:
    lines: list[str] = []
    if delta.op == "update":
        for k in delta.changed_fields():
            if k == "ref":
                continue
            old_val = delta.old_attrs.get(k, "<absent>")
            new_val = delta.new_attrs.get(k, "<absent>")
            lines.append(f"    -{k}: {old_val}")
            lines.append(f"    +{k}: {new_val}")
    elif delta.op == "add":
        for k, v in delta.new_attrs.items():
            if k != "ref":
                lines.append(f"    +{k}: {v}")
    elif delta.op == "remove":
        for k, v in delta.old_attrs.items():
            if k != "ref":
                lines.append(f"    -{k}: {v}")
    return lines


def delta_to_tok(deltas: list[TokDelta]) -> str:
    """Serialize TokDelta list to a compressed Tok format (no outer tag)."""
    if not deltas:
        return ""

    lines = []
    for delta in deltas:
        target = f"@{delta.target_type}|{delta.target_label}"
        file_hint = f" in {delta.file}" if delta.file else ""
        lines.append(f"  {delta.op} {target}{file_hint}")
        lines.extend(_format_delta_attrs(delta))

    return "\n".join(lines).strip()


def format_compact_delta(deltas: list[TokDelta]) -> str:
    """Format as compact single-line changes for minimal token usage."""
    lines = []

    for delta in deltas:
        if delta.op == "update":
            changes = []
            for field in delta.changed_fields():
                old_val = delta.old_attrs.get(field, "")
                new_val = delta.new_attrs.get(field, "")
                changes.append(f"{field}: {old_val} -> {new_val}")
            lines.append(f"@delta {delta.op} {delta.target_type}:{delta.target_label} | {'; '.join(changes)}")
        elif delta.op == "add":
            lines.append(f"@delta add {delta.target_type}:{delta.target_label}")
        elif delta.op == "remove":
            lines.append(f"@delta remove {delta.target_type}:{delta.target_label}")

    return "\n".join(lines)


class TokDeltaTracker:
    """Track before/after snapshots for explicit delta computation."""

    def __init__(self) -> None:
        self.snapshots: dict[str, str] = {}

    def capture(self, name: str, tok_skeleton: str) -> None:
        """Capture a snapshot with a name."""
        self.snapshots[name] = tok_skeleton

    def diff(self, before_name: str, after_name: str) -> list[TokDelta]:
        """Compute delta between two named snapshots."""
        before = self.snapshots.get(before_name, "")
        after = self.snapshots.get(after_name, "")
        return diff_tok(before, after)

    def diff_to_tok(self, before_name: str, after_name: str) -> str:
        """Compute delta and return as Tok string."""
        return delta_to_tok(self.diff(before_name, after_name))


def _apply_single_delta(
    index: dict[tuple[str, str, str], dict[str, Any]],
    delta: TokDelta,
) -> None:
    key = (delta.target_type, delta.target_label, delta.file or "unknown")
    if delta.op == "add":
        index[key] = delta.new_attrs
    elif delta.op == "remove":
        index.pop(key, None)
    elif delta.op == "update":
        if key in index:
            index[key].update(delta.new_attrs)
        else:
            index[key] = delta.new_attrs


def _reconstruct_skeleton(
    index: dict[tuple[str, str, str], dict[str, Any]],
) -> str:
    lines: list[str] = []
    current_file: str | None = None
    sorted_keys = sorted(index.keys(), key=lambda x: (x[2], x[0], x[1]))
    for key in sorted_keys:
        t, label, file = key
        if file not in (current_file, "unknown"):
            lines.append(f"@{file}")
            current_file = file
        attrs = index[key]
        attr_str = " ".join(f"{k}:{v}" for k, v in attrs.items())
        if t == "func":
            lines.append(f"  @func {label}({attrs.get('params', '')}) {attr_str}")
        elif t == "class":
            lines.append(f"  @class {label} {attr_str}")
    return "\n".join(lines)


def apply_delta(tok_string: str, deltas: list[TokDelta]) -> str:
    """Apply a list of deltas to a Tok representation."""
    if not deltas:
        return tok_string

    index = parse_skeleton(tok_string)
    for delta in deltas:
        _apply_single_delta(index, delta)
    return _reconstruct_skeleton(index)
