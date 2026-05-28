from __future__ import annotations

import json
from pathlib import Path

from tok.protocol.handoff import export_handoff, inspect_handoff
from tok.protocol.session_receipt import ArtifactReference, SessionIdentity, TokSessionReceipt


def test_local_agent_handoff_requires_exact_reacquisition_before_edit_like_work(tmp_path: Path) -> None:
    workspace_file = tmp_path / "src" / "target.py"
    workspace_file.parent.mkdir()
    workspace_file.write_text("def answer() -> str:\n    return 'old'\n", encoding="utf-8")

    # Agent A records a local receipt with only non-exact evidence for the target file.
    receipt = TokSessionReceipt(
        receipt_id="tsr_agent_a",
        created_at="2026-05-26T00:00:00Z",
        session=SessionIdentity(session_id="session-agent-a"),
        artifacts=[
            ArtifactReference(
                kind="summary",
                path=str(workspace_file),
                exactness="summary",
                available=False,
            )
        ],
        warnings=["Agent A only handed off summary evidence for the target file."],
    )
    handoff = export_handoff(
        session_receipt=receipt,
        source_agent="agent-a",
        target_agent="agent-b",
        goal="Change answer() after exact reacquisition.",
        completed=["Found likely target file from summary evidence."],
        next_steps=["Reacquire exact target file before editing."],
    )
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(handoff.model_dump(by_alias=True)), encoding="utf-8")

    # Agent B inspects the packet and is blocked from edit-like work on summary evidence.
    inspection = inspect_handoff(handoff_path)
    assert inspection.passed is True
    assert inspection.required_reacquisitions
    assert inspection.required_reacquisitions[0].path == str(workspace_file)
    assert _may_edit_after_handoff(inspection, exact_reacquired_paths=set()) is False

    # Agent B reacquires exact local source material, then may continue the bounded task.
    exact_source = workspace_file.read_text(encoding="utf-8")
    assert "return 'old'" in exact_source
    assert _may_edit_after_handoff(inspection, exact_reacquired_paths={str(workspace_file)}) is True

    new_source = exact_source.replace("return 'old'", "return 'new'")
    workspace_file.write_text(new_source, encoding="utf-8")

    assert workspace_file.read_text(encoding="utf-8") == "def answer() -> str:\n    return 'new'\n"


def _may_edit_after_handoff(inspection, *, exact_reacquired_paths: set[str]) -> bool:
    required = {item.path for item in inspection.required_reacquisitions}
    return required <= exact_reacquired_paths
