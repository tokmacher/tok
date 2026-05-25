"""tok.deterministic — experimental deterministic action receipts.

Not part of the 0.2.x public API. Import via explicit submodule:

    from tok.deterministic.exact_patch import apply_exact_patch
    from tok.deterministic.models import ExactPatchPreconditions
    from tok.deterministic.action_decision import classify_action
"""

from tok.deterministic.action_decision import check_preconditions, classify_action
from tok.deterministic.exact_patch import apply_exact_patch
from tok.deterministic.models import (
    DeterministicActionDecision,
    DeterministicActionReceipt,
    ExactPatchPreconditions,
)
from tok.deterministic.receipts import (
    append_deterministic_receipt,
    deterministic_receipt_path,
    read_deterministic_receipts,
)

__all__ = [
    "apply_exact_patch",
    "check_preconditions",
    "classify_action",
    "DeterministicActionDecision",
    "DeterministicActionReceipt",
    "ExactPatchPreconditions",
    "append_deterministic_receipt",
    "deterministic_receipt_path",
    "read_deterministic_receipts",
]
