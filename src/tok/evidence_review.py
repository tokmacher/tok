"""Backward-compatible facade for tok.analysis.evidence_review."""

from __future__ import annotations

from .analysis.evidence_review import *  # noqa: F403
from .analysis.evidence_review import (  # noqa: F401
    build_coverage_report,
    load_stress_evidence,
    rank_candidates,
    review_capture_dir,
    summarize_capture_file,
)
