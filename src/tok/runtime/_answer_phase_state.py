"""Answer-phase signalling state extracted from RuntimeSession."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnswerPhaseState:
    """Tracks answer-ready and late-answer-assembly repair signals."""

    answer_ready_repair_pending: bool = False
    answer_ready_repair_active: bool = False
    late_assembly_repair_pending: bool = False
    late_assembly_repair_active: bool = False
    late_assembly_repair_mode_pending: str = ""
    late_assembly_repair_mode_active: str = ""
    late_followthrough_pending: bool = False
    late_followthrough_active: bool = False
    answer_phase_expected_this_turn: bool = False
    natural_response_acceptable_this_turn: bool = False

    def reset(self) -> None:
        self.answer_ready_repair_pending = False
        self.answer_ready_repair_active = False
        self.late_assembly_repair_pending = False
        self.late_assembly_repair_active = False
        self.late_assembly_repair_mode_pending = ""
        self.late_assembly_repair_mode_active = ""
        self.late_followthrough_pending = False
        self.late_followthrough_active = False
        self.answer_phase_expected_this_turn = False
        self.natural_response_acceptable_this_turn = False
