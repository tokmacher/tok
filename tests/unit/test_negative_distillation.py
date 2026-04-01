from typing import cast

from tok.neuro.ir import Instruction, TokIR, Macro, MacroRegistry
from tok.neuro.distill import MemoryDistiller
from tok.neuro.llm_clients import StubClient
from tok.neuro.memory import EpisodeMemory, ConstraintMemory


def test_negative_distillation_redundancy():
    registry = MacroRegistry()

    # Define an existing macro in the registry
    ins = (Instruction(op="read", args=("path",), target="res"),)
    m_existing = Macro(name="base_read", instructions=ins, inputs=("path",))
    registry.register(m_existing)

    # Create episodes where the agent redundantly defines the same macro
    memory = []
    tokens = frozenset(["test"])
    for i in range(3):
        ir = TokIR()
        ir.add_macro(m_existing)  # Redundant definition attempt

        memory.append(
            EpisodeMemory(
                tokens=tokens,
                question=f"task_{i}",
                answer="done",
                ok=True,
                metadata={"ir": ir},
            )
        )

    distiller = MemoryDistiller(llm=StubClient(), threshold=10)
    distiller.registry = registry  # Use our primed registry

    typed_memory = cast(list[EpisodeMemory], memory)
    negative_lessons = distiller.mine_negative_patterns(typed_memory)

    assert len(negative_lessons) == 1
    assert isinstance(negative_lessons[0], ConstraintMemory)
    assert "avoid re-defining @base_read" in negative_lessons[0].constraint
    assert negative_lessons[0].tokens == frozenset(["redundancy", "base_read"])
