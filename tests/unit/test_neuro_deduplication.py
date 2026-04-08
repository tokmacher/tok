from typing import cast

from tok.neuro.distill import MemoryDistiller
from tok.neuro.ir import Instruction, Macro, MacroRegistry
from tok.neuro.llm_clients import StubClient
from tok.neuro.memory import EpisodeMemory, LessonMemory
from tok.neuro.metrics import estimated_token_count


def test_production_macro_registry_deduplication() -> None:
    registry = MacroRegistry()

    # Define two macros with same instructions but different names
    ins = (Instruction(op="add", args=("$p0", "$p1"), target="res"),)
    m1 = Macro(name="add_macro", instructions=ins, inputs=("p0", "p1"))
    m2 = Macro(name="duplicate_add", instructions=ins, inputs=("p0", "p1"))

    name1 = registry.register(m1)
    name2 = registry.register(m2)

    assert name1 == "add_macro"
    assert name2 == "add_macro"  # Should return the first one's name
    assert len(registry.macros) == 1


def test_op_sequence_deduplication_across_args() -> None:
    """Macros with the same op sequence but different concrete args are op-seq duplicates."""
    registry = MacroRegistry()

    ins1 = (Instruction(op="grep", args=("parse_error", "src/tok/parser.py"), target=None),)
    ins2 = (Instruction(op="grep", args=("other_term", "src/tok/lexer.py"), target=None),)
    m1 = Macro(name="macro_session_1", instructions=ins1, inputs=("p0", "p1"))
    m2 = Macro(name="macro_session_2", instructions=ins2, inputs=("p0", "p1"))

    registry.register(m1)
    dup = registry.find_op_sequence_duplicate(m2)

    assert dup == "macro_session_1"
    # Registering m2 via the miner path (which now uses find_op_sequence_duplicate)
    # should not add a second macro
    assert len(registry.macros) == 1


def test_production_distiller_compaction() -> None:
    class MockLLM(StubClient):
        def chat(self, system: str, user: str) -> str:
            return "Rule: Distilled Skill"

    llm = MockLLM()
    distiller = MemoryDistiller(llm=llm, threshold=3)

    # Create 5 identical episodes with enough text to show compaction
    tokens = frozenset(["test_token"])
    memory = []
    for i in range(5):
        memory.append(
            EpisodeMemory(
                tokens=tokens,
                question=f"This is a very long question about complex task {i} that requires a lot of context and reasoning.",
                answer=f"The answer to this complex task {i} involves several steps and a detailed explanation of the logic applied.",
                ok=True,
            )
        )

    initial_tokens = estimated_token_count(memory)
    typed_memory = cast("list[EpisodeMemory]", memory)
    new_memory = distiller.compress(typed_memory)
    final_tokens = estimated_token_count(new_memory)

    assert len(new_memory) == 1
    assert isinstance(new_memory[0], LessonMemory)
    assert final_tokens < initial_tokens
