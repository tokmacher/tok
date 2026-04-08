import datetime
import os
import shutil
import sys
import unittest
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tok.neuro.ir import Instruction, Macro, MacroProvenance, MacroRegistry
from tok.neuro.miner import IRPatternMiner


class TestMacroPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path("/tmp/tok_test_macros")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)
        self.test_path = self.test_dir / "macros.tok"

    def tearDown(self) -> None:
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_serialization_round_trip(self) -> None:
        instruction = Instruction(op="add", args=("$p0", "$p1"), target="res")
        macro = Macro(
            name="test_macro",
            instructions=(instruction,),
            inputs=("p0", "p1"),
            hit_count=5,
            last_seen=datetime.datetime.now().isoformat(),
            is_durable=True,
            provenance=MacroProvenance(source_code="add($p0, $p1)"),
        )

        data = macro.to_dict()
        restored = Macro.from_dict(data)

        assert restored.name == "test_macro"
        assert len(restored.instructions) == 1
        assert restored.instructions[0].op == "add"
        assert restored.hit_count == 5
        assert restored.is_durable
        assert restored.provenance is not None
        if restored.provenance is not None:
            assert restored.provenance.source_code == "add($p0, $p1)"

    def test_registry_persistence(self) -> None:
        registry = MacroRegistry()
        m1 = Macro(
            name="m1",
            instructions=(Instruction(op="add", args=()),),
            inputs=(),
            hit_count=10,
        )
        m2 = Macro(
            name="m2",
            instructions=(Instruction(op="grep", args=()),),
            inputs=(),
            hit_count=2,
            is_durable=False,
        )
        registry.register(m1)
        registry.register(m2)

        registry.save_global(str(self.test_path))
        assert self.test_path.exists()

        new_registry = MacroRegistry()
        new_registry.load_global(str(self.test_path))

        assert "m1" in new_registry.macros
        assert "m2" in new_registry.macros
        m1_macro = new_registry.get("m1")
        assert m1_macro is not None
        if m1_macro is not None:
            assert m1_macro.hit_count == 10

    def test_decay_logic(self) -> None:
        registry = MacroRegistry()
        old_date = (datetime.datetime.now() - datetime.timedelta(days=10)).isoformat()
        recent_date = datetime.datetime.now().isoformat()

        # 1. Old and low hits -> should be pruned
        registry.register(
            Macro(
                name="old_low",
                instructions=(Instruction(op="op1", args=()),),
                inputs=(),
                hit_count=1,
                last_seen=old_date,
            )
        )
        # 2. Old but high hits -> should stay (min_hits=3 in test)
        registry.register(
            Macro(
                name="old_high",
                instructions=(Instruction(op="op2", args=()),),
                inputs=(),
                hit_count=10,
                last_seen=old_date,
            )
        )
        # 3. Old but durable -> should stay
        registry.register(
            Macro(
                name="old_durable",
                instructions=(Instruction(op="op3", args=()),),
                inputs=(),
                hit_count=1,
                last_seen=old_date,
                is_durable=True,
            )
        )
        # 4. Recent and low hits -> should stay
        registry.register(
            Macro(
                name="recent_low",
                instructions=(Instruction(op="op4", args=()),),
                inputs=(),
                hit_count=1,
                last_seen=recent_date,
            )
        )

        registry.apply_decay(max_age_days=7, min_hits=3)

        assert "old_low" not in registry.macros
        assert "old_high" in registry.macros
        assert "old_durable" in registry.macros
        assert "recent_low" in registry.macros

    def test_hierarchical_mining_naming(self) -> None:
        miner = IRPatternMiner(min_frequency=2)
        existing_registry = MacroRegistry()
        existing_registry.register(Macro(name="auto_macro_0", instructions=(), inputs=()))

        from tok.neuro.ir import TokIR

        history = TokIR(
            instructions=(
                Instruction(op="@auto_macro_0", args=()),
                Instruction(op="pytest", args=()),
                Instruction(op="@auto_macro_0", args=()),
                Instruction(op="pytest", args=()),
            )
        )

        discovered = miner.mine([history], registry=existing_registry)

        assert len(discovered) == 1
        new_macro = discovered[0]
        assert new_macro.name == "auto_macro_1"
        assert new_macro.provenance is not None
        if new_macro.provenance is not None:
            assert "auto_macro_0" in new_macro.provenance.composed_of
            assert new_macro.provenance.source_code is not None
            if new_macro.provenance.source_code is not None:
                assert "@auto_macro_0 -> pytest" in new_macro.provenance.source_code


if __name__ == "__main__":
    unittest.main()
