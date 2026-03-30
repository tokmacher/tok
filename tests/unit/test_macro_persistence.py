import unittest
import os
import datetime
import shutil
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from tok.neuro.ir import Instruction, Macro, MacroRegistry, MacroProvenance
from tok.neuro.miner import IRPatternMiner


class TestMacroPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("/tmp/tok_test_macros")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)
        self.test_path = self.test_dir / "macros.tok"

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_serialization_round_trip(self):
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

        self.assertEqual(restored.name, "test_macro")
        self.assertEqual(len(restored.instructions), 1)
        self.assertEqual(restored.instructions[0].op, "add")
        self.assertEqual(restored.hit_count, 5)
        self.assertTrue(restored.is_durable)
        self.assertEqual(restored.provenance.source_code, "add($p0, $p1)")

    def test_registry_persistence(self):
        registry = MacroRegistry()
        m1 = Macro(name="m1", instructions=(), inputs=(), hit_count=10)
        m2 = Macro(
            name="m2",
            instructions=(),
            inputs=(),
            hit_count=2,
            is_durable=False,
        )
        registry.register(m1)
        registry.register(m2)

        registry.save_global(str(self.test_path))
        self.assertTrue(self.test_path.exists())

        new_registry = MacroRegistry()
        new_registry.load_global(str(self.test_path))

        self.assertIn("m1", new_registry.macros)
        self.assertIn("m2", new_registry.macros)
        self.assertEqual(new_registry.get("m1").hit_count, 10)

    def test_decay_logic(self):
        registry = MacroRegistry()
        old_date = (
            datetime.datetime.now() - datetime.timedelta(days=10)
        ).isoformat()
        recent_date = datetime.datetime.now().isoformat()

        # 1. Old and low hits -> should be pruned
        registry.register(
            Macro(
                name="old_low",
                instructions=(),
                inputs=(),
                hit_count=1,
                last_seen=old_date,
            )
        )
        # 2. Old but high hits -> should stay (min_hits=3 in test)
        registry.register(
            Macro(
                name="old_high",
                instructions=(),
                inputs=(),
                hit_count=10,
                last_seen=old_date,
            )
        )
        # 3. Old but durable -> should stay
        registry.register(
            Macro(
                name="old_durable",
                instructions=(),
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
                instructions=(),
                inputs=(),
                hit_count=1,
                last_seen=recent_date,
            )
        )

        registry.apply_decay(max_age_days=7, min_hits=3)

        self.assertNotIn("old_low", registry.macros)
        self.assertIn("old_high", registry.macros)
        self.assertIn("old_durable", registry.macros)
        self.assertIn("recent_low", registry.macros)

    def test_hierarchical_mining_naming(self):
        miner = IRPatternMiner(min_frequency=2)
        # Mock an existing macro from a previous session
        existing = {
            "auto_macro_0": Macro(
                name="auto_macro_0", instructions=(), inputs=()
            )
        }

        # Pattern that calls an existing macro
        from tok.neuro.ir import TokIR

        history = TokIR(
            instructions=(
                Instruction(op="@auto_macro_0", args=()),
                Instruction(op="pytest", args=()),
                Instruction(op="@auto_macro_0", args=()),
                Instruction(op="pytest", args=()),
            )
        )

        discovered = miner.mine([history], existing_macros=existing)

        self.assertEqual(len(discovered), 1)
        new_macro = discovered[0]
        # Should be named auto_macro_1 to avoid collision
        self.assertEqual(new_macro.name, "auto_macro_1")
        # Should show it is composed of auto_macro_0
        self.assertIn("auto_macro_0", new_macro.provenance.composed_of)
        self.assertEqual(
            new_macro.provenance.source_code, "@auto_macro_0 -> pytest"
        )


if __name__ == "__main__":
    unittest.main()
