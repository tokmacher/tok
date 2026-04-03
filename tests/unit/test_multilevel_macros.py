"""Test multi-level macro resolution up to 4 layers deep.

Architecture:
  Layer 0 (Primitives): add, mul, filter, sort — raw IR ops
  Layer 1: @double(x) -> add($x, $x)
  Layer 2: @quad(x)   -> @double(@double($x))  i.e. calls @double twice
  Layer 3: @octet(x)  -> @double(@quad($x))     calls @quad then @double
  Layer 4: @hex(x)    -> @double(@octet($x))    calls @octet then @double

Expected: @hex(3) = 3 * 16 = 48
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tok.neuro.ir import (
    Instruction,
    TokIR,
    Macro,
    MacroProvenance,
    MacroRegistry,
    execute_ir,
)


def build_layered_registry() -> MacroRegistry:
    """Build a MacroRegistry with 4 layers of nested macros."""
    registry = MacroRegistry()

    # ── Layer 1: @double(x) = x + x ──
    registry.register(
        Macro(
            name="double",
            inputs=("x",),
            instructions=(
                Instruction(op="add", args=("$x", "$x"), target="result"),
                Instruction(op="identity", args=("$result",)),
            ),
            provenance=MacroProvenance(source_code="add($x, $x)"),
        )
    )

    # ── Layer 2: @quad(x) = @double(@double(x)) ──
    # Step 1: t1 = @double(x)   -> 2x
    # Step 2: t2 = @double(t1)  -> 4x
    registry.register(
        Macro(
            name="quad",
            inputs=("x",),
            instructions=(
                Instruction(op="@double", args=("$x",), target="t1"),
                Instruction(op="@double", args=("$t1",), target="result"),
                Instruction(op="identity", args=("$result",)),
            ),
            provenance=MacroProvenance(
                source_code="@double(@double($x))",
                composed_of=("double",),
            ),
        )
    )

    # ── Layer 3: @octet(x) = @double(@quad(x)) ──
    # Step 1: t1 = @quad(x)   -> 4x
    # Step 2: t2 = @double(t1) -> 8x
    registry.register(
        Macro(
            name="octet",
            inputs=("x",),
            instructions=(
                Instruction(op="@quad", args=("$x",), target="t1"),
                Instruction(op="@double", args=("$t1",), target="result"),
                Instruction(op="identity", args=("$result",)),
            ),
            provenance=MacroProvenance(
                source_code="@double(@quad($x))",
                composed_of=("quad", "double"),
            ),
        )
    )

    # ── Layer 4: @hex(x) = @double(@octet(x)) ──
    # Step 1: t1 = @octet(x)  -> 8x
    # Step 2: t2 = @double(t1) -> 16x
    registry.register(
        Macro(
            name="hex",
            inputs=("x",),
            instructions=(
                Instruction(op="@octet", args=("$x",), target="t1"),
                Instruction(op="@double", args=("$t1",), target="result"),
                Instruction(op="identity", args=("$result",)),
            ),
            provenance=MacroProvenance(
                source_code="@double(@octet($x))",
                composed_of=("octet", "double"),
            ),
        )
    )

    return registry


def test_layer_1() -> None:
    """@double(5) = 10"""
    registry = build_layered_registry()
    ir = TokIR(instructions=(Instruction(op="@double", args=(5,)),))
    result = execute_ir(ir, {}, registry)
    assert result == 10, f"Expected 10, got {result}"
    print(f"  ✅ Layer 1: @double(5) = {result}")


def test_layer_2() -> None:
    """@quad(5) = 20"""
    registry = build_layered_registry()
    ir = TokIR(instructions=(Instruction(op="@quad", args=(5,)),))
    result = execute_ir(ir, {}, registry)
    assert result == 20, f"Expected 20, got {result}"
    print(f"  ✅ Layer 2: @quad(5) = {result}")


def test_layer_3() -> None:
    """@octet(5) = 40"""
    registry = build_layered_registry()
    ir = TokIR(instructions=(Instruction(op="@octet", args=(5,)),))
    result = execute_ir(ir, {}, registry)
    assert result == 40, f"Expected 40, got {result}"
    print(f"  ✅ Layer 3: @octet(5) = {result}")


def test_layer_4() -> None:
    """@hex(5) = 80"""
    registry = build_layered_registry()
    ir = TokIR(instructions=(Instruction(op="@hex", args=(5,)),))
    result = execute_ir(ir, {}, registry)
    assert result == 80, f"Expected 80, got {result}"
    print(f"  ✅ Layer 4: @hex(5) = {result}")


def test_provenance_chain() -> None:
    """Verify that provenance tracks the full composition chain."""
    registry = build_layered_registry()
    hex_macro = registry.get("hex")
    octet_macro = registry.get("octet")
    quad_macro = registry.get("quad")
    double_macro = registry.get("double")

    assert hex_macro is not None, "hex macro not found"
    assert octet_macro is not None, "octet macro not found"
    assert quad_macro is not None, "quad macro not found"
    assert double_macro is not None, "double macro not found"

    assert hex_macro.provenance is not None
    assert "octet" in hex_macro.provenance.composed_of
    assert "double" in hex_macro.provenance.composed_of
    print(
        f"  ✅ Provenance: @hex composed_of = {hex_macro.provenance.composed_of}"
    )

    assert octet_macro.provenance is not None
    assert "quad" in octet_macro.provenance.composed_of
    print(
        f"  ✅ Provenance: @octet composed_of = {octet_macro.provenance.composed_of}"
    )

    assert quad_macro.provenance is not None
    assert "double" in quad_macro.provenance.composed_of
    print(
        f"  ✅ Provenance: @quad composed_of = {quad_macro.provenance.composed_of}"
    )

    assert double_macro.provenance is not None
    assert double_macro.provenance.composed_of == ()
    print("  ✅ Provenance: @double is a leaf macro (no composition)")


def test_mixed_ir_with_macros() -> None:
    """Test an IR program that mixes raw ops with multi-level macro calls."""
    registry = build_layered_registry()

    # Program: take input x, quadruple it, then add 1
    # Result for x=3: @quad(3) = 12, then 12 + 1 = 13
    ir = TokIR(
        instructions=(
            Instruction(op="@quad", args=(3,), target="t1"),
            Instruction(op="add", args=("$t1", 1), target="t2"),
            Instruction(op="identity", args=("$t2",)),
        )
    )
    result = execute_ir(ir, {}, registry)
    assert result == 13, f"Expected 13, got {result}"
    print(f"  ✅ Mixed IR: @quad(3) + 1 = {result}")


def test_deep_chain_with_variable_input() -> None:
    """Full 4-layer chain from variable scope."""
    registry = build_layered_registry()

    ir = TokIR(
        instructions=(
            Instruction(op="const", args=(7,), target="val"),
            Instruction(op="@hex", args=("$val",), target="result"),
            Instruction(op="identity", args=("$result",)),
        )
    )
    result = execute_ir(ir, {}, registry)
    assert result == 112, f"Expected 112 (7*16), got {result}"
    print(f"  ✅ Deep chain with variable: const(7) -> @hex = {result}")


def test_serialization_round_trip() -> None:
    """Test that multi-level macros survive BridgeMemoryState serialization."""
    from tok.runtime.memory.bridge_memory import BridgeMemoryState

    state = BridgeMemoryState()
    registry = build_layered_registry()
    state.macro_registry = registry

    # Serialize to tok format
    tok_text = state.to_tok()
    print("\n  --- Serialized Tok (macros section): ---")
    for line in tok_text.splitlines():
        if "@macro" in line.lower() or "|>" in line:
            print(f"    {line}")

    # Deserialize
    restored = BridgeMemoryState.from_tok(tok_text, load_global_macros=False)
    assert len(restored.macro_registry.macros) == len(registry.macros), (
        f"Expected {len(registry.macros)} macros, got {len(restored.macro_registry.macros)}"
    )
    print(
        f"\n  ✅ Round-trip: {len(restored.macro_registry.macros)} macros survived serialization"
    )

    # Verify we can still execute after round-trip
    ir = TokIR(instructions=(Instruction(op="@double", args=(5,)),))
    result = execute_ir(ir, {}, restored.macro_registry)
    assert result == 10, f"Expected 10 after round-trip, got {result}"
    print(f"  ✅ Round-trip execution: @double(5) = {result}")


if __name__ == "__main__":
    tests = [
        ("Layer 1: @double", test_layer_1),
        ("Layer 2: @quad", test_layer_2),
        ("Layer 3: @octet", test_layer_3),
        ("Layer 4: @hex", test_layer_4),
        ("Provenance Chain", test_provenance_chain),
        ("Mixed IR + Macros", test_mixed_ir_with_macros),
        ("Deep Chain w/ Variable", test_deep_chain_with_variable_input),
        ("Serialization Round-Trip", test_serialization_round_trip),
    ]

    print("\n=== Multi-Level Macro Test Suite ===\n")
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"[{name}]")
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
        print()

    print(f"=== Results: {passed}/{passed + failed} passed ===")
    if failed:
        sys.exit(1)
