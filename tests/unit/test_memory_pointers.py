from tok.memory.pointers import PointerRegistry
from tok.runtime.memory.bridge_memory import BridgeMemoryState, MemoryEntry
from tok.universal_runtime import SemanticValidator


class TestPointerRegistry:
    def test_basic_pointers(self):
        reg = PointerRegistry()
        p1 = reg.get_pointer("/path/to/very/long/file/name.py")
        assert p1 == "*A"
        assert reg.resolve("*A") == "/path/to/very/long/file/name.py"

        p2 = reg.get_pointer("/another/long/path/here.py")
        assert p2 == "*B"

        # Redundant call returns same pointer
        assert reg.get_pointer("/path/to/very/long/file/name.py") == "*A"

    def test_extension(self):
        reg = PointerRegistry()
        for i in range(30):
            reg.get_pointer(f"path_{i}")
        assert "*A1" in reg.map  # Should have wrapped or extended

    def test_serialization(self):
        reg = PointerRegistry()
        reg.get_pointer("long_path_1")
        reg.get_pointer("long_path_2")
        tok = reg.to_tok()
        assert "@pointers" in tok
        assert "*A=long_path_1" in tok

        reg2 = PointerRegistry.from_tok(tok)
        assert reg2.resolve("*A") == "long_path_1"
        assert reg2.resolve("*B") == "long_path_2"


class TestSemanticValidator:
    def test_redundant_prose_detection(self):
        validator = SemanticValidator()
        text = "I have successfully read the file. Here is the content of the file you requested."
        signals = validator.validate_drift(text, {})
        assert signals.get("semantic_drift_detected")

    def test_protocol_reinforcement(self):
        validator = SemanticValidator()
        text = "### Analysis\nThe code looks good."  # Raw markdown header
        signals = validator.validate_drift(text, {})
        assert signals.get("semantic_pressure_detected")


class TestBridgeMemoryPointers:
    def test_wire_state_with_pointers(self):
        state = BridgeMemoryState()
        long_path = "src/tok/universal_runtime.py"
        state.hot["files"] = [MemoryEntry(value=long_path)]

        wire = state.wire_state()
        assert "*A" in wire
        # The path should be hidden from the protocol line (the first line),
        # even if it's present in the @pointers footer.
        header = wire.split("\n")[0]
        assert long_path not in header
        assert state.pointers.resolve("*A") == long_path
