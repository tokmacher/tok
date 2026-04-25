from tok.memory.pointers import PointerRegistry
from tok.runtime.memory.bridge_memory import BridgeMemoryState, MemoryEntry
from tok.universal_runtime import SemanticValidator


class TestPointerRegistry:
    def test_basic_pointers(self) -> None:
        reg = PointerRegistry()
        p1 = reg.get_pointer("/path/to/very/long/file/name.py")
        assert p1 == "*A"
        assert reg.resolve("*A") == "/path/to/very/long/file/name.py"

        p2 = reg.get_pointer("/another/long/path/here.py")
        assert p2 == "*B"

        # Redundant call returns same pointer
        assert reg.get_pointer("/path/to/very/long/file/name.py") == "*A"

    def test_extension(self) -> None:
        reg = PointerRegistry()
        for i in range(30):
            reg.get_pointer(f"path_{i}")
        assert "*A1" in reg.map  # Should have wrapped or extended

    def test_serialization(self) -> None:
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
    def test_redundant_prose_detection(self) -> None:
        validator = SemanticValidator()
        # Unambiguous filler phrase with no Tok markers — should flag drift.
        text = "Certainly! I'll be happy to help you with that right away."
        signals = validator.validate_drift(text, {})
        assert signals.get("semantic_drift_detected")

    def test_protocol_reinforcement(self) -> None:
        validator = SemanticValidator()
        text = "### Analysis\nThe code looks good."  # Raw markdown header
        signals = validator.validate_drift(text, {})
        assert signals.get("semantic_pressure_detected")

    def test_no_false_positive_on_tok_response_with_bullets(self) -> None:
        validator = SemanticValidator()
        # Well-formed Tok response: has >>> marker and bullet lines — not drift.
        text = ">>> turns:3|goal:fix bug\n@msg role:assistant\n|> Here are the results:\n- file.py fixed\n- tests pass"
        signals = validator.validate_drift(text, {})
        assert not signals.get("semantic_drift_detected")

    def test_no_false_positive_on_long_tok_response(self) -> None:
        validator = SemanticValidator()
        # Long response with @msg block — should not trigger Case B.
        text = ">>> turns:5|goal:refactor\n@msg role:assistant\n|> " + " ".join(["word"] * 50)
        signals = validator.validate_drift(text, {})
        assert not signals.get("semantic_drift_detected")

    def test_no_false_positive_on_prose_with_legitimate_words(self) -> None:
        validator = SemanticValidator()
        # "successfully" and "investigate" appeared in old phrase list but are
        # legitimate in Tok-formatted responses. Verify they no longer trigger drift.
        text = ">>> turns:2\n@msg role:assistant\n|> The tests ran successfully. Please investigate the logs."
        signals = validator.validate_drift(text, {})
        assert not signals.get("semantic_drift_detected")

    def test_long_prose_without_tok_markers_is_drift(self) -> None:
        validator = SemanticValidator()
        # >40 words, no Tok markers — genuine prose leak.
        text = " ".join(["word"] * 50)
        signals = validator.validate_drift(text, {})
        assert signals.get("semantic_drift_detected")

    def test_bullets_without_tok_markers_is_drift(self) -> None:
        validator = SemanticValidator()
        text = "Here is my analysis:\n- first point about the code\n- second point about tests"
        signals = validator.validate_drift(text, {})
        assert signals.get("semantic_drift_detected")


class TestBridgeMemoryPointers:
    def test_wire_state_with_pointers(self) -> None:
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
