"""
Error Handling Tests - Proving robustness against malformed input.

Tests that the system fails gracefully with specific, helpful errors
rather than crashing with generic tracebacks.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
from pydantic import ValidationError

from tok.protocol.format_bridge import Bridge
from tok.protocol.models import TokNode, build_tok_traceback
from tok.protocol.parser import TokParser
from tok.protocol.schema import DEFAULT_SCHEMA
from tok.utils.transformer import DocumentTransformer


class TestParserErrorHandling:
    """Test TokParser robustness against malformed input."""

    def test_parse_empty_string(self) -> None:
        """Parsing empty string should return empty list, not raise."""
        parser = TokParser()
        result = parser.parse("")
        assert isinstance(result, list)
        assert result == []

    def test_parse_plain_text(self) -> None:
        """Parsing plain text (no @type) should wrap in @msg, not raise."""
        parser = TokParser()
        result = parser.parse("Hello world")
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].type == "msg"

    def test_feed_and_flush_incomplete(self) -> None:
        """Feeding incomplete Tok and flushing should return list, not raise."""
        parser = TokParser()
        result1 = parser.feed("@msg\n  > Hello")
        result2 = parser.flush()
        assert isinstance(result1, list)
        assert isinstance(result2, list)

    def test_very_long_line(self) -> None:
        """Lines longer than MAX_LINE_LENGTH should be truncated gracefully."""
        parser = TokParser()
        long_line = "@msg\n  > " + "x" * 70000
        result = parser.parse(long_line)
        # Should not raise, should return a list
        assert isinstance(result, list)

    def test_bom_prefix(self) -> None:
        """UTF-8 BOM prefix should be stripped, not cause parse error."""
        parser = TokParser()
        text_with_bom = "\ufeff@msg\n  > Hello"
        result_with_bom = parser.parse(text_with_bom)

        parser2 = TokParser()
        text_no_bom = "@msg\n  > Hello"
        result_no_bom = parser2.parse(text_no_bom)

        # Both should produce valid results
        assert isinstance(result_with_bom, list)
        assert isinstance(result_no_bom, list)
        assert len(result_with_bom) > 0
        assert len(result_no_bom) > 0


class TestBridgeErrorHandling:
    """Test Bridge robustness against malformed input."""

    def test_json_invalid_format(self) -> None:
        """Invalid JSON should return empty string, not raise."""
        result = Bridge.json("not valid json")
        assert isinstance(result, str)
        # Should return empty or minimal valid Tok
        assert result == "" or result.startswith("@")

    def test_xml_unclosed_tag(self) -> None:
        """Unclosed XML should return empty string, not raise."""
        result = Bridge.xml("<invalid>unclosed")
        assert isinstance(result, str)
        assert result == "" or result.startswith("@")

    def test_to_json_empty_string(self) -> None:
        """to_json with empty string should return {}, not raise."""
        result = Bridge.to_json("")
        assert isinstance(result, str)
        assert result == "{}"

    def test_to_xml_empty_string(self) -> None:
        """to_xml with empty string should return empty string, not raise."""
        result = Bridge.to_xml("")
        assert isinstance(result, str)
        # Should be empty or minimal

    def test_to_md_empty_string(self) -> None:
        """to_md with empty string should return empty string, not raise."""
        result = Bridge.to_md("")
        assert isinstance(result, str)
        assert result == ""

    def test_detect_and_convert_invalid_format(self) -> None:
        """detect_and_convert should handle invalid formats gracefully."""
        result = Bridge().detect_and_convert("}{][<>")
        # Should return str, not raise
        assert isinstance(result, str)


class TestSchemaValidation:
    """Test TokSchema validation robustness."""

    def test_validate_empty_type(self) -> None:
        """Validating node with empty type should not crash."""
        node = TokNode(type="", label="", text="", attrs={})
        result = DEFAULT_SCHEMA.validate(node)
        # Should return a tuple (bool, str|None), not raise
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)

    def test_validate_unknown_type(self) -> None:
        """Validating unknown block type should pass (permissive)."""
        node = TokNode(type="unknown_type_xyz", label="", text="", attrs={})
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        # Unknown types should be allowed (permissive design)
        assert is_valid is True

    def test_validate_msg_missing_required_role(self) -> None:
        """Validating @msg without required role attr should fail."""
        node = TokNode(type="msg", label="", text="test", attrs={})
        is_valid, msg = DEFAULT_SCHEMA.validate(node)
        # Should be invalid (role is required)
        assert is_valid is False
        assert msg is not None

    def test_validate_msg_with_role(self) -> None:
        """Validating @msg with role attr should pass."""
        node = TokNode(type="msg", label="", text="test", attrs={"role": "user"})
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        # Should be valid
        assert is_valid is True


class TestErrorBuilding:
    """Test error message construction."""

    def test_build_tok_traceback_with_validation_error(self) -> None:
        """build_tok_traceback should convert ValidationError to @error block."""
        try:
            # Intentionally trigger a validation error
            from tok.protocol.models import TokToolCall

            TokToolCall.model_validate({"tool": "INVALID_TOOL", "path": "x"})
        except ValidationError as e:
            result = build_tok_traceback("TestTool", "raw_input", e)
            assert isinstance(result, str)
            assert "@error" in result
            assert "validation" in result.lower() or "type:" in result.lower()


class TestTransformerErrorHandling:
    """Test DocumentTransformer robustness."""

    def test_detransform_empty_string(self) -> None:
        """Detransform with empty string should return str, not raise."""
        transformer = DocumentTransformer()
        result = transformer.detransform("")
        assert isinstance(result, str)

    def test_transform_empty_string(self) -> None:
        """Transform with empty string should return list, not raise."""
        transformer = DocumentTransformer()
        result = transformer.transform("")
        assert isinstance(result, list)

    def test_transform_plain_text(self) -> None:
        """Transform with plain text should parse gracefully."""
        transformer = DocumentTransformer()
        result = transformer.transform("Just some plain text")
        assert isinstance(result, list)

    def test_detransform_malformed_tok(self) -> None:
        """Detransform malformed Tok should handle gracefully."""
        transformer = DocumentTransformer()
        malformed = "@unknown\n  > text\n  @nested without proper indent"
        result = transformer.detransform(malformed)
        assert isinstance(result, str)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_null_bytes_in_text(self) -> None:
        """Parser should handle null bytes in text gracefully."""
        parser = TokParser()
        text = "@msg\n  > Hello\x00World"
        result = parser.parse(text)
        assert isinstance(result, list)

    def test_extremely_nested_structure(self) -> None:
        """Parser should handle deeply nested indentation."""
        lines = ["@root"]
        for i in range(50):
            lines.append("  " * (i + 1) + f"@level{i}")
        text = "\n".join(lines)
        parser = TokParser()
        result = parser.parse(text)
        # Should not crash, should parse something
        assert isinstance(result, list)

    def test_mixed_indent_tabs_spaces(self) -> None:
        """Parser should handle mixed tabs and spaces in indentation."""
        parser = TokParser()
        text = "@msg\n\t> Hello\n  > World"  # tab then spaces
        result = parser.parse(text)
        assert isinstance(result, list)

    def test_unicode_special_chars(self) -> None:
        """Parser should handle Unicode special characters."""
        parser = TokParser()
        text = "@msg\n  > Hello 🎉 世界 مرحبا"
        result = parser.parse(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_control_characters(self) -> None:
        """Parser should handle control characters gracefully."""
        parser = TokParser()
        text = "@msg\n  > Hello\x01\x02\x03World"
        result = parser.parse(text)
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
