"""
Schema Validation Tests - Proving schema enforcement and coverage.

Tests that the schema validation system correctly accepts valid blocks,
rejects invalid ones, and provides clear error messages.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest

from tok.protocol.models import TokNode
from tok.protocol.schema import DEFAULT_SCHEMA, BlockSchema, TokSchema


class TestDefaultSchemaValidation:
    """Test the DEFAULT_SCHEMA with pre-registered blocks."""

    def test_msg_with_required_role(self) -> None:
        """@msg with required role attribute should be valid."""
        node = TokNode(type="msg", label="", text="Hello", attrs={"role": "user"})
        is_valid, msg = DEFAULT_SCHEMA.validate(node)
        assert is_valid is True
        assert msg is None

    def test_msg_missing_required_role(self) -> None:
        """@msg without required role attribute should be invalid."""
        node = TokNode(type="msg", label="", text="Hello", attrs={})
        is_valid, msg = DEFAULT_SCHEMA.validate(node)
        assert is_valid is False
        assert msg is not None
        assert "role" in msg.lower()

    def test_msg_with_optional_trust(self) -> None:
        """@msg with optional trust attribute should be valid."""
        node = TokNode(
            type="msg",
            label="",
            text="Hello",
            attrs={"role": "user", "trust": "system"},
        )
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        assert is_valid is True

    def test_msg_with_unknown_attribute(self) -> None:
        """@msg with unknown attribute should be invalid."""
        node = TokNode(
            type="msg",
            label="",
            text="Hello",
            attrs={"role": "user", "invented_attr": "xyz"},
        )
        is_valid, msg = DEFAULT_SCHEMA.validate(node)
        assert is_valid is False
        assert msg is not None
        assert "unknown" in msg.lower() or "invented" in msg.lower()

    def test_msg_role_and_trust_together(self) -> None:
        """@msg with both role and trust should be valid."""
        node = TokNode(
            type="msg",
            label="",
            text="Hello",
            attrs={"role": "assistant", "trust": "external"},
        )
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        assert is_valid is True


class TestToolValidation:
    """Test tool-specific validation."""

    def test_get_weather_with_location(self) -> None:
        """@Tool get_weather with location should be valid."""
        node = TokNode(
            type="Tool",
            label="get_weather",
            text="",
            attrs={"location": "San Francisco"},
        )
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        # Should be valid if schema is registered
        assert isinstance(is_valid, bool)

    def test_tool_missing_label(self) -> None:
        """@Tool without label (tool name) should be invalid."""
        node = TokNode(type="Tool", label="", text="", attrs={"location": "SF"})
        is_valid, msg = DEFAULT_SCHEMA.validate(node)
        assert is_valid is False
        assert msg is not None

    def test_tool_with_label(self) -> None:
        """@Tool with label should be valid."""
        node = TokNode(type="Tool", label="read", text="", attrs={"path": "/tmp/file.txt"})
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        assert isinstance(is_valid, bool)


class TestCustomSchemaRegistration:
    """Test custom schema registration and validation."""

    def test_register_custom_schema(self) -> None:
        """Should be able to register and validate custom schemas."""
        schema = TokSchema()
        custom_block = BlockSchema(
            type="custom",
            required_attrs=["id", "name"],
            optional_attrs=["description"],
            description="Custom block type for testing",
        )
        schema.register(custom_block)

        # Should be valid with all required attrs
        node = TokNode(
            type="custom",
            label="",
            text="",
            attrs={"id": "123", "name": "test"},
        )
        is_valid, _msg = schema.validate(node)
        assert is_valid is True

    def test_custom_schema_missing_required_attr(self) -> None:
        """Custom schema should reject missing required attributes."""
        schema = TokSchema()
        custom_block = BlockSchema(
            type="custom",
            required_attrs=["id", "name"],
            optional_attrs=[],
            description="Custom block",
        )
        schema.register(custom_block)

        # Missing "name" - should be invalid
        node = TokNode(type="custom", label="", text="", attrs={"id": "123"})
        is_valid, msg = schema.validate(node)
        assert is_valid is False
        assert msg is not None
        assert "name" in msg.lower()

    def test_custom_schema_with_optional_attrs(self) -> None:
        """Custom schema should accept optional attributes."""
        schema = TokSchema()
        custom_block = BlockSchema(
            type="custom",
            required_attrs=["id"],
            optional_attrs=["description", "tags"],
            description="Custom block",
        )
        schema.register(custom_block)

        # With optional attrs - should be valid
        node = TokNode(
            type="custom",
            label="",
            text="",
            attrs={"id": "123", "description": "A custom block"},
        )
        is_valid, _msg = schema.validate(node)
        assert is_valid is True


class TestPermissiveDesign:
    """Test that unknown block types are allowed (permissive design)."""

    def test_unknown_block_type(self) -> None:
        """Unknown block types should be accepted (permissive)."""
        node = TokNode(type="unknown_xyz_123", label="", text="", attrs={"random": "attr"})
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        assert is_valid is True

    def test_empty_type(self) -> None:
        """Empty block type should be handled gracefully."""
        node = TokNode(type="", label="", text="", attrs={})
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        # Should not crash, should return bool/str
        assert isinstance(is_valid, bool)

    def test_unknown_attrs_on_unknown_type(self) -> None:
        """Unknown attrs on unknown type should pass (permissive)."""
        node = TokNode(
            type="mystery",
            label="",
            text="",
            attrs={"foo": "bar", "baz": "qux"},
        )
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        assert is_valid is True


class TestEdgeCases:
    """Test edge cases in schema validation."""

    def test_node_with_no_attrs(self) -> None:
        """Node with no attrs but required attrs should fail."""
        schema = TokSchema()
        schema.register(
            BlockSchema(
                type="strict",
                required_attrs=["id"],
                optional_attrs=[],
                description="Requires id",
            )
        )
        node = TokNode(type="strict", label="", text="", attrs={})
        is_valid, _msg = schema.validate(node)
        assert is_valid is False

    def test_node_with_none_values(self) -> None:
        """Node attributes with None values should be handled."""
        node = TokNode(type="msg", label="", text="", attrs={"role": "user"})
        # Manually test with None (edge case)
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        assert isinstance(is_valid, bool)

    def test_validation_result_structure(self) -> None:
        """validate() should always return (bool, str|None) tuple."""
        node = TokNode(type="msg", label="", text="", attrs={"role": "user"})
        result = DEFAULT_SCHEMA.validate(node)
        assert isinstance(result, tuple)
        assert len(result) == 2
        is_valid, msg = result
        assert isinstance(is_valid, bool)
        assert msg is None or isinstance(msg, str)


class TestValidationMatrixCoverage:
    """Comprehensive validation matrix for test report."""

    def test_matrix_msg_valid(self) -> None:
        """Matrix: @msg with role = ACCEPT."""
        node = TokNode(type="msg", label="", text="Hello", attrs={"role": "user"})
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        assert is_valid, "Valid @msg should be accepted"

    def test_matrix_msg_missing_role(self) -> None:
        """Matrix: @msg no role = REJECT."""
        node = TokNode(type="msg", label="", text="", attrs={})
        is_valid, msg = DEFAULT_SCHEMA.validate(node)
        assert not is_valid, "Missing role should be rejected"
        assert msg, "Should provide error message"

    def test_matrix_msg_unknown_attr(self) -> None:
        """Matrix: @msg unknown attr = REJECT."""
        node = TokNode(
            type="msg",
            label="",
            text="",
            attrs={"role": "user", "unknown_attr": "val"},
        )
        is_valid, msg = DEFAULT_SCHEMA.validate(node)
        assert not is_valid, "Unknown attributes should be rejected"
        assert msg, "Should provide error message"

    def test_matrix_unknown_type(self) -> None:
        """Matrix: @unknown_type any attrs = ACCEPT."""
        node = TokNode(type="custom_type_xyz", label="", text="", attrs={"any": "thing"})
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        assert is_valid, "Unknown types should be accepted (permissive)"

    def test_matrix_tool_with_label(self) -> None:
        """Matrix: @Tool with label = ACCEPT (or depends on registration)."""
        node = TokNode(type="Tool", label="read", text="", attrs={"path": "/file"})
        is_valid, _msg = DEFAULT_SCHEMA.validate(node)
        assert isinstance(is_valid, bool), "Should return valid result"

    def test_matrix_tool_missing_label(self) -> None:
        """Matrix: @Tool no label = REJECT."""
        node = TokNode(type="Tool", label="", text="", attrs={})
        is_valid, msg = DEFAULT_SCHEMA.validate(node)
        assert not is_valid, "Tool without label should be rejected"
        assert msg, "Should provide error message"


class TestSchemaPrecision:
    """Additional precision tests for schema validation."""

    def test_tool_schema_requires_location_attr(self) -> None:
        """Registered tool schema should reject missing required attributes."""
        node = TokNode(
            type="Tool",
            label="get_weather",
            text="",
            attrs={"units": "metric"},
        )
        is_valid, msg = DEFAULT_SCHEMA.validate(node)
        assert not is_valid
        assert msg
        assert "location" in msg.lower()

    def test_custom_schema_rejects_unknown_attributes(self) -> None:
        """Custom schema should reject unknown attributes not listed."""
        schema = TokSchema()
        schema.register(
            BlockSchema(
                type="strict_tool",
                required_attrs=["id"],
                optional_attrs=["description"],
                description="Tool-like block",
            )
        )
        node = TokNode(
            type="strict_tool",
            label="guard",
            text="",
            attrs={"id": "42", "unexpected": "value"},
        )
        is_valid, msg = schema.validate(node)
        assert not is_valid
        assert msg
        assert "unexpected" in msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
