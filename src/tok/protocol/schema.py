"""
Tok Schema: Definition and validation of Tok blocks.

Inverted design: Instead of "flexible" parsing, we enforce strict semantic
boundaries to prevent model hallucination and attribute drift.
"""

from dataclasses import dataclass, field

from .models import TokNode


@dataclass
class BlockSchema:
    """Schema definition for a Tok block type."""

    type: str
    required_attrs: list[str] = field(default_factory=list)
    optional_attrs: list[str] = field(default_factory=list)
    description: str = ""


class TokSchema:
    """Registry and validator for Tok block schemas."""

    def __init__(self) -> None:
        self._registry: dict[str, BlockSchema] = {}

    def register(self, schema: BlockSchema) -> None:
        """Register a block schema."""
        self._registry[schema.type.lower()] = schema

    def validate(self, node: TokNode) -> tuple[bool, str | None]:
        """
        Validate a node against registered schemas.

        Returns:
            Tuple of (is_valid, error_message).
            If valid, error_message is None.

        """
        node_type = node.type.lower()

        # We only validate blocks that have a schema (e.g., @Tool, @msg)
        # Custom blocks without a schema are ignored for flexibility.
        if node_type == "tool":
            # Special case for @Tool: label is the tool name
            return self._validate_tool(node)

        schema = self._registry.get(node_type)
        if not schema:
            return True, None

        # Check required attributes
        for req in schema.required_attrs:
            if req not in node.attrs:
                return False, f"Missing required attribute: {req!r}"

        # Check for unknown attributes (hallucinations)
        all_allowed = set(schema.required_attrs) | set(schema.optional_attrs)
        for attr in node.attrs:
            if attr not in all_allowed:
                msg = f"Unknown attribute {attr!r}. Allowed: "
                return False, f"{msg}{', '.join(all_allowed)}"

        return True, None

    def _validate_tool(self, node: TokNode) -> tuple[bool, str | None]:
        """Validate a @Tool block."""
        if not node.label:
            return False, "Tool name (label) is missing."

        # Check if we have a specific schema for this tool
        schema = self._registry.get(f"tool:{node.label.lower()}")
        if not schema:
            return True, None  # Allow unknown tools

        # Standard attribute check
        for req in schema.required_attrs:
            if req not in node.attrs:
                err = f"Tool {node.label!r} is missing required attribute: {req!r}"
                return False, err

        all_allowed = set(schema.required_attrs) | set(schema.optional_attrs)
        for attr in node.attrs:
            if attr not in all_allowed:
                msg = f"Tool {node.label!r} has unknown attribute {attr!r}. Allowed: "
                return False, f"{msg}{', '.join(all_allowed)}"

        return True, None


# ── Registry ──────────────────────────────────────────────────────────────────

DEFAULT_SCHEMA = TokSchema()

# Example Tool: get_weather
DEFAULT_SCHEMA.register(
    BlockSchema(
        type="tool:get_weather",
        required_attrs=["location"],
        optional_attrs=["days", "units"],
        description="Fetch weather forecast data.",
    )
)

# Global message block
DEFAULT_SCHEMA.register(
    BlockSchema(
        type="msg",
        required_attrs=["role"],
        optional_attrs=["trust"],
        description="A conversation message.",
    )
)
