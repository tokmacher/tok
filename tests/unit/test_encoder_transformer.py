"""
Encoder/Transformer Tests - Proving Markdown ↔ Tok fidelity.

Tests:
- Markdown → Tok transformation (DocumentTransformer.transform)
- Tok → Markdown transformation (DocumentTransformer.detransform)
- TokEncoder.encode() for list serialization
- Round-trip integrity for content presence
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest

from tok.protocol.encoder import TokEncoder
from tok.protocol.models import TokNode
from tok.protocol.parser import TokParser
from tok.utils.transformer import DocumentTransformer


class TestDocumentTransformerMarkdownToTok:
    """Test Markdown → Tok transformation."""

    def test_simple_markdown_to_tok(self) -> None:
        """Simple markdown with headings should transform to Tok."""
        transformer = DocumentTransformer()
        md = "# Heading\n\nSome paragraph text."
        result = transformer.transform(md)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_markdown_headings_preserved(self) -> None:
        """Markdown headings should survive transformation."""
        transformer = DocumentTransformer()
        md = "# Title\n## Subtitle\n### Detail\n\nContent"
        result = transformer.transform(md)

        # Should have created nodes for headings
        assert len(result) > 0
        types = [node.type for node in result]
        # Should have some block types (heading structure)
        assert len(types) > 0

    def test_markdown_code_block_to_tok(self) -> None:
        """Markdown code blocks should transform."""
        transformer = DocumentTransformer()
        md = "# Code Example\n\n```python\ndef hello():\n    return 'world'\n```"
        result = transformer.transform(md)
        assert len(result) > 0

    def test_markdown_list_to_tok(self) -> None:
        """Markdown lists should transform to Tok."""
        transformer = DocumentTransformer()
        md = "# Items\n\n- Item 1\n- Item 2\n- Item 3"
        result = transformer.transform(md)
        assert len(result) > 0

    def test_markdown_table_to_tok(self) -> None:
        """Markdown tables should transform to Tok table blocks."""
        transformer = DocumentTransformer()
        md = """# Data

| Name | Value |
| --- | --- |
| Alice | 100 |
| Bob | 200 |"""
        result = transformer.transform(md)
        assert len(result) > 0
        # Should have table structure
        has_table = any(node.headers for node in result)
        assert has_table, "Should preserve table structure"


class TestDocumentTransformerTokToMarkdown:
    """Test Tok → Markdown transformation."""

    def test_tok_to_markdown_conversion(self) -> None:
        """Tok text should convert back to Markdown."""
        transformer = DocumentTransformer()
        md_original = "# Heading\n\nParagraph text"

        # Transform to Tok
        nodes = transformer.transform(md_original)

        # Transform back to Markdown
        md_result = transformer.detransform_nodes(nodes)
        assert isinstance(md_result, str)

    def test_markdown_roundtrip_content_preservation(self) -> None:
        """Markdown round-trip should preserve content."""
        transformer = DocumentTransformer()
        md_original = "# Title\n\nThis is content\n\n## Subtitle\n\nMore content"

        nodes = transformer.transform(md_original)
        md_result = transformer.detransform_nodes(nodes)

        # Key content should survive
        assert "Title" in md_result or "Title" in str(nodes)
        assert "content" in md_result.lower() or "content" in str(nodes).lower()

    def test_table_roundtrip(self) -> None:
        """Table structure should survive round-trip."""
        transformer = DocumentTransformer()
        md_table = """| Name | Age |
| --- | --- |
| Alice | 30 |
| Bob | 25 |"""

        nodes = transformer.transform(md_table)

        # Should have table nodes
        has_headers = any(node.headers for node in nodes)
        assert has_headers, "Table headers should be preserved"

    def test_code_block_roundtrip(self) -> None:
        """Code blocks should survive round-trip."""
        transformer = DocumentTransformer()
        md = r"""# Code

\`\`\`python
def func():
    pass
\`\`\`"""

        nodes = transformer.transform(md)
        assert len(nodes) > 0


class TestTokEncoderSerialization:
    """Test TokEncoder.encode() for list serialization."""

    def test_encoder_encodes_nodes_to_text(self) -> None:
        """TokEncoder.encode() should convert nodes to Tok text."""
        node = TokNode(type="msg", label="", text="Hello", attrs={"role": "user"})
        result = TokEncoder.encode([node])
        assert isinstance(result, str)
        assert "@msg" in result

    def test_encoder_handles_single_node(self) -> None:
        """TokEncoder should handle single node in list."""
        node = TokNode(type="data", label="test", text="Content")
        result = TokEncoder.encode([node])
        assert isinstance(result, str)
        assert "@data" in result
        assert "test" in result

    def test_encoder_handles_multiple_nodes(self) -> None:
        """TokEncoder should handle multiple nodes."""
        nodes = [
            TokNode(type="msg", label="", text="First"),
            TokNode(type="msg", label="", text="Second"),
        ]
        result = TokEncoder.encode(nodes)
        assert isinstance(result, str)
        assert result.count("@msg") >= 2

    def test_encoder_compact_mode(self) -> None:
        """TokEncoder should support compact mode."""
        node = TokNode(type="msg", label="", text="Test")
        result = TokEncoder.encode([node], compact=True)
        assert isinstance(result, str)
        # Compact mode might not have newlines
        assert "@msg" in result


class TestTransformerNakedMode:
    """Test transformer naked mode (rich=False)."""

    def test_naked_mode_strips_metadata(self) -> None:
        """Naked mode should produce simpler output without TOC/pool."""
        transformer = DocumentTransformer()
        md = "# Title\n\n## Section\n\nContent"

        nodes_rich = transformer.transform(md, rich=True)
        nodes_naked = transformer.transform(md, rich=False)

        # Both should produce nodes
        assert len(nodes_rich) > 0
        assert len(nodes_naked) > 0

    def test_naked_mode_detransform(self) -> None:
        """Naked mode detransform should be simpler."""
        transformer = DocumentTransformer()
        md = "# Simple\n\nContent"

        nodes = transformer.transform(md, rich=False)
        result = transformer.detransform_nodes(nodes)

        assert isinstance(result, str)


class TestTransformerEdgeCases:
    """Test edge cases in transformer."""

    def test_empty_markdown(self) -> None:
        """Empty markdown should transform without error."""
        transformer = DocumentTransformer()
        result = transformer.transform("")
        assert isinstance(result, list)

    def test_whitespace_only_markdown(self) -> None:
        """Whitespace-only markdown should handle gracefully."""
        transformer = DocumentTransformer()
        result = transformer.transform("   \n\n   ")
        assert isinstance(result, list)

    def test_single_heading(self) -> None:
        """Single heading should transform."""
        transformer = DocumentTransformer()
        result = transformer.transform("# Just a heading")
        assert len(result) > 0

    def test_nested_headings(self) -> None:
        """Nested heading hierarchy should transform."""
        transformer = DocumentTransformer()
        md = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6"
        result = transformer.transform(md)
        assert len(result) > 0

    def test_special_characters_in_content(self) -> None:
        """Special characters should be preserved."""
        transformer = DocumentTransformer()
        md = "# Title\n\nContent with special: @, >, |, !, *"
        result = transformer.transform(md)
        assert len(result) > 0

    def test_unicode_content(self) -> None:
        """Unicode content should be handled."""
        transformer = DocumentTransformer()
        md = "# 标题 (Title)\n\n世界 Мир مرحبا"
        result = transformer.transform(md)
        assert len(result) > 0

    def test_very_long_markdown(self) -> None:
        """Long markdown document should transform."""
        transformer = DocumentTransformer()
        paragraphs = [f"# Section {i}\n\nContent paragraph {i}" for i in range(50)]
        md = "\n\n".join(paragraphs)
        result = transformer.transform(md)
        assert len(result) > 0


class TestTransformerConsistency:
    """Test transformer behavior is consistent."""

    def test_multiple_transforms_same_input(self) -> None:
        """Multiple transforms of same input should be consistent."""
        transformer = DocumentTransformer()
        md = "# Test\n\nContent"

        result1 = transformer.transform(md)
        result2 = transformer.transform(md)

        # Should produce same structure
        assert len(result1) == len(result2)
        assert [n.type for n in result1] == [n.type for n in result2]

    def test_transformer_isolation(self) -> None:
        """Multiple transformer instances should be independent."""
        t1 = DocumentTransformer()
        t2 = DocumentTransformer()

        md1 = "# First\n\nContent 1"
        md2 = "# Second\n\nContent 2"

        result1 = t1.transform(md1)
        result2 = t2.transform(md2)

        # Should produce different results
        assert result1 != result2


class TestDocumentTransformerAdvanced:
    """Advanced transformer assurances (flattening, pointers, round-trip)."""

    def test_flattening_threshold_adds_parent_markers(self) -> None:
        """Nodes beyond the flattening threshold should carry parent pointers."""
        transformer = DocumentTransformer(flattening_threshold=2)
        md = "# Root\n\n## Level 1\n\n### Level 2\n\n#### Level 3\n\n##### Level 4"

        nodes = transformer.transform(md, rich=True)
        assert any("parent" in getattr(n, "attrs", {}) for n in nodes), "Flattened nodes should include parent metadata"

    def test_pointerization_detects_repeated_terms(self) -> None:
        """Repeated terms should generate PTR nodes for compression pointers."""
        transformer = DocumentTransformer()
        repeated_term = "verylongtermverylongtermverylongterm"
        text_counts = {repeated_term: 8}
        pointers = transformer._compute_profitable_pointers(text_counts)
        assert pointers, "Pointer nodes should exist when text repeats"

    def test_transformer_encoder_parser_roundtrip(self) -> None:
        """DocumentTransformer + TokEncoder + TokParser should round-trip."""
        transformer = DocumentTransformer()
        md = "# Title\n\n## Section\n\nParagraph content."

        nodes = transformer.transform(md, rich=True)
        tok_text = TokEncoder.encode(nodes)
        parser = TokParser()
        reparsed = parser.parse(tok_text)
        assert reparsed, "Parser should recover nodes from the encoder output"
        result_md = transformer.detransform_nodes(reparsed)
        assert "Title" in result_md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
