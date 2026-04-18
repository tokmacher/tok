"""
Tok encoding utilities.

Provides encoding functionality for Tok nodes with optional compact
serialization.
"""

from .models import TokNode
from .parser import serialize


class TokEncoder:
    """Encoder for Tok nodes with optional compact serialization."""

    @staticmethod
    def encode(nodes: list[TokNode], compact: bool = False) -> str:
        """
        Encode Tok nodes to string format.

        Args:
            nodes: List of Tok nodes to encode.
            compact: Whether to use compact serialization.

        Returns:
            Encoded string representation of the nodes.

        """
        return serialize(nodes, compact=compact)
