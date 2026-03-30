"""
SerializationProtocol — Interface contract for encoding/decoding operations.

This module defines the abstract interface that all serialization components
must implement, enforcing a clean architectural contract.
"""

from abc import ABC, abstractmethod
from typing import Any


class SerializationProtocol(ABC):
    """Abstract base class for serialization components.

    Defines the contract that any serializer/deserializer must satisfy:
    - encode: convert data to text
    - decode: convert text back to data
    """

    @abstractmethod
    def encode(self, data: Any) -> str:
        """Serialize data to text format.

        Args:
            data: Data to serialize (type depends on implementation)

        Returns:
            Serialized text representation
        """
        ...

    @abstractmethod
    def decode(self, text: str) -> Any:
        """Deserialize text to data.

        Args:
            text: Text to deserialize

        Returns:
            Deserialized data
        """
        ...
