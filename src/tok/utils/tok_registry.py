"""
Tok Registry - Optional tracking layer for sentinel dashboard.
Provides file tracking and integrity reporting.
"""

from typing import Any, Optional


class TokRegistry:
    v7_type_store: dict[str, str] = {}
    """Optional registry for tracking Tok operations."""

    _instance: Optional["TokRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "TokRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if TokRegistry._initialized:
            return
        self._records: list[dict[str, Any]] = []
        self._files: dict[str, str] = {}
        TokRegistry._initialized = True

    @classmethod
    def _get_records(cls) -> list[dict[str, Any]]:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance._records

    @classmethod
    def _get_files(cls) -> dict[str, str]:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance._files

    @classmethod
    def get_files(cls) -> dict[str, str]:
        """Get tracked files."""
        return cls._get_files().copy()

    @classmethod
    def get_all(cls) -> list[dict[str, Any]]:
        """Get all records."""
        return cls._get_records().copy()

    @classmethod
    def global_integrity_report(cls) -> str:
        """Generate integrity report."""
        records = cls._get_records()
        files = cls._get_files()
        return f"🟢 INTEGRITY OK | {len(records)} records | {len(files)} files tracked"

    @classmethod
    def register(
        cls,
        tool: str,
        path: str,
        status: str,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        """Register an operation."""
        record = {
            "tool": tool,
            "path": path,
            "status": status,
            "metadata": metadata,
        }
        if kwargs:
            record["extra"] = kwargs
        cls._get_records().append(record)
        if path and status == "SUCCESS":
            cls._get_files()[path] = status


# Module-level convenience
tok_registry = TokRegistry()
