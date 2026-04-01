import pytest
from typing import Any

from tok.universal_runtime import apply_schema_adaptations


def test_apply_schema_adaptations_merging():
    # Role merging was removed in consolidation
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "world"},
    ]
    adapted = apply_schema_adaptations(messages)

    assert len(adapted) == 2
    assert adapted[0]["content"] == "Hello"
    assert adapted[1]["content"] == "world"


def test_apply_schema_adaptations_placeholders():
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": []},
    ]
    adapted = apply_schema_adaptations(messages)

    assert len(adapted) == 2
    assert adapted[0]["content"] == " "
    assert adapted[1]["content"] == " "


def test_apply_schema_adaptations_flattening():
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    ]
    adapted = apply_schema_adaptations(messages)

    assert len(adapted) == 1
    assert adapted[0]["content"] == "Hello"


if __name__ == "__main__":
    pytest.main([__file__])
