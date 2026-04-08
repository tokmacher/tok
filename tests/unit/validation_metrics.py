"""Validation metric tests comparing Tok, JSON, and XML."""

import json
from pathlib import Path
from typing import Any

import tiktoken

from tok.protocol.format_bridge import Bridge

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "validation_dataset.json"
ENC = tiktoken.get_encoding("cl100k_base")


def _load_dataset() -> list[dict[str, Any]]:
    with DATASET_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _flatten_strings(
    value: str | float | bool | None | dict[str, Any] | list[Any],
    collector: list[str],
) -> None:
    if isinstance(value, str):
        collector.append(value)
    elif isinstance(value, (int, float, bool)):
        collector.append(str(value))
    elif isinstance(value, dict):
        for child in value.values():
            _flatten_strings(child, collector)
    elif isinstance(value, list):
        for child in value:
            _flatten_strings(child, collector)
    elif value is None:
        collector.append("null")
    else:
        collector.append(str(value))


def _payload_text(sample: dict[str, Any]) -> str:
    pieces: list[str] = []
    _flatten_strings(sample, pieces)
    return " ".join(pieces)


def _list_to_xml(value: dict[str, Any] | list[Any]) -> str:
    if isinstance(value, list):
        return "".join(f"<item>{_list_to_xml(v)}</item>" for v in value)
    if isinstance(value, dict):
        return "".join(f"<{k}>{_list_to_xml(v)}</{k}>" for k, v in value.items())
    return str(value)


def _to_xml(sample: dict[str, Any], root: str = "data") -> str:
    return f"<{root}>{_list_to_xml(sample)}</{root}>"


def _overhead_ratio(text: str, payload_length: int) -> float:
    payload_length = max(1, payload_length)
    return (len(text) - payload_length) / payload_length


def _token_efficiency(text: str, payload_tokens: int) -> float:
    tokens = max(1, len(ENC.encode(text)))
    return payload_tokens / tokens


def _measure(sample: dict[str, Any]) -> dict[str, float]:
    payload_text = _payload_text(sample)
    payload_length = len(payload_text)
    payload_tokens = len(ENC.encode(payload_text)) if payload_text else 1

    json_text = json.dumps(sample, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    xml_text = _to_xml(sample)
    tok_text = Bridge.json(json.dumps(sample, ensure_ascii=False))

    return {
        "tok_overhead": _overhead_ratio(tok_text, payload_length),
        "json_overhead": _overhead_ratio(json_text, payload_length),
        "xml_overhead": _overhead_ratio(xml_text, payload_length),
        "tok_efficiency": _token_efficiency(tok_text, payload_tokens),
        "json_efficiency": _token_efficiency(json_text, payload_tokens),
        "xml_efficiency": _token_efficiency(xml_text, payload_tokens),
    }


def _table_heavy_sample() -> dict[str, Any]:
    return {
        "table": [
            {
                "name": "alice",
                "value": i,
                "status": "ok" if i % 2 else "pending",
            }
            for i in range(30)
        ],
        "metadata": {"rows": 30, "summary": "repeated names only once"},
    }


# Minimum expected number of samples in validation dataset
MIN_DATASET_SAMPLES = 50


def test_validation_dataset_loads() -> None:
    """Test that validation dataset loads correctly and has expected structure."""
    samples = _load_dataset()
    assert len(samples) >= MIN_DATASET_SAMPLES
    assert isinstance(samples[0], dict)


def test_tok_metrics_are_non_negative() -> None:
    """Test that all Tok metrics are non-negative for sample data."""
    samples = _load_dataset()
    for sample in samples[:5]:
        metrics = _measure(sample)
        assert all(value >= 0 for value in metrics.values())


def test_tok_overhead_wins_over_json_and_xml_for_table_sample() -> None:
    """Test that Tok has lower overhead than JSON and XML for table-heavy data."""
    sample = _table_heavy_sample()
    metrics = _measure(sample)
    assert metrics["tok_overhead"] < metrics["json_overhead"]
    assert metrics["tok_overhead"] < metrics["xml_overhead"]


def test_tok_token_efficiency_wins_over_json_and_xml_for_table_sample() -> None:
    """Test that Tok has better token efficiency than JSON and XML for table-heavy data."""
    sample = _table_heavy_sample()
    metrics = _measure(sample)
    assert metrics["tok_efficiency"] > metrics["json_efficiency"]
    assert metrics["tok_efficiency"] > metrics["xml_efficiency"]
