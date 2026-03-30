"""Bridge fidelity tests ensuring Tok ↔ JSON round-trips."""

import json
from pathlib import Path
from typing import Any

from tok.protocol.format_bridge import Bridge

DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "validation_dataset.json"
)


def _load_dataset() -> list[dict[str, Any]]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def _key_coverage(sample: dict[str, Any], decoded: Any) -> float:
    if not isinstance(decoded, dict):
        return 0.0
    sample_keys = set(sample.keys())
    if not sample_keys:
        return 1.0
    decoded_keys = set(decoded.keys())
    return len(sample_keys & decoded_keys) / len(sample_keys)


def test_bridge_round_trip_fidelity() -> None:
    samples = _load_dataset()
    assert samples, "Validation dataset should contain samples"

    total = 0.0
    for sample in samples:
        tok_payload = Bridge.json(json.dumps(sample, ensure_ascii=False))
        decoded = Bridge.decode(tok_payload)
        total += _key_coverage(sample, decoded)

    fidelity = total / len(samples)
    assert (
        fidelity >= 0.95
    ), f"Fidelity was {fidelity:.2%}, expected ≥95% key coverage"
