from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAYERS_DOC = ROOT / "docs" / "spec" / "tok_protocol_layers_v0_1.md"
ROADMAP_DOC = ROOT / "docs" / "spec" / "tok_trace_roadmap_v0_1.md"


def test_protocol_layers_document_routing_as_design_axis_not_0_1_layer() -> None:
    text = LAYERS_DOC.read_text()

    assert "Tok Routing is a cross-cutting concern, not a 0.1.x protocol layer" in text
    assert "The default route is local-only" in text
    assert "Remote routing requires explicit configuration" in text
    assert "global routing" in text
    assert "tables" in text
    assert "a DHT" in text
    assert "ambient public discovery" in text


def test_protocol_layers_define_routing_questions_and_future_conformance_levels() -> None:
    text = LAYERS_DOC.read_text()

    for phrase in (
        "What is being requested?",
        "Who is asking?",
        "Who may know?",
        "What may be returned?",
        "L3a",
        "L3b",
        "L3c",
    ):
        assert phrase in text


def test_trace_roadmap_keeps_0_1_7_routing_out_of_release_claims() -> None:
    text = ROADMAP_DOC.read_text()

    assert "**0.1.7:** draft bridge trace/audit only; no Resolver, Routing, Capability, or Session" in text
    assert "**0.2.x:** scoped resolver routing" in text
    assert "no DHT, no ambient discovery" in text


def test_trace_roadmap_lists_routing_adversarial_cases() -> None:
    text = ROADMAP_DOC.read_text()

    for phrase in (
        "resolver referral loop",
        "unauthorized resolver request",
        "hash exists but capability missing",
        "resolver returns wrong bytes",
        "remote unavailable but trace remains valid",
        "conflicting resolver manifests",
    ):
        assert phrase in text
