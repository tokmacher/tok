from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAYERS_DOC = ROOT / "docs" / "spec" / "tok_protocol_layers_v0_1.md"
ROADMAP_DOC = ROOT / "docs" / "spec" / "tok_trace_roadmap_v0_1.md"
FORMAT_DOC = ROOT / "docs" / "spec" / "tok_trace_format_v0_1.md"
CONFORMANCE_DOC = ROOT / "docs" / "spec" / "tok_trace_conformance_v0_1.md"
BRIDGE_STANDARD_DOC = ROOT / "docs" / "bridge-standard.md"
ADVERSARIAL_PACKS = ROOT / "docs" / "spec" / "fixtures" / "adversarial_packs.json"
TRACE_READER = ROOT / "src" / "tok" / "spec" / "trace.py"


def _squash_ws(text: str) -> str:
    return " ".join(text.split())


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


def test_protocol_docs_name_adversarial_pack_manifest_and_release_ladder() -> None:
    layers = LAYERS_DOC.read_text()
    roadmap = ROADMAP_DOC.read_text()

    assert "fixtures/adversarial_packs.json" in layers
    assert "fixtures/adversarial_packs.json" in roadmap
    assert "trace-l1-l2-core-adversarial" in layers
    assert "trace-l1-l2-core-adversarial" in roadmap
    assert "**0.1.8:** named adversarial fixture packs" in roadmap
    assert "**0.1.9:** standalone reference reader and Resolver design" in roadmap
    assert "**0.2.0:** local Tok Resolver beta" in roadmap


def test_bridge_profile_boundary_keeps_sigils_out_of_session_core() -> None:
    bridge = _squash_ws(BRIDGE_STANDARD_DOC.read_text())
    trace_format = _squash_ws(FORMAT_DOC.read_text())

    for phrase in (
        "The bridge grammar is a profile-local adapter contract",
        "must not be treated as Tok Session core semantics",
    ):
        assert phrase in bridge

    for phrase in (
        "Bridge syntax is not part of the trace core",
        "without canonizing the current text sigils",
    ):
        assert phrase in trace_format


def test_conformance_doc_defines_l0_l2_reader_boundary_without_future_claims() -> None:
    text = CONFORMANCE_DOC.read_text()
    prose = _squash_ws(text)

    for phrase in (
        "without importing Tok gateway, runtime, compression, CLI, benchmark, or analysis internals",
        "JSON is the first fixture encoding, not the protocol identity",
        "L3, L4, and L5 remain out of scope",
        "must not claim cross-cache resolution",
        "must not claim",
        "agent-to-agent compact-state exchange",
    ):
        assert phrase in prose

    for phrase in ("| L0", "| L1", "| L2"):
        assert phrase in text


def test_trace_roadmap_defines_0_1_8_as_l0_l2_protocol_hardening() -> None:
    prose = _squash_ws(ROADMAP_DOC.read_text())

    for phrase in (
        "0.1.8 protocol claim is Trace L0-L2 only",
        "parse JSON arrays and JSONL traces",
        "validate required fields, enum values, extension namespaces, canonical payload digests, and pass/warn/fail outcomes",
        "verify local artifacts, exact hashes and sizes, exact versus non-exact references, fallback/degradation reasons, sequence consistency, and supported unified_diff deltas",
        "L3-L5 remain future design only",
    ):
        assert phrase in prose


def test_conformance_doc_names_exact_0_1_8_validation_claims() -> None:
    prose = _squash_ws(CONFORMANCE_DOC.read_text())

    for phrase in (
        "For 0.1.8, Tok validates exactly this L0-L2 set",
        "L0: parse JSON fixture arrays and JSONL traces, reject malformed JSON or malformed fixture structure, and preserve block order",
        "L1: validate required fields, enum values, extension namespace rules, canonical payload digests, and pass/warn/fail audit outcomes",
        "L2: verify local artifact hashes and sizes, exact versus non-exact content claims, fallback/degradation reasons, sequence consistency, and supported unified_diff deltas",
        "Everything above L2 is documentation-only in 0.1.8",
    ):
        assert phrase in prose


def test_adversarial_pack_manifest_groups_local_and_future_cases() -> None:
    manifest = json.loads(ADVERSARIAL_PACKS.read_text())
    packs = {pack["id"]: pack for pack in manifest["packs"]}

    local = packs["trace-l1-l2-core-adversarial"]
    future = packs["resolver-routing-future-adversarial"]

    assert local["status"] == "implemented-local"
    assert local["conformance_levels"] == ["L0", "L1", "L2"]
    assert future["status"] == "future-design-only"
    assert future["conformance_levels"] == ["L3", "L4", "L5"]

    local_cases = {case["id"]: case for case in local["cases"]}
    for case_id in (
        "forged_payload_digest",
        "resolver_uri_path_escape",
        "exactness_lie",
        "resolver_state_lie",
        "unsupported_delta_algorithm",
        "malformed_jsonl_line",
        "duplicate_block_ids",
        "out_of_order_turns",
        "unknown_required_field_or_version",
        "extension_override_core_semantics",
    ):
        assert local_cases[case_id]["expected_status"] == "fail"
        assert local_cases[case_id]["expected_error"]


def test_future_adversarial_pack_is_not_a_0_1_x_supported_outcome_pack() -> None:
    manifest = json.loads(ADVERSARIAL_PACKS.read_text())
    packs = {pack["id"]: pack for pack in manifest["packs"]}

    local = packs["trace-l1-l2-core-adversarial"]
    future = packs["resolver-routing-future-adversarial"]
    local_case_ids = {case["id"] for case in local["cases"]}

    assert future["status"] == "future-design-only"
    assert set(future["conformance_levels"]) == {"L3", "L4", "L5"}
    assert not local_case_ids.intersection({case["id"] for case in future["cases"]})
    for case in future["cases"]:
        assert "expected_status" not in case
        assert "expected_error" not in case


def test_trace_reader_has_no_bridge_runtime_or_cli_imports() -> None:
    source = TRACE_READER.read_text()

    forbidden_imports = (
        "tok.gateway",
        "tok.runtime",
        "tok.compression",
        "tok.cli",
        "tok.testing",
        "tok.analysis",
    )
    for forbidden_import in forbidden_imports:
        assert forbidden_import not in source


def test_spec_package_exports_trace_reader_without_runtime_bridge_imports() -> None:
    source = (ROOT / "src" / "tok" / "spec" / "__init__.py").read_text()

    assert "from .trace import" in source
    for forbidden_import in ("tok.gateway", "tok.runtime", "tok.cli", "tok.compression"):
        assert forbidden_import not in source
