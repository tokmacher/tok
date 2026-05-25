"""TDD tests for tok.compression._file_integrity — file integrity manifest.

Sub-stages:
  1. format_file_integrity_manifest returns None when empty
  2. Single entry formatted correctly
  3. Multiple entries sorted by turn, then path
  4. Turn fallback when not in files_fully_delivered
  5. inject_system_additions_impl injects @reads block
  6. FileDeliveryState has files_read_fingerprints field
  7. _mark_verbatim_file_observation populates fingerprints
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stage 1 — returns None when empty
# ---------------------------------------------------------------------------


class TestFormatManifestEmpty:
    def test_empty_fingerprints_returns_none(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        assert format_file_integrity_manifest({}, {}) is None

    def test_empty_fingerprints_with_nonempty_delivered_still_none(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        assert format_file_integrity_manifest({}, {"x.py": 3}) is None


# ---------------------------------------------------------------------------
# Stage 2 — single entry formatted correctly
# ---------------------------------------------------------------------------


class TestFormatManifestSingleEntry:
    def test_single_entry_present_in_output(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {"src/tok/compression/__init__.py": "a3f2c1d4"}
        delivered = {"src/tok/compression/__init__.py": 3}
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        assert "compression/__init__.py" in result

    def test_single_entry_has_turn(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {"src/tok/compression/__init__.py": "a3f2c1d4"}
        delivered = {"src/tok/compression/__init__.py": 3}
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        assert "t:3" in result

    def test_single_entry_has_fingerprint(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {"src/tok/compression/__init__.py": "a3f2c1d4"}
        delivered = {"src/tok/compression/__init__.py": 3}
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        assert "fp:a3f2c1d4" in result

    def test_single_entry_format_exact(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {"src/tok/compression/__init__.py": "a3f2c1d4"}
        delivered = {"src/tok/compression/__init__.py": 3}
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        assert result.strip() == "compression/__init__.py  t:3  fp:a3f2c1d4"

    def test_path_with_one_component_uses_that_component(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {"standalone.py": "deadbeef"}
        delivered = {"standalone.py": 1}
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        assert "standalone.py" in result

    def test_deep_path_uses_last_two_components(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {"a/b/c/d/e/deep.py": "cafebabe"}
        delivered = {"a/b/c/d/e/deep.py": 5}
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        assert "e/deep.py" in result
        assert "a/b" not in result


# ---------------------------------------------------------------------------
# Stage 3 — multiple entries sorted by turn, then path
# ---------------------------------------------------------------------------


class TestFormatManifestOrdering:
    def test_earlier_turn_sorts_first(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {
            "src/tok/utils/b.py": "11111111",
            "src/tok/utils/a.py": "22222222",
        }
        delivered = {
            "src/tok/utils/b.py": 5,
            "src/tok/utils/a.py": 2,
        }
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        lines = result.strip().splitlines()
        assert "t:2" in lines[0]
        assert "t:5" in lines[1]

    def test_same_turn_sorted_alphabetically(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {
            "src/tok/utils/zebra.py": "aaaaaaaa",
            "src/tok/utils/alpha.py": "bbbbbbbb",
        }
        delivered = {
            "src/tok/utils/zebra.py": 3,
            "src/tok/utils/alpha.py": 3,
        }
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        lines = result.strip().splitlines()
        assert "alpha.py" in lines[0]
        assert "zebra.py" in lines[1]


# ---------------------------------------------------------------------------
# Stage 4 — turn fallback when not in files_fully_delivered
# ---------------------------------------------------------------------------


class TestFormatManifestTurnFallback:
    def test_missing_from_delivered_uses_zero(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {"src/tok/foo.py": "12345678"}
        result = format_file_integrity_manifest(fps, {})
        assert result is not None
        assert "t:0" in result

    def test_zero_turn_sorts_before_positive_turns(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {
            "src/tok/known.py": "aaaaaaaa",
            "src/tok/unknown.py": "bbbbbbbb",
        }
        delivered = {"src/tok/known.py": 4}
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        lines = result.strip().splitlines()
        assert "t:0" in lines[0]
        assert "t:4" in lines[1]


# ---------------------------------------------------------------------------
# Stage 5 — inject_system_additions_impl injects @reads block
# ---------------------------------------------------------------------------


class TestInjectManifest:
    def test_reads_block_injected_when_manifest_present(self) -> None:
        from tok.compression._history_pipeline import inject_system_additions_impl

        body: dict = {"system": "You are helpful."}
        result = inject_system_additions_impl(
            body,
            file_integrity_manifest="foo.py  t:1  fp:abcd1234",
        )
        assert "@reads" in result["system"]

    def test_manifest_content_injected(self) -> None:
        from tok.compression._history_pipeline import inject_system_additions_impl

        body: dict = {"system": "You are helpful."}
        result = inject_system_additions_impl(
            body,
            file_integrity_manifest="bar.py  t:2  fp:deadbeef",
        )
        assert "bar.py  t:2  fp:deadbeef" in result["system"]

    def test_reads_absent_when_manifest_is_none(self) -> None:
        from tok.compression._history_pipeline import inject_system_additions_impl

        body: dict = {"system": "You are helpful."}
        result = inject_system_additions_impl(body, file_integrity_manifest=None)
        assert "@reads" not in result.get("system", "")

    def test_reads_absent_when_manifest_is_empty_string(self) -> None:
        from tok.compression._history_pipeline import inject_system_additions_impl

        body: dict = {"system": "You are helpful."}
        result = inject_system_additions_impl(body, file_integrity_manifest="")
        assert "@reads" not in result.get("system", "")

    def test_reads_block_injected_with_list_system(self) -> None:
        from tok.compression._history_pipeline import inject_system_additions_impl

        body: dict = {"system": [{"type": "text", "text": "You are helpful."}]}
        result = inject_system_additions_impl(
            body,
            file_integrity_manifest="baz.py  t:1  fp:cafebabe",
        )
        system = result["system"]
        assert isinstance(system, list)
        full_text = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
        assert "@reads" in full_text


# ---------------------------------------------------------------------------
# Stage 6 — FileDeliveryState has files_read_fingerprints field
# ---------------------------------------------------------------------------


class TestFidelityStateField:
    def test_files_read_fingerprints_exists(self) -> None:
        from tok.runtime._fidelity_state import FidelityState

        state = FidelityState()
        assert hasattr(state, "files_read_fingerprints")

    def test_files_read_fingerprints_default_is_empty_dict(self) -> None:
        from tok.runtime._fidelity_state import FidelityState

        state = FidelityState()
        assert state.files_read_fingerprints == {}
        assert isinstance(state.files_read_fingerprints, dict)

    def test_reset_clears_files_read_fingerprints(self) -> None:
        from tok.runtime._fidelity_state import FidelityState

        state = FidelityState()
        state.files_read_fingerprints["some/path.py"] = "deadbeef"
        state.reset()
        assert state.files_read_fingerprints == {}


# ---------------------------------------------------------------------------
# Stage 7 — _mark_verbatim_file_observation populates fingerprints
# ---------------------------------------------------------------------------


def _make_file_tool_result_message(tool_use_id: str, content: str, path: str) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
        ],
    }


def _make_tool_use_message(tool_use_id: str, path: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": "read",
                "input": {"path": path},
            }
        ],
    }


def _make_file_context(tool_use_id: str, path: str) -> dict:
    """Build the tool_use_id_to_context entry that the pipeline needs to extract a path."""
    return {
        "name": "read",
        "args": {"path": path},
        "tool_use_id": tool_use_id,
    }


class TestMarkVerbatimPopulatesFingerprints:
    def test_fingerprints_dict_populated_after_compress(self) -> None:
        from tok.compression import compress_tool_results

        fingerprints: dict[str, str] = {}
        path = "src/tok/foo.py"
        large_content = "def foo():\n    return 42\n" * 50
        msgs = [
            _make_tool_use_message("t1", path),
            _make_file_tool_result_message("t1", large_content, path),
        ]
        ctx = {"t1": _make_file_context("t1", path)}
        compress_tool_results(
            msgs,
            bypass_result_cache=True,
            tool_use_id_to_context=ctx,
            session_files_read=set(),  # enables _first_exact_guard
            files_read_fingerprints=fingerprints,
        )
        assert len(fingerprints) > 0

    def test_fingerprint_value_is_8_char_hex(self) -> None:
        from tok.compression import compress_tool_results

        fingerprints: dict[str, str] = {}
        path = "src/tok/bar.py"
        large_content = "x = 1\n" * 100
        msgs = [
            _make_tool_use_message("t1", path),
            _make_file_tool_result_message("t1", large_content, path),
        ]
        ctx = {"t1": _make_file_context("t1", path)}
        compress_tool_results(
            msgs,
            bypass_result_cache=True,
            tool_use_id_to_context=ctx,
            session_files_read=set(),
            files_read_fingerprints=fingerprints,
        )
        for fp in fingerprints.values():
            assert len(fp) == 8
            assert all(c in "0123456789abcdef" for c in fp)

    def test_empty_norm_path_does_not_add_entry(self) -> None:
        from tok.compression import compress_tool_results

        fingerprints: dict[str, str] = {}
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "some output",
                    }
                ],
            }
        ]
        # No tool_use_id_to_context → no path resolved → no fingerprint
        compress_tool_results(
            msgs,
            bypass_result_cache=True,
            files_read_fingerprints=fingerprints,
        )
        assert len(fingerprints) == 0
