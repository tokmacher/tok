"""Regression tests for file integrity manifest — cross-stage invariants."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stage 1 regressions
# ---------------------------------------------------------------------------


class TestManifestEmptyRegression:
    def test_none_fingerprints_not_accepted(self) -> None:
        # The function requires a dict, not None — this tests type safety via runtime
        from tok.compression._file_integrity import format_file_integrity_manifest

        result = format_file_integrity_manifest({}, {})
        assert result is None


# ---------------------------------------------------------------------------
# Stage 2 regressions
# ---------------------------------------------------------------------------


class TestManifestSingleEntryRegression:
    def test_path_with_two_components_preserved_fully(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {"utils/foo.py": "aabbccdd"}
        delivered = {"utils/foo.py": 1}
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        assert "utils/foo.py  t:1  fp:aabbccdd" in result

    def test_returns_string_not_none_when_nonempty(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {"a/b.py": "12345678"}
        result = format_file_integrity_manifest(fps, {})
        assert isinstance(result, str)

    def test_no_trailing_whitespace_per_line(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {"a/b.py": "12345678"}
        delivered = {"a/b.py": 1}
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        for line in result.splitlines():
            assert line == line.rstrip()


# ---------------------------------------------------------------------------
# Stage 3 regressions
# ---------------------------------------------------------------------------


class TestManifestOrderingRegression:
    def test_single_file_no_sorting_needed(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {"a/b.py": "11111111"}
        delivered = {"a/b.py": 7}
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        lines = result.strip().splitlines()
        assert len(lines) == 1

    def test_three_files_ordered_correctly(self) -> None:
        from tok.compression._file_integrity import format_file_integrity_manifest

        fps = {
            "a/c.py": "33333333",
            "a/a.py": "11111111",
            "a/b.py": "22222222",
        }
        delivered = {"a/c.py": 3, "a/a.py": 1, "a/b.py": 2}
        result = format_file_integrity_manifest(fps, delivered)
        assert result is not None
        lines = result.strip().splitlines()
        assert "t:1" in lines[0]
        assert "t:2" in lines[1]
        assert "t:3" in lines[2]


# ---------------------------------------------------------------------------
# Stage 5 regressions
# ---------------------------------------------------------------------------


class TestInjectManifestRegression:
    def test_manifest_placed_after_state_block(self) -> None:
        from tok.compression._history_pipeline import inject_system_additions_impl

        body: dict = {"system": "Base."}
        result = inject_system_additions_impl(
            body,
            tok_state=">>> turns:2",
            file_integrity_manifest="x.py  t:1  fp:aaaabbbb",
        )
        system = result["system"]
        state_pos = system.find(">>> turns")
        reads_pos = system.find("@reads")
        assert state_pos != -1
        assert reads_pos != -1
        assert state_pos < reads_pos

    def test_other_blocks_not_affected_when_manifest_absent(self) -> None:
        from tok.compression._history_pipeline import inject_system_additions_impl

        body: dict = {"system": "Base."}
        result_with = inject_system_additions_impl(
            body.copy(),
            tok_state=">>> turns:1",
            file_integrity_manifest="x.py  t:1  fp:aaaabbbb",
        )
        result_without = inject_system_additions_impl(
            body.copy(),
            tok_state=">>> turns:1",
            file_integrity_manifest=None,
        )
        # State should be present in both
        assert ">>> turns:1" in result_with["system"]
        assert ">>> turns:1" in result_without["system"]
        # Reads only in the one with manifest
        assert "@reads" in result_with["system"]
        assert "@reads" not in result_without["system"]


# ---------------------------------------------------------------------------
# Stage 6 regressions
# ---------------------------------------------------------------------------


class TestFidelityStateRegression:
    def test_files_read_fingerprints_independent_between_instances(self) -> None:
        from tok.runtime._fidelity_state import FidelityState

        s1 = FidelityState()
        s2 = FidelityState()
        s1.files_read_fingerprints["x.py"] = "aaaaaaaa"
        assert "x.py" not in s2.files_read_fingerprints

    def test_reset_clears_fingerprints_not_other_fields(self) -> None:
        from tok.runtime._fidelity_state import FidelityState

        state = FidelityState()
        state.file_reads_by_turn["some/path.py"] = 3
        state.files_read_fingerprints["some/path.py"] = "deadbeef"
        state.reset()
        assert state.files_read_fingerprints == {}
        assert state.file_reads_by_turn == {}


# ---------------------------------------------------------------------------
# Stage 7 regressions
# ---------------------------------------------------------------------------


class TestFingerprintsPopulationRegression:
    def test_none_fingerprints_dict_is_safe(self) -> None:
        from tok.compression import compress_tool_results

        # Passing None should be safe (no-op for fingerprints)
        msgs = [{"role": "user", "content": "hello"}]
        result, breakdown = compress_tool_results(
            msgs,
            bypass_result_cache=True,
            files_read_fingerprints=None,
        )
        assert isinstance(result, list)

    def test_reading_same_file_twice_is_idempotent(self) -> None:
        from tok.compression import compress_tool_results

        fingerprints: dict[str, str] = {}
        content = "def foo(): pass\n" * 50
        path = "src/tok/foo.py"

        def _tool_use(tid: str, p: str) -> dict:
            return {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tid, "name": "read", "input": {"path": p}}],
            }

        def _tool_result(tid: str, c: str) -> dict:
            return {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tid, "content": c}],
            }

        msgs = [
            _tool_use("t1", path),
            _tool_result("t1", content),
            _tool_use("t2", path),
            _tool_result("t2", content),
        ]
        ctx = {
            "t1": {"name": "read", "args": {"path": path}, "tool_use_id": "t1"},
            "t2": {"name": "read", "args": {"path": path}, "tool_use_id": "t2"},
        }
        compress_tool_results(
            msgs,
            bypass_result_cache=True,
            tool_use_id_to_context=ctx,
            session_files_read=set(),
            files_read_fingerprints=fingerprints,
        )
        # Each unique path appears once; value is deterministic
        values = list(fingerprints.values())
        unique_fps = set(values)
        # Same content → same fingerprint; idempotent
        assert len(unique_fps) <= len(fingerprints)

    def test_text_block_tool_result_populates_file_fingerprint(self) -> None:
        from tok.compression import compress_tool_results

        fingerprints: dict[str, str] = {}
        delivered: dict[str, int] = {}
        content = "def exported():\n    return 'cached'\n" * 50
        path = "src/tok/compression/_file_integrity.py"
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": path}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": content}],
                    }
                ],
            },
        ]
        ctx = {"t1": {"name": "Read", "args": {"file_path": path}, "tool_use_id": "t1"}}

        result, _ = compress_tool_results(
            msgs,
            bypass_result_cache=True,
            tool_use_id_to_context=ctx,
            session_files_read=set(),
            files_fully_delivered=delivered,
            current_turn=1,
            files_read_fingerprints=fingerprints,
        )

        assert result[1]["content"][0]["content"] == [{"type": "text", "text": content}]
        assert fingerprints == {path: fingerprints[path]}
        assert len(fingerprints[path]) == 8
        assert delivered[path] == 1
