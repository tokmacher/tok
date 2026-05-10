"""Tests for macro-to-tool_use expansion.

RED-GREEN-RED TDD: all tests are written first against unimplemented modules.
Phase 1-4 implementations will turn these green one by one.
"""

import pytest

from tok.macros.ir import Instruction, Macro, MacroRegistry  # noqa: I001

# ---------------------------------------------------------------------------
# Phase 1: tool_map.py — op-to-tool registry
# ---------------------------------------------------------------------------


class TestToolMap:
    def test_tool_map_view_to_read(self):
        from tok.macros.tool_map import OP_TOOL_MAP

        m = OP_TOOL_MAP["view"]
        assert m.tool_name == "Read"
        assert m.arg_map[0] == "file_path"

    def test_tool_map_cat_to_read(self):
        from tok.macros.tool_map import OP_TOOL_MAP

        m = OP_TOOL_MAP["cat"]
        assert m.tool_name == "Read"
        assert m.arg_map[0] == "file_path"

    def test_tool_map_edit(self):
        from tok.macros.tool_map import OP_TOOL_MAP

        m = OP_TOOL_MAP["edit"]
        assert m.tool_name == "Edit"
        assert m.arg_map[0] == "file_path"

    def test_tool_map_grep(self):
        from tok.macros.tool_map import OP_TOOL_MAP

        m = OP_TOOL_MAP["grep"]
        assert m.tool_name == "Grep"
        assert m.arg_map[0] == "pattern"
        assert m.arg_map[1] == "path"

    def test_tool_map_pytest_to_bash(self):
        from tok.macros.tool_map import OP_TOOL_MAP

        m = OP_TOOL_MAP["pytest"]
        assert m.tool_name == "Bash"
        assert m.shell_template is not None
        assert "pytest" in m.shell_template

    def test_tool_map_ls_to_bash(self):
        from tok.macros.tool_map import OP_TOOL_MAP

        m = OP_TOOL_MAP["ls"]
        assert m.tool_name == "Bash"
        assert m.shell_template is not None

    def test_tool_map_lookup_known(self):
        from tok.macros.tool_map import lookup

        assert lookup("view") is not None
        assert lookup("view").tool_name == "Read"

    def test_tool_map_lookup_unknown_returns_none(self):
        from tok.macros.tool_map import lookup

        assert lookup("nonexistent_op_xyz") is None

    def test_tool_map_unknown_op_not_in_registry(self):
        from tok.macros.tool_map import OP_TOOL_MAP

        assert "nonexistent_op_xyz" not in OP_TOOL_MAP


class TestResolveArgs:
    def test_resolve_placeholder_args(self):
        from tok.macros.tool_map import resolve_args

        ins = Instruction(op="view", args=("$p0",))
        bindings = {"p0": "src/foo.py"}
        result = resolve_args(ins, bindings)
        assert result == {"file_path": "src/foo.py"}

    def test_resolve_literal_args(self):
        from tok.macros.tool_map import resolve_args

        ins = Instruction(op="grep", args=("pattern_value", "$p0"))
        bindings = {"p0": "src/search_path.py"}
        result = resolve_args(ins, bindings)
        assert result == {"pattern": "pattern_value", "path": "src/search_path.py"}

    def test_resolve_shell_template_args(self):
        from tok.macros.tool_map import resolve_args

        ins = Instruction(op="pytest", args=("-v", "$p0"))
        bindings = {"p0": "tests/test_foo.py"}
        result = resolve_args(ins, bindings)
        assert result == {"command": "pytest -v tests/test_foo.py"}

    def test_resolve_missing_binding_raises(self):
        from tok.macros.tool_map import resolve_args

        ins = Instruction(op="view", args=("$p0",))
        with pytest.raises(KeyError):
            resolve_args(ins, {})


# ---------------------------------------------------------------------------
# Phase 2: expansion.py — macro-to-tool_use expander
# ---------------------------------------------------------------------------


class TestExpandMacro:
    def _make_macro(self, name, instructions, inputs):
        return Macro(name=name, instructions=instructions, inputs=inputs, hit_count=5)

    def test_expand_single_instruction_macro(self):
        from tok.macros.expansion import expand_macro

        macro = self._make_macro(
            "view_only",
            (Instruction(op="view", args=("$p0",)),),
            ("p0",),
        )
        blocks = expand_macro(macro, {"p0": "src/calculator.py"})
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["name"] == "Read"
        assert blocks[0]["input"]["file_path"] == "src/calculator.py"

    def test_expand_multi_instruction_macro(self):
        from tok.macros.expansion import expand_macro

        macro = self._make_macro(
            "grep_view",
            (
                Instruction(op="grep", args=("$pattern",)),
                Instruction(op="view", args=("$file",)),
            ),
            ("pattern", "file"),
        )
        blocks = expand_macro(macro, {"pattern": "reactor", "file": "src/tok/neuro/ir.py"})
        assert len(blocks) == 2
        assert blocks[0]["name"] == "Grep"
        assert blocks[0]["input"]["pattern"] == "reactor"
        assert blocks[1]["name"] == "Read"
        assert blocks[1]["input"]["file_path"] == "src/tok/neuro/ir.py"

    def test_expand_with_literal_args(self):
        from tok.macros.expansion import expand_macro

        macro = self._make_macro(
            "pytest_file",
            (Instruction(op="pytest", args=("-v", "$p0")),),
            ("p0",),
        )
        blocks = expand_macro(macro, {"p0": "tests/test_foo.py"})
        assert len(blocks) == 1
        assert blocks[0]["name"] == "Bash"
        assert blocks[0]["input"]["command"] == "pytest -v tests/test_foo.py"

    def test_expand_mixed_known_unknown_ops(self):
        from tok.macros.expansion import expand_macro

        macro = self._make_macro(
            "mixed",
            (
                Instruction(op="view", args=("$p0",)),
                Instruction(op="mystery_op", args=("$p0",)),
                Instruction(op="grep", args=("TODO", "$p0")),
            ),
            ("p0",),
        )
        blocks = expand_macro(macro, {"p0": "src/main.py"})
        assert len(blocks) == 2
        assert blocks[0]["name"] == "Read"
        assert blocks[1]["name"] == "Grep"

    def test_expand_all_unknown_ops_returns_empty(self):
        from tok.macros.expansion import expand_macro

        macro = self._make_macro(
            "all_unknown",
            (Instruction(op="foo_bar", args=()),),
            (),
        )
        blocks = expand_macro(macro, {})
        assert blocks == []

    def test_expanded_blocks_have_valid_ids(self):
        import re

        from tok.macros.expansion import expand_macro

        macro = self._make_macro(
            "grep_view",
            (
                Instruction(op="grep", args=("$pattern",)),
                Instruction(op="view", args=("$file",)),
            ),
            ("pattern", "file"),
        )
        blocks = expand_macro(macro, {"pattern": "test", "file": "a.py"})
        for block in blocks:
            assert re.match(r"^[A-Za-z0-9_-]+$", block["id"]), f"Invalid ID: {block['id']}"

    def test_expanded_block_ids_are_unique(self):
        from tok.macros.expansion import expand_macro

        macro = self._make_macro(
            "triple",
            (
                Instruction(op="view", args=("$p0",)),
                Instruction(op="view", args=("$p1",)),
                Instruction(op="view", args=("$p2",)),
            ),
            ("p0", "p1", "p2"),
        )
        blocks = expand_macro(macro, {"p0": "a.py", "p1": "b.py", "p2": "c.py"})
        ids = [b["id"] for b in blocks]
        assert len(ids) == len(set(ids))


class TestExpandMacroToolUseBlock:
    def test_intercept_at_sign_tool_use(self):
        from tok.macros.expansion import expand_macro_tool_use_block

        registry = MacroRegistry()
        registry.register(
            Macro(
                name="grep_view",
                instructions=(
                    Instruction(op="grep", args=("$pattern",)),
                    Instruction(op="view", args=("$file",)),
                ),
                inputs=("pattern", "file"),
            )
        )

        block = {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "@grep_view",
            "input": {"pattern": "reactor", "file": "src/tok/neuro/ir.py"},
        }
        expanded = expand_macro_tool_use_block(block, registry)
        assert len(expanded) == 2
        assert expanded[0]["name"] == "Grep"
        assert expanded[1]["name"] == "Read"

    def test_non_macro_blocks_untouched(self):
        from tok.macros.expansion import expand_macro_tool_use_block

        registry = MacroRegistry()
        block = {
            "type": "tool_use",
            "id": "toolu_456",
            "name": "Read",
            "input": {"file_path": "src/foo.py"},
        }
        expanded = expand_macro_tool_use_block(block, registry)
        assert len(expanded) == 1
        assert expanded[0] is block

    def test_unknown_macro_name_returns_original(self):
        from tok.macros.expansion import expand_macro_tool_use_block

        registry = MacroRegistry()
        block = {
            "type": "tool_use",
            "id": "toolu_789",
            "name": "@nonexistent_macro",
            "input": {},
        }
        expanded = expand_macro_tool_use_block(block, registry)
        assert len(expanded) == 1
        assert expanded[0]["name"] == "@nonexistent_macro"

    def test_non_tool_use_block_passthrough(self):
        from tok.macros.expansion import expand_macro_tool_use_block

        registry = MacroRegistry()
        block = {"type": "text", "text": "hello"}
        expanded = expand_macro_tool_use_block(block, registry)
        assert expanded == [block]

    def test_text_block_not_touched(self):
        from tok.macros.expansion import expand_macro_tool_use_block

        registry = MacroRegistry()
        block = {"type": "thinking", "thinking": "pondering"}
        expanded = expand_macro_tool_use_block(block, registry)
        assert expanded == [block]


# ---------------------------------------------------------------------------
# Phase 3: Response pipeline integration
# ---------------------------------------------------------------------------


class TestExpandToolUseBlocks:
    def test_expand_in_block_list(self):
        from tok.macros.expansion import expand_tool_use_blocks

        registry = MacroRegistry()
        registry.register(
            Macro(
                name="grep_view",
                instructions=(
                    Instruction(op="grep", args=("$pattern",)),
                    Instruction(op="view", args=("$file",)),
                ),
                inputs=("pattern", "file"),
            )
        )

        blocks = [
            {"type": "text", "text": "Let me search."},
            {"type": "tool_use", "id": "toolu_1", "name": "@grep_view", "input": {"pattern": "test", "file": "a.py"}},
            {"type": "text", "text": "Done."},
        ]
        expanded = expand_tool_use_blocks(blocks, registry)
        assert len(expanded) == 4
        assert expanded[0]["type"] == "text"
        assert expanded[1]["name"] == "Grep"
        assert expanded[2]["name"] == "Read"
        assert expanded[3]["type"] == "text"

    def test_no_macros_all_passthrough(self):
        from tok.macros.expansion import expand_tool_use_blocks

        registry = MacroRegistry()
        blocks = [
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a.py"}},
        ]
        expanded = expand_tool_use_blocks(blocks, registry)
        assert len(expanded) == 1
        assert expanded[0]["name"] == "Read"

    def test_multiple_macros_in_one_list(self):
        from tok.macros.expansion import expand_tool_use_blocks

        registry = MacroRegistry()
        registry.register(
            Macro(
                name="view_only",
                instructions=(Instruction(op="view", args=("$p0",)),),
                inputs=("p0",),
            )
        )
        registry.register(
            Macro(
                name="grep_only",
                instructions=(Instruction(op="grep", args=("$pattern",)),),
                inputs=("pattern",),
            )
        )

        blocks = [
            {"type": "tool_use", "id": "toolu_1", "name": "@view_only", "input": {"p0": "a.py"}},
            {"type": "tool_use", "id": "toolu_2", "name": "@grep_only", "input": {"pattern": "TODO"}},
        ]
        expanded = expand_tool_use_blocks(blocks, registry)
        assert len(expanded) == 2
        assert expanded[0]["name"] == "Read"
        assert expanded[1]["name"] == "Grep"


# ---------------------------------------------------------------------------
# Phase 4: Prompt instruction injection
# ---------------------------------------------------------------------------


class TestPromptHintInjection:
    def test_macro_hint_in_runtime_hints_when_jit_available(self):
        from tok.macros.expansion import macro_hint_for_session

        registry = MacroRegistry()
        registry.register(
            Macro(
                name="grep_view",
                instructions=(
                    Instruction(op="grep", args=("$pattern",)),
                    Instruction(op="view", args=("$file",)),
                ),
                inputs=("pattern", "file"),
                hit_count=5,
            )
        )
        hint = macro_hint_for_session(registry)
        assert hint is not None
        assert "@grep_view" in hint
        assert "grep" in hint
        assert "view" in hint

    def test_no_hint_when_registry_empty(self):
        from tok.macros.expansion import macro_hint_for_session

        registry = MacroRegistry()
        hint = macro_hint_for_session(registry)
        assert hint is None

    def test_hint_includes_usage_instruction(self):
        from tok.macros.expansion import macro_hint_for_session

        registry = MacroRegistry()
        registry.register(
            Macro(
                name="view_edit",
                instructions=(
                    Instruction(op="view", args=("$p0",)),
                    Instruction(op="edit", args=("$p0",)),
                ),
                inputs=("p0",),
                hit_count=10,
            )
        )
        hint = macro_hint_for_session(registry)
        assert hint is not None
        assert "tool" in hint.lower() or "call" in hint.lower()


# ---------------------------------------------------------------------------
# Phase 3 regression: savings attribution still works
# ---------------------------------------------------------------------------


class TestSavingsAttribution:
    def test_savings_counted_after_expansion(self):
        from tok.macros.expansion import expand_macro

        macro = Macro(
            name="grep_view",
            instructions=(
                Instruction(op="grep", args=("$pattern",)),
                Instruction(op="view", args=("$file",)),
            ),
            inputs=("pattern", "file"),
        )
        blocks = expand_macro(macro, {"pattern": "reactor", "file": "src/tok/neuro/ir.py"})
        reference = "@grep_view(pattern=reactor, file=src/tok/neuro/ir.py)"
        expanded_text = " | ".join(f"{b['name']}({b['input']})" for b in blocks)
        assert len(expanded_text) > len(reference)
