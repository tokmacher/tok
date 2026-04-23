"""
System prompt bloat analyzer for the Tok runtime.

Measures token and character counts of each prompt component across scenarios,
directive escalation levels, memory profiles, and history compression.
"""

from __future__ import annotations

from typing import Any

from tok.utils.token_utils import count_tokens


def count_chars(text: str) -> int:
    return len(text)


def measure(text: str, label: str) -> dict[str, str | int]:
    return {
        "label": label,
        "chars": count_chars(text),
        "tokens": count_tokens(text),
    }


def measure_base_prompts() -> dict[str, dict[str, str | int]]:
    """Measure token counts of TOK_SYSTEM_PROMPT and all variants."""
    from tok.analysis.prompt import (
        MINIMAL_PULSE_PROMPT,
        NAKED_TOK_SYSTEM_PROMPT,
        TOK_EXPLORE_PROMPT,
        TOK_SYSTEM_PROMPT,
    )

    return {
        "TOK_SYSTEM_PROMPT": measure(TOK_SYSTEM_PROMPT, "TOK_SYSTEM_PROMPT"),
        "NAKED_TOK_SYSTEM_PROMPT": measure(NAKED_TOK_SYSTEM_PROMPT, "NAKED_TOK_SYSTEM_PROMPT"),
        "TOK_EXPLORE_PROMPT": measure(TOK_EXPLORE_PROMPT, "TOK_EXPLORE_PROMPT"),
        "MINIMAL_PULSE_PROMPT": measure(MINIMAL_PULSE_PROMPT, "MINIMAL_PULSE_PROMPT"),
    }


def measure_directive_sizes() -> dict[str, dict[str, str | int]]:
    """Measure all TOK_OUTPUT_DIRECTIVE variants and TOK_PROTOCOL_LAW."""
    from tok.compression import (
        TOK_OUTPUT_DIRECTIVE,
        TOK_OUTPUT_DIRECTIVE_MINIMAL,
        TOK_OUTPUT_DIRECTIVE_REINFORCED,
        TOK_PROTOCOL_LAW,
        TOK_TOOL_COMPAT_DIRECTIVE,
    )

    return {
        "TOK_PROTOCOL_LAW": measure(TOK_PROTOCOL_LAW, "TOK_PROTOCOL_LAW"),
        "TOK_OUTPUT_DIRECTIVE": measure(TOK_OUTPUT_DIRECTIVE, "TOK_OUTPUT_DIRECTIVE"),
        "TOK_OUTPUT_DIRECTIVE_MINIMAL": measure(TOK_OUTPUT_DIRECTIVE_MINIMAL, "TOK_OUTPUT_DIRECTIVE_MINIMAL"),
        "TOK_OUTPUT_DIRECTIVE_REINFORCED": measure(TOK_OUTPUT_DIRECTIVE_REINFORCED, "TOK_OUTPUT_DIRECTIVE_REINFORCED"),
        "TOK_TOOL_COMPAT_DIRECTIVE": measure(TOK_TOOL_COMPAT_DIRECTIVE, "TOK_TOOL_COMPAT_DIRECTIVE"),
    }


def measure_grammar_snippets() -> dict[str, dict[str, str | int]]:
    """Measure grammar snippet sizes at each bootstrap level."""
    from tok.analysis.prompt import get_grammar_snippet

    levels = ["essentials", "restricted", "full", "pulse", "explore"]
    return {lvl: measure(get_grammar_snippet(lvl), f"grammar:{lvl}") for lvl in levels}


def analyze_directive_overlap() -> dict[str, dict[str, list[str] | int]]:
    """Identify token overlap between directive components using line diffing."""
    from tok.compression import (
        TOK_OUTPUT_DIRECTIVE,
        TOK_OUTPUT_DIRECTIVE_MINIMAL,
        TOK_OUTPUT_DIRECTIVE_REINFORCED,
        TOK_PROTOCOL_LAW,
    )

    def _lines(s: str) -> set[str]:
        return {ln.strip() for ln in s.splitlines() if ln.strip()}

    law_lines = _lines(TOK_PROTOCOL_LAW)
    full_lines = _lines(TOK_OUTPUT_DIRECTIVE)
    minimal_lines = _lines(TOK_OUTPUT_DIRECTIVE_MINIMAL)
    reinforced_lines = _lines(TOK_OUTPUT_DIRECTIVE_REINFORCED)

    law_full_overlap = law_lines & full_lines
    full_minimal_overlap = full_lines & minimal_lines
    law_reinforced_overlap = law_lines & reinforced_lines
    full_reinforced_overlap = full_lines & reinforced_lines

    return {
        "law_vs_full_directive": {
            "shared_lines": list(law_full_overlap),
            "shared_count": len(law_full_overlap),
        },
        "full_vs_minimal_directive": {
            "shared_lines": list(full_minimal_overlap),
            "shared_count": len(full_minimal_overlap),
        },
        "law_vs_reinforced": {
            "shared_lines": list(law_reinforced_overlap),
            "shared_count": len(law_reinforced_overlap),
        },
        "full_vs_reinforced": {
            "shared_lines": list(full_reinforced_overlap),
            "shared_count": len(full_reinforced_overlap),
        },
    }


def measure_pressure_impact() -> dict[str, dict[str, str | int]]:
    """Measure the injected system prompt size at pressure levels 0, 25, 50, 75, 100."""
    from tok.compression import inject_system_additions

    results = {}
    for pressure in [0, 25, 50, 75, 100]:
        body = inject_system_additions(
            {"system": ""},
            tok_state=None,
            tool_compatible=False,
            grammar=None,
            pressure=pressure,
        )
        injected = body.get("system", "")
        results[f"pressure_{pressure}"] = measure(injected, f"pressure={pressure}")
    return results


def measure_dynamic_injections() -> dict[str, dict[str, str | int]]:
    """Measure the total injected system additions with grammar at each level."""
    from tok.analysis.prompt import get_grammar_snippet
    from tok.compression import inject_system_additions

    results = {}
    # grammar only (no tok_state)
    for level in ["essentials", "restricted", "full", "pulse", "explore"]:
        grammar = get_grammar_snippet(level)
        body = inject_system_additions(
            {"system": ""},
            tok_state=None,
            tool_compatible=False,
            grammar=grammar,
            pressure=0,
        )
        injected = body.get("system", "")
        results[f"grammar_only:{level}"] = measure(injected, f"grammar_only:{level}")

    # tok_state only (representative wire state, no grammar)
    sample_tok_state = ">>> turns:5|goal:implement feature X|files:src/foo.py,src/bar.py|errs:test_x failed"
    body = inject_system_additions(
        {"system": ""},
        tok_state=sample_tok_state,
        tool_compatible=False,
        grammar=None,
        pressure=0,
    )
    results["tok_state_only"] = measure(body.get("system", ""), "tok_state_only")

    # combined: grammar + tok_state + pressure
    grammar = get_grammar_snippet("restricted")
    body = inject_system_additions(
        {"system": ""},
        tok_state=sample_tok_state,
        tool_compatible=False,
        grammar=grammar,
        pressure=75,
    )
    results["combined_grammar_state_pressure75"] = measure(body.get("system", ""), "combined_grammar_state_pressure75")

    return results


_TYPICAL_TOK_STATE = (
    ">>> turns:8|goal:refactor auth|files:src/auth.py,src/tokens.py|"
    "cmds:pytest tests/|errs:TokenExpired|constraints:no breaking changes|"
    "next:update refresh logic"
)


def measure_per_turn_actual() -> dict[str, Any]:
    """
    Measure what is actually injected each turn by the gateway path.

    The gateway always passes grammar=None, todo=None, deltas=None.
    Only pressure and tok_state vary turn-to-turn.

    Scenarios:
      cold_start      — first turn, no memory, pressure=0
      warm_no_drift   — turn 2+, state present, pressure=0
      warm_low_drift  — state present, pressure=25 (law fires, minimal directive)
      warm_high_drift — state present, pressure=75 (law + reinforced directive)
    """
    from tok.compression import inject_system_additions

    def _inject(tok_state: str | None, pressure: int) -> str:
        body = inject_system_additions(
            {"system": ""},
            tok_state=tok_state,
            tool_compatible=False,
            grammar=None,  # gateway never passes grammar
            todo=None,
            deltas=None,
            pressure=pressure,
        )
        return str(body.get("system", ""))

    cold = _inject(None, 0)
    warm_no_drift = _inject(_TYPICAL_TOK_STATE, 0)
    warm_low_drift = _inject(_TYPICAL_TOK_STATE, 25)
    warm_high_drift = _inject(_TYPICAL_TOK_STATE, 75)

    results: dict[str, Any] = {
        "cold_start": measure(cold, "per_turn:cold_start"),
        "warm_no_drift": measure(warm_no_drift, "per_turn:warm_no_drift"),
        "warm_low_drift": measure(warm_low_drift, "per_turn:warm_low_drift"),
        "warm_high_drift": measure(warm_high_drift, "per_turn:warm_high_drift"),
    }

    # Breakdown: token cost of each additive component
    from tok.compression import (
        TOK_OUTPUT_DIRECTIVE_MINIMAL,
        TOK_OUTPUT_DIRECTIVE_REINFORCED,
        TOK_PROTOCOL_LAW,
    )

    results["_component_costs"] = {
        "mode_header": measure("=== MODE: TOK-NATIVE ===", "mode_header"),
        "minimal_directive": measure(TOK_OUTPUT_DIRECTIVE_MINIMAL, "minimal_directive"),
        "protocol_law": measure(TOK_PROTOCOL_LAW, "protocol_law"),
        "reinforced_directive": measure(TOK_OUTPUT_DIRECTIVE_REINFORCED, "reinforced_directive"),
        "typical_tok_state": measure(
            f"[Tok compressed history]\n{_TYPICAL_TOK_STATE}",
            "tok_state_wrapper",
        ),
        "note": (
            "grammar/todo/deltas are always None from the gateway — "
            "TOK_SYSTEM_PROMPT (891t) is never injected per-turn."
        ),
    }

    return results


def simulate_memory_growth(
    turns: list[int] | None = None,
) -> dict[str, Any]:
    """Simulate bridge memory accumulation over N turns and measure wire_state size."""
    from tok.runtime.memory.bridge_memory import BridgeMemoryState

    if turns is None:
        turns = [1, 5, 10, 20, 50]

    results = {}
    for n in turns:
        state = BridgeMemoryState()
        for i in range(n):
            wire = (
                f">>> turns:{i}|goal:implement feature X|"
                f"files:src/module_{i % 5}.py,src/util.py|"
                f"cmds:pytest tests/|"
                f"errs:AssertionError in test_{i % 3}|"
                f"constraints:never break API|"
                f"next:fix failing test"
            )
            state.ingest_wire_state(wire)
        projected = state.wire_state()
        results[f"turns_{n}"] = measure(projected, f"wire_state after {n} turns")
        # Also capture raw to_tok for comparison (full serialization with scores)
        full_tok = state.to_tok()
        results[f"turns_{n}_full_tok"] = measure(full_tok, f"full_tok after {n} turns")
    return results


def measure_memory_profiles() -> dict[str, dict[str, str | int]]:
    """Measure wire_state size under each memory projection profile."""
    from tok.runtime.memory.bridge_memory import BridgeMemoryState
    from tok.runtime.policy.smart_policy import policy_for_model

    # Prime a state with realistic data
    state = BridgeMemoryState()
    for i in range(20):
        wire = (
            f">>> turns:{i}|goal:refactor auth module|"
            f"files:src/auth.py,src/tokens.py,tests/test_auth.py|"
            f"cmds:pytest tests/test_auth.py -v|"
            f"errs:TokenExpired in test_refresh,ValueError in test_create|"
            f"constraints:no breaking changes,keep backward compat|"
            f"questions:should we use redis or memcached?|"
            f"next:update token refresh logic"
        )
        state.ingest_wire_state(wire)

    results = {}
    for model_name, model_key in [
        ("claude-3-5-sonnet-20240620", "claude"),
        ("gpt-4o", "gpt"),
        ("gemini-1.5-pro", "gemini"),
    ]:
        policy = policy_for_model(model_name)
        for mode in ["aggressive", "balanced", "recovery"]:
            profile = policy.memory_profiles[mode]
            projected = state.wire_state(profile=profile)
            key = f"{model_key}:{mode}"
            results[key] = measure(projected, key)

    return results


def _make_conversation(n_turns: int) -> list[dict[str, Any]]:
    """Build a synthetic n-turn conversation for compression testing."""
    msgs: list[dict[str, Any]] = []
    for i in range(n_turns):
        msgs.append(
            {
                "role": "user",
                "content": (
                    f"Turn {i}: Please implement step {i} of the auth module. "
                    "Make sure to update src/auth.py and run pytest tests/test_auth.py. "
                    "Do not break backward compatibility."
                ),
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"tool_{i}",
                        "name": "edit_file",
                        "input": {
                            "path": "src/auth.py",
                            "content": f"# edited at turn {i}\ndef authenticate(): pass\n",
                        },
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"tool_{i}",
                        "content": f"OK - wrote src/auth.py (turn {i})",
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": f"Done with step {i}. Next I will implement step {i + 1}.",
            }
        )
    return msgs


def analyze_history_compression() -> dict[str, Any]:
    """Measure compression effectiveness at different conversation lengths."""
    from tok.compression import compress_history

    results = {}
    for n_turns in [2, 5, 10, 20]:
        for keep_turns in [1, 2, 3]:
            if keep_turns >= n_turns:
                continue
            msgs = _make_conversation(n_turns)
            original_text = " ".join(str(m.get("content", "")) for m in msgs)
            recent, tok_state = compress_history(msgs, keep_turns=keep_turns)
            recent_text = " ".join(str(m.get("content", "")) for m in recent)
            key = f"{n_turns}turns_keep{keep_turns}"
            results[key] = {
                "original_tokens": count_tokens(original_text),
                "original_chars": count_chars(original_text),
                "recent_tokens": count_tokens(recent_text),
                "tok_state": measure(tok_state, f"tok_state:{key}"),
                "messages_dropped": len(msgs) - len(recent),
                "compression_ratio": round(
                    count_tokens(recent_text + tok_state) / max(1, count_tokens(original_text)),
                    3,
                ),
            }
    return results


def measure_tool_compression_impact() -> dict[str, Any]:
    """Measure savings from tool result compression in the recent window."""
    from tok.compression import compress_recent_window

    # Build messages with large tool results of various types
    def _make_msg_with_tool_result(content: str, tool_id: str = "t1") -> list[dict[str, Any]]:
        return [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "view_file",
                        "input": {"path": "src/auth.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": content,
                    }
                ],
            },
        ]

    pytest_output = (
        "tests/test_auth.py::test_login PASSED\n"
        "tests/test_auth.py::test_logout PASSED\n"
        "tests/test_auth.py::test_refresh FAILED\n"
        "  AssertionError: token expired\n"
        "tests/test_auth.py::test_create PASSED\n"
        "============================== 3 passed, 1 failed in 0.52s ==============================\n"
    )

    git_diff_output = (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "index abc1234..def5678 100644\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
        "@@ -10,7 +10,10 @@ class Auth:\n"
        "-    def login(self, user):\n"
        "+    def login(self, user, mfa=False):\n"
        "+        if mfa:\n"
        "+            return self._mfa_login(user)\n"
        "         return self._basic_login(user)\n"
    )

    file_read_output = (
        "class Auth:\n"
        "    def __init__(self):\n"
        "        self.tokens = {}\n"
        "\n"
        "    def login(self, user):\n"
        "        token = self._create_token(user)\n"
        "        self.tokens[token] = user\n"
        "        return token\n"
        "\n"
        "    def logout(self, token):\n"
        "        self.tokens.pop(token, None)\n"
        "\n"
        "    def refresh(self, token):\n"
        "        if token not in self.tokens:\n"
        "            raise ValueError('token expired')\n"
        "        user = self.tokens.pop(token)\n"
        "        return self.login(user)\n"
    ) * 5  # repeat to make it large

    results = {}
    for name, content in [
        ("pytest", pytest_output),
        ("git_diff", git_diff_output),
        ("file_read", file_read_output),
    ]:
        msgs = _make_msg_with_tool_result(content)
        compressed_msgs, breakdown = compress_recent_window(msgs)
        compressed_content = ""
        for m in compressed_msgs:
            c = m.get("content", "")
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        raw = b.get("content", "")
                        compressed_content = raw if isinstance(raw, str) else str(raw)
        results[name] = {
            "original": measure(content, f"original:{name}"),
            "compressed": measure(compressed_content, f"compressed:{name}"),
            "breakdown": breakdown,
            "ratio": round(
                count_tokens(compressed_content) / max(1, count_tokens(content)),
                3,
            ),
        }
    return results


def measure_cold_start() -> dict[str, str | int]:
    """Cold-start scenario: no memory, no directives beyond minimal."""
    from tok.compression import inject_system_additions

    body = inject_system_additions(
        {"system": ""},
        tok_state=None,
        tool_compatible=False,
        grammar=None,
        pressure=0,
    )
    return measure(body.get("system", ""), "cold_start")


def measure_typical_session() -> dict[str, str | int]:
    """Typical session: memory state + minimal directive."""
    from tok.compression import inject_system_additions

    tok_state = (
        ">>> turns:8|goal:refactor auth|files:src/auth.py,src/tokens.py|"
        "cmds:pytest tests/|errs:TokenExpired|constraints:no breaking changes|"
        "next:update refresh logic"
    )
    body = inject_system_additions(
        {"system": ""},
        tok_state=tok_state,
        tool_compatible=False,
        grammar=None,
        pressure=0,
    )
    return measure(body.get("system", ""), "typical_session")


def measure_high_pressure() -> dict[str, str | int]:
    """High-pressure scenario: protocol law + reinforced directive."""
    from tok.compression import inject_system_additions

    tok_state = ">>> turns:15|goal:fix protocol drift|files:src/gateway.py|errs:protocol_drift,json_blobs_detected"
    body = inject_system_additions(
        {"system": ""},
        tok_state=tok_state,
        tool_compatible=False,
        grammar=None,
        pressure=75,
    )
    return measure(body.get("system", ""), "high_pressure")


def measure_memory_heavy() -> dict[str, Any]:
    """Memory-heavy scenario: simulate a state loaded with many entries."""
    from tok.compression import inject_system_additions
    from tok.runtime.memory.bridge_memory import BridgeMemoryState

    state = BridgeMemoryState()
    for i in range(50):
        wire = (
            f">>> turns:{i}|goal:large migration|"
            f"files:src/module_{i % 10}.py,src/util_{i % 5}.py|"
            f"cmds:pytest tests/test_{i % 8}.py|"
            f"errs:AssertionError turn {i}|"
            f"constraints:no breaking changes|"
            f"next:fix test_{i % 3}"
        )
        state.ingest_wire_state(wire)

    tok_state = state.wire_state()
    body = inject_system_additions(
        {"system": ""},
        tok_state=tok_state,
        tool_compatible=False,
        grammar=None,
        pressure=0,
    )
    return {
        "wire_state": measure(tok_state, "wire_state_memory_heavy"),
        "injected_system": measure(body.get("system", ""), "injected_memory_heavy"),
    }


def run_all_baselines() -> dict[str, Any]:
    return {
        "cold_start": measure_cold_start(),
        "typical_session": measure_typical_session(),
        "high_pressure": measure_high_pressure(),
        "memory_heavy": measure_memory_heavy(),
    }


if __name__ == "__main__":
    pass
