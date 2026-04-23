from __future__ import annotations

from pathlib import Path

from ._models import BenchmarkDefinition

DEFAULT_BENCHMARKS: dict[str, BenchmarkDefinition] = {
    "coding-loop": BenchmarkDefinition(
        name="coding-loop",
        fixture_path=Path("tests/fixtures/replay/claude_coding_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed coding session. Answer briefly and cite exact filenames when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the file that was changed>\n"
            "Verification=<the command or result that verified the fix>"
        ),
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "1 passed", "pytest"),
        default_turns=3,
    ),
    "coding-loop-5": BenchmarkDefinition(
        name="coding-loop-5",
        fixture_path=Path("tests/fixtures/replay/claude_coding_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed coding session. Answer briefly and cite exact filenames when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the file that was changed>\n"
            "Verification=<the command or result that verified the fix>"
        ),
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "1 passed", "pytest"),
        default_turns=5,
    ),
    "research-loop": BenchmarkDefinition(
        name="research-loop",
        fixture_path=Path("tests/fixtures/replay/research_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed codebase research session. "
            "Answer briefly and cite exact filenames and identifiers when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>"
        ),
        success_terms=("compression.py", "compress_history"),
        min_success_terms=2,
        expected_file_terms=("compression.py",),
        expected_verification_terms=("compress_history",),
        default_turns=3,
        prompt_sequence=(
            "Respond in exactly one line: File=<the primary file that answered the original question>",
            "Respond in exactly one line: Verification=<the function, class, or finding that supports the answer>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
        ),
    ),
    "research-loop-current": BenchmarkDefinition(
        name="research-loop-current",
        fixture_path=Path("tests/fixtures/replay/research_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed codebase research session. "
            "Answer briefly and cite exact filenames and identifiers when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>"
        ),
        success_terms=(
            "compression/__init__.py",
            "bridge_memory.py",
            "compress_history",
            "BridgeMemoryState",
        ),
        min_success_terms=2,
        expected_file_terms=("compression/__init__.py", "runtime/memory/bridge_memory.py"),
        expected_verification_terms=("compress_history", "BridgeMemoryState"),
        default_turns=3,
        prompt_sequence=(
            "Respond in exactly one line: File=<the primary file that answered the original question>",
            "Respond in exactly one line: Verification=<the function, class, or finding that supports the answer>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
        ),
    ),
    "research-loop-5": BenchmarkDefinition(
        name="research-loop-5",
        fixture_path=Path("tests/fixtures/replay/research_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed codebase research session. "
            "Answer briefly and cite exact filenames and identifiers when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>"
        ),
        # Accept both original (compression.py) and related (bridge_memory.py) findings
        # since fixture's final question asks about related memory structure
        success_terms=(
            "compression.py",
            "bridge_memory.py",
            "compress_history",
            "BridgeMemoryState",
        ),
        min_success_terms=2,
        expected_file_terms=("compression.py", "bridge_memory.py"),
        expected_verification_terms=("compress_history", "BridgeMemoryState"),
        default_turns=5,
        prompt_sequence=(
            "Respond in exactly one line: File=<the primary file that answered the original question>",
            "Respond in exactly one line: Verification=<the function, class, or finding that supports the answer>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
            "Respond in exactly one line: Related=<the related file or class mentioned during the investigation>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
        ),
    ),
    # --- 8-turn probes: not part of the validated 5-turn release set ---
    "coding-loop-8": BenchmarkDefinition(
        name="coding-loop-8",
        fixture_path=Path("tests/fixtures/replay/claude_coding_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed coding session. Answer briefly and cite exact filenames when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the file that was changed>\n"
            "Verification=<the command or result that verified the fix>"
        ),
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "1 passed", "pytest"),
        default_turns=8,
    ),
    "research-loop-8": BenchmarkDefinition(
        name="research-loop-8",
        fixture_path=Path("tests/fixtures/replay/research_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed codebase research session. "
            "Answer briefly and cite exact filenames and identifiers when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>"
        ),
        # Accept both original (compression.py) and related (bridge_memory.py) findings
        # since fixture's final question asks about related memory structure
        success_terms=(
            "compression.py",
            "bridge_memory.py",
            "compress_history",
            "BridgeMemoryState",
        ),
        min_success_terms=2,
        expected_file_terms=("compression.py", "bridge_memory.py"),
        expected_verification_terms=("compress_history", "BridgeMemoryState"),
        default_turns=8,
        prompt_sequence=(
            "Respond in exactly one line: File=<the primary file that answered the original question>",
            "Respond in exactly one line: Verification=<the function, class, or finding that supports the answer>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
            "Respond in exactly one line: Related=<the related file or class mentioned during the investigation>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
            "Respond in exactly one line: File=<the primary file that answered the original question>",
            "Respond in exactly one line: Verification=<the function, class, or finding that supports the answer>",
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>",
        ),
    ),
    "neuro-loop": BenchmarkDefinition(
        name="neuro-loop",
        fixture_path=Path("tests/fixtures/replay/neuro_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed coding session. Answer briefly and cite exact filenames when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>"
        ),
        success_terms=("parser.py", "parse_error"),
        min_success_terms=2,
        expected_file_terms=("parser.py",),
        expected_verification_terms=("parse_error",),
        default_turns=3,
        prompt_sequence=(
            "Respond in exactly one line: File=<the primary file that answered the original question>",
            "Respond in exactly one line: Verification=<the function, class, or finding that supports the answer>",
        ),
    ),
    "jit-loop": BenchmarkDefinition(
        name="jit-loop",
        fixture_path=Path("tests/fixtures/replay/jit_loop.jsonl"),
        system_prompt=(
            "You are evaluating a repetitive coding task. "
            "Identify patterns and use JIT macros when offered to save time."
        ),
        followup_prompt=("Check src/tok/cli.py for the same pattern."),
        success_terms=("cli.py", "parse_error", "pytest src/tok/cli.py"),
        min_success_terms=2,
        expected_verification_terms=(),
        default_turns=1,
    ),
    "grammar_drift": BenchmarkDefinition(
        name="grammar_drift",
        fixture_path=Path("tests/fixtures/replay/grammar_drift.jsonl"),
        system_prompt="Analyze the following session for grammar drift.",
        followup_prompt="Has the grammar drifted?",
        success_terms=("yes", "no"),
        min_success_terms=1,
        default_turns=3,
    ),
    # --- 15-turn probes: scaling validation ---
    "coding-loop-15": BenchmarkDefinition(
        name="coding-loop-15",
        fixture_path=Path("tests/fixtures/replay/claude_coding_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed coding session. Answer briefly and cite exact filenames when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the file that was changed>\n"
            "Verification=<the command or result that verified the fix>"
        ),
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "1 passed", "pytest"),
        default_turns=15,
    ),
    "research-loop-15": BenchmarkDefinition(
        name="research-loop-15",
        fixture_path=Path("tests/fixtures/replay/research_loop_extended.jsonl"),
        system_prompt=(
            "You are evaluating a completed codebase research session. "
            "Answer briefly and cite exact filenames and identifiers when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>"
        ),
        # Accept all files discovered during extended research session
        success_terms=(
            "compression.py",
            "bridge_memory.py",
            "core.py",
            "smart_policy.py",
            "compress_history",
            "BridgeMemoryState",
            "RuntimeSession",
            "MemoryProjectionProfile",
        ),
        min_success_terms=3,
        expected_file_terms=(
            "compression.py",
            "bridge_memory.py",
            "core.py",
            "smart_policy.py",
        ),
        expected_verification_terms=(
            "compress_history",
            "BridgeMemoryState",
            "RuntimeSession",
            "MemoryProjectionProfile",
        ),
        default_turns=15,
    ),
    # --- 25-turn probes: long-session validation ---
    "coding-loop-25": BenchmarkDefinition(
        name="coding-loop-25",
        fixture_path=Path("tests/fixtures/replay/claude_coding_loop.jsonl"),
        system_prompt=(
            "You are evaluating a completed coding session. Answer briefly and cite exact filenames when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the file that was changed>\n"
            "Verification=<the command or result that verified the fix>"
        ),
        success_terms=("gateway.py", "passed"),
        min_success_terms=2,
        expected_file_terms=("gateway.py",),
        expected_verification_terms=("passed", "1 passed", "pytest"),
        default_turns=25,
    ),
    "research-loop-25": BenchmarkDefinition(
        name="research-loop-25",
        fixture_path=Path("tests/fixtures/replay/research_loop_extended.jsonl"),
        system_prompt=(
            "You are evaluating a completed codebase research session. "
            "Answer briefly and cite exact filenames and identifiers when possible."
        ),
        followup_prompt=(
            "Based on the conversation so far, respond in exactly two lines:\n"
            "File=<the primary file that answered the question>\n"
            "Verification=<the function, class, or finding that supports the answer>"
        ),
        # Accept all files discovered during extended research session
        success_terms=(
            "compression.py",
            "bridge_memory.py",
            "core.py",
            "smart_policy.py",
            "compress_history",
            "BridgeMemoryState",
            "RuntimeSession",
            "MemoryProjectionProfile",
        ),
        min_success_terms=3,
        expected_file_terms=(
            "compression.py",
            "bridge_memory.py",
            "core.py",
            "smart_policy.py",
        ),
        expected_verification_terms=(
            "compress_history",
            "BridgeMemoryState",
            "RuntimeSession",
            "MemoryProjectionProfile",
        ),
        default_turns=25,
    ),
}


DEFAULT_MULTI_TURN_PROMPTS: tuple[str, ...] = (
    "Respond in exactly one line: File=<the file that was changed>",
    "Respond in exactly one line: Verification=<the command or result that verified the fix>",
    "Based on the conversation so far, respond in exactly two lines:\n"
    "File=<the file that was changed>\n"
    "Verification=<the command or result that verified the fix>",
)


def load_benchmark_definition(name: str) -> BenchmarkDefinition:
    try:
        return DEFAULT_BENCHMARKS[name]
    except KeyError as exc:
        msg = f"Unknown benchmark: {name}"
        raise ValueError(msg) from exc
