from __future__ import annotations

from dataclasses import dataclass
import re

from .coding_grow import tokify_code
from .coding_tasks import CodingTask
from .llm_clients import ChatLLM
from .growth_modes import GrowthMode
from .memory import TokMemory, EpisodeMemory, LessonMemory, RepairMemory
from collections.abc import Sequence
from .ir import MacroRegistry, TokIR, execute_ir
from .. import verify


CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Attempt:
    ok: bool
    code: str
    error: str
    trace: tuple[str, ...]


def extract_code(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip() + "\n"
    # Fallback: return raw text (some models omit fences)
    return text.strip() + "\n"


# safe_verify has been moved to seed_lab/verify.py as verify_coding_task


def memory_key(prompt: str) -> tuple[str, ...]:
    tok = tokify_code(prompt)
    # Keep structure-heavy tokens for retrieval
    keep = [
        t
        for t in tok.tokens
        if t.startswith(
            ("cue:", "type:", "out:", "schema:", "meta:order:", "ex:")
        )
    ]
    return tuple(sorted(set(keep)))[:12]


class TokLLMGrowEngine:
    def __init__(
        self, llm: ChatLLM, registry: MacroRegistry | None = None
    ) -> None:
        self.llm = llm
        self.registry = registry or MacroRegistry()
        self.memory: list[TokMemory] = []

    def _retrieve(self, key: tuple[str, ...], k: int = 3) -> list[str]:
        query = frozenset(key)
        scored: list[tuple[int, str]] = []
        for m in self.memory:
            if isinstance(m, EpisodeMemory):
                scored.append((len(query & m.tokens), m.answer))
        scored.sort(reverse=True)
        return [code for score, code in scored[:k] if score > 0]

    def _hint(self, key: tuple[str, ...]) -> str:
        key_set = frozenset(key)
        # Check if we have a learned hint in memory
        for m in self.memory:
            if isinstance(m, LessonMemory) and m.tokens == key_set:
                return m.lesson

        # Default hints derived from Tok structure.
        key_set = frozenset(key)

        # Check for past repair lessons
        repairs = [
            m
            for m in self.memory
            if isinstance(m, RepairMemory) and m.tokens == key_set
        ]
        if repairs:
            # Show the most recent success story if possible
            for r in reversed(repairs):
                if r.final_ok:
                    return f"Past fix for this skill: {r.history[-1][0].strip()}\n(Context: {r.history[-2][1] if len(r.history) > 1 else 'initial attempt'})"

        if "cue:normalize" in key_set:
            return "Remember to normalize: lowercase and remove non-alphanumeric characters before reasoning."
        if "cue:dedupe" in key_set or "cue:unique" in key_set:
            return "If duplicates matter, consider using dict.fromkeys (order-preserving) or set (unordered) depending on the spec."
        if "cue:even" in key_set:
            return "If the spec mentions even numbers, apply an even filter before aggregating."
        return "Keep it simple and directly follow the signature and examples."

    def propose(
        self, task: CodingTask, retrieved: list[str], hint: str, failure: str
    ) -> str:
        system = (
            "You are a careful Python programmer. Output ONLY a single Python code block.\n"
            "Do not import anything. Do not access files or network. Use pure Python.\n"
            "Match the signature exactly and pass the examples.\n\n"
            "SPECIAL: You have access to symbolic macros. IF A MACRO EXISTS FOR THIS TASK, "
            "YOU MUST OUTPUT ONLY THE MACRO CALL (e.g., `@macro_name(args)` ). "
            "THIS IS FASTER AND MORE RELIABLE THAN WRITING PYTHON CODE."
        )
        if self.registry.macros:
            system += "\n\nAvailable Macros:\n"
            for name, m in self.registry.macros.items():
                system += f"- @{name}: inputs={m.inputs}\n"
        user = f"{task.prompt}\n\nHint: {hint}\n"
        if retrieved:
            user += "\nRetrieved prior solutions (may or may not apply):\n"
            for idx, code in enumerate(retrieved, start=1):
                snippet = code.strip()
                if len(snippet) > 900:
                    snippet = snippet[:900] + "\n# ...truncated...\n"
                user += f"\n# Retrieved {idx}\n```python\n{snippet}\n```\n"
        if failure:
            user += f"\nPrevious attempt failed with: {failure}\nPlease fix.\n"
        raw = self.llm.chat(system=system, user=user)
        return extract_code(raw)

    def solve(
        self,
        task: CodingTask,
        *,
        max_attempts: int = 4,
        memory_k: int = 3,
        growth_mode: GrowthMode = GrowthMode.EPISODE,
    ) -> Attempt:
        key = memory_key(task.prompt)
        hint = self._hint(key)
        history: list[tuple[str, str]] = []
        failure = ""
        trace: list[str] = []
        for attempt_idx in range(1, max_attempts + 1):
            retrieved = self._retrieve(key, k=memory_k)
            code = self.propose(task, retrieved, hint, failure)

            # 2. Execute (Symbolic or Python)
            if code.strip().startswith("@"):
                # Symbolic Macro execution
                macro_invocation = code.strip()[
                    1:
                ]  # e.g. "auto_macro_0($data)"
                # Simple parser for @macro($var)
                match = re.search(r"(\w+)\((.*)\)", macro_invocation)
                if match:
                    m_name = match.group(1)

                    macro = self.registry.get(m_name)
                    if macro:
                        try:
                            # Run on first sample to verify
                            sample_input = task.tests[0][0]
                            # Wrap in $data if needed
                            result = execute_ir(
                                TokIR(macro.instructions),
                                {"data": sample_input},
                                self.registry,
                            )
                            expected = task.tests[0][1]
                            ok = result == expected
                            err = (
                                ""
                                if ok
                                else f"Macro {m_name} returned {result}, expected {expected}"
                            )
                        except Exception as e:
                            ok = False
                            err = str(e)
                    else:
                        ok = False
                        err = f"Macro {m_name} not found in registry"
                else:
                    ok = False
                    err = "Invalid macro invocation format"
            else:
                # Normal Python execution
                v_res = verify.verify_coding_task(task, code)
                ok = v_res.ok
                err = (
                    v_res.diagnostics
                    if v_res.diagnostics
                    else v_res.failure_type or ""
                )

            trace.append(f"attempt{attempt_idx}:{'ok' if ok else 'fail'}")
            history.append((code, err))

            if ok:
                if growth_mode != GrowthMode.NONE:
                    is_repair = attempt_idx > 1
                    if growth_mode != GrowthMode.REPAIR or is_repair:
                        self.memory.append(
                            EpisodeMemory(
                                tokens=frozenset(key),
                                question=task.prompt,
                                answer=code,
                                ok=True,
                            )
                        )
                        if is_repair:
                            self.memory.append(
                                RepairMemory(
                                    tokens=frozenset(key),
                                    history=tuple(history),
                                    final_ok=True,
                                )
                            )
                        self.memory.append(
                            LessonMemory(tokens=frozenset(key), lesson=hint)
                        )
                return Attempt(True, code, "", tuple(trace))
            failure = err

        if growth_mode != GrowthMode.NONE:
            self.memory.append(
                RepairMemory(
                    tokens=frozenset(key),
                    history=tuple(history),
                    final_ok=False,
                )
            )
        return Attempt(False, code, failure, tuple(trace))


def save_tok(
    path: str,
    memory: Sequence[TokMemory],
    registry: MacroRegistry | None = None,
) -> None:
    """Saves memory and registry to a portable .tok file."""
    with open(path, "w") as f:
        if registry:
            f.write("# Macros\n")
            for name, m in registry.macros.items():
                f.write(f"@{name}({', '.join(m.inputs)}):\n")
                for inst in m.instructions:
                    f.write(
                        f"  {inst.target} = {inst.op}({', '.join(str(a) for a in inst.args)})\n"
                    )
                f.write("\n")

        f.write("# Rules & Lessons\n")
        for mem in memory:
            if isinstance(mem, LessonMemory):
                f.write(f"TOK: {mem.lesson}\n")


def load_tok(_path: str) -> tuple[list[TokMemory], MacroRegistry]:
    """Basic parser for .tok files (POC)."""
    # ... In a real system, this would be a full grammar parser ...
    # For now, we return empty structure to show intent.
    return [], MacroRegistry()
