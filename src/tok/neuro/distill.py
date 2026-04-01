from __future__ import annotations
import collections
import logging
from typing import TYPE_CHECKING

from ..runtime.memory.tok_state import (
    TokMemory,
    EpisodeMemory,
    LessonMemory,
    ConstraintMemory,
    RepairMemory,
)
from .ir import MacroRegistry, TokIR
from .miner import IRPatternMiner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .llm_clients import ChatLLM

from .metrics import estimated_token_count


class MemoryDistiller:
    def __init__(self, llm: ChatLLM, threshold: int = 5):
        self.llm = llm
        self.threshold = threshold
        self.registry = MacroRegistry()
        self.miner = IRPatternMiner()

    def compress(self, memory: list[TokMemory]) -> list[TokMemory]:
        """
        Analyzes memory and replaces redundant episodes with distilled lessons.
        """
        initial_tokens = estimated_token_count(memory)
        episodes = [m for m in memory if isinstance(m, EpisodeMemory) and m.ok]
        repairs = [
            m for m in memory if isinstance(m, RepairMemory) and m.final_ok
        ]

        if len(episodes) < self.threshold and not repairs:
            return memory

        # Group by tokens (our conceptual signature)
        groups = collections.defaultdict(list)
        repair_groups = collections.defaultdict(list)

        for m in memory:
            if isinstance(m, EpisodeMemory) and m.ok:
                groups[m.tokens].append(m)
            elif isinstance(m, RepairMemory) and m.final_ok:
                repair_groups[m.tokens].append(m)

        logger.debug(
            f"Distiller: Found {len(groups)} episode groups and {len(repair_groups)} repair groups."
        )

        # 1. Autonomous IR Pattern Mining (Success)
        ir_histories = []
        for m in memory:
            if (
                isinstance(m, EpisodeMemory)
                and "ir" in m.metadata
                and isinstance(m.metadata["ir"], TokIR)
            ):
                ir_histories.append(m.metadata["ir"])

        discovered_macros = self.miner.mine(ir_histories, self.registry)
        for macro in discovered_macros:
            self.registry.register(macro)
            logger.debug(
                f"Autonomous Inversion: Registered macro @{macro.name}"
            )

        # 2. Negative Distillation (Failure-Driven)
        negative_lessons = self.mine_negative_patterns(memory)

        # 3. Rule Synthesis and Compression
        new_memory = [
            m
            for m in memory
            if not isinstance(m, EpisodeMemory | RepairMemory)
        ]
        new_memory.extend(negative_lessons)

        # Distill episodic groups
        for _tokens, batch in groups.items():
            if len(batch) >= self.threshold:
                # Use a combined label for the skill
                label = f"group_{'_'.join(sorted(_tokens)[:3])}"
                lesson = self.distill_batch(label, batch)
                if lesson:
                    new_memory.append(lesson)
                    logger.debug(
                        f"Distiller: Collapsed {len(batch)} episodes into 1 lesson."
                    )
                    continue
            new_memory.extend(batch)

        # Distill repair traces
        for _tokens, r_batch in repair_groups.items():
            for r in r_batch:
                lesson = self.distill_repair(r)
                if lesson:
                    new_memory.append(lesson)
                else:
                    new_memory.append(r)

        # Hierarchical Inversion
        new_memory = self.hierarchical_compress(new_memory)

        final_tokens = estimated_token_count(new_memory)
        savings = initial_tokens - final_tokens
        if savings > 0:
            logger.info(
                f"Distifflation complete: {initial_tokens} -> {final_tokens} tokens ({savings} saved)."
            )

        return new_memory

    def mine_negative_patterns(
        self, memory: list[TokMemory]
    ) -> list[ConstraintMemory]:
        """
        Detects redundant logic attempts (shadowing) and creates constraints.
        """
        redundancies: dict[str, int] = collections.defaultdict(int)
        for m in memory:
            if not isinstance(m, EpisodeMemory) or "ir" not in m.metadata:
                continue
            ir = m.metadata["ir"]
            if not isinstance(ir, TokIR):
                continue

            # Check for macros defined in this turn that were already redundant
            for macro in ir.get_defined_macros():
                existing = self.registry.find_duplicate(macro)
                if existing:
                    redundancies[existing] += 1

        constraints = []
        for macro_name, count in redundancies.items():
            if count >= 2:  # Pattern repeat threshold
                constraints.append(
                    ConstraintMemory(
                        tokens=frozenset(["redundancy", macro_name]),
                        constraint=f"TOK-NEG: avoid re-defining @{macro_name}; use existing definition to save tokens.",
                    )
                )
                logger.debug(
                    f"Negative Distillation: Mining constraint for @{macro_name}"
                )

        return constraints

    def distill_batch(
        self, key: str, episodes: list[EpisodeMemory]
    ) -> LessonMemory | None:
        """
        Uses LLM to synthesize a batch of episodes into a single principle.
        """
        if not episodes:
            return None

        system = (
            "You are a knowledge distillation engine. You are given a group of solved tasks "
            "that all share the same underlying skill or pattern. Your goal is to write a "
            "compressed, symbolic rule (a Tok rule) that explains how to solve this class of problems. "
            "Format: 'TOK: $symbolic_pattern -> $logic' "
            "Avoid natural language filler. Use $var for parameters. Keep it as dense as possible."
        )

        examples = []
        all_tokens: set[str] = set()
        for ep in episodes[:5]:  # Use up to 5 examples for context
            examples.append(f"Q: {ep.question}\nA: {ep.answer}")
            all_tokens.update(ep.tokens)

        user = f"Skill: {key}\n\nExamples:\n" + "\n---\n".join(examples)

        try:
            lesson_text = self.llm.chat(system=system, user=user)
            if lesson_text:
                return LessonMemory(
                    tokens=frozenset(all_tokens),  # Broad coverage
                    lesson=lesson_text.strip(),
                )
        except Exception:
            pass
        return None

    def distill_repair(self, repair: RepairMemory) -> LessonMemory | None:
        """
        Synthesizes a single multi-turn repair trace into a 'fixing strategy'.
        """
        system = (
            "You are a programming mentor. Extract a dense 'Pitfall/Fix' rule from "
            "a debugging trace. "
            "Format: 'TOK-REP: $pitfall_pattern -> $correct_fix' "
            "Use pseudo-code or symbolic notation where possible. NO conversational text."
        )

        history_str = []
        for idx, (code, err) in enumerate(repair.history, start=1):
            status = (
                "FINAL FIX"
                if (idx == len(repair.history) and repair.final_ok)
                else f"Attempt {idx}"
            )
            history_str.append(
                f"--- {status} ---\nCode:\n{code}\nError: {err}"
            )

        user = "Repair History:\n" + "\n".join(history_str)

        try:
            lesson_text = self.llm.chat(system=system, user=user)
            if lesson_text:
                return LessonMemory(
                    tokens=repair.tokens, lesson=lesson_text.strip()
                )
        except Exception as exc:
            logger.error(f"Distill error: {exc}")
        return None

    def hierarchical_compress(
        self, memory: list[TokMemory]
    ) -> list[TokMemory]:
        """
        Takes distilled lessons and attempts to invert them into higher-level abstractions.
        """
        lessons = [m for m in memory if isinstance(m, LessonMemory)]
        if len(lessons) < 2:
            return memory

        system = (
            "You are a meta-learning engine. You are given a list of specific 'Tok rules'. "
            "Your goal is to identify common high-level patterns (e.g., recursion, stack usage, "
            "sliding windows) and synthesize them into universal 'Metaprinciples'. "
            "Format: 'TOK-META: $concept -> $generic_strategy' "
            "If multiple rules can be unified by a single metaprinciple, suggest the metaprinciple "
            "and list which rules it replaces. "
            "Return a JSON list of objects: [{'metaprinciple': '...', 'replaces': ['rule text 1', 'rule text 2']}]"
        )

        rules_list = [f"Rule {i}: {m.lesson}" for i, m in enumerate(lessons)]
        user = "Existing Rules:\n" + "\n".join(rules_list)

        try:
            import json

            response = self.llm.chat(system=system, user=user)
            # Try to find JSON in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end != -1:
                inversions = json.loads(response[start:end])

                new_memory = [
                    m for m in memory if not isinstance(m, LessonMemory)
                ]
                replaced_indices = set()

                for inv in inversions:
                    meta_text = inv.get("metaprinciple")
                    replaces = inv.get("replaces", [])

                    if meta_text:
                        new_memory.append(
                            LessonMemory(
                                tokens=frozenset(),  # Meta-lessons have broad tokens
                                lesson=meta_text,
                            )
                        )
                        # Identify which specific lessons were abstracted
                        for r_text in replaces:
                            for idx, m in enumerate(lessons):
                                if (
                                    r_text.strip() == m.lesson.strip()
                                    or r_text in m.lesson
                                ):
                                    replaced_indices.add(idx)

                # Keep lessons that weren't replaced
                for idx, m in enumerate(lessons):
                    if idx not in replaced_indices:
                        new_memory.append(m)

                return new_memory
        except Exception as exc:
            logger.error(f"Meta-distill error: {exc}")

        return memory
