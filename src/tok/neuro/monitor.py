import logging
import re
import time
import shlex
from pathlib import Path
from typing import cast
from .ir import TokIR, Instruction
from .miner import IRPatternMiner
from .planner import save_tok
from .memory import EpisodeMemory, TokMemory

logger = logging.getLogger(__name__)

BLOCK_RE = re.compile(
    r"\[(.*?)\] COMMAND: (.*?)\nEXIT CODE: (\d+)\nSTDOUT:\n(.*?)\n\nSTDERR:",
    re.DOTALL,
)


class LiveNeuroMonitor:
    def __init__(self, log_path: str, target_tok: str) -> None:
        self.log_path = Path(log_path)
        self.target_tok = Path(target_tok)
        self.history: list[EpisodeMemory] = []
        self.last_pos = 0

    def parse_command_to_ir(self, cmd_str: str) -> TokIR:
        parts = shlex.split(cmd_str)
        if not parts:
            return TokIR(())
        op = parts[0]
        args = ["$" + str(i) for i in range(len(parts[1:]))]
        inst = Instruction(op=op, args=tuple(args), target="res")
        return TokIR(instructions=(inst,))

    def step(self) -> None:
        if not self.log_path.exists():
            return

        with open(self.log_path) as f:
            f.seek(self.last_pos)
            content = f.read()
            self.last_pos = f.tell()

        blocks = BLOCK_RE.findall(content)
        new_episodes = []
        for _, cmd, code, _ in blocks:
            if code == "0":
                ir = self.parse_command_to_ir(cmd)
                new_episodes.append(
                    EpisodeMemory(
                        tokens=frozenset(),
                        question=cmd,
                        answer="ok",
                        ok=True,
                        metadata={"ir": ir},
                    )
                )

        if new_episodes:
            logger.debug(
                "Distilled %d new actions into TokIR", len(new_episodes)
            )
            self.history.extend(new_episodes)

            # Mine and Sync
            miner = IRPatternMiner(min_frequency=2)
            histories = [
                e.metadata["ir"]
                for e in self.history
                if isinstance(e.metadata.get("ir"), TokIR)
            ]
            macros = miner.mine(histories)

            if macros:
                from .ir import MacroRegistry

                reg = MacroRegistry()
                for m in macros:
                    reg.register(m)
                save_tok(
                    str(self.target_tok),
                    cast("list[TokMemory]", self.history),
                    reg,
                )
                logger.info(
                    "Synced %d macros to %s", len(macros), self.target_tok
                )

    def run(self) -> None:
        logger.info("Neuro-Monitor started on %s", self.log_path)
        while True:
            self.step()
            time.sleep(5)


if __name__ == "__main__":
    monitor = LiveNeuroMonitor(
        str(Path(__file__).parent.parent.parent / "execution.log"),
        str(Path(__file__).parent.parent.parent / "macros.tok"),
    )
    monitor.run()
