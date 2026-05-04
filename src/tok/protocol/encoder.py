from .models import TokNode
from .parser import serialize


class TokEncoder:
    @staticmethod
    def encode(nodes: list[TokNode], compact: bool = False) -> str:
        return serialize(nodes, compact=compact)
