from .models import TokNode
from .parser import serialize


# Stub out missing neural module
class NeuralCompressor:
    def compress(self, text: str) -> str:
        return text


class TokEncoder:
    @staticmethod
    def encode(nodes: list[TokNode], compact: bool = False) -> str:
        raw_tok = serialize(nodes, compact=compact)
        if compact:
            return NeuralCompressor().compress(raw_tok)
        return raw_tok
