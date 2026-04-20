"""Memory growth measurements for BridgeMemoryState."""

import json

try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional
    plt = None

from tok.runtime.memory.bridge_memory import BridgeMemoryState


def test_bridge_memory_growth_stays_bounded() -> None:
    state = BridgeMemoryState()
    tok_lengths: list[int] = []
    json_lengths: list[int] = []
    history: list[dict[str, int]] = []

    for turn in range(1, 1001):
        tok_state = f">>> turns:{turn}|facts:fact_{turn}|goal:goal_{turn}|next:next_{turn}"
        state.ingest_wire_state(tok_state)
        tok_lengths.append(len(state.to_tok()))
        history.append({"turn": turn, "facts": turn})
        json_lengths.append(len(json.dumps(history)))

    assert tok_lengths, "Tok lengths should be recorded"
    assert tok_lengths[-1] < 25000, "Tok growth should stay below threshold"
    assert tok_lengths[-1] - tok_lengths[0] < 22000, "Tok growth should flatten"
    assert json_lengths[-1] > tok_lengths[-1] * 1.1, "JSON log should grow faster"

    if plt:
        plt.figure()
        plt.plot(tok_lengths, label="tok")
        plt.plot(json_lengths, label="json")
        plt.legend()
        plt.close()
