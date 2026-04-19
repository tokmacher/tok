"""Model-specific compression strategy profiles.

Each profile defines how aggressively Tok should compress for a given model,
based on empirical benchmark data about repair-loop susceptibility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """Compression profile for a specific model or model family."""

    name: str
    repair_loop_susceptibility: float
    compression_aggressiveness: float
    prefer_stable_file_stubs: bool


_DEEPSEEK_V3_2 = ModelProfile(
    name="deepseek-v3.2",
    repair_loop_susceptibility=0.76,
    compression_aggressiveness=0.5,
    prefer_stable_file_stubs=True,
)

_GPT_4_1 = ModelProfile(
    name="gpt-4.1",
    repair_loop_susceptibility=0.24,
    compression_aggressiveness=1.0,
    prefer_stable_file_stubs=True,
)

_CLAUDE_SONNET = ModelProfile(
    name="claude-sonnet",
    repair_loop_susceptibility=0.1,
    compression_aggressiveness=1.0,
    prefer_stable_file_stubs=True,
)

_QWEN3_CODER = ModelProfile(
    name="qwen3-coder",
    repair_loop_susceptibility=0.7,
    compression_aggressiveness=0.4,
    prefer_stable_file_stubs=True,
)

_GLM_5 = ModelProfile(
    name="glm-5",
    repair_loop_susceptibility=0.4,
    compression_aggressiveness=0.6,
    prefer_stable_file_stubs=True,
)

_DEFAULT = ModelProfile(
    name="default",
    repair_loop_susceptibility=0.3,
    compression_aggressiveness=0.8,
    prefer_stable_file_stubs=True,
)

_MODEL_PROFILES: dict[str, ModelProfile] = {
    "deepseek-v3": _DEEPSEEK_V3_2,
    "deepseek-v3.2": _DEEPSEEK_V3_2,
    "deepseek/deepseek-v3": _DEEPSEEK_V3_2,
    "deepseek/deepseek-v3.2": _DEEPSEEK_V3_2,
    "deepseek-chat": _DEEPSEEK_V3_2,
    "gpt-4.1": _GPT_4_1,
    "gpt-4-1": _GPT_4_1,
    "openai/gpt-4.1": _GPT_4_1,
    "claude-sonnet": _CLAUDE_SONNET,
    "claude-sonnet-4": _CLAUDE_SONNET,
    "claude-sonnet-4.6": _CLAUDE_SONNET,
    "anthropic/claude-sonnet-4": _CLAUDE_SONNET,
    "qwen3-coder": _QWEN3_CODER,
    "qwen3-coder-next": _QWEN3_CODER,
    "qwen/qwen3-coder": _QWEN3_CODER,
    "qwen/qwen3-coder-next": _QWEN3_CODER,
    "glm-5": _GLM_5,
    "z-ai/glm-5": _GLM_5,
}


def get_model_profile(model: str) -> ModelProfile:
    """Look up the model profile for a given model string.

    Falls back to a fuzzy prefix match. Returns the default profile
    if no match is found.
    """
    if model in _MODEL_PROFILES:
        return _MODEL_PROFILES[model]
    model_lower = model.lower()
    for key, profile in _MODEL_PROFILES.items():
        if model_lower.startswith(key.lower()) or key.lower().startswith(model_lower):
            return profile
    return _DEFAULT
