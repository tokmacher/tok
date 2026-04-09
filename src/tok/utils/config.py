"""Configuration for the Tok Protocol."""

import os

# LLM Configuration
MODEL_NAME: str = "gpt-4o-mini"
API_BASE: str = "https://openrouter.ai/api/v1"
API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
DEFAULT_TEMPERATURE: float = 0.7

# Token Thresholds
MIN_TOK_LENGTH: int = 500

# State Management
MAX_STEPS: int = 20
PING_INTERVAL: int = 10
