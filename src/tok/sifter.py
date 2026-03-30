"""Backward-compatible facade for tok.utils.sifter."""

from __future__ import annotations

from .utils.sifter import *  # noqa: F403
from .utils.sifter import DirectoryWalker, Sifter  # noqa: F401
