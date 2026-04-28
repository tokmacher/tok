"""Backward-compatible facade for the gateway module layout."""

from __future__ import annotations

from .gateway import _response_contract_for_mode, create_app

__all__ = [
    "_response_contract_for_mode",
    "create_app",
]
