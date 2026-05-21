"""Adapter selection and unstable-adapter gating."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterConfig:
    name: str
    bridge_mode: str
    unstable: bool = False


_ALIASES = {
    "claude": "claude",
    "claude-code": "claude",
    "claude-bridge": "claude",
    "opencode": "opencode",
    "codex": "codex-cli",
    "codex-cli": "codex-cli",
}


def resolve_adapter_config(adapter: str | None = None) -> AdapterConfig:
    raw = (adapter or os.getenv("TOK_ADAPTER", "claude")).strip() or "claude"
    name = _ALIASES.get(raw, raw)
    if name == "claude":
        return AdapterConfig(name="claude", bridge_mode="claude")
    if name in {"opencode", "codex-cli"}:
        if os.getenv("TOK_UNSTABLE_ADAPTERS", "").strip() != "1":
            raise ValueError(f"Adapter {name!r} requires TOK_UNSTABLE_ADAPTERS=1")
        return AdapterConfig(name=name, bridge_mode=name, unstable=True)
    raise ValueError(f"Unknown adapter {raw!r}")


def get_adapter_probe(adapter: str) -> Any:
    config = resolve_adapter_config(adapter)
    if config.name == "opencode":
        from tok.adapters.opencode_probe import OpenCodeAdapter

        return OpenCodeAdapter()
    if config.name == "codex-cli":
        from tok.adapters.codex_probe import CodexAdapter

        return CodexAdapter()
    from tok.adapters import ClaudeBridgeAdapter

    return ClaudeBridgeAdapter()


def parse_or_fail_open(adapter: Any, raw_bytes: bytes) -> Any | None:
    try:
        return adapter.parse_inbound_request(raw_bytes)
    except Exception:
        return None


__all__ = ["AdapterConfig", "get_adapter_probe", "parse_or_fail_open", "resolve_adapter_config"]
