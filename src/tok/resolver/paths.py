from __future__ import annotations

import os
from pathlib import Path


def resolver_root() -> Path:
    root = os.getenv("TOK_RESOLVER_ROOT", "").strip()
    if root:
        return Path(root).expanduser()
    return Path.home() / ".tok" / "resolver"


def global_manifest_path() -> Path:
    return resolver_root() / "manifest.tok"
