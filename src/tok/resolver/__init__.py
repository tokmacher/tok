from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .paths import global_manifest_path as global_manifest_path
    from .paths import resolver_root as resolver_root
    from .store import ContentStore as ContentStore
    from .store import format_resolver_uri as format_resolver_uri
    from .store import parse_resolver_uri as parse_resolver_uri


__all__ = [
    "ContentStore",
    "format_resolver_uri",
    "global_manifest_path",
    "parse_resolver_uri",
    "resolver_root",
]


def __getattr__(name: str):
    if name == "ContentStore":
        from .store import ContentStore

        return ContentStore
    if name == "format_resolver_uri":
        from .store import format_resolver_uri

        return format_resolver_uri
    if name == "parse_resolver_uri":
        from .store import parse_resolver_uri

        return parse_resolver_uri
    if name == "resolver_root":
        from .paths import resolver_root

        return resolver_root
    if name == "global_manifest_path":
        from .paths import global_manifest_path

        return global_manifest_path
    raise AttributeError(name)
