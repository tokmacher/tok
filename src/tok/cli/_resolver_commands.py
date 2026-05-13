from __future__ import annotations

from pathlib import Path

import typer

from tok.resolver.manifest import ResolverManifest
from tok.resolver.paths import global_manifest_path, resolver_root
from tok.resolver.store import ContentStore, parse_resolver_uri

from ._cli_support import console

resolver_app = typer.Typer(help="Local resolver beta commands")


@resolver_app.command("init")
def init_resolver() -> None:
    """Initialize the local resolver store and manifest."""
    root = resolver_root()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = global_manifest_path()
    if manifest_path.exists():
        console.print(f"Resolver manifest already exists: {manifest_path}")
        raise typer.Exit(code=0)
    manifest = ResolverManifest.default()
    manifest.save(manifest_path)
    console.print(f"Initialized resolver at {root}")
    console.print(f"Manifest: {manifest_path}")


@resolver_app.command("status")
def status() -> None:
    """Show resolver root and manifest status."""
    root = resolver_root()
    manifest_path = global_manifest_path()
    console.print(f"Resolver root: {root}")
    if manifest_path.exists():
        console.print("Manifest: present")
    else:
        console.print("Manifest: missing (run `tok resolver init`)")


@resolver_app.command("store")
def store_status() -> None:
    """Show basic store stats."""
    root = resolver_root()
    store = ContentStore(root)
    objects_dir = store.root / "objects"
    count = 0
    if objects_dir.exists():
        count = sum(1 for p in objects_dir.rglob("*") if p.is_file())
    console.print(f"Objects: {count}")


@resolver_app.command("get")
def get(
    uri: str,
    out: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        help="Write bytes to a file instead of printing",
    ),
) -> None:
    """Fetch resolver content by tok-resolver URI."""
    digest = parse_resolver_uri(uri)
    store = ContentStore(resolver_root())
    data = store.get(digest)
    if data is None:
        console.print(f"[red]Missing object for {digest}[/red]")
        raise typer.Exit(code=1)
    if out is None:
        console.print(data.decode("utf-8", errors="replace"))
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    console.print(f"Wrote {len(data)} bytes to {out}")


def register(app: typer.Typer) -> None:
    app.add_typer(resolver_app, name="resolver")
