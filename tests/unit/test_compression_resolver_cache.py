"""Section 5.3.3: Resolver as Compression Cache — RED/GREEN tests.

The resolver is a local content-addressed store. When exact evidence is observed,
content should be stored in the resolver so it survives session restarts.

Target behavior:
- record_exact() with register_exact_content=True writes content to resolver.
- The resolver can retrieve content by digest.
- A file path manifest tracks path -> (digest, mtime) for cache invalidation.
- Modified files (mtime change) must be re-read, not served from cache.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def test_runtime_record_exact_evidence_writes_resolver_cache(tmp_path: Path) -> None:
    from tok.runtime.core import RuntimeSession

    session = RuntimeSession(memory_dir=tmp_path)
    content = b"def answer():\n    return 42\n"
    signals = session.record_exact_evidence("file:src/example.py", digest="", content=content)

    assert signals["evidence_resolver_cache_stored"] == 1
    assert session.evidence_safety.ledger["src/example.py"].latest_digest.startswith("sha256:")
    assert session.resolver_cache is not None
    assert session.resolver_cache.get(path="src/example.py") == content


# ---------------------------------------------------------------------------
# ResolverCache: new module for resolver-backed compression cache
# ---------------------------------------------------------------------------

class TestResolverCacheModuleExists:
    def test_resolver_cache_module_importable(self) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        assert ResolverCache is not None

    def test_resolver_cache_has_put_method(self) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache.__new__(ResolverCache)
        assert hasattr(rc, "put")

    def test_resolver_cache_has_get_method(self) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache.__new__(ResolverCache)
        assert hasattr(rc, "get")

    def test_resolver_cache_has_invalidate_method(self) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache.__new__(ResolverCache)
        assert hasattr(rc, "invalidate")


# ---------------------------------------------------------------------------
# Store and retrieve content
# ---------------------------------------------------------------------------

class TestResolverCacheStoreRetrieve:
    def test_put_returns_digest(self, tmp_path: Path) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        content = b"def foo():\n    return 42\n"
        digest = rc.put(path="/repo/foo.py", content=content)
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert digest == expected

    def test_get_returns_stored_content(self, tmp_path: Path) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        content = b"file content here"
        rc.put(path="/repo/bar.py", content=content)
        retrieved = rc.get(path="/repo/bar.py")
        assert retrieved == content

    def test_get_missing_path_returns_none(self, tmp_path: Path) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        result = rc.get(path="/repo/never_stored.py")
        assert result is None

    def test_put_overwrites_same_path(self, tmp_path: Path) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        content1 = b"version 1"
        content2 = b"version 2"
        rc.put(path="/repo/foo.py", content=content1)
        rc.put(path="/repo/foo.py", content=content2)
        result = rc.get(path="/repo/foo.py")
        assert result == content2


# ---------------------------------------------------------------------------
# Mtime-based invalidation
# ---------------------------------------------------------------------------

class TestResolverCacheMtimeInvalidation:
    def test_cache_miss_when_mtime_changed(self, tmp_path: Path) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        content = b"original content"
        mtime_old = 1000.0
        mtime_new = 2000.0
        rc.put(path="/repo/foo.py", content=content, mtime=mtime_old)
        # Retrieve with different mtime -> cache miss
        result = rc.get(path="/repo/foo.py", mtime=mtime_new)
        assert result is None

    def test_cache_hit_when_mtime_unchanged(self, tmp_path: Path) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        content = b"stable content"
        mtime = 1234.5
        rc.put(path="/repo/foo.py", content=content, mtime=mtime)
        result = rc.get(path="/repo/foo.py", mtime=mtime)
        assert result == content

    def test_get_without_mtime_returns_content(self, tmp_path: Path) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        content = b"some content"
        rc.put(path="/repo/foo.py", content=content)
        # No mtime check — always returns if path is known
        result = rc.get(path="/repo/foo.py")
        assert result == content

    def test_invalidate_removes_entry(self, tmp_path: Path) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        content = b"cached content"
        rc.put(path="/repo/foo.py", content=content)
        rc.invalidate(path="/repo/foo.py")
        result = rc.get(path="/repo/foo.py")
        assert result is None


# ---------------------------------------------------------------------------
# EvidenceSafetyState.record_exact integration
# ---------------------------------------------------------------------------

class TestEvidenceSafetyResolverIntegration:
    def test_record_exact_without_resolver_still_works(self) -> None:
        from tok.runtime.evidence_safety import EvidenceSafetyState

        state = EvidenceSafetyState()
        signals = state.record_exact(
            "src/foo.py",
            digest="sha256:abc",
            turn=1,
        )
        assert signals.get("evidence_exact_observed") == 1

    def test_record_exact_with_resolver_stores_content(self, tmp_path: Path) -> None:
        from tok.runtime.evidence_safety import EvidenceSafetyState
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        state = EvidenceSafetyState()
        content = b"def foo():\n    return 42\n" * 10
        state.record_exact(
            "src/foo.py",
            digest="sha256:" + hashlib.sha256(content).hexdigest(),
            turn=1,
            content=content,
            resolver_cache=rc,
        )
        retrieved = rc.get(path="src/foo.py")
        assert retrieved == content

    def test_record_exact_without_content_does_not_write_resolver(self, tmp_path: Path) -> None:
        from tok.runtime.evidence_safety import EvidenceSafetyState
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        state = EvidenceSafetyState()
        state.record_exact(
            "src/bar.py",
            digest="sha256:abc123",
            turn=1,
            resolver_cache=rc,
        )
        result = rc.get(path="src/bar.py")
        assert result is None  # No content provided, nothing stored


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestResolverCacheEdgeCases:
    def test_empty_content_can_be_stored(self, tmp_path: Path) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        rc.put(path="/repo/empty.py", content=b"")
        result = rc.get(path="/repo/empty.py")
        assert result == b""

    def test_large_content_stored_correctly(self, tmp_path: Path) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        content = b"x" * (100 * 1024)  # 100 KB
        rc.put(path="/repo/large.py", content=content)
        result = rc.get(path="/repo/large.py")
        assert result == content

    def test_different_paths_same_content_stored_independently(self, tmp_path: Path) -> None:
        from tok.utils.resolver_cache import ResolverCache  # type: ignore[import]

        rc = ResolverCache(tmp_path)
        content = b"same content"
        rc.put(path="/repo/a.py", content=content)
        rc.put(path="/repo/b.py", content=content)
        assert rc.get(path="/repo/a.py") == content
        assert rc.get(path="/repo/b.py") == content
        rc.invalidate(path="/repo/a.py")
        assert rc.get(path="/repo/a.py") is None
        assert rc.get(path="/repo/b.py") == content
