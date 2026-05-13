"""TDD tests for ResultCache (Plan 2).

Tests the cache interface: lookup, store, invalidate, should_dedup, compute_key.
"""

from __future__ import annotations

from tok.compression._result_cache import CacheEntry, ResultCache


class TestResultCacheLookup:
    def test_miss_returns_none(self):
        cache = ResultCache()
        assert cache.lookup("nonexistent") is None

    def test_hit_returns_entry(self):
        cache = ResultCache()
        entry = CacheEntry(key="k1", content_hash="h1", compressed_content="stub")
        cache.store("k1", entry)
        result = cache.lookup("k1")
        assert result is not None
        assert result.content_hash == "h1"


class TestResultCacheStore:
    def test_store_and_retrieve(self):
        cache = ResultCache()
        entry = CacheEntry(key="k1", content_hash="h1", compressed_content="c1")
        cache.store("k1", entry)
        assert cache.lookup("k1").compressed_content == "c1"

    def test_store_overwrites(self):
        cache = ResultCache()
        cache.store("k1", CacheEntry(key="k1", content_hash="h1", compressed_content="old"))
        cache.store("k1", CacheEntry(key="k1", content_hash="h2", compressed_content="new"))
        assert cache.lookup("k1").compressed_content == "new"


class TestResultCacheInvalidate:
    def test_invalidate_removes_entry(self):
        cache = ResultCache()
        cache.store("k1", CacheEntry(key="k1", content_hash="h1", compressed_content="c1"))
        cache.invalidate("k1")
        assert cache.lookup("k1") is None

    def test_invalidate_nonexistent_is_noop(self):
        cache = ResultCache()
        cache.invalidate("nonexistent")


class TestResultCacheShouldDedup:
    def test_dedup_true_on_hash_match(self):
        cache = ResultCache()
        cache.store("k1", CacheEntry(key="k1", content_hash="h1", compressed_content="c1"))
        assert cache.should_dedup("bash", "h1", "k1") is True

    def test_dedup_false_on_hash_mismatch(self):
        cache = ResultCache()
        cache.store("k1", CacheEntry(key="k1", content_hash="h1", compressed_content="c1"))
        assert cache.should_dedup("bash", "h2", "k1") is False

    def test_dedup_false_on_miss(self):
        cache = ResultCache()
        assert cache.should_dedup("bash", "h1", "k1") is False


class TestResultCacheComputeKey:
    def test_deterministic(self):
        key1 = ResultCache.compute_key("read", {"path": "/foo.py"}, "content")
        key2 = ResultCache.compute_key("read", {"path": "/foo.py"}, "content")
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        key1 = ResultCache.compute_key("read", {"path": "/foo.py"}, "content")
        key2 = ResultCache.compute_key("read", {"path": "/bar.py"}, "content")
        assert key1 != key2


class TestCacheEntry:
    def test_frozen(self):
        entry = CacheEntry(key="k1", content_hash="h1", compressed_content="c1")
        assert entry.key == "k1"
        assert entry.content_hash == "h1"

    def test_defaults(self):
        entry = CacheEntry(key="k1", content_hash="h1", compressed_content="c1")
        assert entry.timestamp > 0
        assert entry.tool_name == ""
        assert entry.turn_stored == 0
