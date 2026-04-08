from __future__ import annotations

import pytest
from brain_agent.memory.embedding_cache import EmbeddingCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_vec(seed: int, dim: int = 4) -> list[float]:
    """Tiny deterministic vector for testing."""
    import math
    base = [float(seed + i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in base))
    return [x / norm for x in base]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cache_miss_returns_none():
    """get() on an empty cache must return None."""
    cache = EmbeddingCache(max_size=10)
    assert cache.get("anything") is None


def test_put_and_get():
    """put() followed by get() returns the same vector."""
    cache = EmbeddingCache(max_size=10)
    vec = _dummy_vec(1)
    cache.put("hello", vec)
    result = cache.get("hello")
    assert result == vec


def test_get_or_compute_caches():
    """compute_fn should be called only once for the same text."""
    cache = EmbeddingCache(max_size=10)
    call_count = 0

    def compute(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        return _dummy_vec(42)

    result1 = cache.get_or_compute("same text", compute)
    result2 = cache.get_or_compute("same text", compute)

    assert result1 == result2
    assert call_count == 1


def test_lru_eviction():
    """When max_size is exceeded the oldest (least-recently-used) entry is evicted."""
    cache = EmbeddingCache(max_size=3)
    cache.put("a", _dummy_vec(1))
    cache.put("b", _dummy_vec(2))
    cache.put("c", _dummy_vec(3))

    # Insert a 4th entry — "a" should be evicted (LRU)
    cache.put("d", _dummy_vec(4))

    assert cache.get("a") is None        # evicted
    assert cache.get("b") is not None    # still present
    assert cache.get("c") is not None
    assert cache.get("d") is not None
    assert cache.size == 3


def test_lru_access_refreshes():
    """Accessing an entry with get() should refresh it so it is not evicted next."""
    cache = EmbeddingCache(max_size=3)
    cache.put("a", _dummy_vec(1))
    cache.put("b", _dummy_vec(2))
    cache.put("c", _dummy_vec(3))

    # Touch "a" so it becomes the most-recently-used
    cache.get("a")

    # Insert a 4th entry — "b" should now be LRU and get evicted
    cache.put("d", _dummy_vec(4))

    assert cache.get("b") is None        # evicted
    assert cache.get("a") is not None    # refreshed, still present
    assert cache.get("c") is not None
    assert cache.get("d") is not None


def test_size_property():
    """size property reflects current number of cached entries."""
    cache = EmbeddingCache(max_size=10)
    assert cache.size == 0
    cache.put("x", _dummy_vec(1))
    assert cache.size == 1
    cache.put("y", _dummy_vec(2))
    assert cache.size == 2
    # Updating an existing key must not increase size
    cache.put("x", _dummy_vec(99))
    assert cache.size == 2


def test_clear():
    """clear() empties the cache completely."""
    cache = EmbeddingCache(max_size=10)
    cache.put("a", _dummy_vec(1))
    cache.put("b", _dummy_vec(2))
    cache.clear()
    assert cache.size == 0
    assert cache.get("a") is None
    assert cache.get("b") is None
