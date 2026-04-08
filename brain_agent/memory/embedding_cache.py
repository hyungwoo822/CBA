"""embedding_cache.py — SHA256-keyed LRU cache for embedding vectors.

Maps to Long-Term Potentiation: previously computed embeddings are reactivated
faster on repeated exposure. LRU eviction mirrors synaptic homeostasis.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Callable


class EmbeddingCache:
    """In-memory LRU cache keyed by SHA256 hash of input text.

    Parameters
    ----------
    max_size:
        Maximum number of embeddings to hold before LRU eviction begins.
        Defaults to 10 000 entries.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._max_size: int = max_size
        self._store: OrderedDict[str, list[float]] = OrderedDict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_hash(text: str) -> str:
        """Return the SHA256 hex-digest of *text*."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str) -> list[float] | None:
        """Look up *text* in the cache.

        Returns the cached embedding if present (moving the entry to the
        most-recently-used position), or ``None`` on a cache miss.
        """
        key = self._content_hash(text)
        if key not in self._store:
            return None
        # Refresh position: move to end == most recently used
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, text: str, embedding: list[float]) -> None:
        """Insert or update the embedding for *text*.

        If the cache is already at *max_size* and the key is new, the
        least-recently-used entry is evicted first.
        """
        key = self._content_hash(text)
        if key in self._store:
            # Update in-place and refresh position
            self._store.move_to_end(key)
            self._store[key] = embedding
        else:
            if len(self._store) >= self._max_size:
                # Evict LRU (first item in OrderedDict)
                self._store.popitem(last=False)
            self._store[key] = embedding

    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], list[float]],
    ) -> list[float]:
        """Cache-through pattern.

        Returns the cached embedding when available, otherwise calls
        *compute_fn(text)*, stores the result, and returns it.
        """
        cached = self.get(text)
        if cached is not None:
            return cached
        embedding = compute_fn(text)
        self.put(text, embedding)
        return embedding

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of entries held in the cache."""
        return len(self._store)
