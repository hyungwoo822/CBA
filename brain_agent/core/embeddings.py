from __future__ import annotations
import numpy as np

from brain_agent.memory.embedding_cache import EmbeddingCache

EMBEDDING_DIM = 384
PATTERN_SEPARATION_SIGMA = 0.01


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_mock: bool = False):
        self._use_mock = use_mock
        self._model = None
        self._model_name = model_name
        self._cache = EmbeddingCache(max_size=10_000)

    def _get_model(self):
        if self._model is None and not self._use_mock:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _compute_embed(self, text: str) -> list[float]:
        model = self._get_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed(self, text: str) -> list[float]:
        if self._use_mock:
            return self._cache.get_or_compute(text, self._mock_embed)
        return self._cache.get_or_compute(text, self._compute_embed)

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = np.dot(va, vb)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def pattern_separate(self, embedding: list[float]) -> list[float]:
        vec = np.array(embedding, dtype=np.float32)
        noise = np.random.normal(0, PATTERN_SEPARATION_SIGMA, size=vec.shape)
        separated = vec + noise.astype(np.float32)
        separated = separated / np.linalg.norm(separated)
        return separated.tolist()

    @staticmethod
    def _mock_embed(text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
