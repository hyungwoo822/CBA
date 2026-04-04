# tests/core/test_embeddings.py
import numpy as np
from brain_agent.core.embeddings import EmbeddingService

def test_embed_returns_correct_dimension():
    svc = EmbeddingService(use_mock=True)
    vec = svc.embed("hello world")
    assert len(vec) == 384

def test_embed_is_normalized():
    svc = EmbeddingService(use_mock=True)
    vec = np.array(svc.embed("test"))
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 0.01

def test_same_input_same_output():
    svc = EmbeddingService(use_mock=True)
    v1 = svc.embed("hello")
    v2 = svc.embed("hello")
    assert v1 == v2

def test_different_input_different_output():
    svc = EmbeddingService(use_mock=True)
    v1 = svc.embed("hello")
    v2 = svc.embed("world")
    assert v1 != v2

def test_cosine_similarity():
    svc = EmbeddingService(use_mock=True)
    sim = svc.cosine_similarity(svc.embed("test"), svc.embed("test"))
    assert abs(sim - 1.0) < 0.01

def test_pattern_separation_adds_noise():
    svc = EmbeddingService(use_mock=True)
    base = svc.embed("same input")
    separated = svc.pattern_separate(base)
    sim = svc.cosine_similarity(base, separated)
    assert 0.95 < sim < 1.0
