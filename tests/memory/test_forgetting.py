import math

from brain_agent.memory.forgetting import ForgettingEngine


def test_retention_full_at_zero_distance():
    engine = ForgettingEngine()
    assert engine.retention(distance=0.0, strength=1.0) == 1.0


def test_retention_decreases_with_distance():
    engine = ForgettingEngine()
    r1 = engine.retention(distance=1.0, strength=1.0)
    r5 = engine.retention(distance=5.0, strength=1.0)
    r10 = engine.retention(distance=10.0, strength=1.0)
    assert r1 > r5 > r10


def test_higher_strength_slower_decay():
    engine = ForgettingEngine()
    assert engine.retention(distance=5.0, strength=5.0) > engine.retention(
        distance=5.0, strength=1.0
    )


def test_interference_reduces_strength():
    engine = ForgettingEngine()
    assert engine.apply_interference(old_strength=1.0, similarity=0.9) < 1.0


def test_no_interference_below_threshold():
    engine = ForgettingEngine()
    assert engine.apply_interference(old_strength=1.0, similarity=0.5) == 1.0


def test_retrieval_induced_forgetting():
    engine = ForgettingEngine()
    suppressed = engine.retrieval_induced_forgetting(1.0)
    assert 0.80 <= suppressed <= 0.90


def test_homeostatic_scaling():
    engine = ForgettingEngine()
    scaled = engine.homeostatic_scale(
        [1.0, 0.5, 0.1, 0.05], factor=0.95, threshold=0.08
    )
    assert len(scaled) == 3  # 0.05*0.95=0.0475 < 0.08 pruned
