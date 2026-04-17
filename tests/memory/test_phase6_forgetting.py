"""Tests for Phase 6 ForgettingEngine extensions."""

from brain_agent.memory.forgetting import ForgettingEngine


def test_apply_interference_backwards_compatible():
    """Old call site with 2 args still works."""
    engine = ForgettingEngine()
    reduced = engine.apply_interference(old_strength=1.0, similarity=0.9)
    assert reduced < 1.0


def test_apply_interference_policy_none_passthrough():
    engine = ForgettingEngine()
    result = engine.apply_interference(
        old_strength=1.0,
        similarity=0.95,
        decay_policy="none",
    )
    assert result == 1.0


def test_apply_interference_never_decay_passthrough():
    engine = ForgettingEngine()
    result = engine.apply_interference(
        old_strength=1.0,
        similarity=0.95,
        decay_policy="normal",
        never_decay=True,
    )
    assert result == 1.0


def test_apply_interference_never_decay_wins_over_policy():
    engine = ForgettingEngine()
    result = engine.apply_interference(
        old_strength=1.0,
        similarity=0.9,
        decay_policy="normal",
        never_decay=True,
        importance_score=0.0,
    )
    assert result == 1.0


def test_apply_interference_importance_reduces_decay():
    engine = ForgettingEngine()
    low = engine.apply_interference(
        old_strength=1.0,
        similarity=0.9,
        decay_policy="normal",
        importance_score=0.0,
    )
    high = engine.apply_interference(
        old_strength=1.0,
        similarity=0.9,
        decay_policy="normal",
        importance_score=1.0,
    )
    assert low < high < 1.0


def test_apply_interference_importance_full_double_rate():
    engine = ForgettingEngine()
    base_loss = 1.0 - engine.apply_interference(
        old_strength=1.0,
        similarity=0.9,
        decay_policy="normal",
        importance_score=0.0,
    )
    imp_loss = 1.0 - engine.apply_interference(
        old_strength=1.0,
        similarity=0.9,
        decay_policy="normal",
        importance_score=1.0,
    )
    assert abs(imp_loss - base_loss * 0.5) < 1e-6


def test_apply_interference_below_threshold_still_passthrough():
    engine = ForgettingEngine()
    result = engine.apply_interference(
        old_strength=1.0,
        similarity=0.5,
        decay_policy="normal",
        importance_score=0.0,
    )
    assert result == 1.0
