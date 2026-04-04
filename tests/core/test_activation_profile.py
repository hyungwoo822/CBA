"""Tests for content-driven activation profiling."""
import pytest
from brain_agent.core.activation_profile import compute_activation_profile, should_full_process


def test_emotional_input_boosts_amygdala():
    profile = compute_activation_profile(
        comprehension={"intent": "sharing", "complexity": "simple"},
        emotional_tag={"valence": -0.8, "arousal": 0.9},
    )
    assert profile.get("amygdala", 0) > 0.7
    assert profile.get("insula", 0) > 0.7


def test_emotional_input_suppresses_analytical():
    profile = compute_activation_profile(
        comprehension={"intent": "venting", "complexity": "simple"},
        emotional_tag={"valence": -0.8, "arousal": 0.9},
    )
    # Analytical regions should not be high for emotional venting
    assert profile.get("acc", 0) < profile.get("amygdala", 0)


def test_analytical_input_boosts_pfc():
    profile = compute_activation_profile(
        comprehension={"intent": "question", "complexity": "complex"},
        emotional_tag={"valence": 0.0, "arousal": 0.2},
    )
    assert profile.get("prefrontal_cortex", 0) > 0.7
    assert profile.get("acc", 0) > 0.6


def test_simple_greeting_suppresses_most():
    profile = compute_activation_profile(
        comprehension={"intent": "greeting", "complexity": "simple"},
        emotional_tag={"valence": 0.1, "arousal": 0.1},
    )
    assert profile.get("prefrontal_cortex", 0) < 0.5
    assert profile.get("amygdala", 0) < 0.5


def test_procedural_hit_suppresses_analytical():
    profile = compute_activation_profile(
        comprehension={"intent": "greeting", "complexity": "simple"},
        has_procedure=True,
    )
    assert profile.get("acc", 0) <= 0.3


def test_should_full_process_threshold():
    profile = {"amygdala": 0.8, "acc": 0.2}
    assert should_full_process("amygdala", profile) is True
    assert should_full_process("acc", profile) is False


def test_gains_clamped():
    profile = compute_activation_profile(
        comprehension={"intent": "question", "complexity": "very_complex"},
        emotional_tag={"valence": -1.0, "arousal": 1.0},
    )
    for gain in profile.values():
        assert 0.1 <= gain <= 1.0


def test_social_intent_boosts_tpj():
    profile = compute_activation_profile(
        comprehension={"intent": "social", "complexity": "moderate"},
    )
    assert profile.get("tpj", 0) >= 0.7
