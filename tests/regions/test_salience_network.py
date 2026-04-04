import pytest
from brain_agent.regions.salience_network import SalienceNetworkRegion
from brain_agent.core.signals import Signal, SignalType, EmotionalTag
from brain_agent.core.network_modes import NetworkMode, TripleNetworkController


@pytest.fixture
def sn():
    return SalienceNetworkRegion(network_ctrl=TripleNetworkController())


async def test_high_salience_switches_to_ecn(sn):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus", payload={"text": "urgent"},
                 emotional_tag=EmotionalTag(valence=-0.5, arousal=0.8))
    await sn.process(sig)
    assert sn._network_ctrl.current_mode == NetworkMode.ECN


async def test_low_salience_stays_dmn(sn):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus", payload={"text": "ok"},
                 emotional_tag=EmotionalTag(valence=0, arousal=0.05))
    await sn.process(sig)
    assert sn._network_ctrl.current_mode == NetworkMode.DMN


async def test_task_complete_switches_to_dmn(sn):
    sn._network_ctrl.switch_to(NetworkMode.ECN)
    sig = Signal(type=SignalType.GWT_BROADCAST, source="workspace", payload={"status": "task_complete"})
    await sn.process(sig)
    assert sn._network_ctrl.current_mode == NetworkMode.DMN


# ── Creative Mode Trigger tests (Beaty 2018) ──────────────────────


async def test_creative_trigger_all_conditions_met(sn):
    """Creative mode when: high ACC errors + no procedure + high arousal."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "novel problem"},
        emotional_tag=EmotionalTag(valence=0.2, arousal=0.6),
        metadata={"acc_error_ratio": 0.7, "cached_procedure": None},
    )
    result = await sn.process(sig)
    assert sn._network_ctrl.current_mode == NetworkMode.CREATIVE
    assert result is not None
    assert result.payload["to"] == "creative"


async def test_creative_not_triggered_low_error_ratio(sn):
    """No creative mode when ACC error ratio is low."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "easy problem"},
        emotional_tag=EmotionalTag(valence=0.2, arousal=0.6),
        metadata={"acc_error_ratio": 0.2, "cached_procedure": None},
    )
    await sn.process(sig)
    assert sn._network_ctrl.current_mode != NetworkMode.CREATIVE


async def test_creative_not_triggered_with_cached_procedure(sn):
    """No creative mode when a cached procedure exists (not novel)."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "known problem"},
        emotional_tag=EmotionalTag(valence=0.2, arousal=0.6),
        metadata={
            "acc_error_ratio": 0.7,
            "cached_procedure": {"action_sequence": ["do_thing"], "stage": "autonomous"},
        },
    )
    await sn.process(sig)
    assert sn._network_ctrl.current_mode != NetworkMode.CREATIVE


async def test_creative_not_triggered_low_arousal(sn):
    """No creative mode when arousal is too low."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "boring novel problem"},
        emotional_tag=EmotionalTag(valence=0.0, arousal=0.1),
        metadata={"acc_error_ratio": 0.7, "cached_procedure": None},
    )
    await sn.process(sig)
    assert sn._network_ctrl.current_mode != NetworkMode.CREATIVE


async def test_creative_not_triggered_no_emotional_tag(sn):
    """No creative mode when no emotional tag (arousal defaults to 0)."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "novel problem"},
        metadata={"acc_error_ratio": 0.7, "cached_procedure": None},
    )
    await sn.process(sig)
    assert sn._network_ctrl.current_mode != NetworkMode.CREATIVE


async def test_creative_takes_priority_over_ecn_switch(sn):
    """Creative mode should be checked before normal DMN->ECN switch."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "important novel problem"},
        emotional_tag=EmotionalTag(valence=0.3, arousal=0.8),
        metadata={"acc_error_ratio": 0.8, "cached_procedure": None},
    )
    result = await sn.process(sig)
    # Should enter CREATIVE, not ECN, even though salience > threshold
    assert sn._network_ctrl.current_mode == NetworkMode.CREATIVE
    assert result.payload["to"] == "creative"


# ── Memory-Based Novelty Assessment tests (Sridharan 2008) ────────


async def test_novelty_high_when_no_memories(sn):
    """No retrieved memories should yield high novelty (0.8)."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "new topic"},
        emotional_tag=EmotionalTag(valence=0.0, arousal=0.0),
    )
    salience = sn._compute_salience(sig)
    # arousal=0 * 0.6 + novelty=0.8 * 0.4 = 0.32
    assert salience == pytest.approx(0.32)


async def test_novelty_low_when_high_retrieval_score(sn):
    """High retrieval score should yield low novelty."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "known topic"},
        emotional_tag=EmotionalTag(valence=0.0, arousal=0.0),
        metadata={"retrieved_memories": [{"score": 0.95, "content": "known fact"}]},
    )
    salience = sn._compute_salience(sig)
    # novelty = 1.0 - 0.95 = 0.05
    # salience = 0.0 * 0.6 + 0.05 * 0.4 = 0.02
    assert salience == pytest.approx(0.02)


async def test_novelty_medium_when_moderate_retrieval_score(sn):
    """Moderate retrieval score should yield moderate novelty."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "somewhat familiar"},
        emotional_tag=EmotionalTag(valence=0.0, arousal=0.0),
        metadata={"retrieved_memories": [
            {"score": 0.5, "content": "partial match"},
            {"score": 0.3, "content": "weak match"},
        ]},
    )
    salience = sn._compute_salience(sig)
    # best_score = 0.5, novelty = 1.0 - 0.5 = 0.5
    # salience = 0.0 * 0.6 + 0.5 * 0.4 = 0.2
    assert salience == pytest.approx(0.2)


async def test_novelty_uses_best_score_from_multiple_memories(sn):
    """Should use the best (highest) retrieval score for novelty."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "topic"},
        emotional_tag=EmotionalTag(valence=0.0, arousal=0.0),
        metadata={"retrieved_memories": [
            {"score": 0.2, "content": "weak"},
            {"score": 0.8, "content": "strong"},
            {"score": 0.4, "content": "medium"},
        ]},
    )
    salience = sn._compute_salience(sig)
    # best_score = 0.8, novelty = 0.2
    # salience = 0.0 * 0.6 + 0.2 * 0.4 = 0.08
    assert salience == pytest.approx(0.08)


async def test_novelty_non_external_input_without_memories(sn):
    """Non-external-input signals without memories should default to low novelty (0.1)."""
    sig = Signal(
        type=SignalType.GWT_BROADCAST,
        source="workspace",
        payload={"status": "update"},
    )
    salience = sn._compute_salience(sig)
    # arousal=0, novelty=0.1 (non-external-input, no memories)
    # salience = 0.0 * 0.6 + 0.1 * 0.4 = 0.04
    assert salience == pytest.approx(0.04)


async def test_novelty_clamped_at_zero(sn):
    """Retrieval score of 1.0 should yield novelty of 0.0."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "exact match"},
        emotional_tag=EmotionalTag(valence=0.0, arousal=0.0),
        metadata={"retrieved_memories": [{"score": 1.0, "content": "exact"}]},
    )
    salience = sn._compute_salience(sig)
    assert salience == pytest.approx(0.0)
