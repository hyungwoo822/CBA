"""Tests for Insula — interoceptive monitoring (Craig 2009)."""
import pytest
from brain_agent.regions.insula import Insula
from brain_agent.core.signals import Signal, SignalType, EmotionalTag
from brain_agent.core.neuromodulators import Neuromodulators


@pytest.fixture
def nm():
    return Neuromodulators()

@pytest.fixture
def insula(nm):
    return Insula(neuromodulators=nm)

def test_name_and_position(insula):
    assert insula.name == "insula"

@pytest.mark.asyncio
async def test_computes_interoceptive_state(insula, nm):
    nm.cortisol = 0.8
    nm.epinephrine = 0.7
    signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "test"}, emotional_tag=EmotionalTag(valence=-0.5, arousal=0.8))
    result = await insula.process(signal)
    state = result.metadata["interoceptive_state"]
    assert state["stress_level"] > 0.5
    assert state["energy_level"] < 0.6
    assert state["emotional_awareness"] > 0.5

@pytest.mark.asyncio
async def test_low_arousal_low_awareness(insula, nm):
    nm.cortisol = 0.3
    nm.epinephrine = 0.3
    signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "test"}, emotional_tag=EmotionalTag(valence=0.0, arousal=0.1))
    result = await insula.process(signal)
    state = result.metadata["interoceptive_state"]
    assert state["stress_level"] < 0.4
    assert state["emotional_awareness"] < 0.4

@pytest.mark.asyncio
async def test_risk_assessment(insula, nm):
    nm.cortisol = 0.75
    nm.gaba = 0.3
    signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "test"}, emotional_tag=EmotionalTag(valence=-0.7, arousal=0.9))
    result = await insula.process(signal)
    assert result.metadata["interoceptive_state"]["risk_sensitivity"] > 0.5

@pytest.mark.asyncio
async def test_activation_level(insula, nm):
    signal = Signal(type=SignalType.EXTERNAL_INPUT, source="test", payload={"text": "test"}, emotional_tag=EmotionalTag(valence=0.0, arousal=0.5))
    await insula.process(signal)
    assert insula.activation_level > 0.0
