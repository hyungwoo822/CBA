"""Tests for Sylvian Parietal-Temporal (Spt) — auditory-motor interface."""
import pytest
from brain_agent.regions.spt import SylvianParietalTemporal
from brain_agent.regions.base import Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType, EmotionalTag


@pytest.fixture
def spt():
    return SylvianParietalTemporal()


def test_anatomy(spt):
    assert spt.name == "spt"
    assert spt.lobe == Lobe.PARIETAL
    assert spt.hemisphere == Hemisphere.LEFT


async def test_generates_production_plan(spt):
    """Spt should generate a production plan from comprehension data."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="wernicke",
        payload={
            "text": "How do I deploy to production?",
            "comprehension": {"intent": "question", "complexity": "simple", "keywords": ["deploy", "production"]},
        },
    )
    result = await spt.process(sig)
    plan = result.metadata["production_plan"]
    assert "register" in plan
    assert "structure" in plan
    assert "tone" in plan
    assert "language" in plan
    assert plan["response_intent"] == "answer"


async def test_command_gets_technical_register(spt):
    """Commands should map to technical register."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="wernicke",
        payload={
            "text": "run the build script",
            "comprehension": {"intent": "command", "keywords": ["run", "build", "script"]},
        },
    )
    result = await spt.process(sig)
    plan = result.metadata["production_plan"]
    assert plan["register"] == "technical"
    assert plan["response_intent"] == "confirmation_and_result"


async def test_emotional_tone_modulation(spt):
    """High arousal negative should produce urgent tone."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="wernicke",
        payload={
            "text": "Help, the server is down!",
            "comprehension": {"intent": "emotional_expression", "keywords": ["help", "server", "down"]},
        },
    )
    sig.emotional_tag = EmotionalTag(valence=-0.6, arousal=0.8)
    result = await spt.process(sig)
    plan = result.metadata["production_plan"]
    assert plan["tone"] == "urgent"


async def test_greeting_gets_informal_register(spt):
    """Greetings should get informal register."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="wernicke",
        payload={
            "text": "Hi there!",
            "comprehension": {"intent": "greeting", "keywords": ["hi"]},
        },
    )
    result = await spt.process(sig)
    plan = result.metadata["production_plan"]
    assert plan["register"] == "informal"
    assert plan["response_intent"] == "reciprocate"
