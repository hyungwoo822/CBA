import pytest
from brain_agent.regions.thalamus import Thalamus
from brain_agent.core.signals import EmotionalTag, Signal, SignalType


@pytest.fixture
def thalamus():
    return Thalamus()


async def test_preprocesses_external_input(thalamus):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="user",
                 payload={"text": "  hello world  "})
    result = await thalamus.process(sig)
    assert result is not None
    assert result.payload["text"] == "hello world"


async def test_adds_default_modality(thalamus):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="user",
                 payload={"text": "test"})
    result = await thalamus.process(sig)
    assert result.payload["modality"] == "text"


async def test_preserves_explicit_modality(thalamus):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="user",
                 payload={"text": "test", "modality": "image"})
    result = await thalamus.process(sig)
    assert result.payload["modality"] == "image"


async def test_emits_activation_on_input(thalamus):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="user",
                 payload={"text": "test"})
    await thalamus.process(sig)
    assert thalamus.activation_level == pytest.approx(0.6)


async def test_passes_through_non_input(thalamus):
    sig = Signal(type=SignalType.PLAN, source="pfc",
                 payload={"goal": "x"})
    result = await thalamus.process(sig)
    assert result is sig  # pass-through unchanged


async def test_name_and_position(thalamus):
    assert thalamus.name == "thalamus"
    assert thalamus.position.x == 0
    assert thalamus.position.y == 0
    assert thalamus.position.z == 0


async def test_attention_gate_high_relevance(thalamus):
    signal = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="test",
        payload={"text": "explain quantum computing"},
    )
    signal.metadata["goal_keywords"] = ["quantum", "computing"]
    result = await thalamus.process_with_attention(
        signal, goal_embedding=None, current_arousal=0.5
    )
    assert result is not None
    assert result.metadata.get("attention_weight", 0) > 0.3


async def test_attention_gate_bottom_up_salience(thalamus):
    signal = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="test",
        payload={"text": "fire!!!"},
    )
    signal.emotional_tag = EmotionalTag(valence=-0.8, arousal=0.95)
    result = await thalamus.process_with_attention(
        signal, goal_embedding=None, current_arousal=0.95
    )
    assert result is not None
    assert result.metadata.get("attention_weight", 0) > 0.5


async def test_attention_gate_low_relevance_passes(thalamus):
    signal = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="test",
        payload={"text": "hmm"},
    )
    result = await thalamus.process_with_attention(
        signal, goal_embedding=None, current_arousal=0.1
    )
    assert result is not None
    assert "attention_weight" in result.metadata
