# tests/core/test_signals.py
from brain_agent.core.signals import Signal, EmotionalTag, SignalType
import uuid

def test_emotional_tag_creation():
    tag = EmotionalTag(valence=0.5, arousal=0.8)
    assert tag.valence == 0.5
    assert tag.arousal == 0.8

def test_emotional_tag_clamps_values():
    tag = EmotionalTag(valence=-1.5, arousal=2.0)
    assert tag.valence == -1.0
    assert tag.arousal == 1.0

def test_emotional_tag_neutral():
    tag = EmotionalTag.neutral()
    assert tag.valence == 0.0
    assert tag.arousal == 0.0

def test_signal_creation():
    sig = Signal(type=SignalType.PLAN, source="pfc", payload={"content": "analyze auth module"})
    assert sig.type == SignalType.PLAN
    assert sig.source == "pfc"
    assert sig.priority == 0.5
    assert sig.emotional_tag is None
    assert isinstance(sig.id, uuid.UUID)

def test_signal_with_emotional_tag():
    tag = EmotionalTag(valence=-0.8, arousal=0.9)
    sig = Signal(type=SignalType.ACTION_RESULT, source="cerebellum", payload={"error": 0.02}, emotional_tag=tag)
    assert sig.emotional_tag.arousal == 0.9

def test_signal_types_exist():
    assert SignalType.PLAN
    assert SignalType.ACTION_SELECTED
    assert SignalType.ACTION_RESULT
    assert SignalType.CONFLICT_DETECTED
    assert SignalType.STRATEGY_SWITCH
    assert SignalType.PREDICTION_ERROR
    assert SignalType.EMOTIONAL_TAG
    assert SignalType.GWT_BROADCAST
    assert SignalType.CONSOLIDATION_TRIGGER
    assert SignalType.NETWORK_SWITCH
    assert SignalType.RESOURCE_STATUS
    assert SignalType.EXTERNAL_INPUT
    assert SignalType.ENCODE
    assert SignalType.RETRIEVE
