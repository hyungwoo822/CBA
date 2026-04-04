import pytest
from brain_agent.core.workspace import GlobalWorkspace
from brain_agent.core.signals import Signal, SignalType, EmotionalTag

def test_submit_and_compete():
    gw = GlobalWorkspace()
    gw.submit(Signal(type=SignalType.PLAN, source="pfc", payload={"x": 1}), salience=0.8, goal_relevance=0.7)
    gw.submit(Signal(type=SignalType.EMOTIONAL_TAG, source="amygdala", payload={"x": 2}), salience=0.3, goal_relevance=0.2)
    winner = gw.compete()
    assert winner is not None
    assert winner.source == "pfc"

def test_no_winner_below_threshold():
    gw = GlobalWorkspace(ignition_threshold=0.9)
    gw.submit(Signal(type=SignalType.PLAN, source="pfc", payload={}), salience=0.1, goal_relevance=0.1)
    assert gw.compete() is None

def test_competition_clears_queue():
    gw = GlobalWorkspace()
    gw.submit(Signal(type=SignalType.PLAN, source="pfc", payload={}), salience=0.8, goal_relevance=0.8)
    gw.compete()
    assert gw.compete() is None

def test_high_arousal_boosts_score():
    gw = GlobalWorkspace()
    sig_calm = Signal(type=SignalType.PLAN, source="pfc", payload={}, emotional_tag=EmotionalTag(valence=0, arousal=0.1))
    sig_excited = Signal(type=SignalType.PLAN, source="pfc", payload={}, emotional_tag=EmotionalTag(valence=0, arousal=0.9))
    s1 = gw._compute_score(sig_calm, salience=0.5, goal_relevance=0.5)
    s2 = gw._compute_score(sig_excited, salience=0.5, goal_relevance=0.5)
    assert s2 > s1
