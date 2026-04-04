# tests/core/test_temporal.py
from brain_agent.core.temporal import TemporalModel

def test_temporal_model_initial_state():
    tm = TemporalModel()
    assert tm.interaction_count == 0
    assert tm.current_session_id == ""

def test_increment_interaction():
    tm = TemporalModel()
    tm.tick()
    assert tm.interaction_count == 1
    tm.tick()
    assert tm.interaction_count == 2

def test_distance_zero_for_current():
    tm = TemporalModel()
    tm.tick()
    d = tm.distance(last_interaction=1, last_session=tm.current_session_id)
    assert d == 0.0

def test_distance_increases_with_interaction_gap():
    tm = TemporalModel()
    for _ in range(10):
        tm.tick()
    d = tm.distance(last_interaction=1, last_session=tm.current_session_id)
    assert d > 0.0

def test_distance_increases_with_session_gap():
    tm = TemporalModel()
    tm.start_session("s1")
    tm.tick()
    tm.close_session()
    tm.start_session("s2")
    tm.tick()
    tm.close_session()
    tm.start_session("s3")
    tm.tick()
    d = tm.distance(last_interaction=1, last_session="s1")
    d_same = tm.distance(last_interaction=tm.interaction_count, last_session="s3")
    assert d > d_same

def test_closed_sessions_tracked():
    tm = TemporalModel()
    tm.start_session("s1")
    tm.close_session()
    tm.start_session("s2")
    tm.close_session()
    assert tm.count_sessions_since("s1") == 2
    assert tm.count_sessions_since("s2") == 1
