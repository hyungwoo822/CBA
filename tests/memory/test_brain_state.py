"""Tests for BrainStateStore — neuromodulator and activation persistence."""
import pytest
from brain_agent.memory.brain_state import BrainStateStore
from brain_agent.core.neuromodulators import Neuromodulators


@pytest.fixture
async def store(tmp_path):
    s = BrainStateStore(db_path=str(tmp_path / "brain_state.db"))
    await s.initialize()
    yield s
    await s.close()


async def test_default_neuromodulators(store):
    """Fresh DB should return baseline 0.5 for all NTs."""
    state = await store.load_neuromodulators()
    assert state["dopamine"] == 0.5
    assert state["serotonin"] == 0.5
    assert state["cortisol"] == 0.5


async def test_save_and_load_neuromodulators(store):
    """Saved NT state should persist across load calls."""
    await store.save_neuromodulators({
        "dopamine": 0.8,
        "norepinephrine": 0.3,
        "serotonin": 0.6,
        "acetylcholine": 0.9,
        "cortisol": 0.7,
        "epinephrine": 0.2,
    })
    state = await store.load_neuromodulators()
    assert state["dopamine"] == 0.8
    assert state["norepinephrine"] == 0.3
    assert state["acetylcholine"] == 0.9
    assert state["cortisol"] == 0.7


async def test_save_and_load_region_activations(store):
    """Region activations should persist."""
    activations = {
        "prefrontal_cortex": 0.9,
        "amygdala": 0.4,
        "wernicke_area": 0.7,
    }
    await store.save_region_activations(activations)
    loaded = await store.load_region_activations()
    assert loaded["prefrontal_cortex"] == 0.9
    assert loaded["amygdala"] == 0.4
    assert loaded["wernicke_area"] == 0.7


async def test_neuromodulators_load_from(store):
    """Neuromodulators.load_from should restore state from DB data."""
    await store.save_neuromodulators({
        "dopamine": 0.7,
        "norepinephrine": 0.3,
        "serotonin": 0.8,
        "acetylcholine": 0.2,
        "cortisol": 0.6,
        "epinephrine": 0.1,
    })
    nm = Neuromodulators()
    state = await store.load_neuromodulators()
    nm.load_from(state)
    assert nm.dopamine == 0.7
    assert nm.norepinephrine == 0.3
    assert nm.serotonin == 0.8
    assert nm.acetylcholine == 0.2
    assert nm.cortisol == 0.6
    assert nm.epinephrine == 0.1


async def test_interaction_count_persistence(store):
    """Interaction count and session ID should persist."""
    await store.update_interaction_count(42, "session-abc")
    count, session = await store.load_interaction_count()
    assert count == 42
    assert session == "session-abc"


async def test_gaba_persists(store):
    """GABA should survive save + load roundtrip."""
    await store.save_neuromodulators({
        "dopamine": 0.5,
        "norepinephrine": 0.5,
        "serotonin": 0.5,
        "acetylcholine": 0.5,
        "cortisol": 0.5,
        "epinephrine": 0.5,
        "gaba": 0.72,
    })
    state = await store.load_neuromodulators()
    assert state["gaba"] == 0.72


async def test_gaba_defaults_if_missing(store):
    """Fresh DB should return gaba=0.5 by default."""
    state = await store.load_neuromodulators()
    assert state["gaba"] == 0.5


async def test_gaba_in_history(store):
    """GABA should appear in neuromodulator history."""
    await store.save_neuromodulators({
        "dopamine": 0.5,
        "norepinephrine": 0.5,
        "serotonin": 0.5,
        "acetylcholine": 0.5,
        "cortisol": 0.5,
        "epinephrine": 0.5,
        "gaba": 0.65,
    })
    history = await store.get_neuromodulator_history(limit=1)
    assert len(history) == 1
    assert history[0]["gaba"] == 0.65
