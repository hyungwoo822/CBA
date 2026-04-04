import pytest
from brain_agent.memory.episodic_store import EpisodicStore


@pytest.fixture
async def store(tmp_db_path):
    s = EpisodicStore(db_path=tmp_db_path)
    await s.initialize()
    yield s
    await s.close()


async def test_save_and_get(store):
    ep_id = await store.save(
        content="found auth bug in line 42",
        context_embedding=[0.1] * 384,
        entities={"what": "auth bug", "where": "line 42"},
        emotional_tag={"valence": -0.3, "arousal": 0.6},
        interaction_id=5,
        session_id="s1",
    )
    ep = await store.get_by_id(ep_id)
    assert ep is not None
    assert "auth bug" in ep["content"]


async def test_search_by_interaction_range(store):
    for i in range(5):
        await store.save(
            content=f"event {i}",
            context_embedding=[0.1] * 384,
            entities={},
            emotional_tag={"valence": 0, "arousal": 0},
            interaction_id=i + 1,
            session_id="s1",
        )
    results = await store.get_by_interaction_range(2, 4)
    assert len(results) == 3


async def test_get_recent(store):
    for i in range(10):
        await store.save(
            content=f"event {i}",
            context_embedding=[0.1] * 384,
            entities={},
            emotional_tag={"valence": 0, "arousal": 0},
            interaction_id=i,
            session_id="s1",
        )
    recent = await store.get_recent(limit=3)
    assert len(recent) == 3
    assert recent[0]["content"] == "event 9"
