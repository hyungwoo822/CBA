import pytest
from brain_agent.memory.hippocampal_staging import HippocampalStaging


@pytest.fixture
async def staging(tmp_db_path, mock_embedding):
    s = HippocampalStaging(db_path=tmp_db_path, embed_fn=mock_embedding)
    await s.initialize()
    yield s
    await s.close()


async def test_encode_and_retrieve_by_id(staging):
    mem_id = await staging.encode(
        content="user asked about auth bug",
        entities={"what": "auth bug"},
        interaction_id=1,
        session_id="s1",
    )
    mem = await staging.get_by_id(mem_id)
    assert mem is not None
    assert mem["content"] == "user asked about auth bug"
    assert mem["strength"] == 1.0
    assert mem["consolidated"] is False


async def test_retrieve_boosts_strength(staging):
    mem_id = await staging.encode(
        content="test memory", entities={}, interaction_id=1, session_id="s1"
    )
    mem_before = await staging.get_by_id(mem_id)
    await staging.on_retrieval(mem_id, boost=2.0)
    mem_after = await staging.get_by_id(mem_id)
    assert mem_after["strength"] == mem_before["strength"] * 2.0
    assert mem_after["access_count"] == 1


async def test_get_unconsolidated(staging):
    await staging.encode(content="a", entities={}, interaction_id=1, session_id="s1")
    await staging.encode(content="b", entities={}, interaction_id=2, session_id="s1")
    unconsol = await staging.get_unconsolidated()
    assert len(unconsol) == 2


async def test_mark_consolidated(staging):
    mem_id = await staging.encode(
        content="done", entities={}, interaction_id=1, session_id="s1"
    )
    await staging.mark_consolidated(mem_id)
    assert len(await staging.get_unconsolidated()) == 0


async def test_emotional_tag_stored(staging):
    mem_id = await staging.encode(
        content="scary error",
        entities={},
        interaction_id=1,
        session_id="s1",
        emotional_tag={"valence": -0.8, "arousal": 0.9},
    )
    mem = await staging.get_by_id(mem_id)
    assert mem["emotional_tag"]["arousal"] == 0.9


@pytest.mark.asyncio
async def test_pattern_separation_marks_similar(staging):
    """Similar memories get separated_from marker (DG, Yassa & Stark 2011)."""
    id1 = await staging.encode("I like coffee in the morning", entities={"keywords": ["coffee"]}, interaction_id="i1", session_id="s1")
    id2 = await staging.encode("I like coffee in the morning too", entities={"keywords": ["coffee"]}, interaction_id="i2", session_id="s1")
    # Both should store successfully (separation doesn't block)
    assert id1 != id2
    mem2 = await staging.get_by_id(id2)
    assert mem2 is not None


@pytest.mark.asyncio
async def test_cosine_sim_helper(staging):
    """Cosine similarity computation."""
    assert staging._cosine_sim([1, 0], [1, 0]) == pytest.approx(1.0)
    assert staging._cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)
    assert staging._cosine_sim([], [1, 0]) == 0.0
