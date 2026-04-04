import pytest
from brain_agent.memory.manager import MemoryManager
from brain_agent.core.session import SessionManager


@pytest.fixture
async def system(tmp_path, mock_embedding):
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    sm = SessionManager(
        db_path=str(tmp_path / "sessions.db"), embed_fn=mock_embedding
    )
    await mm.initialize()
    await sm.initialize()
    yield mm, sm
    await mm.close()
    await sm.close()


async def test_full_request_lifecycle(system):
    mm, sm = system

    # Start a session
    session = await sm.start_session()
    interaction = await sm.on_interaction("find auth bug")
    mm.set_context(interaction, session.id)

    # Sensory registration and attention
    mm.sensory.new_cycle()
    mm.sensory.register({"text": "find auth bug"}, modality="text")
    attended = mm.sensory.attend(lambda x: True)
    assert len(attended) == 1

    # Load into working memory
    mm.working.load(mm._to_wm_item(attended[0]))
    assert len(mm.working.get_slots()) == 1

    # Encode to hippocampal staging
    mem_id = await mm.encode(
        content="find auth bug",
        entities={"task": "find bug", "target": "auth"},
        emotional_tag={"valence": -0.3, "arousal": 0.5},
    )

    # Verify staging contents
    staging_mem = await mm.staging.get_by_id(mem_id)
    assert staging_mem["content"] == "find auth bug"
    assert staging_mem["emotional_tag"]["arousal"] == 0.5

    # Generate more interactions
    for i in range(20):
        interaction = await sm.on_interaction(f"step {i}")
        mm.set_context(interaction, session.id)
        await mm.encode(content=f"step {i} result", entities={"step": i})

    # Consolidate staging -> episodic
    result = await mm.consolidate()
    assert result.transferred == 21

    # Verify final state
    stats = await mm.stats()
    assert stats["episodic"] == 21
    assert stats["staging"] == 0

    # Close session
    await sm.close_session()
