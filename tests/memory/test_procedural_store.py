import pytest
from brain_agent.memory.procedural_store import ProceduralStore, ProcedureStage


@pytest.fixture
async def store(tmp_db_path):
    s = ProceduralStore(db_path=tmp_db_path)
    await s.initialize()
    yield s
    await s.close()


async def test_save_and_match(store):
    await store.save(
        trigger_pattern="read file *",
        action_sequence=[{"tool": "read_file", "args": {"path": "{0}"}}],
    )
    match = await store.match("read file auth.py")
    assert match is not None
    assert match["stage"] == ProcedureStage.COGNITIVE.value


async def test_no_match_returns_none(store):
    assert await store.match("completely different thing") is None


async def test_record_success_increments_count(store):
    proc_id = await store.save(
        trigger_pattern="test *",
        action_sequence=[{"tool": "run_tests"}],
    )
    await store.record_execution(proc_id, success=True)
    await store.record_execution(proc_id, success=True)
    proc = await store.get_by_id(proc_id)
    assert proc["execution_count"] == 2
    assert proc["success_rate"] == 1.0


async def test_stage_promotion(store):
    proc_id = await store.save(
        trigger_pattern="build *",
        action_sequence=[{"tool": "build"}],
    )
    for _ in range(10):
        await store.record_execution(proc_id, success=True)
    assert (await store.get_by_id(proc_id))["stage"] == ProcedureStage.ASSOCIATIVE.value


async def test_autonomous_after_many_successes(store):
    proc_id = await store.save(
        trigger_pattern="deploy *",
        action_sequence=[{"tool": "deploy"}],
    )
    for _ in range(50):
        await store.record_execution(proc_id, success=True)
    assert (await store.get_by_id(proc_id))["stage"] == ProcedureStage.AUTONOMOUS.value


async def test_intent_trigger_matches_similar_input(tmp_path):
    """Intent-based triggers should match semantically similar inputs."""
    store = ProceduralStore(db_path=str(tmp_path / "proc.db"))
    await store.initialize()

    # Save with intent-based trigger
    await store.save(
        trigger_pattern="command:build,deploy,project",
        action_sequence=[{"tool": "respond", "args": {"text": "Building..."}}],
    )

    # Should match same intent pattern
    result = await store.match("command:build,deploy,project")
    assert result is not None
    assert result["trigger_pattern"] == "command:build,deploy,project"

    # Should NOT match completely different intent
    result2 = await store.match("question:weather,today")
    assert result2 is None

    await store.close()
