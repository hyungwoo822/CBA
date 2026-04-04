import pytest
from brain_agent.core.session import SessionManager


@pytest.fixture
async def manager(tmp_db_path, mock_embedding):
    sm = SessionManager(db_path=tmp_db_path, embed_fn=mock_embedding)
    await sm.initialize()
    yield sm
    await sm.close()


async def test_start_creates_session(manager):
    session = await manager.start_session()
    assert session.id != ""
    assert session.start_interaction == manager.temporal.interaction_count


async def test_close_session(manager):
    session = await manager.start_session()
    sid = session.id
    await manager.close_session()
    assert manager.temporal.current_session_id == ""
    assert manager.temporal.count_sessions_since(sid) == 1


async def test_detect_idle_timeout(manager):
    session = await manager.start_session()
    from datetime import timedelta, timezone, datetime
    manager.temporal._last_wall_clock = datetime.now(timezone.utc) - timedelta(minutes=31)
    assert manager.should_start_new_session("any input")


async def test_interaction_ticks(manager):
    await manager.start_session()
    await manager.on_interaction("hello")
    await manager.on_interaction("world")
    assert manager.temporal.interaction_count == 2
