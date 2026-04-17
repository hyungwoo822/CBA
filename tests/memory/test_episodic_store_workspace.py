"""Tests for workspace-aware EpisodicStore."""
import pytest

from brain_agent.memory.episodic_store import EpisodicStore
from brain_agent.migrations.steps import m001_workspace_columns as m001


@pytest.fixture
async def episodic(tmp_path):
    s = EpisodicStore(db_path=str(tmp_path / "episodic.db"))
    await s.initialize()
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=str(tmp_path),
    )
    yield s
    await s.close()


async def test_save_default_workspace_personal(episodic):
    await episodic.save(
        content="had coffee",
        context_embedding=[0.0] * 8,
        entities={},
        emotional_tag={"valence": 0, "arousal": 0},
        interaction_id=1,
        session_id="s1",
    )
    rows = await episodic.get_recent(limit=10)
    assert rows[0]["workspace_id"] == "personal"


async def test_save_explicit_workspace(episodic):
    await episodic.save(
        content="deployed billing",
        context_embedding=[0.0] * 8,
        entities={},
        emotional_tag={"valence": 0, "arousal": 0},
        interaction_id=2,
        session_id="s2",
        workspace_id="ws_biz",
        source_id="src_abc123",
        event_type="deployment",
        actor="alice",
    )
    rows = await episodic.get_recent(limit=10, workspace_id="ws_biz")
    assert len(rows) == 1
    assert rows[0]["workspace_id"] == "ws_biz"
    assert rows[0]["source_id"] == "src_abc123"
    assert rows[0]["event_type"] == "deployment"
    assert rows[0]["actor"] == "alice"


async def test_save_importance_and_never_decay(episodic):
    await episodic.save(
        content="business critical event",
        context_embedding=[0.0] * 8,
        entities={},
        emotional_tag={"valence": 0, "arousal": 0},
        interaction_id=3,
        session_id="s3",
        workspace_id="ws_biz",
        importance_score=0.95,
        never_decay=True,
    )
    rows = await episodic.get_recent(limit=10, workspace_id="ws_biz")
    assert rows[0]["importance_score"] == 0.95
    assert rows[0]["never_decay"] == 1


async def test_get_recent_filters_by_workspace(episodic):
    await episodic.save(
        content="personal thing",
        context_embedding=[0.0] * 8,
        entities={},
        emotional_tag={"valence": 0, "arousal": 0},
        interaction_id=10,
        session_id="s",
        workspace_id="personal",
    )
    await episodic.save(
        content="biz thing",
        context_embedding=[0.0] * 8,
        entities={},
        emotional_tag={"valence": 0, "arousal": 0},
        interaction_id=11,
        session_id="s",
        workspace_id="ws_biz",
    )
    personal = await episodic.get_recent(limit=5, workspace_id="personal")
    biz = await episodic.get_recent(limit=5, workspace_id="ws_biz")
    assert len(personal) == 1
    assert personal[0]["content"] == "personal thing"
    assert len(biz) == 1
    assert biz[0]["content"] == "biz thing"


async def test_get_recent_none_returns_all(episodic):
    await episodic.save(
        content="a",
        context_embedding=[0.0] * 8,
        entities={},
        emotional_tag={"valence": 0, "arousal": 0},
        interaction_id=1,
        session_id="s",
        workspace_id="personal",
    )
    await episodic.save(
        content="b",
        context_embedding=[0.0] * 8,
        entities={},
        emotional_tag={"valence": 0, "arousal": 0},
        interaction_id=2,
        session_id="s",
        workspace_id="ws_biz",
    )
    both = await episodic.get_recent(limit=5, workspace_id=None)
    assert len(both) == 2
