"""Tests for workspace-aware HippocampalStaging."""
import pytest

from brain_agent.memory.hippocampal_staging import HippocampalStaging
from brain_agent.migrations.steps import m001_workspace_columns as m001


@pytest.fixture
async def staging(tmp_path, mock_embedding):
    s = HippocampalStaging(
        db_path=str(tmp_path / "staging.db"),
        embed_fn=mock_embedding,
    )
    await s.initialize()
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=str(tmp_path),
    )
    yield s
    await s.close()


async def test_encode_default_workspace(staging):
    mid = await staging.encode(
        content="a fresh memory",
        entities={},
        interaction_id=1,
        session_id="s1",
    )
    row = await staging.get_by_id(mid)
    assert row["workspace_id"] == "personal"


async def test_encode_explicit_workspace(staging):
    mid = await staging.encode(
        content="a biz memory",
        entities={},
        interaction_id=2,
        session_id="s1",
        workspace_id="ws_biz",
    )
    row = await staging.get_by_id(mid)
    assert row["workspace_id"] == "ws_biz"


async def test_get_unconsolidated_filters_by_workspace(staging):
    await staging.encode(
        "personal-a",
        entities={},
        interaction_id=1,
        session_id="s",
        workspace_id="personal",
    )
    await staging.encode(
        "biz-a",
        entities={},
        interaction_id=2,
        session_id="s",
        workspace_id="ws_biz",
    )
    personal = await staging.get_unconsolidated(workspace_id="personal")
    biz = await staging.get_unconsolidated(workspace_id="ws_biz")
    assert len(personal) == 1 and personal[0]["content"] == "personal-a"
    assert len(biz) == 1 and biz[0]["content"] == "biz-a"


async def test_get_unconsolidated_none_returns_all(staging):
    await staging.encode(
        "p", entities={}, interaction_id=1, session_id="s", workspace_id="personal"
    )
    await staging.encode(
        "b", entities={}, interaction_id=2, session_id="s", workspace_id="ws_biz"
    )
    both = await staging.get_unconsolidated(workspace_id=None)
    assert len(both) == 2


async def test_encode_edge_stores_edge_staging_row(staging):
    eid = await staging.encode_edge(
        source="alice",
        relation="likes",
        target="coffee",
        interaction_id=5,
        session_id="s",
        workspace_id="personal",
    )
    rows = await staging.get_unconsolidated_edges(workspace_id="personal")
    assert len(rows) == 1
    assert rows[0]["id"] == eid
    assert rows[0]["source_node"] == "alice"
    assert rows[0]["relation"] == "likes"
    assert rows[0]["target_node"] == "coffee"


async def test_encode_edge_distinct_from_encode_content(staging):
    mid = await staging.encode("content mem", {}, 1, "s", workspace_id="personal")
    eid = await staging.encode_edge(
        "a", "r", "b", interaction_id=2, session_id="s", workspace_id="personal"
    )
    contents = await staging.get_unconsolidated(workspace_id="personal")
    edges = await staging.get_unconsolidated_edges(workspace_id="personal")
    assert len(contents) == 1 and contents[0]["id"] == mid
    assert len(edges) == 1 and edges[0]["id"] == eid


async def test_reinforce_increments_access_count(staging):
    mid = await staging.encode(
        "repeat me", {}, 1, "s", workspace_id="personal"
    )
    before = await staging.get_by_id(mid)
    await staging.reinforce(mid, boost=2.0)
    after = await staging.get_by_id(mid)
    assert after["strength"] > before["strength"]
    assert after["access_count"] == before["access_count"] + 1
