"""Tests for workspace-aware ProceduralStore."""
import json

import pytest

from brain_agent.memory.procedural_store import ProceduralStore
from brain_agent.migrations.steps import m001_workspace_columns as m001


@pytest.fixture
async def proc(tmp_path):
    p = ProceduralStore(db_path=str(tmp_path / "procedural.db"))
    await p.initialize()
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=str(tmp_path),
    )
    yield p
    await p.close()


async def test_save_default_workspace(proc):
    pid = await proc.save(
        trigger_pattern="*hello*",
        action_sequence=[{"op": "greet"}],
    )
    row = await proc.get_by_id(pid)
    assert row["workspace_id"] == "personal"


async def test_save_explicit_workspace(proc):
    pid = await proc.save(
        trigger_pattern="*deploy*",
        action_sequence=[{"op": "run_ci"}],
        workspace_id="ws_biz",
        trigger_embedding=json.dumps([0.1, 0.2]),
    )
    row = await proc.get_by_id(pid)
    assert row["workspace_id"] == "ws_biz"
    assert row["trigger_embedding"] == json.dumps([0.1, 0.2])


async def test_match_filters_by_workspace(proc):
    await proc.save("*hi*", [{"op": "greet_personal"}], workspace_id="personal")
    await proc.save("*hi*", [{"op": "greet_biz"}], workspace_id="ws_biz")
    biz = await proc.match("hi there", workspace_id="ws_biz")
    assert biz is not None
    assert biz["workspace_id"] == "ws_biz"
    assert biz["action_sequence"] == [{"op": "greet_biz"}]


async def test_match_none_returns_first_of_any_workspace(proc):
    await proc.save("*hi*", [{"op": "greet_personal"}], workspace_id="personal")
    match = await proc.match("hi there", workspace_id=None)
    assert match is not None
    assert match["action_sequence"] == [{"op": "greet_personal"}]


async def test_save_same_trigger_different_workspaces_independent(proc):
    a = await proc.save("*t*", [{"op": "A"}], workspace_id="personal")
    b = await proc.save("*t*", [{"op": "B"}], workspace_id="ws_biz")
    assert a != b
    row_a = await proc.get_by_id(a)
    row_b = await proc.get_by_id(b)
    assert row_a["workspace_id"] == "personal"
    assert row_b["workspace_id"] == "ws_biz"
