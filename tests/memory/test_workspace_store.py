"""Tests for WorkspaceStore."""
import pytest

from brain_agent.memory.workspace_store import WorkspaceStore


@pytest.fixture
async def store(tmp_path):
    store = WorkspaceStore(db_path=str(tmp_path / "workspaces.db"))
    await store.initialize()
    yield store
    await store.close()


async def test_personal_workspace_auto_created(store):
    ws = await store.get_workspace("personal")
    assert ws is not None
    assert ws["id"] == "personal"
    assert ws["name"] == "Personal Knowledge"
    assert ws["decay_policy"] == "normal"


async def test_existing_personal_workspace_name_is_canonicalized(store):
    await store.update_workspace("personal", name="personal")

    ws = await store.get_workspace("personal")
    by_name = await store.get_workspace("Personal Knowledge")
    items = await store.list_workspaces()

    assert ws["name"] == "Personal Knowledge"
    assert by_name["id"] == "personal"
    assert "Personal Knowledge" in {workspace["name"] for workspace in items}


async def test_create_workspace(store):
    ws = await store.create_workspace(
        name="Billing Service",
        description="Payment flow domain",
        decay_policy="none",
    )
    assert ws["name"] == "Billing Service"
    assert ws["decay_policy"] == "none"
    assert ws["id"]
    assert ws["id"] != "personal"


async def test_create_workspace_duplicate_name_raises(store):
    await store.create_workspace(name="Proj A")
    with pytest.raises(ValueError, match="already exists"):
        await store.create_workspace(name="Proj A")


async def test_list_workspaces_includes_personal(store):
    await store.create_workspace(name="Proj X")
    items = await store.list_workspaces()
    names = {workspace["name"] for workspace in items}
    assert "Personal Knowledge" in names
    assert "Proj X" in names


async def test_get_workspace_by_id_and_name(store):
    ws = await store.create_workspace(name="Proj Y")
    by_name = await store.get_workspace("Proj Y")
    by_id = await store.get_workspace(ws["id"])
    assert by_name["id"] == ws["id"]
    assert by_id["id"] == ws["id"]


async def test_get_workspace_missing_returns_none(store):
    assert await store.get_workspace("nonexistent") is None


async def test_update_workspace(store):
    ws = await store.create_workspace(name="Proj Z", decay_policy="normal")
    await store.update_workspace(ws["id"], decay_policy="slow", description="updated")
    updated = await store.get_workspace(ws["id"])
    assert updated["decay_policy"] == "slow"
    assert updated["description"] == "updated"


async def test_delete_workspace(store):
    ws = await store.create_workspace(name="Proj Temp")
    await store.delete_workspace(ws["id"])
    assert await store.get_workspace(ws["id"]) is None


async def test_delete_personal_raises(store):
    with pytest.raises(ValueError, match="Cannot delete personal"):
        await store.delete_workspace("personal")


async def test_get_session_workspace_default_personal(store):
    assert await store.get_session_workspace("session-xyz") == "personal"


async def test_set_and_get_session_workspace(store):
    ws = await store.create_workspace(name="Proj Session")
    await store.set_session_workspace("session-abc", ws["id"])
    assert await store.get_session_workspace("session-abc") == ws["id"]


async def test_set_session_workspace_unknown_id_raises(store):
    with pytest.raises(ValueError, match="Workspace not found"):
        await store.set_session_workspace("session-def", "no-such-workspace")


async def test_get_last_workspace_returns_most_recent(store):
    ws_a = await store.create_workspace(name="Proj Last A")
    ws_b = await store.create_workspace(name="Proj Last B")
    await store.set_session_workspace("s1", ws_a["id"])
    await store.set_session_workspace("s2", ws_b["id"])
    assert await store.get_last_workspace() == ws_b["id"]


async def test_get_last_workspace_empty_returns_personal(store):
    assert await store.get_last_workspace() == "personal"
