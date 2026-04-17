"""API tests for /api/workspaces/*."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from brain_agent.agent import BrainAgent
from brain_agent.dashboard.server import create_app, event_bus


@pytest.fixture
async def app_and_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_AGENT_DATA_DIR", str(tmp_path))
    agent = BrainAgent(use_mock_embeddings=True, data_dir=str(tmp_path))
    await agent.initialize()
    app = create_app(agent=agent)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client, agent
    await agent.close()


async def test_list_workspaces_includes_personal(app_and_agent):
    client, _ = app_and_agent
    response = await client.get("/api/workspaces")
    assert response.status_code == 200
    names = {item["name"] for item in response.json()["workspaces"]}
    assert "Personal Knowledge" in names


async def test_create_workspace(app_and_agent):
    client, _ = app_and_agent
    response = await client.post(
        "/api/workspaces",
        json={
            "name": "Billing Service",
            "decay_policy": "none",
            "description": "pay",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Billing Service"
    assert body["decay_policy"] == "none"


async def test_get_workspace_returns_stats(app_and_agent):
    client, _ = app_and_agent
    created = await client.post("/api/workspaces", json={"name": "Proj Stats"})
    workspace_id = created.json()["id"]
    response = await client.get(f"/api/workspaces/{workspace_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == workspace_id
    assert "node_count" in body
    assert "edge_count" in body
    assert "pending_count" in body


async def test_get_missing_workspace_404(app_and_agent):
    client, _ = app_and_agent
    response = await client.get("/api/workspaces/does-not-exist")
    assert response.status_code == 404


async def test_set_current_emits_workspace_changed(app_and_agent):
    client, _ = app_and_agent
    event_bus._buffer.clear()
    created = await client.post("/api/workspaces", json={"name": "Proj Switch"})
    workspace_id = created.json()["id"]

    response = await client.put(
        "/api/workspaces/current",
        json={"session_id": "sess-A", "workspace_id": workspace_id},
    )

    assert response.status_code == 200
    event = next(
        item
        for item in event_bus.get_recent(10)
        if item["type"] == "workspace_changed"
    )
    assert event["payload"]["workspace_id"] == workspace_id
    assert event["payload"]["session_id"] == "sess-A"


async def test_get_current_returns_bound_workspace(app_and_agent):
    client, _ = app_and_agent
    created = await client.post("/api/workspaces", json={"name": "Proj Bind"})
    workspace_id = created.json()["id"]
    await client.put(
        "/api/workspaces/current",
        json={"session_id": "sess-B", "workspace_id": workspace_id},
    )

    response = await client.get("/api/workspaces/current?session_id=sess-B")

    assert response.status_code == 200
    assert response.json()["workspace"]["id"] == workspace_id


async def test_update_workspace(app_and_agent):
    client, _ = app_and_agent
    created = await client.post("/api/workspaces", json={"name": "Proj Patch"})
    workspace_id = created.json()["id"]
    response = await client.patch(
        f"/api/workspaces/{workspace_id}",
        json={"decay_policy": "slow", "description": "updated"},
    )
    assert response.status_code == 200
    assert response.json()["decay_policy"] == "slow"
    assert response.json()["description"] == "updated"


async def test_delete_personal_rejected(app_and_agent):
    client, _ = app_and_agent
    response = await client.delete("/api/workspaces/personal")
    assert response.status_code == 400
    assert "personal" in response.json()["detail"].lower()


async def test_delete_custom_workspace(app_and_agent):
    client, _ = app_and_agent
    created = await client.post("/api/workspaces", json={"name": "Proj Gone"})
    workspace_id = created.json()["id"]
    response = await client.delete(f"/api/workspaces/{workspace_id}")
    assert response.status_code == 200
    missing = await client.get(f"/api/workspaces/{workspace_id}")
    assert missing.status_code == 404
