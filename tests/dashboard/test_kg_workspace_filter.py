"""Knowledge graph endpoint workspace filter + cross-ref edges."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from brain_agent.agent import BrainAgent
from brain_agent.dashboard.server import create_app


@pytest.fixture
async def client_with_two_workspaces(tmp_path):
    agent = BrainAgent(use_mock_embeddings=True, data_dir=str(tmp_path))
    await agent.initialize()
    ws_a = await agent.memory.workspace.create_workspace(name="wsA")
    ws_b = await agent.memory.workspace.create_workspace(name="wsB")
    await agent.memory.semantic.add_edge(
        subject="node-A1",
        relation="refs",
        target="node-B1",
        target_workspace_id=ws_b["id"],
        workspace_id=ws_a["id"],
        confidence="STABLE",
    )
    await agent.memory.semantic.add_edge(
        subject="node-A1",
        relation="owns",
        target="node-A2",
        workspace_id=ws_a["id"],
        confidence="STABLE",
    )
    app = create_app(agent=agent)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client, agent, ws_a["id"], ws_b["id"]
    await agent.close()


async def test_kg_without_workspace_returns_all(client_with_two_workspaces):
    client, *_ = client_with_two_workspaces
    response = await client.get("/api/memory/knowledge-graph")
    assert response.status_code == 200
    assert len(response.json()["edges"]) >= 2


async def test_kg_with_workspace_filters_edges(client_with_two_workspaces):
    client, _, workspace_a, _ = client_with_two_workspaces
    response = await client.get(
        f"/api/memory/knowledge-graph?workspace_id={workspace_a}"
    )
    assert response.status_code == 200
    for edge in response.json()["edges"]:
        assert edge.get("cross_ref", False) is False


async def test_kg_include_cross_refs_true(client_with_two_workspaces):
    client, _, workspace_a, _ = client_with_two_workspaces
    response = await client.get(
        "/api/memory/knowledge-graph"
        f"?workspace_id={workspace_a}&include_cross_refs=true"
    )
    assert response.status_code == 200
    cross_refs = [edge for edge in response.json()["edges"] if edge.get("cross_ref")]
    assert len(cross_refs) >= 1
