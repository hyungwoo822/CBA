"""Export preview filter matrix tests."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from brain_agent.agent import BrainAgent
from brain_agent.dashboard.server import create_app


@pytest.fixture
async def client_with_seeded_facts(tmp_path):
    agent = BrainAgent(use_mock_embeddings=True, data_dir=str(tmp_path))
    await agent.initialize()
    workspace = await agent.memory.workspace.create_workspace(name="Bill")
    await agent.memory.semantic.add_edge(
        subject="pay",
        relation="implements",
        target="idem",
        workspace_id=workspace["id"],
        confidence="STABLE",
        importance_score=0.9,
        never_decay=True,
        epistemic_source="asserted",
    )
    await agent.memory.semantic.add_edge(
        subject="x",
        relation="relates_to",
        target="y",
        workspace_id=workspace["id"],
        confidence="PROVISIONAL",
        importance_score=0.1,
        never_decay=False,
    )
    app = create_app(agent=agent)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as api_client:
        yield api_client, workspace["id"]
    await agent.close()


async def test_preview_shape(client_with_seeded_facts):
    api_client, workspace_id = client_with_seeded_facts
    response = await api_client.post(
        "/api/export/preview",
        json={"workspace_id": workspace_id, "filters": {}},
    )
    assert response.status_code == 200
    body = response.json()
    assert "workspace" in body
    assert "ontology" in body
    assert "facts" in body
    assert "open_questions" in body
    assert "unresolved_contradictions" in body


async def test_preview_never_decay_only(client_with_seeded_facts):
    api_client, workspace_id = client_with_seeded_facts
    response = await api_client.post(
        "/api/export/preview",
        json={"workspace_id": workspace_id, "filters": {"never_decay_only": True}},
    )
    assert response.status_code == 200
    for fact in response.json()["facts"]:
        assert fact["never_decay"] is True


async def test_preview_min_importance(client_with_seeded_facts):
    api_client, workspace_id = client_with_seeded_facts
    response = await api_client.post(
        "/api/export/preview",
        json={"workspace_id": workspace_id, "filters": {"min_importance": 0.5}},
    )
    assert response.status_code == 200
    for fact in response.json()["facts"]:
        assert fact["importance_score"] >= 0.5


async def test_preview_min_confidence(client_with_seeded_facts):
    api_client, workspace_id = client_with_seeded_facts
    response = await api_client.post(
        "/api/export/preview",
        json={"workspace_id": workspace_id, "filters": {"min_confidence": "STABLE"}},
    )
    assert response.status_code == 200
    for fact in response.json()["facts"]:
        assert fact["confidence"] in ("STABLE", "CANONICAL", "USER_GROUND_TRUTH")


async def test_preview_include_raw_vault(client_with_seeded_facts):
    api_client, workspace_id = client_with_seeded_facts
    response = await api_client.post(
        "/api/export/preview",
        json={
            "workspace_id": workspace_id,
            "filters": {"include_raw_vault": True},
        },
    )
    assert response.status_code == 200
    assert "raw_vault_refs" in response.json()


async def test_preview_filter_combinations(client_with_seeded_facts):
    api_client, workspace_id = client_with_seeded_facts
    response = await api_client.post(
        "/api/export/preview",
        json={
            "workspace_id": workspace_id,
            "filters": {
                "never_decay_only": True,
                "min_importance": 0.5,
                "min_confidence": "STABLE",
                "include_raw_vault": False,
            },
        },
    )
    assert response.status_code == 200
    assert all(
        fact["never_decay"] and fact["importance_score"] >= 0.5
        for fact in response.json()["facts"]
    )
