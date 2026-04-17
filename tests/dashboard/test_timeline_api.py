"""Temporal timeline API tests."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from brain_agent.agent import BrainAgent
from brain_agent.dashboard.server import create_app


@pytest.fixture
async def client_with_temporal_chain(tmp_path):
    agent = BrainAgent(use_mock_embeddings=True, data_dir=str(tmp_path))
    await agent.initialize()
    old = await agent.memory.semantic.add_edge(
        subject="hw",
        relation="is",
        target="laptop",
        workspace_id="personal",
        confidence="STABLE",
    )
    await agent.memory.semantic.add_edge(
        subject="hw",
        relation="is",
        target="desktop",
        workspace_id="personal",
        confidence="STABLE",
        supersedes=old["id"],
    )
    app = create_app(agent=agent)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as api_client:
        yield api_client
    await agent.close()


async def test_timeline_chain(client_with_temporal_chain):
    response = await client_with_temporal_chain.get(
        "/api/memory/timeline?workspace_id=personal&subject=hw"
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["chain"]) == 2
    assert body["chain"][0]["target"] == "laptop"
    assert body["chain"][1]["target"] == "desktop"
    assert body["chain"][0]["valid_to"] is not None
    assert body["chain"][1]["valid_to"] is None
