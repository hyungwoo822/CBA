"""Raw vault source API tests."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from brain_agent.agent import BrainAgent
from brain_agent.dashboard.server import create_app


@pytest.fixture
async def client(tmp_path):
    agent = BrainAgent(use_mock_embeddings=True, data_dir=str(tmp_path))
    await agent.initialize()
    app = create_app(agent=agent)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as api_client:
        yield api_client, agent
    await agent.close()


async def test_source_meta_404_when_missing(client):
    api_client, _ = client
    response = await api_client.get("/api/sources/nope")
    assert response.status_code == 404


async def test_source_meta_ok(client):
    api_client, agent = client
    source = await agent.memory.raw_vault.ingest(
        workspace_id="personal",
        kind="user_utterance",
        data=b"hello",
        filename="h.txt",
    )
    response = await api_client.get(f"/api/sources/{source['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == source["id"]


async def test_source_raw_503_on_integrity_fail(client):
    api_client, agent = client
    source = await agent.memory.raw_vault.ingest(
        workspace_id="personal",
        kind="user_utterance",
        data=b"hi",
        filename="h.txt",
    )
    await agent.memory.raw_vault.mark_integrity_invalid(source["id"])
    response = await api_client.get(f"/api/sources/{source['id']}/raw")
    assert response.status_code == 503
    assert "integrity" in response.json()["detail"].lower()


async def test_source_text(client):
    api_client, agent = client
    source = await agent.memory.raw_vault.ingest(
        workspace_id="personal",
        kind="user_utterance",
        data=b"hello world",
        filename="h.txt",
        extracted_text="hello world",
    )
    response = await api_client.get(f"/api/sources/{source['id']}/text")
    assert response.status_code == 200
    assert response.json()["text"] == "hello world"
