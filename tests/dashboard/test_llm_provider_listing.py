"""LLM provider inventory tests."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from brain_agent.agent import BrainAgent
from brain_agent.dashboard.server import create_app


@pytest.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    agent = BrainAgent(use_mock_embeddings=True, data_dir=str(tmp_path))
    await agent.initialize()
    app = create_app(agent=agent)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as api_client:
        yield api_client
    await agent.close()


async def test_providers_default_model_present(client):
    response = await client.get("/api/llm/providers")
    assert response.status_code == 200
    body = response.json()
    assert body["default_model"]
    assert isinstance(body["available"], list)


async def test_providers_marks_missing_keys_unavailable(client):
    response = await client.get("/api/llm/providers")
    body = response.json()
    openai_entries = [item for item in body["available"] if item["vendor"] == "openai"]
    if openai_entries:
        assert all(not item["available"] for item in openai_entries)
        assert any("OPENAI_API_KEY" in (item.get("reason") or "") for item in openai_entries)


async def test_providers_anthropic_reachable(client):
    response = await client.get("/api/llm/providers")
    body = response.json()
    anthropic_entries = [
        item for item in body["available"] if item["vendor"] == "anthropic"
    ]
    if anthropic_entries:
        assert any(item["available"] for item in anthropic_entries)


async def test_providers_litellm_mock(client):
    with patch(
        "brain_agent.dashboard.providers_inventory._fetch_model_list",
        return_value=[
            {"litellm_provider": "openai", "model_name": "openai/gpt-4o-mini"},
            {"litellm_provider": "anthropic", "model_name": "anthropic/claude-sonnet-4-6"},
            {"litellm_provider": "ollama", "model_name": "ollama/llama3"},
        ],
    ):
        response = await client.get("/api/llm/providers")
    body = response.json()
    vendors = {item["vendor"] for item in body["available"]}
    assert {"openai", "anthropic", "ollama"} <= vendors
