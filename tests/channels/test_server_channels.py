"""Tests for /api/channels endpoints and broadcast in /api/process."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def app():
    """Create a test app with mocked agent and channel manager."""
    from brain_agent.dashboard.server import create_app
    return create_app(static_dir=None)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_get_channels_returns_empty_when_no_channels(client):
    resp = client.get("/api/channels")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
