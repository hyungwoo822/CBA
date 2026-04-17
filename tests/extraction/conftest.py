"""Shared fixtures for extraction tests."""
import pytest

from brain_agent.extraction._mock_llm import RecordingLLMProvider


@pytest.fixture
def mock_llm():
    return RecordingLLMProvider(default_model="mock-default")


@pytest.fixture
def mock_llm_factory():
    def _build(responses=None, default_model="mock-default"):
        return RecordingLLMProvider(default_model=default_model, responses=responses)

    return _build
