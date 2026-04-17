"""Tests for Stage 5 Broca Refine."""
import pytest

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.refiner import Refiner


@pytest.fixture
def refiner(mock_llm):
    return Refiner(mock_llm, ExtractionConfig())


async def test_personal_workspace_refines(refiner, mock_llm):
    mock_llm.enqueue_content("polished answer")
    output = await refiner.refine(
        agent_response="draft response",
        language="ko",
        workspace={"id": "personal", "name": "Personal Knowledge"},
    )
    assert output == "polished answer"
    assert len(mock_llm.calls) == 1


async def test_non_personal_workspace_skips(refiner, mock_llm):
    output = await refiner.refine(
        agent_response="draft",
        language="en",
        workspace={"id": "billing-service", "name": "Billing Service"},
    )
    assert output == ""
    assert mock_llm.calls == []


async def test_empty_response_returns_empty(refiner, mock_llm):
    output = await refiner.refine("", "ko", {"id": "personal", "name": "Personal Knowledge"})
    assert output == ""
    assert mock_llm.calls == []


async def test_null_response_from_llm_returns_original(refiner, mock_llm):
    mock_llm.enqueue_content("null")
    output = await refiner.refine(
        agent_response="draft",
        language="ko",
        workspace={"id": "personal", "name": "Personal Knowledge"},
    )
    assert output == "draft"


async def test_uses_configured_model(refiner, mock_llm):
    mock_llm.enqueue_content("polished")
    refiner._cfg.refine_model = "claude-haiku-4-5"
    await refiner.refine("draft", "ko", {"id": "personal", "name": "Personal Knowledge"})
    assert mock_llm.calls[0]["model"] == "claude-haiku-4-5"


async def test_auto_resolves_default_model(refiner, mock_llm):
    mock_llm.enqueue_content("polished")
    refiner._cfg.refine_model = "auto"
    await refiner.refine("draft", "ko", {"id": "personal", "name": "Personal Knowledge"})
    assert mock_llm.calls[0]["model"] == "mock-default"
