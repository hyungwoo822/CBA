"""Expression mode block override coverage."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from brain_agent.config.schema import ExtractionConfig
from brain_agent.pipeline import ProcessingPipeline


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.sensory = MagicMock()
    mem.working = MagicMock()
    mem.working.get_context.return_value = ""
    mem.brain_state = MagicMock()
    mem.recall_tracker = MagicMock()
    mem.workspace = MagicMock()
    mem.workspace.get_session_workspace = AsyncMock(return_value="personal")
    mem.workspace.get_workspace = AsyncMock(return_value={"id": "personal", "name": "Personal Knowledge"})
    mem.set_neuromodulators = MagicMock()
    mem.set_cortisol_accessor = MagicMock()
    mem.retrieve_identity = AsyncMock(return_value={"self_model": [], "user_model": []})
    mem.retrieve_with_contradictions = AsyncMock(return_value={
        "candidates": [], "edges": [], "nodes": [],
        "contradictions": [], "gaps": [], "inference_fill": [],
    })
    mem.procedural = MagicMock()
    mem.procedural.match = AsyncMock(return_value=None)
    mem.encode = AsyncMock(return_value="ep1")
    mem.staging = MagicMock()
    mem.staging.count_unconsolidated = AsyncMock(return_value=0)
    mem.staging.get_unconsolidated = AsyncMock(return_value=[])
    mem.consolidation = MagicMock()
    mem.consolidation.should_consolidate = AsyncMock(return_value=False)
    mem.consolidate = AsyncMock()
    return mem


def _block_result():
    return MagicMock(
        response_mode="block",
        clarification_questions=["Did you move?"],
        workspace_id="personal",
        workspace_ask=None,
    )


@pytest.mark.asyncio
async def test_expression_override_demotes_block_to_append(mock_memory):
    pipe = ProcessingPipeline(
        memory=mock_memory,
        llm_provider=None,
        extraction_config=ExtractionConfig(expression_override_block=True),
    )
    pipe.extraction_orchestrator.extract = AsyncMock(return_value=_block_result())

    result = await pipe.process_request(text="Where do I live?", interaction_mode="expression")

    assert result.response_mode == "append"
    assert result.response
    assert result.clarification_questions == ["Did you move?"]


@pytest.mark.asyncio
async def test_expression_override_disabled_still_blocks(mock_memory):
    pipe = ProcessingPipeline(
        memory=mock_memory,
        llm_provider=None,
        extraction_config=ExtractionConfig(expression_override_block=False),
    )
    pipe.extraction_orchestrator.extract = AsyncMock(return_value=_block_result())

    result = await pipe.process_request(text="Where do I live?", interaction_mode="expression")

    assert result.response_mode == "block"
    assert result.response == "Did you move?"


@pytest.mark.asyncio
async def test_question_mode_always_blocks(mock_memory):
    pipe = ProcessingPipeline(
        memory=mock_memory,
        llm_provider=None,
        extraction_config=ExtractionConfig(expression_override_block=True),
    )
    pipe.extraction_orchestrator.extract = AsyncMock(return_value=_block_result())

    result = await pipe.process_request(text="Where do I live?", interaction_mode="question")

    assert result.response_mode == "block"
