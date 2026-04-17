"""Phase 5 end-to-end smoke tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def wired_pipeline():
    from brain_agent.pipeline import ProcessingPipeline

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
    pipe = ProcessingPipeline(memory=mem, llm_provider=None)
    return pipe, mem


def _result(mode, questions=None):
    return MagicMock(
        response_mode=mode,
        clarification_questions=questions or [],
        workspace_id="personal",
        workspace_ask=None,
    )


@pytest.mark.asyncio
async def test_full_normal_path(wired_pipeline):
    pipe, _ = wired_pipeline
    pipe.extraction_orchestrator.extract = AsyncMock(return_value=_result("normal"))

    result = await pipe.process_request(text="hello")

    assert result.response_mode == "normal"
    assert result.workspace_id == "personal"


@pytest.mark.asyncio
async def test_full_block_path(wired_pipeline):
    pipe, _ = wired_pipeline
    pipe.extraction_orchestrator.extract = AsyncMock(
        return_value=_result("block", ["Where did you move?"])
    )

    result = await pipe.process_request(text="I moved")

    assert result.response == "Where did you move?"
    assert result.response_mode == "block"


@pytest.mark.asyncio
async def test_full_append_path(wired_pipeline):
    pipe, _ = wired_pipeline
    pipe.extraction_orchestrator.extract = AsyncMock(
        return_value=_result("append", ["Since when?"])
    )

    result = await pipe.process_request(text="I prefer Go")

    assert result.response_mode == "append"
    assert result.clarification_questions == ["Since when?"]
