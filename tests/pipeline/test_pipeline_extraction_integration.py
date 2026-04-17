"""Phase 5 integration tests for ExtractionOrchestrator wiring."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from brain_agent.config.schema import BrainAgentConfig
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
    mem.workspace.get_or_create_personal = AsyncMock(return_value={"id": "personal"})
    mem.workspace.get_session_workspace = AsyncMock(return_value="personal")
    mem.workspace.get_workspace = AsyncMock(return_value={"id": "personal", "name": "Personal Knowledge"})
    mem.set_neuromodulators = MagicMock()
    mem.set_cortisol_accessor = MagicMock()
    mem.retrieve_identity = AsyncMock(return_value={"self_model": [], "user_model": []})
    mem.retrieve_with_contradictions = AsyncMock(return_value={
        "candidates": [],
        "edges": [],
        "nodes": [],
        "contradictions": [],
        "gaps": [],
        "inference_fill": [],
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
    mem.semantic = MagicMock()
    return mem


def _orch_result(mode="normal", questions=None, workspace_ask=None):
    return MagicMock(
        response_mode=mode,
        clarification_questions=questions or [],
        workspace_id="personal",
        workspace_ask=workspace_ask,
        response_text="",
        contradictions=[],
        open_questions=[],
    )


def test_pipeline_has_extraction_orchestrator(mock_memory):
    from brain_agent.extraction.orchestrator import ExtractionOrchestrator

    pipe = ProcessingPipeline(memory=mock_memory, llm_provider=MagicMock())
    assert isinstance(pipe.extraction_orchestrator, ExtractionOrchestrator)


def test_pipeline_accepts_extraction_config_injection(mock_memory):
    cfg = BrainAgentConfig()
    cfg.extraction.enable_severity_block = False
    pipe = ProcessingPipeline(
        memory=mock_memory,
        llm_provider=MagicMock(),
        extraction_config=cfg.extraction,
    )
    assert pipe.extraction_orchestrator.config.enable_severity_block is False


def test_legacy_psc_methods_removed(mock_memory):
    pipe = ProcessingPipeline(memory=mock_memory, llm_provider=None)
    assert not hasattr(pipe, "_post_synaptic_consolidation")
    assert not hasattr(pipe, "_extract_user_facts")


@pytest.mark.asyncio
async def test_block_mode_short_circuits_with_questions(mock_memory):
    pipe = ProcessingPipeline(memory=mock_memory, llm_provider=None)
    questions = ["Did you move?", "Where should I store this?"]
    pipe.extraction_orchestrator.extract = AsyncMock(
        return_value=_orch_result(mode="block", questions=questions)
    )

    result = await pipe.process_request(text="I moved but not sure where")

    assert result.response == "- Did you move?\n- Where should I store this?"
    assert result.response_mode == "block"
    assert result.clarification_questions == questions
    assert result.workspace_id == "personal"
    mock_memory.retrieve_with_contradictions.assert_not_awaited()


@pytest.mark.asyncio
async def test_append_mode_returns_response_plus_questions(mock_memory):
    pipe = ProcessingPipeline(memory=mock_memory, llm_provider=None)
    questions = ["Since when?"]
    pipe.extraction_orchestrator.extract = AsyncMock(
        return_value=_orch_result(mode="append", questions=questions)
    )

    result = await pipe.process_request(text="I prefer Go now")

    assert result.response_mode == "append"
    assert result.response
    assert result.clarification_questions == questions


@pytest.mark.asyncio
async def test_normal_mode_workspace_ask_propagated(mock_memory):
    pipe = ProcessingPipeline(memory=mock_memory, llm_provider=None)
    pipe.extraction_orchestrator.extract = AsyncMock(
        return_value=_orch_result(
            mode="normal",
            workspace_ask="Switch to billing-service workspace?",
        )
    )

    result = await pipe.process_request(text="/payments endpoint fails")

    assert result.response_mode == "normal"
    assert result.workspace_ask == "Switch to billing-service workspace?"


@pytest.mark.asyncio
async def test_retrieval_contradictions_propagate_to_result(mock_memory):
    pipe = ProcessingPipeline(memory=mock_memory, llm_provider=None)
    pipe.extraction_orchestrator.extract = AsyncMock(return_value=_orch_result())
    mock_memory.retrieve_with_contradictions = AsyncMock(return_value={
        "candidates": [],
        "edges": [],
        "nodes": [],
        "contradictions": [{"subject": "user", "key": "lives_in", "value_a": "Seoul", "value_b": "Busan"}],
        "gaps": [{"node": "hospital_visit", "missing": "happened_at", "type": "Event"}],
        "inference_fill": [],
    })

    result = await pipe.process_request(text="where do I live?")

    assert result.retrieval_contradictions[0]["value_b"] == "Busan"
    assert result.retrieval_gaps[0]["missing"] == "happened_at"
