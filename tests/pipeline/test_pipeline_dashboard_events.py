"""Block mode emits a clarification_requested event."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_clarification_requested_emitted_on_block():
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
    mem.procedural = MagicMock()
    mem.procedural.match = AsyncMock(return_value=None)
    emitter = MagicMock()
    emitter.region_activation = AsyncMock()
    emitter.signal_flow = AsyncMock()
    emitter.region_io = AsyncMock()
    emitter.region_processing = AsyncMock()
    emitter.neuromodulator_update = AsyncMock()
    emitter.network_switch = AsyncMock()
    emitter.clarification_requested = AsyncMock()
    pipe = ProcessingPipeline(memory=mem, llm_provider=None, emitter=emitter)
    pipe.extraction_orchestrator.extract = AsyncMock(return_value=MagicMock(
        response_mode="block",
        clarification_questions=["Where did you move?"],
        workspace_id="personal",
        workspace_ask=None,
    ))

    await pipe.process_request(text="I moved")

    emitter.clarification_requested.assert_awaited_once()
