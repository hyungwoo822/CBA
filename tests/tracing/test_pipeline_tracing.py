"""Tests for Pipeline trace context threading."""
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field

from brain_agent.pipeline import ProcessingPipeline
from brain_agent.memory.manager import MemoryManager
from brain_agent.providers.base import LLMResponse
from brain_agent.providers.myelinated import MyelinatedProvider
from brain_agent.middleware.base import MiddlewareChain


def _mock_embed(text: str) -> list[float]:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(384).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


class MockProvider:
    def __init__(self):
        self.call_count = 0
        self._trace_parent = None
        self._trace_region = None

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        self.call_count += 1
        return LLMResponse(
            content='{"intent":"greeting","complexity":"simple","keywords":[],'
                    '"entities":[],"language":"en","word_count":1,"avg_word_length":5,'
                    '"confidence":0.9,"response":"Hello!","plan":"greet user",'
                    '"appraisal":{"valence":0.5,"arousal":0.3,"dominance":0.5},'
                    '"metacognition":{"confidence":0.9,"uncertainty":"low","reasoning_quality":"adequate"}}',
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

    def get_default_model(self):
        return "mock-model"

    def set_trace_context(self, parent, region):
        self._trace_parent = parent
        self._trace_region = region

    def clear_trace_context(self):
        self._trace_parent = None
        self._trace_region = None


@pytest.fixture
async def memory(tmp_path):
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=_mock_embed)
    await mm.initialize()
    yield mm
    await mm.close()


@pytest.fixture
def pipeline(memory):
    provider = MockProvider()
    myelinated = MyelinatedProvider(inner=provider, myelin=MiddlewareChain())
    return ProcessingPipeline(memory=memory, llm_provider=myelinated)


async def test_pipeline_accepts_trace_run_param(pipeline):
    """process_request should accept an optional trace_run parameter."""
    mock_run = MagicMock()
    mock_run.create_child.return_value = MagicMock()
    result = await pipeline.process_request("hello", trace_run=mock_run)
    assert result.response


async def test_pipeline_creates_phase_child_runs(pipeline):
    """Phase child runs should be created when trace_run is provided."""
    mock_run = MagicMock()
    mock_phase = MagicMock()
    mock_run.create_child.return_value = mock_phase

    await pipeline.process_request("hello", trace_run=mock_run)

    assert mock_run.create_child.call_count >= 1
    phase_names = [
        call.kwargs.get("name", "")
        for call in mock_run.create_child.call_args_list
    ]
    assert any("phase." in name for name in phase_names)


async def test_pipeline_no_trace_when_none(pipeline):
    """Pipeline should work normally when trace_run is None."""
    result = await pipeline.process_request("hello", trace_run=None)
    assert result.response


async def test_pipeline_trace_run_cleared_after_request(pipeline):
    """_current_trace_run should be None after process_request completes."""
    mock_run = MagicMock()
    mock_run.create_child.return_value = MagicMock()

    await pipeline.process_request("hello", trace_run=mock_run)

    assert pipeline._current_trace_run is None


async def test_pipeline_phase_runs_ended(pipeline):
    """Phase runs should have end() and post() called on them."""
    mock_run = MagicMock()
    mock_phase = MagicMock()
    mock_run.create_child.return_value = mock_phase

    await pipeline.process_request("hello", trace_run=mock_run)

    # At least one phase run should have been ended
    assert mock_phase.end.call_count >= 1
    assert mock_phase.post.call_count >= 1


async def test_pipeline_trace_run_instance_variables(pipeline):
    """ProcessingPipeline should have _current_trace_run and _current_phase_run attributes."""
    assert hasattr(pipeline, "_current_trace_run")
    assert hasattr(pipeline, "_current_phase_run")
    assert pipeline._current_trace_run is None
    assert pipeline._current_phase_run is None


async def test_pipeline_helper_methods_exist(pipeline):
    """Pipeline should have _start_phase, _end_phase, _set_llm_trace, _clear_llm_trace."""
    assert callable(getattr(pipeline, "_start_phase", None))
    assert callable(getattr(pipeline, "_end_phase", None))
    assert callable(getattr(pipeline, "_set_llm_trace", None))
    assert callable(getattr(pipeline, "_clear_llm_trace", None))


async def test_set_llm_trace_sets_context(pipeline):
    """_set_llm_trace should call set_trace_context on llm_provider."""
    mock_phase = MagicMock()
    pipeline._set_llm_trace(mock_phase, "pfc")
    assert pipeline._llm_provider._trace_parent is mock_phase
    assert pipeline._llm_provider._trace_region == "pfc"


async def test_clear_llm_trace_clears_context(pipeline):
    """_clear_llm_trace should clear the trace context on llm_provider."""
    mock_phase = MagicMock()
    pipeline._set_llm_trace(mock_phase, "broca")
    pipeline._clear_llm_trace()
    assert pipeline._llm_provider._trace_parent is None
    assert pipeline._llm_provider._trace_region is None


async def test_start_phase_returns_none_when_no_trace_run(pipeline):
    """_start_phase should return None if no current trace run is set."""
    pipeline._current_trace_run = None
    result = pipeline._start_phase("phase.test")
    assert result is None


async def test_start_phase_creates_child_run(pipeline):
    """_start_phase should create a child run on the current trace run."""
    mock_run = MagicMock()
    mock_child = MagicMock()
    mock_run.create_child.return_value = mock_child

    pipeline._current_trace_run = mock_run
    result = pipeline._start_phase("phase.sensory", inputs={"text": "hello"})

    assert result is mock_child
    mock_run.create_child.assert_called_once_with(
        name="phase.sensory", run_type="chain",
        inputs={"text": "hello"}, extra={},
    )
    assert pipeline._current_phase_run is mock_child


async def test_end_phase_calls_end_and_post(pipeline):
    """_end_phase should call end() and post() on the run."""
    mock_run = MagicMock()
    pipeline._end_phase(mock_run, outputs={"result": "done"})
    mock_run.end.assert_called_once_with(outputs={"result": "done"})
    mock_run.post.assert_called_once()
    assert pipeline._current_phase_run is None


async def test_end_phase_noop_when_none(pipeline):
    """_end_phase with None run should not raise."""
    pipeline._end_phase(None)  # should not raise
    assert pipeline._current_phase_run is None


async def test_multiple_requests_do_not_leak_trace_state(pipeline):
    """Subsequent requests should not carry over trace state from previous requests."""
    mock_run = MagicMock()
    mock_run.create_child.return_value = MagicMock()

    await pipeline.process_request("first", trace_run=mock_run)
    # Second request without trace_run — should have None state
    await pipeline.process_request("second", trace_run=None)

    assert pipeline._current_trace_run is None
