"""Tests for BrainAgent root trace lifecycle."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from brain_agent.agent import BrainAgent
from brain_agent.config.schema import BrainAgentConfig, TracingConfig


class TestTracingManagerInit:
    @patch("brain_agent.tracing.langsmith_tracer.LangSmithTracer")
    def test_agent_creates_tracing_manager_when_enabled(self, MockTracer, tmp_path):
        config = BrainAgentConfig(tracing=TracingConfig(enabled=True, project_name="test-brain"))
        agent = BrainAgent(
            config=config,
            data_dir=str(tmp_path),
            use_mock_embeddings=True,
        )
        assert agent.tracing is not None
        assert agent.tracing._enabled is True

    def test_agent_creates_tracing_manager_when_disabled(self, tmp_path):
        config = BrainAgentConfig(tracing=TracingConfig(enabled=False))
        agent = BrainAgent(
            config=config,
            data_dir=str(tmp_path),
            use_mock_embeddings=True,
        )
        assert agent.tracing is not None
        assert agent.tracing._enabled is False


def _make_agent_ready(agent):
    """Patch agent internals so process() can run without a real DB."""
    from unittest.mock import AsyncMock, MagicMock
    agent._initialized = True
    agent.session_manager = MagicMock()
    agent.session_manager.should_start_new_session.return_value = False
    agent.session_manager.on_interaction = AsyncMock(return_value="i1")
    agent.session_manager.current_session = MagicMock(id="s1")
    agent.memory.set_context = MagicMock()
    agent.memory.consolidation.should_consolidate = AsyncMock(return_value=False)


class TestRootTraceLifecycle:
    @patch("brain_agent.tracing.langsmith_tracer.LangSmithTracer")
    @pytest.mark.asyncio
    async def test_process_creates_and_ends_root_trace(self, MockTracer, tmp_path):
        mock_tracer = MagicMock()
        mock_run = MagicMock()
        mock_run.create_child.return_value = MagicMock()
        mock_tracer.create_root_run.return_value = mock_run
        MockTracer.return_value = mock_tracer

        config = BrainAgentConfig(tracing=TracingConfig(enabled=True, project_name="test"))
        agent = BrainAgent(
            config=config, model="openai/gpt-4o-mini",
            data_dir=str(tmp_path), use_mock_embeddings=True,
        )

        from brain_agent.pipeline import PipelineResult
        agent.pipeline.process_request = AsyncMock(
            return_value=PipelineResult(response="hi", network_mode="ECN")
        )
        _make_agent_ready(agent)

        await agent.process("hello")

        mock_tracer.create_root_run.assert_called_once()
        create_kwargs = mock_tracer.create_root_run.call_args.kwargs
        assert create_kwargs["inputs"]["text"] == "hello"
        assert create_kwargs["extra"]["session_id"] == "s1"

        mock_tracer.end_run.assert_called_once()

        pipeline_call = agent.pipeline.process_request.call_args
        assert pipeline_call.kwargs.get("trace_run") is mock_run

    @pytest.mark.asyncio
    async def test_process_works_without_tracing(self, tmp_path):
        config = BrainAgentConfig(tracing=TracingConfig(enabled=False))
        agent = BrainAgent(
            config=config, data_dir=str(tmp_path), use_mock_embeddings=True,
        )

        from brain_agent.pipeline import PipelineResult
        agent.pipeline.process_request = AsyncMock(
            return_value=PipelineResult(response="hi")
        )
        _make_agent_ready(agent)

        result = await agent.process("hello")
        assert result.response == "hi"

        pipeline_call = agent.pipeline.process_request.call_args
        assert pipeline_call.kwargs.get("trace_run") is None
