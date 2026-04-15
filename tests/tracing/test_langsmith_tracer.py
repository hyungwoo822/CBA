"""Tests for LangSmithTracer — thin RunTree wrapper."""
from unittest.mock import patch, MagicMock, call

from brain_agent.tracing.langsmith_tracer import LangSmithTracer


class TestCreateRootRun:
    @patch("brain_agent.tracing.langsmith_tracer.RunTree")
    def test_creates_run_tree_with_correct_params(self, MockRunTree):
        mock_run = MagicMock()
        MockRunTree.return_value = mock_run

        tracer = LangSmithTracer(project_name="test-project")
        run = tracer.create_root_run(
            name="brain_agent.process",
            inputs={"text": "hello"},
            extra={"session_id": "s1"},
        )

        MockRunTree.assert_called_once_with(
            name="brain_agent.process",
            run_type="chain",
            inputs={"text": "hello"},
            extra={"session_id": "s1"},
            project_name="test-project",
        )
        assert run is mock_run


class TestCreateChildRun:
    @patch("brain_agent.tracing.langsmith_tracer.RunTree")
    def test_creates_child_on_parent(self, MockRunTree):
        parent = MagicMock()
        child = MagicMock()
        parent.create_child.return_value = child

        tracer = LangSmithTracer(project_name="test-project")
        result = tracer.create_child_run(
            parent=parent,
            name="llm.chat",
            run_type="llm",
            inputs={"messages": [{"role": "user", "content": "hi"}]},
            extra={"region": "wernicke"},
        )

        parent.create_child.assert_called_once_with(
            name="llm.chat",
            run_type="llm",
            inputs={"messages": [{"role": "user", "content": "hi"}]},
            extra={"region": "wernicke"},
        )
        assert result is child


class TestEndRun:
    def test_ends_and_posts_run(self):
        mock_run = MagicMock()
        tracer = LangSmithTracer(project_name="test-project")

        tracer.end_run(mock_run, outputs={"content": "hello"})

        mock_run.end.assert_called_once_with(outputs={"content": "hello"}, error=None)
        mock_run.post.assert_called_once()

    def test_ends_with_error(self):
        mock_run = MagicMock()
        tracer = LangSmithTracer(project_name="test-project")

        tracer.end_run(mock_run, error="LLM call failed")

        mock_run.end.assert_called_once_with(outputs=None, error="LLM call failed")
        mock_run.post.assert_called_once()

    def test_noop_on_none_run(self):
        tracer = LangSmithTracer(project_name="test-project")
        tracer.end_run(None)  # should not raise
