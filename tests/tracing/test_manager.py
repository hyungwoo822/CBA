"""Tests for TracingManager — enable/disable orchestrator."""
from unittest.mock import patch, MagicMock

from brain_agent.config.schema import TracingConfig
from brain_agent.tracing.manager import TracingManager


class TestDisabled:
    def test_start_returns_none(self):
        mgr = TracingManager(TracingConfig(enabled=False))
        run = mgr.start_request_trace("hello", "s1", "i1", "text")
        assert run is None

    def test_end_is_noop(self):
        mgr = TracingManager(TracingConfig(enabled=False))
        mgr.end_request_trace(None, None)  # should not raise

    def test_create_child_returns_none(self):
        mgr = TracingManager(TracingConfig(enabled=False))
        child = mgr.create_child(None, "phase.sensory", "chain", {})
        assert child is None

    def test_end_child_is_noop(self):
        mgr = TracingManager(TracingConfig(enabled=False))
        mgr.end_child(None)  # should not raise

    def test_does_not_import_langsmith(self):
        mgr = TracingManager(TracingConfig(enabled=False))
        assert mgr._tracer is None


class TestEnabled:
    @patch("brain_agent.tracing.langsmith_tracer.LangSmithTracer")
    def test_start_creates_root_run(self, MockTracer):
        mock_tracer = MagicMock()
        mock_run = MagicMock()
        mock_tracer.create_root_run.return_value = mock_run
        MockTracer.return_value = mock_tracer

        mgr = TracingManager(TracingConfig(enabled=True, project_name="test"))
        run = mgr.start_request_trace("hello", "s1", "i1", "text")

        assert run is mock_run
        mock_tracer.create_root_run.assert_called_once_with(
            name="brain_agent.process",
            inputs={"text": "hello", "modality": "text"},
            extra={"session_id": "s1", "interaction_id": "i1"},
        )

    @patch("brain_agent.tracing.langsmith_tracer.LangSmithTracer")
    def test_end_finalizes_root_run(self, MockTracer):
        mock_tracer = MagicMock()
        MockTracer.return_value = mock_tracer
        mock_run = MagicMock()

        mgr = TracingManager(TracingConfig(enabled=True))
        mgr.end_request_trace(mock_run, {"response": "hi", "network_mode": "ECN"})

        mock_tracer.end_run.assert_called_once_with(
            mock_run,
            outputs={"response": "hi", "network_mode": "ECN"},
        )

    @patch("brain_agent.tracing.langsmith_tracer.LangSmithTracer")
    def test_create_child_delegates_to_tracer(self, MockTracer):
        mock_tracer = MagicMock()
        mock_child = MagicMock()
        mock_tracer.create_child_run.return_value = mock_child
        MockTracer.return_value = mock_tracer

        mgr = TracingManager(TracingConfig(enabled=True))
        parent = MagicMock()
        child = mgr.create_child(parent, "phase.sensory", "chain", {"text": "hi"})

        assert child is mock_child

    @patch("brain_agent.tracing.langsmith_tracer.LangSmithTracer")
    def test_create_child_returns_none_when_parent_is_none(self, MockTracer):
        MockTracer.return_value = MagicMock()
        mgr = TracingManager(TracingConfig(enabled=True))
        child = mgr.create_child(None, "phase.sensory", "chain", {})
        assert child is None

    @patch("brain_agent.tracing.langsmith_tracer.LangSmithTracer")
    def test_end_child_delegates_to_tracer(self, MockTracer):
        mock_tracer = MagicMock()
        MockTracer.return_value = mock_tracer

        mgr = TracingManager(TracingConfig(enabled=True))
        mock_child = MagicMock()
        mgr.end_child(mock_child, outputs={"signals": 3})

        mock_tracer.end_run.assert_called_once_with(mock_child, outputs={"signals": 3}, error=None)
