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

    def test_does_not_import_tracer(self):
        mgr = TracingManager(TracingConfig(enabled=False))
        assert mgr._tracer is None


class TestEnabledLangSmith:
    @patch("brain_agent.tracing.langsmith_tracer.LangSmithTracer")
    def test_start_creates_root_run(self, MockTracer):
        mock_tracer = MagicMock()
        mock_run = MagicMock()
        mock_tracer.create_root_run.return_value = mock_run
        MockTracer.return_value = mock_tracer

        mgr = TracingManager(TracingConfig(enabled=True, provider="langsmith", project_name="test"))
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

        mgr = TracingManager(TracingConfig(enabled=True, provider="langsmith"))
        mgr.end_request_trace(mock_run, {"response": "hi", "network_mode": "ECN"})

        mock_tracer.end_run.assert_called_once_with(
            mock_run,
            outputs={"response": "hi", "network_mode": "ECN"},
        )


class TestEnabledLangFuse:
    @patch("brain_agent.tracing.langfuse_tracer.LangFuseTracer")
    def test_langfuse_is_default_provider(self, MockTracer):
        mock_tracer = MagicMock()
        MockTracer.return_value = mock_tracer

        mgr = TracingManager(TracingConfig(enabled=True, project_name="test"))
        assert mgr._provider == "langfuse"
        MockTracer.assert_called_once()

    @patch("brain_agent.tracing.langfuse_tracer.LangFuseTracer")
    def test_start_creates_root_run(self, MockTracer):
        mock_tracer = MagicMock()
        mock_run = MagicMock()
        mock_tracer.create_root_run.return_value = mock_run
        MockTracer.return_value = mock_tracer

        mgr = TracingManager(TracingConfig(enabled=True, project_name="test"))
        run = mgr.start_request_trace("hello", "s1", "i1", "text")

        assert run is mock_run

    @patch("brain_agent.tracing.langfuse_tracer.LangFuseTracer")
    def test_create_child_returns_none_when_parent_is_none(self, MockTracer):
        MockTracer.return_value = MagicMock()
        mgr = TracingManager(TracingConfig(enabled=True))
        child = mgr.create_child(None, "phase.sensory", "chain", {})
        assert child is None


class TestProviderSelection:
    @patch("brain_agent.tracing.langsmith_tracer.LangSmithTracer")
    def test_selects_langsmith(self, MockTracer):
        MockTracer.return_value = MagicMock()
        mgr = TracingManager(TracingConfig(enabled=True, provider="langsmith"))
        assert mgr._provider == "langsmith"
        MockTracer.assert_called_once()

    @patch("brain_agent.tracing.langfuse_tracer.LangFuseTracer")
    def test_selects_langfuse(self, MockTracer):
        MockTracer.return_value = MagicMock()
        mgr = TracingManager(TracingConfig(enabled=True, provider="langfuse"))
        assert mgr._provider == "langfuse"
        MockTracer.assert_called_once()
