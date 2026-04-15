"""TracingManager — orchestrates LLM/tool tracing lifecycle."""
from __future__ import annotations

import logging
from typing import Any

from brain_agent.config.schema import TracingConfig

logger = logging.getLogger(__name__)


class TracingManager:
    """Manages trace lifecycle. No-op when disabled.

    Callers use ``if run:`` to skip tracing — no separate flag checks needed.
    """

    def __init__(self, config: TracingConfig):
        self._enabled = config.enabled
        self._provider = config.provider
        self._tracer = None
        if self._enabled:
            if config.provider == "langsmith":
                from brain_agent.tracing.langsmith_tracer import LangSmithTracer
                self._tracer = LangSmithTracer(
                    project_name=config.project_name,
                    api_key=config.api_key or None,
                )
            else:  # langfuse (default)
                from brain_agent.tracing.langfuse_tracer import LangFuseTracer
                self._tracer = LangFuseTracer(
                    project_name=config.project_name,
                    api_key=config.api_key or None,
                )
            logger.info("Tracing enabled — provider: %s, project: %s", config.provider, config.project_name)

    def start_request_trace(
        self, text: str, session_id: str, interaction_id: str, modality: str,
    ) -> Any:
        """Create root trace run for a user request. Returns None if disabled."""
        if not self._tracer:
            return None
        return self._tracer.create_root_run(
            name="brain_agent.process",
            inputs={"text": text, "modality": modality},
            extra={"session_id": session_id, "interaction_id": interaction_id},
        )

    def end_request_trace(self, run: Any, result: dict | None) -> None:
        """Finalize and post root trace run."""
        if not self._tracer or run is None:
            return
        self._tracer.end_run(run, outputs=result)

    def create_child(
        self, parent: Any, name: str, run_type: str,
        inputs: dict, extra: dict | None = None,
    ) -> Any:
        """Create a child run under parent. Returns None if parent is None or disabled."""
        if not self._tracer or parent is None:
            return None
        return self._tracer.create_child_run(
            parent=parent, name=name, run_type=run_type,
            inputs=inputs, extra=extra,
        )

    def end_child(self, run: Any, outputs: dict | None = None, error: str | None = None) -> None:
        """End and post a child run."""
        if not self._tracer or run is None:
            return
        self._tracer.end_run(run, outputs=outputs, error=error)
