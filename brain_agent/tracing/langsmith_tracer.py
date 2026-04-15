"""LangSmith tracer — thin wrapper around RunTree API."""
from __future__ import annotations

from typing import Any


class LangSmithTracer:
    """Wraps LangSmith RunTree for hierarchical tracing."""

    def __init__(self, project_name: str, api_key: str | None = None):
        from langsmith import RunTree  # lazy import
        self._RunTree = RunTree
        self._project = project_name
        self._api_key = api_key

    def create_root_run(self, name: str, inputs: dict, extra: dict) -> Any:
        return self._RunTree(
            name=name,
            run_type="chain",
            inputs=inputs,
            extra=extra,
            project_name=self._project,
        )

    def create_child_run(
        self, parent: Any, name: str, run_type: str,
        inputs: dict, extra: dict | None = None,
    ) -> Any:
        return parent.create_child(
            name=name,
            run_type=run_type,
            inputs=inputs,
            extra=extra or {},
        )

    def end_run(self, run: Any | None, outputs: dict | None = None, error: str | None = None) -> None:
        if run is None:
            return
        run.end(outputs=outputs, error=error)
        run.post()
