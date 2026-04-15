"""LangFuse tracer — adapts LangFuse SDK to RunTree-compatible interface."""
from __future__ import annotations

from typing import Any


class LangFuseRunNode:
    """Wraps a LangFuse observation to match RunTree interface.

    MyelinSheath and Pipeline call create_child / end / post on trace
    objects.  This adapter translates those calls to LangFuse's
    start_observation / update / end API so the rest of the codebase
    stays provider-agnostic.
    """

    def __init__(self, obj: Any):
        self._obj = obj
        self.extra: dict = {}

    def create_child(
        self, name: str, run_type: str, inputs: dict, extra: dict | None = None,
    ) -> LangFuseRunNode:
        metadata = extra or {}
        if run_type == "llm":
            # Extract model name from metadata if available
            model = metadata.get("metadata", {}).get("ls_model_name", None)
            child = self._obj.start_observation(
                name=name, as_type="generation",
                input=inputs, metadata=metadata,
                model=model,
            )
        elif run_type == "tool":
            child = self._obj.start_observation(
                name=name, as_type="tool",
                input=inputs, metadata=metadata,
            )
        else:
            child = self._obj.start_observation(
                name=name, as_type="span",
                input=inputs, metadata=metadata,
            )
        return LangFuseRunNode(child)

    def end(self, outputs: dict | None = None, error: str | None = None) -> None:
        update_kwargs: dict[str, Any] = {}
        if outputs:
            update_kwargs["output"] = outputs
        if error:
            update_kwargs["status_message"] = error
            update_kwargs["level"] = "ERROR"

        # Extract usage and model from extra.metadata (set by MyelinSheath)
        meta = self.extra.get("metadata", {})
        usage = meta.get("usage_metadata")
        model = meta.get("ls_model_name")

        if usage:
            update_kwargs["usage_details"] = usage
        if model:
            update_kwargs["model"] = model

        if update_kwargs:
            self._obj.update(**update_kwargs)
        self._obj.end()

    def post(self) -> None:
        # LangFuse auto-flushes, no explicit post needed
        pass


class LangFuseTracer:
    """Manages LangFuse client and creates trace hierarchies."""

    def __init__(self, project_name: str, api_key: str | None = None):
        from langfuse import Langfuse
        # LangFuse reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
        # LANGFUSE_HOST from environment if not provided.
        self._langfuse = Langfuse()
        self._project = project_name

    def create_root_run(self, name: str, inputs: dict, extra: dict) -> LangFuseRunNode:
        obs = self._langfuse.start_observation(
            name=name, as_type="span",
            input=inputs, metadata=extra,
        )
        return LangFuseRunNode(obs)

    def create_child_run(
        self, parent: LangFuseRunNode, name: str, run_type: str,
        inputs: dict, extra: dict | None = None,
    ) -> LangFuseRunNode:
        return parent.create_child(
            name=name, run_type=run_type,
            inputs=inputs, extra=extra,
        )

    def end_run(self, run: LangFuseRunNode | None, outputs: dict | None = None, error: str | None = None) -> None:
        if run is None:
            return
        run.end(outputs=outputs, error=error)

    def shutdown(self) -> None:
        """Flush pending events and shutdown."""
        self._langfuse.flush()
        self._langfuse.shutdown()
