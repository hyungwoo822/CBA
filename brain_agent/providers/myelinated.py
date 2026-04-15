"""Myelinated Provider — wraps an LLM provider with myelin-sheath middleware.

Just as myelin insulates axons to increase conduction velocity and
prevent signal degradation, this wrapper intercepts every LLM call
and runs it through the myelin middleware chain (token counting,
rate limiting, etc.) without modifying the underlying provider.

No brain region needs to change — swapping a bare provider for a
myelinated one is transparent.
"""
from __future__ import annotations

from typing import Any

from brain_agent.middleware.base import MiddlewareChain, MiddlewareContext
from brain_agent.providers.base import LLMProvider, LLMResponse


class MyelinatedProvider(LLMProvider):
    """LLM provider insulated with a myelin middleware chain.

    Delegates to the inner provider while routing every ``chat()``
    call through the middleware onion.
    """

    def __init__(self, inner: LLMProvider, myelin: MiddlewareChain):
        self._inner = inner
        self._myelin = myelin
        self._trace_parent = None
        self._trace_region = None

    def get_default_model(self) -> str:
        return self._inner.get_default_model()

    def set_trace_context(self, parent_run, region_name: str) -> None:
        """Set active trace context for the next LLM call."""
        self._trace_parent = parent_run
        self._trace_region = region_name

    def clear_trace_context(self) -> None:
        """Clear trace context."""
        self._trace_parent = None
        self._trace_region = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        # Resolve model name — regions call chat(model=None),
        # but the actual model is on the inner provider.
        resolved_model = model or self._inner.get_default_model()
        context = MiddlewareContext(data={
            "messages": messages,
            "tools": tools,
            "model": resolved_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "trace_parent": self._trace_parent,
            "trace_region": self._trace_region,
        })

        async def core(ctx: MiddlewareContext) -> MiddlewareContext:
            response = await self._inner.chat(
                messages=ctx["messages"],
                tools=ctx.get("tools"),
                model=ctx.get("model"),
                max_tokens=ctx.get("max_tokens", max_tokens),
                temperature=ctx.get("temperature", temperature),
            )
            ctx["response"] = response
            ctx["usage"] = response.usage
            return ctx

        context = await self._myelin.execute(context, core)
        return context.get("response", LLMResponse(content=None, finish_reason="error"))
