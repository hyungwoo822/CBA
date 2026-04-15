"""Tests for MyelinatedProvider trace context pass-through."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass, field

from brain_agent.middleware.base import Middleware, MiddlewareChain, MiddlewareContext
from brain_agent.providers.base import LLMResponse
from brain_agent.providers.myelinated import MyelinatedProvider


class MockProvider:
    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        return LLMResponse(content="mock", usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8})

    def get_default_model(self):
        return "mock"


class SpyMiddleware(Middleware):
    """Captures context data passed through the middleware chain."""
    def __init__(self):
        self.captured = {}

    async def __call__(self, context, next_fn):
        self.captured["trace_parent"] = context.get("trace_parent")
        self.captured["trace_region"] = context.get("trace_region")
        return await next_fn(context)


async def test_set_trace_context_passes_to_middleware():
    spy = SpyMiddleware()
    chain = MiddlewareChain([spy])
    provider = MyelinatedProvider(inner=MockProvider(), myelin=chain)

    mock_run = MagicMock()
    provider.set_trace_context(mock_run, "wernicke")

    await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert spy.captured["trace_parent"] is mock_run
    assert spy.captured["trace_region"] == "wernicke"


async def test_clear_trace_context():
    spy = SpyMiddleware()
    chain = MiddlewareChain([spy])
    provider = MyelinatedProvider(inner=MockProvider(), myelin=chain)

    provider.set_trace_context(MagicMock(), "pfc")
    provider.clear_trace_context()

    await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert spy.captured["trace_parent"] is None
    assert spy.captured["trace_region"] is None


async def test_default_trace_context_is_none():
    spy = SpyMiddleware()
    chain = MiddlewareChain([spy])
    provider = MyelinatedProvider(inner=MockProvider(), myelin=chain)

    await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert spy.captured["trace_parent"] is None
    assert spy.captured["trace_region"] is None
