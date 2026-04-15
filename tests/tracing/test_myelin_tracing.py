"""Tests for MyelinSheath LangSmith LLM tracing."""
import pytest
from unittest.mock import MagicMock

from brain_agent.middleware.base import MiddlewareContext
from brain_agent.middleware.myelin.sheath import MyelinSheath
from brain_agent.providers.base import LLMResponse


def _make_context(trace_parent=None, trace_region=None):
    return MiddlewareContext(data={
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "trace_parent": trace_parent,
        "trace_region": trace_region,
    })


async def _mock_next_fn(ctx):
    ctx["response"] = LLMResponse(
        content="hello",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    ctx["usage"] = ctx["response"].usage
    return ctx


async def test_creates_llm_child_run_when_trace_parent_present():
    mock_parent = MagicMock()
    mock_child = MagicMock()
    mock_child.extra = {}  # real dict so usage_metadata can be set
    mock_parent.create_child.return_value = mock_child

    sheath = MyelinSheath()
    ctx = _make_context(trace_parent=mock_parent, trace_region="wernicke")

    await sheath(ctx, _mock_next_fn)

    mock_parent.create_child.assert_called_once()
    kwargs = mock_parent.create_child.call_args.kwargs
    assert kwargs["name"] == "llm.chat"
    assert kwargs["run_type"] == "llm"
    assert kwargs["extra"]["region"] == "wernicke"
    assert "messages" in kwargs["inputs"]

    mock_child.end.assert_called_once()
    assert mock_child.extra["usage_metadata"]["input_tokens"] == 10
    mock_child.post.assert_called_once()


async def test_skips_tracing_when_no_trace_parent():
    sheath = MyelinSheath()
    ctx = _make_context(trace_parent=None)

    result = await sheath(ctx, _mock_next_fn)

    # Should still process normally — just no tracing
    assert result["usage"]["total_tokens"] == 15
    assert sheath._call_count == 1


async def test_tracing_does_not_break_on_llm_error():
    mock_parent = MagicMock()
    mock_child = MagicMock()
    mock_parent.create_child.return_value = mock_child

    sheath = MyelinSheath()
    ctx = _make_context(trace_parent=mock_parent, trace_region="pfc")

    async def error_next(c):
        c["usage"] = {"error": "LLM failed"}
        return c

    result = await sheath(ctx, error_next)

    # Tracing child should still end (with error info)
    mock_child.end.assert_called_once()
    mock_child.post.assert_called_once()


async def test_includes_model_in_llm_run_inputs():
    mock_parent = MagicMock()
    mock_child = MagicMock()
    mock_parent.create_child.return_value = mock_child

    sheath = MyelinSheath()
    ctx = _make_context(trace_parent=mock_parent, trace_region="broca")

    await sheath(ctx, _mock_next_fn)

    inputs = mock_parent.create_child.call_args.kwargs["inputs"]
    assert inputs["model"] == "gpt-4o-mini"
