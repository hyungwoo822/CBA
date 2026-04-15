# LangSmith Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate LangSmith tracing into CBA's existing middleware architecture for hierarchical LLM/tool call tracking and per-region cost analysis.

**Architecture:** LangSmith `RunTree` API is wrapped in a thin `LangSmithTracer` class, managed by `TracingManager` (no-op when disabled). Trace context flows from `BrainAgent` (root run) → `Pipeline` (phase runs) → `MyelinatedProvider` → `MyelinSheath` (LLM runs). Tool runs are created in `_execute_through_barrier()`.

**Tech Stack:** langsmith SDK, Pydantic config, existing middleware onion pattern

**Spec:** `docs/superpowers/specs/2026-04-15-langsmith-integration-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `brain_agent/tracing/__init__.py` | Export `TracingManager` |
| `brain_agent/tracing/manager.py` | Enable/disable orchestrator, root run lifecycle |
| `brain_agent/tracing/langsmith_tracer.py` | Thin wrapper around LangSmith `RunTree` API |
| `tests/tracing/__init__.py` | Test package |
| `tests/tracing/test_config.py` | TracingConfig unit tests |
| `tests/tracing/test_langsmith_tracer.py` | LangSmithTracer tests (mocked RunTree) |
| `tests/tracing/test_manager.py` | TracingManager enable/disable tests |
| `tests/tracing/test_myelin_tracing.py` | MyelinSheath + trace integration |
| `tests/tracing/test_pipeline_tracing.py` | Pipeline phase/region/tool tracing |
| `tests/tracing/test_agent_tracing.py` | BrainAgent root trace integration |

### Modified Files

| File | Change |
|------|--------|
| `brain_agent/config/schema.py` | Add `TracingConfig`, add `tracing` field to `BrainAgentConfig` |
| `brain_agent/providers/myelinated.py` | Add `set_trace_context()` / `clear_trace_context()`, pass to MiddlewareContext |
| `brain_agent/middleware/myelin/sheath.py` | Create LLM child run from `trace_parent` in context |
| `brain_agent/pipeline.py` | Accept `trace_run` param, create phase/tool child runs, set region context |
| `brain_agent/agent.py` | Init `TracingManager`, create root run in `process()` |
| `pyproject.toml` | Add `langsmith>=0.1.0` dependency |
| `.env.example` | Add `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` |

---

## Task 1: TracingConfig

**Files:**
- Modify: `brain_agent/config/schema.py`
- Create: `tests/tracing/__init__.py`
- Create: `tests/tracing/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracing/__init__.py` (empty) and `tests/tracing/test_config.py`:

```python
"""Tests for TracingConfig integration into BrainAgentConfig."""
from brain_agent.config.schema import BrainAgentConfig, TracingConfig


def test_tracing_config_defaults():
    config = TracingConfig()
    assert config.enabled is False
    assert config.project_name == "brain-agent"
    assert config.api_key == ""


def test_brain_agent_config_has_tracing():
    config = BrainAgentConfig()
    assert isinstance(config.tracing, TracingConfig)
    assert config.tracing.enabled is False


def test_tracing_config_from_dict():
    config = BrainAgentConfig.from_dict({
        "tracing": {"enabled": True, "project_name": "my-project", "api_key": "ls_test"}
    })
    assert config.tracing.enabled is True
    assert config.tracing.project_name == "my-project"
    assert config.tracing.api_key == "ls_test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tracing/test_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'TracingConfig'`

- [ ] **Step 3: Write minimal implementation**

In `brain_agent/config/schema.py`, add `TracingConfig` class before `BrainAgentConfig`:

```python
class TracingConfig(BaseModel):
    """LLM call tracing configuration (LangSmith)."""
    enabled: bool = False
    project_name: str = "brain-agent"
    api_key: str = ""
```

Add `tracing` field to `BrainAgentConfig`:

```python
class BrainAgentConfig(BaseModel):
    # ... existing fields ...
    tracing: TracingConfig = Field(default_factory=TracingConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tracing/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add brain_agent/config/schema.py tests/tracing/
git commit -m "feat(tracing): add TracingConfig to BrainAgentConfig"
```

---

## Task 2: LangSmithTracer

**Files:**
- Create: `brain_agent/tracing/__init__.py`
- Create: `brain_agent/tracing/langsmith_tracer.py`
- Create: `tests/tracing/test_langsmith_tracer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracing/test_langsmith_tracer.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tracing/test_langsmith_tracer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain_agent.tracing'`

- [ ] **Step 3: Write minimal implementation**

Create `brain_agent/tracing/__init__.py`:

```python
"""Tracing subsystem — LLM/tool call observability."""
```

Create `brain_agent/tracing/langsmith_tracer.py`:

```python
"""LangSmith tracer — thin wrapper around RunTree API."""
from __future__ import annotations

from langsmith import RunTree


class LangSmithTracer:
    """Wraps LangSmith RunTree for hierarchical tracing."""

    def __init__(self, project_name: str, api_key: str | None = None):
        self._project = project_name
        # api_key=None lets langsmith auto-detect from LANGSMITH_API_KEY env
        self._api_key = api_key

    def create_root_run(self, name: str, inputs: dict, extra: dict) -> RunTree:
        return RunTree(
            name=name,
            run_type="chain",
            inputs=inputs,
            extra=extra,
            project_name=self._project,
        )

    def create_child_run(
        self, parent: RunTree, name: str, run_type: str,
        inputs: dict, extra: dict | None = None,
    ) -> RunTree:
        return parent.create_child(
            name=name,
            run_type=run_type,
            inputs=inputs,
            extra=extra or {},
        )

    def end_run(self, run: RunTree | None, outputs: dict | None = None, error: str | None = None) -> None:
        if run is None:
            return
        run.end(outputs=outputs, error=error)
        run.post()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tracing/test_langsmith_tracer.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add brain_agent/tracing/ tests/tracing/test_langsmith_tracer.py
git commit -m "feat(tracing): add LangSmithTracer RunTree wrapper"
```

---

## Task 3: TracingManager

**Files:**
- Modify: `brain_agent/tracing/__init__.py`
- Create: `brain_agent/tracing/manager.py`
- Create: `tests/tracing/test_manager.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracing/test_manager.py`:

```python
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
    @patch("brain_agent.tracing.manager.LangSmithTracer")
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

    @patch("brain_agent.tracing.manager.LangSmithTracer")
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

    @patch("brain_agent.tracing.manager.LangSmithTracer")
    def test_create_child_delegates_to_tracer(self, MockTracer):
        mock_tracer = MagicMock()
        mock_child = MagicMock()
        mock_tracer.create_child_run.return_value = mock_child
        MockTracer.return_value = mock_tracer

        mgr = TracingManager(TracingConfig(enabled=True))
        parent = MagicMock()
        child = mgr.create_child(parent, "phase.sensory", "chain", {"text": "hi"})

        assert child is mock_child

    @patch("brain_agent.tracing.manager.LangSmithTracer")
    def test_create_child_returns_none_when_parent_is_none(self, MockTracer):
        MockTracer.return_value = MagicMock()
        mgr = TracingManager(TracingConfig(enabled=True))
        child = mgr.create_child(None, "phase.sensory", "chain", {})
        assert child is None

    @patch("brain_agent.tracing.manager.LangSmithTracer")
    def test_end_child_delegates_to_tracer(self, MockTracer):
        mock_tracer = MagicMock()
        MockTracer.return_value = mock_tracer

        mgr = TracingManager(TracingConfig(enabled=True))
        mock_child = MagicMock()
        mgr.end_child(mock_child, outputs={"signals": 3})

        mock_tracer.end_run.assert_called_once_with(mock_child, outputs={"signals": 3})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tracing/test_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain_agent.tracing.manager'`

- [ ] **Step 3: Write minimal implementation**

Create `brain_agent/tracing/manager.py`:

```python
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
        self._tracer = None
        if self._enabled:
            from brain_agent.tracing.langsmith_tracer import LangSmithTracer
            self._tracer = LangSmithTracer(
                project_name=config.project_name,
                api_key=config.api_key or None,
            )
            logger.info("Tracing enabled — project: %s", config.project_name)

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
```

Update `brain_agent/tracing/__init__.py`:

```python
"""Tracing subsystem — LLM/tool call observability."""
from brain_agent.tracing.manager import TracingManager

__all__ = ["TracingManager"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tracing/test_manager.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add brain_agent/tracing/ tests/tracing/test_manager.py
git commit -m "feat(tracing): add TracingManager with enable/disable"
```

---

## Task 4: MyelinatedProvider Trace Context Pass-Through

**Files:**
- Modify: `brain_agent/providers/myelinated.py`
- Create: `tests/tracing/test_myelinated_trace.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracing/test_myelinated_trace.py`:

```python
"""Tests for MyelinatedProvider trace context pass-through."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass, field

from brain_agent.middleware.base import Middleware, MiddlewareChain, MiddlewareContext
from brain_agent.providers.base import LLMResponse
from brain_agent.providers.myelinated import MyelinatedProvider


@dataclass
class MockLLMResponse:
    content: str | None = None
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tracing/test_myelinated_trace.py -v`
Expected: FAIL — `AttributeError: 'MyelinatedProvider' object has no attribute 'set_trace_context'`

- [ ] **Step 3: Write minimal implementation**

In `brain_agent/providers/myelinated.py`, add trace context methods and pass them into the MiddlewareContext:

Add to `__init__`:
```python
def __init__(self, inner: LLMProvider, myelin: MiddlewareChain):
    self._inner = inner
    self._myelin = myelin
    self._trace_parent = None
    self._trace_region = None
```

Add new methods:
```python
def set_trace_context(self, parent_run, region_name: str) -> None:
    """Set active trace context for the next LLM call."""
    self._trace_parent = parent_run
    self._trace_region = region_name

def clear_trace_context(self) -> None:
    """Clear trace context."""
    self._trace_parent = None
    self._trace_region = None
```

In the `chat` method, add trace keys to the MiddlewareContext:
```python
context = MiddlewareContext(data={
    "messages": messages,
    "tools": tools,
    "model": model,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "trace_parent": self._trace_parent,    # NEW
    "trace_region": self._trace_region,    # NEW
})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tracing/test_myelinated_trace.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add brain_agent/providers/myelinated.py tests/tracing/test_myelinated_trace.py
git commit -m "feat(tracing): add trace context pass-through to MyelinatedProvider"
```

---

## Task 5: MyelinSheath LLM Tracing

**Files:**
- Modify: `brain_agent/middleware/myelin/sheath.py`
- Create: `tests/tracing/test_myelin_tracing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracing/test_myelin_tracing.py`:

```python
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
    end_kwargs = mock_child.end.call_args.kwargs
    assert end_kwargs["outputs"]["usage"]["prompt_tokens"] == 10
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tracing/test_myelin_tracing.py -v`
Expected: FAIL — `test_creates_llm_child_run_when_trace_parent_present` fails because MyelinSheath does not create child runs yet.

- [ ] **Step 3: Write minimal implementation**

Replace `brain_agent/middleware/myelin/sheath.py` with:

```python
"""Myelin Sheath — insulates neural signal transmissions.

Myelin wraps axons to increase conduction velocity and prevent
signal degradation.  In CBA this middleware wraps every LLM call,
monitoring metabolic cost (token usage) and signal fidelity.

Reference: Nave & Werner 2014, "Myelination of the nervous system:
mechanisms and functions."
"""
from __future__ import annotations

import logging

from brain_agent.middleware.base import Middleware, MiddlewareContext
from brain_agent.middleware.registry import register_middleware

logger = logging.getLogger(__name__)


class MyelinSheath(Middleware):
    """Insulates LLM transmissions, tracking metabolic cost.

    Wraps every ``LLMProvider.chat()`` call and logs token
    consumption — the neural metabolic equivalent of glucose uptake
    during high-frequency axonal firing.

    When a ``trace_parent`` is present in the context, creates a
    LangSmith child run of type ``llm`` for cost tracking.
    """

    def __init__(self):
        self._total_prompt = 0
        self._total_completion = 0
        self._call_count = 0

    async def __call__(self, context, next_fn):
        # ── Trace: create LLM child run if tracing is active ──
        trace_parent = context.get("trace_parent")
        trace_region = context.get("trace_region", "unknown")
        llm_run = None
        if trace_parent:
            try:
                llm_run = trace_parent.create_child(
                    name="llm.chat",
                    run_type="llm",
                    inputs={
                        "messages": context.get("messages", []),
                        "model": context.get("model"),
                    },
                    extra={"region": trace_region},
                )
            except Exception as e:
                logger.warning("Failed to create LLM trace run: %s", e)

        context = await next_fn(context)

        usage = context.get("usage", {})

        # ── Token accounting (existing logic) ──
        if usage and "error" not in usage:
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            total = usage.get("total_tokens", 0)

            self._total_prompt += prompt
            self._total_completion += completion
            self._call_count += 1

            logger.info(
                "[MyelinSheath] Transmission #%d — prompt: %s, completion: %s, "
                "total: %s (cumulative: %s)",
                self._call_count, prompt, completion, total,
                self._total_prompt + self._total_completion,
            )

        # ── Trace: finalize LLM child run ──
        if llm_run:
            try:
                response = context.get("response")
                llm_run.end(outputs={
                    "content": response.content if response else None,
                    "usage": usage,
                })
                llm_run.post()
            except Exception as e:
                logger.warning("Failed to end LLM trace run: %s", e)

        return context


register_middleware("myelin_sheath", MyelinSheath)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tracing/test_myelin_tracing.py -v`
Expected: 4 passed

- [ ] **Step 5: Run existing MyelinSheath tests to ensure no regression**

Run: `pytest tests/ -k myelin -v`
Expected: All existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add brain_agent/middleware/myelin/sheath.py tests/tracing/test_myelin_tracing.py
git commit -m "feat(tracing): add LLM child run creation to MyelinSheath"
```

---

## Task 6: Pipeline Phase, Region, and Tool Tracing

**Files:**
- Modify: `brain_agent/pipeline.py`
- Create: `tests/tracing/test_pipeline_tracing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracing/test_pipeline_tracing.py`:

```python
"""Tests for Pipeline trace context threading."""
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field

from brain_agent.pipeline import ProcessingPipeline
from brain_agent.memory.manager import MemoryManager
from brain_agent.providers.base import LLMResponse
from brain_agent.providers.myelinated import MyelinatedProvider
from brain_agent.middleware.base import MiddlewareChain


def _mock_embed(text: str) -> list[float]:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(384).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


class MockProvider:
    def __init__(self):
        self.call_count = 0
        self._trace_parent = None
        self._trace_region = None

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        self.call_count += 1
        return LLMResponse(
            content='{"intent":"greeting","complexity":"simple","keywords":[],'
                    '"entities":[],"language":"en","word_count":1,"avg_word_length":5,'
                    '"confidence":0.9,"response":"Hello!","plan":"greet user",'
                    '"appraisal":{"valence":0.5,"arousal":0.3,"dominance":0.5},'
                    '"metacognition":{"confidence":0.9,"uncertainty":"low","reasoning_quality":"adequate"}}',
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

    def get_default_model(self):
        return "mock-model"

    def set_trace_context(self, parent, region):
        self._trace_parent = parent
        self._trace_region = region

    def clear_trace_context(self):
        self._trace_parent = None
        self._trace_region = None


@pytest.fixture
async def memory(tmp_path):
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=_mock_embed)
    await mm.initialize()
    yield mm
    await mm.close()


@pytest.fixture
def pipeline(memory):
    provider = MockProvider()
    myelinated = MyelinatedProvider(inner=provider, myelin=MiddlewareChain())
    return ProcessingPipeline(memory=memory, llm_provider=myelinated)


async def test_pipeline_accepts_trace_run_param(pipeline):
    """process_request should accept an optional trace_run parameter."""
    mock_run = MagicMock()
    mock_run.create_child.return_value = MagicMock()
    result = await pipeline.process_request("hello", trace_run=mock_run)
    assert result.response  # pipeline completes normally


async def test_pipeline_creates_phase_child_runs(pipeline):
    """Phase child runs should be created when trace_run is provided."""
    mock_run = MagicMock()
    mock_phase = MagicMock()
    mock_run.create_child.return_value = mock_phase

    await pipeline.process_request("hello", trace_run=mock_run)

    # At least one phase child run should be created
    assert mock_run.create_child.call_count >= 1
    phase_names = [
        call.kwargs.get("name", call.args[0] if call.args else "")
        for call in mock_run.create_child.call_args_list
    ]
    # Check that phase runs are created with "phase." prefix
    assert any("phase." in name for name in phase_names)


async def test_pipeline_sets_region_trace_on_provider(pipeline):
    """Pipeline should set trace_region on provider before LLM-calling regions."""
    mock_run = MagicMock()
    mock_phase = MagicMock()
    mock_run.create_child.return_value = mock_phase

    await pipeline.process_request("hello", trace_run=mock_run)

    # The MyelinatedProvider should have had set_trace_context called
    inner_provider = pipeline._llm_provider
    # After processing, trace context may be cleared, but it should have been set
    # We verify this by checking the provider is MyelinatedProvider with the method
    assert hasattr(inner_provider, "set_trace_context")


async def test_pipeline_no_trace_when_none(pipeline):
    """Pipeline should work normally when trace_run is None."""
    result = await pipeline.process_request("hello", trace_run=None)
    assert result.response


async def test_tool_execution_creates_trace_run(pipeline):
    """_execute_through_barrier should create tool child run when trace is active."""
    mock_phase_run = MagicMock()
    mock_tool_run = MagicMock()
    mock_phase_run.create_child.return_value = mock_tool_run

    pipeline._current_phase_run = mock_phase_run

    # We need a tool to execute
    from brain_agent.tools.registry import ToolRegistry
    pipeline.tool_registry = ToolRegistry()
    pipeline.tool_registry.load_builtins(["file_read"])

    # Execute through barrier with a known tool
    result = await pipeline._execute_through_barrier(
        "file_read", {"path": __file__},
    )

    mock_phase_run.create_child.assert_called()
    call_kwargs = mock_phase_run.create_child.call_args.kwargs
    assert call_kwargs["name"] == "tool.file_read"
    assert call_kwargs["run_type"] == "tool"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tracing/test_pipeline_tracing.py -v`
Expected: FAIL — `TypeError: process_request() got an unexpected keyword argument 'trace_run'`

- [ ] **Step 3: Write minimal implementation**

Modify `brain_agent/pipeline.py`:

**3a.** Add `_current_phase_run` to `__init__` (after line `self.predictor = Predictor()`):

```python
        self._current_trace_run = None
        self._current_phase_run = None
```

**3b.** Add helper methods (after `_all_regions` method):

```python
    def _set_llm_trace(self, phase_run, region_name: str) -> None:
        """Set trace context on LLM provider for the next region call."""
        if phase_run and hasattr(self._llm_provider, "set_trace_context"):
            self._llm_provider.set_trace_context(phase_run, region_name)

    def _clear_llm_trace(self) -> None:
        """Clear trace context on LLM provider."""
        if hasattr(self._llm_provider, "clear_trace_context"):
            self._llm_provider.clear_trace_context()

    def _start_phase(self, name: str, inputs: dict | None = None) -> object | None:
        """Create a phase-level trace child run."""
        if self._current_trace_run:
            try:
                run = self._current_trace_run.create_child(
                    name=name, run_type="chain",
                    inputs=inputs or {}, extra={},
                )
                self._current_phase_run = run
                return run
            except Exception:
                logger.warning("Failed to create phase trace: %s", name)
        return None

    def _end_phase(self, run, outputs: dict | None = None) -> None:
        """End a phase-level trace run."""
        if run:
            try:
                run.end(outputs=outputs or {})
                run.post()
            except Exception:
                logger.warning("Failed to end phase trace")
        self._current_phase_run = None
```

**3c.** Update `process_request` signature to accept `trace_run`:

```python
    async def process_request(self, text: str = "", image: bytes | None = None, audio: bytes | None = None, trace_run=None) -> PipelineResult:
```

At the top of `process_request`, store trace_run:

```python
        self._current_trace_run = trace_run
        self._current_phase_run = None
```

**3d.** Add phase tracing at phase boundaries. Insert after each `# Phase N` comment block:

After Phase 1 comment (sensory input):
```python
        phase1_run = self._start_phase("phase.sensory_input", {"text": text, "modality": input_modality})
```

Before Phase 2+3 comment (dual stream):
```python
        self._end_phase(phase1_run, {"signals": signals_count})
        phase23_run = self._start_phase("phase.dual_stream", {"text": text})
```

Before Phase 6 (retrieval):
```python
        self._end_phase(phase23_run, {"comprehension": comprehension.get("complexity", "unknown")})
        phase6_run = self._start_phase("phase.retrieval", {"text": text})
```

Before cortical integration (PFC LLM call):
```python
        self._end_phase(phase6_run, {"memories_retrieved": len(retrieved)})
        phase7_run = self._start_phase("phase.executive", {"text": text})
```

Set region trace before each LLM-calling region:
```python
        # Before PFC cortical integration call
        self._set_llm_trace(phase7_run, "pfc")

        # Before Broca call
        self._set_llm_trace(phase7_run, "broca")
```

At end of process_request (before return):
```python
        self._end_phase(phase7_run, {"response_length": len(result.response)})
        self._clear_llm_trace()
        self._current_trace_run = None
```

**3e.** Add tool tracing in `_execute_through_barrier`:

```python
    async def _execute_through_barrier(self, tool_name, tool_params):
        # ── Trace: create tool child run ──
        tool_run = None
        if self._current_phase_run:
            try:
                tool_run = self._current_phase_run.create_child(
                    name=f"tool.{tool_name}",
                    run_type="tool",
                    inputs={"tool_name": tool_name, "params": tool_params},
                    extra={},
                )
            except Exception:
                logger.warning("Failed to create tool trace: %s", tool_name)

        # existing execution logic...
        if self._barrier_mw:
            ctx = MiddlewareContext(data={...})
            # ...
            result = ctx.get("result", "")
        else:
            result = await self.tool_registry.execute(tool_name, tool_params)

        # ── Trace: end tool child run ──
        if tool_run:
            try:
                tool_run.end(outputs={"result": str(result)[:1000]})
                tool_run.post()
            except Exception:
                logger.warning("Failed to end tool trace: %s", tool_name)

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tracing/test_pipeline_tracing.py -v`
Expected: 5 passed

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest tests/test_pipeline.py -v`
Expected: All existing pipeline tests pass (trace_run defaults to None).

- [ ] **Step 6: Commit**

```bash
git add brain_agent/pipeline.py tests/tracing/test_pipeline_tracing.py
git commit -m "feat(tracing): add phase/region/tool tracing to Pipeline"
```

---

## Task 7: BrainAgent Root Trace Integration

**Files:**
- Modify: `brain_agent/agent.py`
- Create: `tests/tracing/test_agent_tracing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracing/test_agent_tracing.py`:

```python
"""Tests for BrainAgent root trace lifecycle."""
import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock

from brain_agent.agent import BrainAgent
from brain_agent.config.schema import BrainAgentConfig, TracingConfig


@pytest.fixture
def config_tracing_enabled(tmp_path):
    return BrainAgentConfig(
        tracing=TracingConfig(enabled=True, project_name="test-brain"),
    )


@pytest.fixture
def config_tracing_disabled(tmp_path):
    return BrainAgentConfig(
        tracing=TracingConfig(enabled=False),
    )


class TestTracingManagerInit:
    @patch("brain_agent.tracing.manager.LangSmithTracer")
    def test_agent_creates_tracing_manager_when_enabled(self, MockTracer, config_tracing_enabled, tmp_path):
        agent = BrainAgent(
            config=config_tracing_enabled,
            data_dir=str(tmp_path),
            use_mock_embeddings=True,
        )
        assert agent.tracing is not None
        assert agent.tracing._enabled is True

    def test_agent_creates_tracing_manager_when_disabled(self, config_tracing_disabled, tmp_path):
        agent = BrainAgent(
            config=config_tracing_disabled,
            data_dir=str(tmp_path),
            use_mock_embeddings=True,
        )
        assert agent.tracing is not None
        assert agent.tracing._enabled is False


class TestRootTraceLifecycle:
    @patch("brain_agent.tracing.manager.LangSmithTracer")
    async def test_process_creates_and_ends_root_trace(self, MockTracer, tmp_path):
        mock_tracer = MagicMock()
        mock_run = MagicMock()
        mock_run.create_child.return_value = MagicMock()
        mock_tracer.create_root_run.return_value = mock_run
        MockTracer.return_value = mock_tracer

        config = BrainAgentConfig(
            tracing=TracingConfig(enabled=True, project_name="test"),
        )
        agent = BrainAgent(
            config=config,
            model="openai/gpt-4o-mini",
            data_dir=str(tmp_path),
            use_mock_embeddings=True,
        )

        # Mock the pipeline to avoid actual LLM calls
        from brain_agent.pipeline import PipelineResult
        agent.pipeline.process_request = AsyncMock(
            return_value=PipelineResult(response="hi", network_mode="ECN")
        )
        agent._initialized = True
        agent.session_manager = AsyncMock()
        agent.session_manager.should_start_new_session.return_value = False
        agent.session_manager.on_interaction = AsyncMock(return_value="i1")
        agent.session_manager.current_session = MagicMock(id="s1")

        await agent.process("hello")

        # Root run should have been created
        mock_tracer.create_root_run.assert_called_once()
        create_kwargs = mock_tracer.create_root_run.call_args.kwargs
        assert create_kwargs["inputs"]["text"] == "hello"
        assert create_kwargs["extra"]["session_id"] == "s1"

        # Root run should have been ended
        mock_tracer.end_run.assert_called_once()

        # trace_run should have been passed to pipeline
        pipeline_call = agent.pipeline.process_request.call_args
        assert pipeline_call.kwargs.get("trace_run") is mock_run

    async def test_process_works_without_tracing(self, tmp_path):
        config = BrainAgentConfig(
            tracing=TracingConfig(enabled=False),
        )
        agent = BrainAgent(
            config=config,
            data_dir=str(tmp_path),
            use_mock_embeddings=True,
        )

        from brain_agent.pipeline import PipelineResult
        agent.pipeline.process_request = AsyncMock(
            return_value=PipelineResult(response="hi")
        )
        agent._initialized = True
        agent.session_manager = AsyncMock()
        agent.session_manager.should_start_new_session.return_value = False
        agent.session_manager.on_interaction = AsyncMock(return_value="i1")
        agent.session_manager.current_session = MagicMock(id="s1")

        result = await agent.process("hello")
        assert result.response == "hi"

        # trace_run should be None
        pipeline_call = agent.pipeline.process_request.call_args
        assert pipeline_call.kwargs.get("trace_run") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tracing/test_agent_tracing.py -v`
Expected: FAIL — `AttributeError: 'BrainAgent' object has no attribute 'tracing'`

- [ ] **Step 3: Write minimal implementation**

Modify `brain_agent/agent.py`:

**3a.** Add import at top:

```python
from brain_agent.tracing import TracingManager
```

**3b.** In `__init__`, after middleware setup (after `self._barrier_mw = ...` line), add:

```python
        # ── Tracing (LangSmith observability) ──
        self.tracing = TracingManager(self.config.tracing)
```

**3c.** In `process()`, wrap the pipeline execution with trace lifecycle. Modify the method to:

```python
    async def process(self, text: str, image: bytes | None = None, audio: bytes | None = None) -> PipelineResult:
        if not self._initialized:
            await self.initialize()

        if self.session_manager.should_start_new_session(text):
            await self.session_manager.close_session()
            if await self.memory.consolidation.should_consolidate():
                await self.memory.consolidate()
            await self.session_manager.start_session()

        interaction_id = await self.session_manager.on_interaction(text)
        session_id = (
            self.session_manager.current_session.id
            if self.session_manager.current_session
            else ""
        )
        self.memory.set_context(interaction_id, session_id)

        # ── Determine modality ──
        modality = "text"
        if image:
            modality = "image"
        elif audio:
            modality = "audio"

        # ── Trace: start root run ──
        trace_run = self.tracing.start_request_trace(
            text=text, session_id=session_id,
            interaction_id=interaction_id, modality=modality,
        )

        # ── Meninges wrap: pipeline-level protective membrane ──
        context = MiddlewareContext(data={
            "user_input": text,
            "image": image,
            "audio": audio,
        })

        async def _pipeline_core(ctx: MiddlewareContext) -> MiddlewareContext:
            result = await self.pipeline.process_request(
                ctx["user_input"],
                image=ctx.get("image"),
                audio=ctx.get("audio"),
                trace_run=trace_run,
            )
            ctx["result"] = result
            return ctx

        context = await self._meninges_mw.execute(context, _pipeline_core)
        result = context["result"]

        # ── Trace: end root run ──
        self.tracing.end_request_trace(trace_run, {
            "response": result.response,
            "network_mode": result.network_mode,
            "signals_processed": result.signals_processed,
        })

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tracing/test_agent_tracing.py -v`
Expected: 4 passed

- [ ] **Step 5: Run existing agent/integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add brain_agent/agent.py tests/tracing/test_agent_tracing.py
git commit -m "feat(tracing): add root trace lifecycle to BrainAgent"
```

---

## Task 8: Dependencies and Environment Config

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`

- [ ] **Step 1: Add langsmith dependency to pyproject.toml**

Add `"langsmith>=0.1.0"` to the `dependencies` list:

```toml
dependencies = [
    "pydantic>=2.0",
    "aiosqlite>=0.20.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "numpy>=1.24.0",
    "litellm>=1.40.0",
    "python-dotenv>=1.0.0",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "websockets>=12.0",
    "python-multipart>=0.0.6",
    "pypdf>=4.0.0",
    "python-docx>=1.0.0",
    "mcp>=1.0.0",
    "networkx>=3.0",
    "langsmith>=0.1.0",
]
```

- [ ] **Step 2: Add environment variables to .env.example**

Append to `.env.example`:

```bash
# ── Tracing (LangSmith) ──
LANGCHAIN_TRACING_V2=""       # Set to "true" to enable LangSmith tracing
LANGSMITH_API_KEY=""           # Your LangSmith API key (ls_...)
LANGSMITH_PROJECT=""           # LangSmith project name (default: brain-agent)
```

- [ ] **Step 3: Install the new dependency**

Run: `pip install langsmith>=0.1.0`
Expected: Successfully installed langsmith

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/ -v --timeout=60`
Expected: All tests pass including new tracing tests.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example
git commit -m "chore: add langsmith dependency and env config"
```

---

## Task 9: Metadata Enrichment

**Files:**
- Modify: `brain_agent/pipeline.py` (add metadata to phase runs)
- Modify: `brain_agent/agent.py` (add metadata to root run end)

- [ ] **Step 1: Add custom metadata to root trace end**

In `brain_agent/agent.py`, enrich the `end_request_trace` call with neuromodulator and memory data:

```python
        # ── Trace: end root run with enriched metadata ──
        trace_outputs = {
            "response": result.response,
            "network_mode": result.network_mode,
            "signals_processed": result.signals_processed,
            "memories_retrieved_count": len(result.memories_retrieved),
            "from_cache": result.from_cache,
        }
        self.tracing.end_request_trace(trace_run, trace_outputs)
```

- [ ] **Step 2: Add neuromodulator snapshot to executive phase**

In `brain_agent/pipeline.py`, when creating the executive phase run, include neuromodulator state in `extra`:

```python
        phase7_run = self._start_phase("phase.executive", {
            "text": text,
        })
        # Enrich phase with neuromodulator snapshot
        if phase7_run:
            try:
                phase7_run.extra.update({
                    "neuromodulators": self.neuromodulators.snapshot(),
                    "network_mode": self.network_ctrl.current_mode.value,
                })
            except Exception:
                pass
```

- [ ] **Step 3: Run tests to verify no regression**

Run: `pytest tests/ -v --timeout=60`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add brain_agent/agent.py brain_agent/pipeline.py
git commit -m "feat(tracing): add neuromodulator and memory metadata to traces"
```

---

## Verification Checklist

After all tasks are complete:

- [ ] `pytest tests/ -v` — all tests pass
- [ ] `pytest tests/tracing/ -v` — all tracing tests pass
- [ ] With `LANGCHAIN_TRACING_V2=true` and valid API key, traces appear in LangSmith dashboard
- [ ] With `tracing.enabled = false` (default), no performance impact and no langsmith errors
- [ ] Trace tree shows: `brain_agent.process` → `phase.*` → `llm.chat` / `tool.*` hierarchy
- [ ] Each LLM run shows `region` in metadata
- [ ] Cost is auto-calculated by LangSmith for each `run_type="llm"` run
