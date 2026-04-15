# LangFuse Integration Design

**Date:** 2026-04-15
**Status:** Approved
**Goal:** LangFuse를 기본 tracing provider로 추가, LangSmith도 유지

## Architecture

`LangFuseRunNode` 어댑터가 LangFuse 객체를 RunTree 인터페이스(`create_child`, `end`, `post`, `extra`)로 래핑. MyelinSheath, Pipeline, BrainAgent 변경 없음.

```
TracingManager
  ├─ provider="langfuse" (default) → LangFuseTracer → LangFuseRunNode
  └─ provider="langsmith"          → LangSmithTracer → RunTree
```

## Files

| File | Change |
|------|--------|
| `brain_agent/tracing/langfuse_tracer.py` | **NEW** — LangFuseTracer + LangFuseRunNode |
| `brain_agent/tracing/manager.py` | Select tracer by `config.provider` |
| `brain_agent/config/schema.py` | Default provider → `"langfuse"` |
| `pyproject.toml` | Add `langfuse>=2.0` |
| `.env.example` | Add LangFuse env vars |
| `tests/tracing/test_langfuse_tracer.py` | **NEW** — LangFuseTracer + adapter tests |
| `tests/tracing/test_manager.py` | Update for provider selection |

## LangFuseRunNode Adapter

Maps RunTree interface to LangFuse SDK:
- `create_child(run_type="llm")` → `obj.generation()`
- `create_child(run_type="chain"|"tool")` → `obj.span()`
- `end(outputs)` → reads `self.extra.metadata` for usage/model, calls `obj.end()`
- `post()` → no-op (LangFuse auto-flushes)

## Config

```python
class TracingConfig(BaseModel):
    enabled: bool = True
    provider: str = "langfuse"  # "langfuse" | "langsmith"
    project_name: str = "CBA"
    api_key: str = ""
```

## Environment

```bash
# LangFuse
LANGFUSE_PUBLIC_KEY="pk-..."
LANGFUSE_SECRET_KEY="sk-..."
LANGFUSE_HOST="https://cloud.langfuse.com"  # or self-hosted
```
