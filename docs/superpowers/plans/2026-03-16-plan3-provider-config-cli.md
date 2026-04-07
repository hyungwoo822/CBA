# Plan 3: Provider + Config + CLI + BrainAgent API

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LLM provider abstraction, Pydantic config, tool registry, CLI commands, and the public BrainAgent API class — making the framework usable end-to-end.

**Architecture:** Adapted from nanobot's patterns. LLMProvider ABC with LiteLLM default implementation. PFC's stub replaced with real LLM calls. BrainAgent class as the public facade.

**Tech Stack:** Python 3.11+, litellm, pydantic v2, typer

---

## File Structure

```
brain_agent/
├── providers/
│   ├── __init__.py
│   ├── base.py             # LLMProvider ABC, LLMResponse, ToolCallRequest
│   └── litellm_provider.py # LiteLLM-based universal provider
├── tools/
│   ├── __init__.py
│   ├── base.py             # Tool ABC
│   └── registry.py         # ToolRegistry
├── config/
│   ├── __init__.py
│   └── schema.py           # Pydantic config models
├── agent.py                # BrainAgent — public API class
└── cli/
    ├── __init__.py
    └── commands.py          # CLI entry points
```

---

## Chunk 1: Provider + Tools + Config

### Task 1: LLM Provider Base + LiteLLM

### Task 2: Tool Base + Registry

### Task 3: Config Schema

## Chunk 2: BrainAgent API + CLI

### Task 4: BrainAgent Class (public facade)

### Task 5: CLI Commands + __main__.py
