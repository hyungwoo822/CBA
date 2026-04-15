# LangSmith Integration Design

**Date:** 2026-04-15
**Status:** Approved
**Goal:** LLM/tool call 트레이싱, Region별 비용 추적을 LangSmith로 구현

---

## 1. Overview

CBA의 기존 3-layer neural sheath 미들웨어(Meninges/Myelin/Barrier)에 LangSmith SDK를 통합하여, 모든 LLM 호출, 도구 실행, 파이프라인 단계를 계층적으로 트레이싱한다.

### 추적 대상

| 레벨 | 대상 | LangSmith run_type |
|------|------|--------------------|
| 요청 | `BrainAgent.process()` 1회 호출 | `chain` |
| Phase | 7-phase pipeline 각 단계 | `chain` |
| Region | LLM 호출하는 brain region (Wernicke, Amygdala, PFC, Broca, V1, A1) | `chain` |
| LLM | `litellm.acompletion()` 호출 | `llm` |
| Tool | `ToolRegistry.execute()` 호출 | `tool` |

### 비용 집계 수준

- **요청 단위**: root run의 자식 합산 (LangSmith 자동)
- **Region 단위**: `extra.region` 메타데이터 필터
- **세션 단위**: `extra.session_id` 메타데이터 그룹핑

비용 계산은 LangSmith 플랫폼에 위임. CBA에 별도 비용 로직 없음.

---

## 2. Trace Tree Structure

```
brain_agent.process [chain] ─── session_id, interaction_id, input text
├─ phase.sensory_input [chain]
│  └─ region.visual_cortex [chain]  (이미지 입력 시)
│     └─ llm.chat [llm] ─── model, tokens, cost
├─ phase.dual_stream [chain]
│  └─ region.wernicke [chain]
│     └─ llm.chat [llm]
├─ phase.integration [chain]
│  └─ region.amygdala [chain]
│     ├─ llm.chat [llm] ─── right hemisphere (fast appraisal)
│     └─ llm.chat [llm] ─── left hemisphere (contextual)
├─ phase.retrieval [chain]
│  └─ region.pfc [chain]
│     ├─ llm.chat [llm] ─── reasoning
│     ├─ tool.web_fetch [tool] ─── params, result
│     └─ llm.chat [llm] ─── tool result processing
├─ phase.production [chain]
│  └─ region.broca [chain]
│     └─ llm.chat [llm] ─── language formulation
├─ phase.encoding [chain]
│  └─ post_synaptic_consolidation [chain]
│     └─ llm.chat [llm] ─── fact extraction + refinement
└─ metadata: {network_mode, neuromodulators, memories_retrieved}
```

---

## 3. Architecture

### 3.1 New Files

```
brain_agent/tracing/
├─ __init__.py          # TracingManager export
├─ manager.py           # TracingManager: init, root run, enable/disable
└─ langsmith_tracer.py  # LangSmith RunTree wrapper
```

### 3.2 TracingManager (`manager.py`)

```python
class TracingManager:
    """Manages LangSmith tracing lifecycle.
    
    When disabled, all methods return None — callers use
    `if run:` to skip tracing with zero overhead.
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

    def start_request_trace(self, text, session_id, interaction_id, modality) -> RunTree | None:
        if not self._tracer:
            return None
        return self._tracer.create_root_run(...)

    def end_request_trace(self, run, result) -> None:
        if run:
            self._tracer.end_run(run, outputs=...)
```

### 3.3 LangSmithTracer (`langsmith_tracer.py`)

```python
from langsmith import RunTree

class LangSmithTracer:
    """Thin wrapper around LangSmith RunTree API."""
    
    def __init__(self, project_name: str, api_key: str | None = None):
        self._project = project_name
        self._api_key = api_key

    def create_root_run(self, name, inputs, extra) -> RunTree:
        return RunTree(
            name=name, run_type="chain",
            inputs=inputs, extra=extra,
            project_name=self._project,
        )

    def create_child_run(self, parent, name, run_type, inputs, extra=None) -> RunTree:
        return parent.create_child(
            name=name, run_type=run_type,
            inputs=inputs, extra=extra or {},
        )

    def end_run(self, run, outputs=None, error=None):
        run.end(outputs=outputs, error=error)
        run.post()
```

---

## 4. Integration Points

### 4.1 Config (`config/schema.py`)

```python
class TracingConfig(BaseModel):
    enabled: bool = False
    provider: str = "langsmith"
    project_name: str = "brain-agent"
    api_key: str = ""
```

`BrainAgentConfig`에 `tracing: TracingConfig` 필드 추가.

### 4.2 BrainAgent (`agent.py`)

- `__init__`: `TracingManager(self.config.tracing)` 생성
- `process()`: root run 생성 → `context["trace_run"]`에 주입 → pipeline 실행 → root run 종료

```python
async def process(self, text, image=None, audio=None):
    trace_run = self.tracing.start_request_trace(
        text=text, session_id=session_id,
        interaction_id=interaction_id,
        modality=input_modality,
    )
    context["trace_run"] = trace_run

    # ... pipeline execution ...

    self.tracing.end_request_trace(trace_run, result)
    return result
```

### 4.3 Pipeline (`pipeline.py`)

`process_request()` 내 각 Phase 블록에서 child run 생성:

```python
trace_run = self._get_trace_run()  # pipeline에 전달된 trace context

# Phase 1
phase_run = trace_run.create_child(...) if trace_run else None
# ... phase logic ...
if phase_run:
    phase_run.end(outputs={...})
    phase_run.post()
```

Region이 LLM을 호출하기 전에 `context["trace_region"]` 설정:

```python
context["trace_region"] = "wernicke"
context["trace_phase_run"] = phase_run  # MyelinSheath에서 부모로 사용
```

### 4.4 MyelinSheath (`middleware/myelin/sheath.py`)

기존 토큰 트래킹 로직 뒤에 LangSmith child run 추가:

```python
async def __call__(self, context, next_fn):
    # trace parent: phase_run > trace_run > None (fallback chain)
    parent_run = context.get("trace_phase_run") or context.get("trace_run")
    region = context.get("trace_region", "unknown")
    
    if parent_run:
        llm_run = parent_run.create_child(
            name="llm.chat",
            run_type="llm",
            inputs={"messages": context["messages"], "model": context.get("model")},
            extra={"region": region},
        )

    context = await next_fn(context)

    # existing token tracking...
    usage = context.get("usage", {})
    
    if parent_run and llm_run:
        response = context.get("response")
        llm_run.end(outputs={
            "content": response.content if response else None,
            "usage": usage,
        })
        llm_run.post()

    return context
```

### 4.5 Tool Execution (`pipeline.py` `_execute_through_barrier()`)

```python
async def _execute_through_barrier(self, tool_name, tool_params):
    parent_run = self._current_phase_run  # 현재 phase의 run
    if parent_run:
        tool_run = parent_run.create_child(
            name=f"tool.{tool_name}",
            run_type="tool",
            inputs={"tool_name": tool_name, "params": tool_params},
        )

    result = ...  # existing execution

    if parent_run and tool_run:
        tool_run.end(outputs={"result": str(result)[:1000]})
        tool_run.post()
    return result
```

---

## 5. Custom Metadata

각 run의 `extra` 필드에 CBA 고유 메타데이터 태깅:

| 키 | 레벨 | 값 |
|----|------|-----|
| `session_id` | root | 세션 ID |
| `interaction_id` | root | 인터랙션 ID |
| `network_mode` | root | DMN / ECN / Creative |
| `region` | llm | wernicke / amygdala / pfc / broca / visual_cortex / auditory_cortex |
| `neuromodulators` | root | {dopamine, serotonin, norepinephrine, ...} |
| `memories_retrieved` | root | 검색된 메모리 수 |
| `complexity` | root | fast / standard / full |
| `prediction_surprise` | root | 0.0 ~ 1.0 |

---

## 6. Toggle & Performance

### Toggle

- `config.tracing.enabled = False` (기본값) → `langsmith` import 안 함, 모든 메서드 None 반환
- Pipeline/MyelinSheath: `if parent_run:` 한 줄로 분기
- 환경변수 `LANGCHAIN_TRACING_V2=true`와 연동 가능

### Performance

- `RunTree.post()`는 LangSmith SDK 내부에서 비동기 배치 전송
- Disabled 시 오버헤드: `dict.get()` + `None` 체크 — 무시 가능
- Tool result는 1000자로 truncate하여 전송 크기 제한

---

## 7. Dependencies

```toml
# pyproject.toml
dependencies = [
    ...,
    "langsmith>=0.1.0",
]
```

### Environment Variables

```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY="ls_..."
LANGSMITH_PROJECT="brain-agent"
```

---

## 8. Files Changed

| File | Change |
|------|--------|
| `brain_agent/tracing/__init__.py` | **NEW** — TracingManager export |
| `brain_agent/tracing/manager.py` | **NEW** — TracingManager class |
| `brain_agent/tracing/langsmith_tracer.py` | **NEW** — LangSmith RunTree wrapper |
| `brain_agent/config/schema.py` | ADD TracingConfig + tracing field |
| `brain_agent/agent.py` | ADD TracingManager init + root run in process() |
| `brain_agent/pipeline.py` | ADD phase/region/tool child runs |
| `brain_agent/middleware/myelin/sheath.py` | ADD LLM child run with region metadata |
| `pyproject.toml` | ADD langsmith dependency |
| `.env.example` | ADD LANGSMITH env vars |
