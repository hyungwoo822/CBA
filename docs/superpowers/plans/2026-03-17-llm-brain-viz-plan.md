# LLM Integration + 3D Brain Visualization — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire PFC to a real LLM, fix consolidation, emit pipeline events in real-time, replace the dashboard brain with an anatomical 3D model and particle-based signal flow visualization.

**Architecture:** Backend-first. Tasks 1-5 fix the pipeline (LLM, consolidation, event emission). Tasks 6-8 overhaul the frontend (brain model, curved connections, particle system). Each task is independently testable and committable.

**Tech Stack:** Python (litellm, FastAPI, aiosqlite), React 19, Three.js, @react-three/fiber, @react-three/drei, Zustand

**Spec:** `docs/superpowers/specs/2026-03-17-llm-brain-viz-design.md`

---

## Chunk 1: Backend — LLM Integration + Consolidation Fix

### Task 1: Wire LLMProvider into PrefrontalCortex

**Files:**
- Modify: `brain_agent/regions/prefrontal.py`
- Modify: `tests/regions/test_prefrontal.py`

- [ ] **Step 1: Write failing test — PFC with mock LLMProvider returns LLM response**

Add to `tests/regions/test_prefrontal.py`:

```python
from unittest.mock import AsyncMock
from brain_agent.providers.base import LLMProvider, LLMResponse


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock(spec=LLMProvider)
    provider.chat.return_value = LLMResponse(
        content="Hello! I'm doing well.",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    )
    return provider


@pytest.fixture
def pfc_with_llm(mock_llm_provider):
    return PrefrontalCortex(llm_provider=mock_llm_provider)


async def test_calls_llm_when_provider_present(pfc_with_llm, mock_llm_provider):
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "Hello, how are you?"})
    result = await pfc_with_llm.process(sig)
    assert result is not None
    assert result.type == SignalType.PLAN
    mock_llm_provider.chat.assert_awaited_once()
    action = result.payload["actions"][0]
    assert action["args"]["text"] == "Hello! I'm doing well."


async def test_stub_behavior_without_provider(pfc):
    """Existing stub behavior preserved when no provider."""
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "test"})
    result = await pfc.process(sig)
    assert result.payload["actions"][0]["args"]["text"] == "Processing: test"


async def test_llm_error_falls_back_to_stub(pfc_with_llm, mock_llm_provider):
    mock_llm_provider.chat.return_value = LLMResponse(
        content=None, finish_reason="error", usage={"error": "API timeout"}
    )
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="thalamus",
                 payload={"text": "test"})
    result = await pfc_with_llm.process(sig)
    assert result is not None
    assert result.type == SignalType.PLAN
    # Falls back to stub response
    assert "Processing:" in result.payload["actions"][0]["args"]["text"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/regions/test_prefrontal.py -v`
Expected: FAIL — `PrefrontalCortex()` does not accept `llm_provider` parameter

- [ ] **Step 3: Implement PFC LLM integration**

Replace `brain_agent/regions/prefrontal.py`:

```python
from __future__ import annotations
from brain_agent.regions.base import BrainRegion, Vec3
from brain_agent.core.signals import Signal, SignalType
from brain_agent.providers.base import LLMProvider


class PrefrontalCortex(BrainRegion):
    """Planning and reasoning via LLM. Spec ref: Section 4.2 PFC."""

    def __init__(self, llm_provider: LLMProvider | None = None):
        super().__init__(name="prefrontal_cortex", position=Vec3(0, 60, 20))
        self.llm_provider = llm_provider
        self.goal_stack: list[str] = []

    async def _call_llm(self, text: str, emotional_tag=None) -> str | None:
        """Call LLM and return response text, or None on failure."""
        if not self.llm_provider:
            return None
        system = (
            "You are a brain-inspired AI agent. "
            "Respond concisely and helpfully to the user's message."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        try:
            response = await self.llm_provider.chat(messages)
            if response.content:
                return response.content
        except Exception:
            pass
        return None

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type == SignalType.EXTERNAL_INPUT:
            text = signal.payload.get("text", "")
            self.goal_stack = [text]
            self.emit_activation(0.9)

            llm_response = await self._call_llm(text, signal.emotional_tag)
            response_text = llm_response if llm_response else f"Processing: {text}"

            plan = {
                "goal": text,
                "actions": [
                    {
                        "tool": "respond",
                        "confidence": 0.8,
                        "args": {"text": response_text},
                    }
                ],
            }
            return Signal(
                type=SignalType.PLAN,
                source=self.name,
                payload=plan,
                emotional_tag=signal.emotional_tag,
            )
        elif signal.type == SignalType.CONFLICT_DETECTED:
            self.emit_activation(1.0)
            llm_response = await self._call_llm(
                "Reconsider the previous plan. There was a conflict.",
                signal.emotional_tag,
            )
            response_text = llm_response or "Re-planning due to conflict"
            return Signal(
                type=SignalType.PLAN,
                source=self.name,
                payload={
                    "goal": "re-plan",
                    "actions": [
                        {"tool": "respond", "confidence": 0.7, "args": {"text": response_text}}
                    ],
                },
            )
        elif signal.type == SignalType.STRATEGY_SWITCH:
            self.goal_stack.clear()
            self.emit_activation(1.0)
            return None
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/regions/test_prefrontal.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/regions/prefrontal.py tests/regions/test_prefrontal.py
git commit -m "feat: wire LLMProvider into PrefrontalCortex with fallback to stub"
```

---

### Task 2: Wire LLMProvider through Pipeline and BrainAgent

**Files:**
- Modify: `brain_agent/pipeline.py`
- Modify: `brain_agent/agent.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test — Pipeline accepts llm_provider**

Add to `tests/test_pipeline.py`:

```python
from unittest.mock import AsyncMock
from brain_agent.providers.base import LLMProvider, LLMResponse


async def test_pipeline_accepts_llm_provider():
    provider = AsyncMock(spec=LLMProvider)
    provider.chat.return_value = LLMResponse(
        content="LLM says hello", finish_reason="stop", usage={}
    )
    pipeline = ProcessingPipeline(llm_provider=provider)
    result = await pipeline.process_request("hello")
    assert result.response == "LLM says hello"
    provider.chat.assert_awaited()


async def test_pipeline_works_without_provider():
    pipeline = ProcessingPipeline()
    result = await pipeline.process_request("hello")
    assert "Processing:" in result.response
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py::test_pipeline_accepts_llm_provider -v`
Expected: FAIL — `ProcessingPipeline()` does not accept `llm_provider`

- [ ] **Step 3: Modify ProcessingPipeline to accept llm_provider**

In `brain_agent/pipeline.py`, change `__init__`:

```python
class ProcessingPipeline:
    """Orchestrates the full brain processing flow for a single request."""

    def __init__(
        self,
        llm_provider: "LLMProvider | None" = None,
        emitter: "DashboardEmitter | None" = None,
    ):
        self.neuromodulators = Neuromodulators()
        self.network_ctrl = TripleNetworkController()
        self.router = ThalamicRouter(
            network_ctrl=self.network_ctrl, neuromodulators=self.neuromodulators
        )
        self.workspace = GlobalWorkspace()

        # Memory
        self.sensory = SensoryBuffer()
        self.working = WorkingMemory(capacity=4)

        # Regions
        self.thalamus = Thalamus()
        self.amygdala = Amygdala()
        self.salience = SalienceNetworkRegion(network_ctrl=self.network_ctrl)
        self.pfc = PrefrontalCortex(llm_provider=llm_provider)
        self.acc = AnteriorCingulateCortex()
        self.basal_ganglia = BasalGanglia()
        self.cerebellum = Cerebellum()
        self.hypothalamus = Hypothalamus(neuromodulators=self.neuromodulators)
        self._emitter = emitter
```

Add imports at top of `pipeline.py` using `TYPE_CHECKING` guard (avoids circular imports since emitter imports from server which imports from agent):

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_agent.providers.base import LLMProvider
    from brain_agent.dashboard.emitter import DashboardEmitter
```

- [ ] **Step 4: Modify BrainAgent to create LiteLLMProvider**

In `brain_agent/agent.py`, update `__init__` to create provider and pass it to pipeline. Add after `self.tools = ToolRegistry()`:

```python
from brain_agent.providers.litellm_provider import LiteLLMProvider

# In __init__:
if not self.config.provider.api_key or use_mock_embeddings:
    self._llm_provider = None
else:
    self._llm_provider = LiteLLMProvider(
        model=self.config.agent.model,
        api_key=self.config.provider.api_key,
    )
self.pipeline = ProcessingPipeline(llm_provider=self._llm_provider)
```

Remove the old `self.pipeline = ProcessingPipeline()` line.

- [ ] **Step 5: Run all tests to verify they pass**

Run: `pytest tests/test_pipeline.py tests/regions/test_prefrontal.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add brain_agent/pipeline.py brain_agent/agent.py tests/test_pipeline.py
git commit -m "feat: wire LLMProvider through Pipeline and BrainAgent"
```

---

### Task 3: Fix Consolidation — Wire Config + Post-Request Trigger

**Files:**
- Modify: `brain_agent/config/schema.py`
- Modify: `brain_agent/memory/consolidation.py`
- Modify: `brain_agent/memory/manager.py`
- Modify: `brain_agent/agent.py`
- Modify: `tests/memory/test_consolidation.py`

- [ ] **Step 1: Write failing test — ConsolidationEngine uses configurable threshold**

Add to `tests/memory/test_consolidation.py`:

```python
async def test_should_consolidate_respects_custom_threshold(staging, episodic):
    engine = ConsolidationEngine(
        staging=staging, episodic_store=episodic, forgetting=ForgettingEngine(),
        threshold=3,
    )
    for i in range(3):
        await staging.encode(f"memory {i}", {}, i, "s1")
    assert await engine.should_consolidate() is True


async def test_should_not_consolidate_below_custom_threshold(staging, episodic):
    engine = ConsolidationEngine(
        staging=staging, episodic_store=episodic, forgetting=ForgettingEngine(),
        threshold=10,
    )
    await staging.encode("single memory", {}, 1, "s1")
    assert await engine.should_consolidate() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/memory/test_consolidation.py::test_should_consolidate_respects_custom_threshold -v`
Expected: FAIL — `ConsolidationEngine()` does not accept `threshold`

- [ ] **Step 3: Update ConsolidationEngine to accept threshold parameter**

In `brain_agent/memory/consolidation.py`:

```python
class ConsolidationEngine:
    def __init__(
        self,
        staging: HippocampalStaging,
        episodic_store: EpisodicStore,
        forgetting: ForgettingEngine,
        threshold: int = 5,
    ):
        self._staging = staging
        self._episodic = episodic_store
        self._forgetting = forgetting
        self._threshold = threshold

    async def should_consolidate(self) -> bool:
        return await self._staging.count_unconsolidated() >= self._threshold
```

Delete the `STAGING_PRESSURE_THRESHOLD = 20` constant declaration on line 9. Keep the other constants (`HOMEOSTATIC_FACTOR`, `PRUNING_THRESHOLD`, `EMOTIONAL_BOOST`).

Note: Existing tests (`test_should_consolidate_staging_pressure` with 25 memories, `test_should_not_consolidate_low_count` with 1 memory) remain correct because the `engine` fixture creates `ConsolidationEngine` without explicit threshold, which now defaults to 5. 25 >= 5 still passes, 1 < 5 still fails.

- [ ] **Step 4: Update config default**

In `brain_agent/config/schema.py`, change:

```python
consolidation_threshold: int = Field(default=5, ge=5)
```

- [ ] **Step 5: Wire config through MemoryManager**

In `brain_agent/memory/manager.py`, update `__init__` to accept and pass threshold:

```python
def __init__(
    self,
    db_dir: str,
    embed_fn: Callable[[str], list[float]],
    working_capacity: int = 4,
    consolidation_threshold: int = 5,
):
    ...
    self.consolidation = ConsolidationEngine(
        staging=self.staging,
        episodic_store=self.episodic,
        forgetting=self.forgetting,
        threshold=consolidation_threshold,
    )
```

- [ ] **Step 6: Wire config through BrainAgent**

In `brain_agent/agent.py`, pass threshold when creating MemoryManager:

```python
self.memory = MemoryManager(
    db_dir=self._data_dir,
    embed_fn=self._embed_fn,
    working_capacity=self.config.memory.working_capacity,
    consolidation_threshold=self.config.memory.consolidation_threshold,
)
```

- [ ] **Step 7: Add post-request consolidation in BrainAgent.process()**

In `brain_agent/agent.py`, add after the `await self.memory.encode(...)` block (after line 118), before `return result`:

```python
# Post-request consolidation check
if await self.memory.consolidation.should_consolidate():
    await self.memory.consolidate()
```

- [ ] **Step 8: Run all tests**

Run: `pytest tests/memory/test_consolidation.py tests/test_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add brain_agent/config/schema.py brain_agent/memory/consolidation.py brain_agent/memory/manager.py brain_agent/agent.py tests/memory/test_consolidation.py
git commit -m "fix: wire consolidation threshold to config and trigger after each request"
```

---

## Chunk 2: Backend — Pipeline Event Emission + Server Simplification

### Task 4: Add signal_flow to DashboardEmitter + Emit Events in Pipeline

**Files:**
- Modify: `brain_agent/dashboard/emitter.py`
- Modify: `brain_agent/pipeline.py`
- Create: `tests/dashboard/test_emitter.py`

- [ ] **Step 1: Write failing test — DashboardEmitter.signal_flow()**

Create `tests/dashboard/test_emitter.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from brain_agent.dashboard.emitter import DashboardEmitter


async def test_signal_flow_emits_event():
    emitter = DashboardEmitter()
    with patch("brain_agent.dashboard.emitter.event_bus") as mock_bus:
        mock_bus.emit = AsyncMock()
        await emitter.signal_flow("thalamus", "amygdala", "EXTERNAL_INPUT", 0.8)
        mock_bus.emit.assert_awaited_once_with("signal_flow", {
            "source": "thalamus",
            "target": "amygdala",
            "signal_type": "EXTERNAL_INPUT",
            "strength": 0.8,
        })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/dashboard/test_emitter.py -v`
Expected: FAIL — `DashboardEmitter` has no `signal_flow` method

- [ ] **Step 3: Add signal_flow method to DashboardEmitter**

In `brain_agent/dashboard/emitter.py`, add:

```python
async def signal_flow(self, source: str, target: str, signal_type: str, strength: float) -> None:
    await event_bus.emit("signal_flow", {
        "source": source, "target": target,
        "signal_type": signal_type, "strength": strength,
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/dashboard/test_emitter.py -v`
Expected: PASS

- [ ] **Step 5: Write failing test — Pipeline emits events when emitter present**

Add to `tests/test_pipeline.py`:

```python
from brain_agent.dashboard.emitter import DashboardEmitter


async def test_pipeline_emits_signal_flow_events():
    emitter = DashboardEmitter()
    emitter.signal_flow = AsyncMock()
    emitter.region_activation = AsyncMock()
    emitter.network_switch = AsyncMock()
    emitter.broadcast = AsyncMock()

    pipeline = ProcessingPipeline(emitter=emitter)
    await pipeline.process_request("hello")

    # Should have emitted multiple signal_flow events
    assert emitter.signal_flow.await_count >= 5
    # Should have emitted region activations
    assert emitter.region_activation.await_count >= 3


async def test_pipeline_works_without_emitter():
    pipeline = ProcessingPipeline()
    result = await pipeline.process_request("hello")
    assert result.response != ""
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py::test_pipeline_emits_signal_flow_events -v`
Expected: FAIL — pipeline doesn't emit events

- [ ] **Step 7: Add _emit helper and update process_request() with event emission**

First, ensure `ProcessingPipeline.__init__` already accepts `emitter` parameter (done in Task 2 Step 3 — verify `self._emitter = emitter` is stored).

Note: All `memory_flow` events are deferred to the server (Task 5), since they require `agent.memory.stats()` which is not available inside the pipeline.

In `brain_agent/pipeline.py`, add a helper method and rewrite `process_request`:

```python
async def _emit(self, method: str, *args, **kwargs) -> None:
    """Emit dashboard event if emitter is present."""
    if self._emitter:
        fn = getattr(self._emitter, method, None)
        if fn:
            await fn(*args, **kwargs)

async def process_request(self, text: str) -> PipelineResult:
    result = PipelineResult()
    signals_count = 0

    # 1. Create input signal
    input_signal = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="user",
        payload={"text": text},
    )
    self._route(input_signal)

    # 2. Thalamus: preprocess
    input_signal = await self.thalamus.process(input_signal)
    signals_count += 1
    await self._emit("region_activation", "thalamus", self.thalamus.activation_level, "active")
    await self._emit("signal_flow", "_input", "thalamus", "EXTERNAL_INPUT", 0.6)

    # 3. Sensory buffer: register
    self.sensory.new_cycle()
    self.sensory.register(input_signal.payload, modality="text")

    # 4. Amygdala: emotional tagging
    input_signal = await self.amygdala.process(input_signal)
    signals_count += 1
    await self._emit("region_activation", "amygdala", self.amygdala.activation_level, "active")
    await self._emit("signal_flow", "thalamus", "amygdala", "EXTERNAL_INPUT", self.amygdala.activation_level or 0.3)

    # 5. Salience Network: evaluate -> mode switch
    prev_mode = self.network_ctrl.current_mode.value
    await self.salience.process(input_signal)
    signals_count += 1
    new_mode = self.network_ctrl.current_mode.value
    if prev_mode != new_mode:
        await self._emit("network_switch", prev_mode, new_mode, "salience_evaluation")
    await self._emit("region_activation", "salience_network", self.salience.activation_level, "active")
    await self._emit("signal_flow", "amygdala", "salience_network", "EXTERNAL_INPUT", 0.5)

    # 6. Working Memory: load
    wm_item = WorkingMemoryItem(content=text, slot="phonological")
    self.working.load(wm_item)

    # 7. PFC: plan
    plan_signal = await self.pfc.process(input_signal)
    signals_count += 1
    await self._emit("region_activation", "prefrontal_cortex", self.pfc.activation_level, "high_activity")
    await self._emit("signal_flow", "salience_network", "prefrontal_cortex", "EXTERNAL_INPUT", 0.8)

    if plan_signal and plan_signal.type == SignalType.PLAN:
        self._route(plan_signal)

        # 8. ACC: conflict check
        conflict = await self.acc.process(plan_signal)
        signals_count += 1
        await self._emit("region_activation", "acc", self.acc.activation_level, "active")
        await self._emit("signal_flow", "prefrontal_cortex", "acc", "PLAN", 0.6)

        if conflict and conflict.type == SignalType.CONFLICT_DETECTED:
            plan_signal = await self.pfc.process(conflict)
            signals_count += 1

        # 9. Basal Ganglia: Go/NoGo
        action_signal = await self.basal_ganglia.process(plan_signal)
        signals_count += 1
        await self._emit("region_activation", "basal_ganglia", self.basal_ganglia.activation_level, "active")
        await self._emit("signal_flow", "acc", "basal_ganglia", "PLAN", 0.5)

        if action_signal and action_signal.type == SignalType.ACTION_SELECTED:
            # 10. Cerebellum: predict
            action_signal = await self.cerebellum.process(action_signal)
            signals_count += 1
            await self._emit("region_activation", "cerebellum", self.cerebellum.activation_level, "active")
            await self._emit("signal_flow", "basal_ganglia", "cerebellum", "ACTION_SELECTED", 0.5)

            # 11. Execute
            action = action_signal.payload.get("action", {})
            result.actions_taken.append(action)
            result.response = action.get("args", {}).get("text", "Action executed")
            await self._emit("broadcast", "action_executed", "pipeline")

            # 12. Cerebellum: check prediction error
            await self._emit("signal_flow", "basal_ganglia", "cerebellum", "ACTION_RESULT", 0.5)
            result_signal = Signal(
                type=SignalType.ACTION_RESULT,
                source="executor",
                payload={"predicted": "success", "actual": "success", "error": 0.05},
            )
            error_signal = await self.cerebellum.process(result_signal)
            signals_count += 1

            if error_signal and error_signal.type == SignalType.PREDICTION_ERROR:
                await self.acc.process(error_signal)
                signals_count += 1

            # 13. GWT Broadcast
            self.workspace.submit(
                Signal(
                    type=SignalType.GWT_BROADCAST,
                    source="pipeline",
                    payload={"status": "task_complete", "result": result.response},
                ),
                salience=0.7,
                goal_relevance=0.8,
            )
            broadcast = self.workspace.compete()
            if broadcast:
                await self.salience.process(broadcast)
                signals_count += 1
            await self._emit("broadcast", result.response, "pipeline")
            await self._emit("network_switch", "ECN", "DMN", "task_complete")

    result.network_mode = self.network_ctrl.current_mode.value
    result.signals_processed = signals_count
    return result
```

- [ ] **Step 8: Run all tests**

Run: `pytest tests/test_pipeline.py tests/dashboard/test_emitter.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add brain_agent/dashboard/emitter.py brain_agent/pipeline.py tests/dashboard/test_emitter.py tests/test_pipeline.py
git commit -m "feat: emit signal_flow events from pipeline stages in real-time"
```

---

### Task 5: Simplify Server process_message()

**Files:**
- Modify: `brain_agent/dashboard/server.py`

- [ ] **Step 1: Rewrite process_message() in server.py**

Replace the `process_message` endpoint body in `brain_agent/dashboard/server.py`:

```python
@app.post("/api/process")
async def process_message(body: ProcessRequest):
    agent_inst: BrainAgent | None = _state["agent"]
    if agent_inst is None:
        return {"error": "agent not initialized"}
    text = body.text
    if not text.strip():
        return {"error": "empty message"}

    # 1. Signal entry — emitted by pipeline internally via _emit
    # 2. Process — pipeline emits region_activation + signal_flow events
    result = await agent_inst.process(text)
    pipeline = agent_inst.pipeline

    # 3. Hippocampus activation (encoding happens in agent.py, not pipeline)
    await _emitter.region_activation("hippocampus", 0.7, "active")
    await _emitter.signal_flow("cerebellum", "hippocampus", "ENCODE", 0.6)

    # 4. Memory flow stats
    stats = await agent_inst.memory.stats()
    await _emitter.memory_flow(
        stats["sensory"], stats["working"],
        stats["staging"], stats["episodic"] + stats["semantic"],
    )

    # 5. Neuromodulators
    nm = pipeline.neuromodulators.snapshot()
    await _emitter.neuromodulator_update(
        nm["urgency"], nm["learning_rate"],
        nm["patience"], nm["reward_signal"],
    )

    return {
        "response": result.response,
        "network_mode": result.network_mode,
        "signals_processed": result.signals_processed,
        "actions": result.actions_taken,
    }
```

- [ ] **Step 2: Pass emitter to pipeline via BrainAgent**

In `brain_agent/dashboard/server.py`, inside `create_app()`, update the lifespan to pass emitter to agent's pipeline after agent creation:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    if _state["agent"] is None:
        _state["agent"] = BrainAgent(use_mock_embeddings=True)
        await _state["agent"].initialize()
    # Wire emitter into pipeline
    _state["agent"].pipeline._emitter = _emitter
    yield
    if _state["agent"] and _state["owns_agent"]:
        await _state["agent"].close()
```

- [ ] **Step 3: Run dashboard test**

Run: `pytest tests/dashboard/test_server.py -v`
Expected: PASS (or skip if no test exists for process_message)

- [ ] **Step 4: Manual smoke test**

Run: `python -m brain_agent dashboard --port 3001`
Open `http://localhost:3001` in browser, send a message, verify:
- Events appear in the event log
- Regions activate during processing
- Response comes back

- [ ] **Step 5: Commit**

```bash
git add brain_agent/dashboard/server.py
git commit -m "refactor: simplify server process_message, pipeline emits events directly"
```

---

## Chunk 3: Frontend — 3D Brain Model + Particle Signal Flow

### Task 6: Source Brain Model + Create BrainModel Component + Shared Constants

**Files:**
- Create: `dashboard/public/models/brain.glb`
- Create: `dashboard/src/components/BrainModel.tsx`
- Create: `dashboard/src/constants/brainRegions.ts` (shared positions/config)
- Modify: `dashboard/src/components/BrainScene.tsx`

- [ ] **Step 1: Source and download brain model**

Find a CC-licensed anatomical brain GLB model from Sketchfab (search "brain anatomy glb"). Requirements:
- < 100k faces
- Recognizable brain silhouette (hemispheres, cerebellum, brainstem)
- CC license

Download and place at `dashboard/public/models/brain.glb`.

**Fallback if no suitable model found:** Create a procedural brain shape in code:

```tsx
// Procedural fallback — use scaled ellipsoids for hemispheres + cerebellum
function ProceduralBrain() {
  return (
    <group>
      {/* Left hemisphere */}
      <mesh position={[-8, 5, 0]} scale={[12, 15, 18]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshPhysicalMaterial color="#1a1a3e" transparent opacity={0.18} transmission={0.6} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      {/* Right hemisphere */}
      <mesh position={[8, 5, 0]} scale={[12, 15, 18]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshPhysicalMaterial color="#1a1a3e" transparent opacity={0.18} transmission={0.6} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      {/* Cerebellum */}
      <mesh position={[0, -15, -10]} scale={[10, 7, 8]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshPhysicalMaterial color="#1a1a3e" transparent opacity={0.18} transmission={0.6} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      {/* Brainstem */}
      <mesh position={[0, -22, -5]} scale={[3, 6, 3]}>
        <cylinderGeometry args={[1, 0.7, 1, 16]} />
        <meshPhysicalMaterial color="#1a1a3e" transparent opacity={0.18} transmission={0.6} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
    </group>
  )
}
```

Use this if Sketchfab search yields nothing suitable.

- [ ] **Step 2: Create shared constants file**

Create `dashboard/src/constants/brainRegions.ts` — single source of truth for region positions and config:

```tsx
export const REGION_CONFIG: Record<string, { position: [number, number, number]; color: string; scale: number }> = {
  prefrontal_cortex: { position: [0, 15, 25], color: '#3b82f6', scale: 1.8 },
  acc: { position: [0, 18, 10], color: '#eab308', scale: 0.8 },
  amygdala: { position: [-12, -5, 5], color: '#f43f5e', scale: 0.7 },
  basal_ganglia: { position: [-6, 5, 0], color: '#f97316', scale: 0.9 },
  cerebellum: { position: [0, -20, -15], color: '#8b5cf6', scale: 1.5 },
  thalamus: { position: [0, 5, 0], color: '#ef4444', scale: 1.0 },
  hypothalamus: { position: [0, 0, 3], color: '#ec4899', scale: 0.6 },
  hippocampus: { position: [-14, -3, 0], color: '#06b6d4', scale: 1.0 },
  salience_network: { position: [12, 10, 10], color: '#22c55e', scale: 0.8 },
}

// Extract position-only map for connections and particles
export const POSITIONS: Record<string, [number, number, number]> = Object.fromEntries(
  Object.entries(REGION_CONFIG).map(([k, v]) => [k, v.position])
)

// Virtual input spawn point (outside brain, front-top)
export const INPUT_POSITION: [number, number, number] = [0, 30, 40]
```

Note: Exact positions depend on the GLB model's coordinate system. Adjust after visual inspection.

- [ ] **Step 3: Create BrainModel.tsx**

Create `dashboard/src/components/BrainModel.tsx`:

```tsx
import { useEffect } from 'react'
import { useGLTF } from '@react-three/drei'
import * as THREE from 'three'

const BRAIN_MATERIAL = new THREE.MeshPhysicalMaterial({
  color: '#1a1a3e',
  transparent: true,
  opacity: 0.18,
  roughness: 0.3,
  transmission: 0.6,
  thickness: 2.0,
  side: THREE.DoubleSide,
  depthWrite: false,
})

export function BrainModel() {
  const { scene } = useGLTF('/models/brain.glb')

  useEffect(() => {
    scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        (child as THREE.Mesh).material = BRAIN_MATERIAL
      }
    })
  }, [scene])

  return <primitive object={scene} />
}

useGLTF.preload('/models/brain.glb')
```

- [ ] **Step 4: Replace BrainOutline with BrainModel in BrainScene.tsx**

In `dashboard/src/components/BrainScene.tsx`:
- Remove the `BrainOutline` function entirely
- Remove the local `REGION_CONFIG` constant
- Import `BrainModel` and `REGION_CONFIG` from shared constants
- Replace `<BrainOutline />` with `<BrainModel />`

```tsx
import { BrainModel } from './BrainModel'
import { REGION_CONFIG } from '../constants/brainRegions'

export function BrainScene() {
  return (
    <group>
      <BrainModel />
      <ConnectionLines />
      {Object.entries(REGION_CONFIG).map(([name, config]) => (
        <RegionNode key={name} name={name} config={config} />
      ))}
    </group>
  )
}
```

- [ ] **Step 5: Update lighting and camera in App.tsx**

In `dashboard/src/App.tsx`:

```tsx
<Canvas camera={{ position: [0, 20, 80], fov: 50 }}>
  <ambientLight intensity={0.6} />
  <pointLight position={[100, 100, 100]} intensity={1.2} />
  <pointLight position={[-80, -50, 80]} intensity={0.4} color="#4060ff" />
  <BrainScene />
  <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
</Canvas>
```

- [ ] **Step 6: Build and verify**

```bash
cd dashboard && npm run build
```

Expected: Build succeeds, brain model renders as translucent shell with region nodes visible inside.

- [ ] **Step 7: Commit**

```bash
git add dashboard/public/models/brain.glb dashboard/src/components/BrainModel.tsx dashboard/src/components/BrainScene.tsx dashboard/src/constants/brainRegions.ts dashboard/src/App.tsx
git commit -m "feat: replace wireframe sphere with anatomical brain GLTF model"
```

---

### Task 7: Curved Connections + Extended Topology

**Files:**
- Create: `dashboard/src/components/CurvedConnections.tsx`
- Modify: `dashboard/src/components/BrainScene.tsx`

- [ ] **Step 1: Create CurvedConnections.tsx**

Create `dashboard/src/components/CurvedConnections.tsx`:

```tsx
import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import { useBrainStore } from '../stores/brainState'
import { POSITIONS } from '../constants/brainRegions'
import * as THREE from 'three'

// Full connection topology: pipeline path + anatomical connections
const CONNECTIONS: [string, string][] = [
  // Pipeline path
  ['thalamus', 'amygdala'],
  ['amygdala', 'salience_network'],
  ['salience_network', 'prefrontal_cortex'],
  ['prefrontal_cortex', 'acc'],
  ['acc', 'basal_ganglia'],
  ['basal_ganglia', 'cerebellum'],
  ['cerebellum', 'hippocampus'],
  // Anatomical connections
  ['thalamus', 'hippocampus'],
  ['thalamus', 'cerebellum'],
  ['prefrontal_cortex', 'hypothalamus'],
  ['hippocampus', 'prefrontal_cortex'],
  ['salience_network', 'amygdala'],
  ['amygdala', 'hippocampus'],
  ['thalamus', 'prefrontal_cortex'],
  ['hypothalamus', 'thalamus'],
]

// POSITIONS imported from ../constants/brainRegions (single source of truth)

export function computeCurvePoints(
  from: [number, number, number],
  to: [number, number, number],
  segments: number = 32,
): [number, number, number][] {
  const fromV = new THREE.Vector3(...from)
  const toV = new THREE.Vector3(...to)
  const mid = new THREE.Vector3().addVectors(fromV, toV).multiplyScalar(0.5)

  // Perpendicular offset for curve
  const dir = new THREE.Vector3().subVectors(toV, fromV)
  const up = new THREE.Vector3(0, 1, 0)
  const perp = new THREE.Vector3().crossVectors(dir, up).normalize()
  const offset = dir.length() * 0.3
  mid.add(perp.multiplyScalar(offset))

  const curve = new THREE.CatmullRomCurve3([fromV, mid, toV])
  return curve.getPoints(segments).map(p => [p.x, p.y, p.z] as [number, number, number])
}

export function CurvedConnections() {
  const regions = useBrainStore((s) => s.regions)

  const curves = useMemo(() => {
    return CONNECTIONS.map(([from, to]) => {
      const fromPos = POSITIONS[from]
      const toPos = POSITIONS[to]
      if (!fromPos || !toPos) return null
      return {
        from, to,
        points: computeCurvePoints(fromPos, toPos),
      }
    }).filter(Boolean) as { from: string; to: string; points: [number, number, number][] }[]
  }, [])

  return (
    <group>
      {curves.map(({ from, to, points }) => {
        const fromLevel = regions[from]?.level ?? 0
        const toLevel = regions[to]?.level ?? 0
        const active = fromLevel > 0.1 && toLevel > 0.1

        return (
          <Line
            key={`${from}-${to}`}
            points={points}
            color={active ? '#4ade80' : '#1e293b'}
            transparent
            opacity={active ? 0.5 : 0.08}
            lineWidth={active ? 1.5 : 0.5}
          />
        )
      })}
    </group>
  )
}

export { CONNECTIONS }
```

- [ ] **Step 2: Replace ConnectionLines with CurvedConnections in BrainScene.tsx**

In `dashboard/src/components/BrainScene.tsx`:
- Remove the `ConnectionLines` function and the old hardcoded `connections` array
- Import and use `CurvedConnections`
- `REGION_CONFIG` is already imported from `../constants/brainRegions` (Task 6)

```tsx
import { CurvedConnections } from './CurvedConnections'

// BrainScene now uses:
// - BrainModel (from Task 6)
// - CurvedConnections (replaces ConnectionLines)
// - REGION_CONFIG from shared constants
```

- [ ] **Step 3: Build and verify**

```bash
cd dashboard && npm run build
```

Expected: Build succeeds, curved lines replace straight connections.

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/components/CurvedConnections.tsx dashboard/src/components/BrainScene.tsx
git commit -m "feat: replace straight connections with curved anatomical topology"
```

---

### Task 8: Particle Signal Flow System + Store Extension

**Files:**
- Create: `dashboard/src/components/SignalParticles.tsx`
- Modify: `dashboard/src/stores/brainState.ts`
- Modify: `dashboard/src/hooks/useWebSocket.ts`
- Modify: `dashboard/src/components/BrainScene.tsx`

- [ ] **Step 1: Extend brainState.ts with signal flow state**

In `dashboard/src/stores/brainState.ts`, add:

```typescript
export interface SignalFlowEvent {
  id: string
  source: string
  target: string
  signal_type: string
  strength: number
  timestamp: number
}

export interface Particle {
  id: string
  source: string
  target: string
  signal_type: string
  progress: number       // 0 → 1
  speed: number          // units per second
  color: string
  delay: number          // stagger delay in seconds
}
```

Add to `BrainState` interface:

```typescript
signalFlows: SignalFlowEvent[]
particles: Particle[]
addSignalFlow: (flow: Omit<SignalFlowEvent, 'id' | 'timestamp'>) => void
setParticles: (particles: Particle[]) => void
```

Add to the store:

```typescript
signalFlows: [],
particles: [],
addSignalFlow: (flow) => set((s) => {
  const event: SignalFlowEvent = {
    ...flow,
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    timestamp: Date.now(),
  }

  const SIGNAL_COLORS: Record<string, string> = {
    'EXTERNAL_INPUT': '#06b6d4',
    'PLAN': '#3b82f6',
    'ACTION_SELECTED': '#22c55e',
    'EMOTIONAL_TAG': '#f43f5e',
    'GWT_BROADCAST': '#eab308',
    'ENCODE': '#a855f7',
  }

  const count = Math.max(1, Math.min(5, Math.round(flow.strength * 5)))
  const speed = 0.5 + flow.strength * 1.0
  const color = SIGNAL_COLORS[flow.signal_type] || '#ffffff'

  const newParticles: Particle[] = Array.from({ length: count }, (_, i) => ({
    id: `${event.id}-p${i}`,
    source: flow.source,
    target: flow.target,
    signal_type: flow.signal_type,
    progress: 0,
    speed,
    color,
    delay: i * 0.1,
  }))

  return {
    signalFlows: [...s.signalFlows.slice(-49), event],
    particles: [...s.particles, ...newParticles].slice(-150),
  }
}),
setParticles: (particles) => set({ particles }),
```

- [ ] **Step 2: Handle signal_flow in useWebSocket.ts**

In `dashboard/src/hooks/useWebSocket.ts`, add case in `handleEvent`:

```typescript
case 'signal_flow':
  store.addSignalFlow({
    source: data.payload.source,
    target: data.payload.target,
    signal_type: data.payload.signal_type,
    strength: data.payload.strength,
  })
  break
```

- [ ] **Step 3: Create SignalParticles.tsx**

Create `dashboard/src/components/SignalParticles.tsx`:

```tsx
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useBrainStore, Particle } from '../stores/brainState'
import { POSITIONS, INPUT_POSITION } from '../constants/brainRegions'
import { computeCurvePoints } from './CurvedConnections'

const MAX_PARTICLES = 150
const tempObject = new THREE.Object3D()
const tempColor = new THREE.Color()

function getPosition(name: string): [number, number, number] {
  if (name === '_input') return INPUT_POSITION
  return POSITIONS[name] || [0, 0, 0]
}

// Cache curves to avoid recomputing every frame
const curveCache = new Map<string, THREE.CatmullRomCurve3>()

function getCurve(source: string, target: string): THREE.CatmullRomCurve3 {
  const key = `${source}->${target}`
  if (!curveCache.has(key)) {
    const from = getPosition(source)
    const to = getPosition(target)
    const points = computeCurvePoints(from, to, 32)
    curveCache.set(key, new THREE.CatmullRomCurve3(
      points.map(p => new THREE.Vector3(...p))
    ))
  }
  return curveCache.get(key)!
}

export function SignalParticles() {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  // Use refs for animation state to avoid per-frame store updates
  const particlesRef = useRef<Particle[]>([])

  // Sync new particles from store into ref (only on add, not every frame)
  const storeParticles = useBrainStore((s) => s.particles)
  if (storeParticles !== particlesRef.current) {
    particlesRef.current = storeParticles
  }

  useFrame((_, delta) => {
    if (!meshRef.current) return
    const particles = particlesRef.current
    if (particles.length === 0) {
      meshRef.current.count = 0
      return
    }

    const updated: Particle[] = []
    let idx = 0
    const store = useBrainStore.getState()

    for (const p of particles) {
      if (p.delay > 0) {
        updated.push({ ...p, delay: p.delay - delta })
        continue
      }

      const newProgress = p.progress + delta * p.speed
      if (newProgress >= 1) {
        // Particle arrived — flash target region
        const region = store.regions[p.target]
        if (region) {
          store.setRegionActivation(p.target, Math.min(1, region.level + 0.15), 'high_activity')
        }
        continue // remove particle
      }

      const curve = getCurve(p.source, p.target)
      const point = curve.getPointAt(Math.min(newProgress, 1))
      tempObject.position.copy(point)
      const scale = 0.5 + (1 - Math.abs(newProgress - 0.5) * 2) * 0.3
      tempObject.scale.setScalar(scale)
      tempObject.updateMatrix()
      meshRef.current.setMatrixAt(idx, tempObject.matrix)

      tempColor.set(p.color)
      meshRef.current.setColorAt(idx, tempColor)

      updated.push({ ...p, progress: newProgress })
      idx++
    }

    meshRef.current.count = idx
    meshRef.current.instanceMatrix.needsUpdate = true
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true
    }

    // Update ref directly — only sync back to store when particles added/removed
    particlesRef.current = updated
    if (updated.length !== particles.length) {
      store.setParticles(updated)
    }
  })

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, MAX_PARTICLES]}>
      <sphereGeometry args={[0.5, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  )
}
```

- [ ] **Step 4: Add SignalParticles to BrainScene.tsx**

In `dashboard/src/components/BrainScene.tsx`:

```tsx
import { SignalParticles } from './SignalParticles'

export function BrainScene() {
  return (
    <group>
      <BrainModel />
      <CurvedConnections />
      <SignalParticles />
      {Object.entries(REGION_CONFIG).map(([name, config]) => (
        <RegionNode key={name} name={name} config={config} />
      ))}
    </group>
  )
}
```

- [ ] **Step 5: Build**

```bash
cd dashboard && npm run build
```

Expected: Build succeeds.

- [ ] **Step 6: End-to-end test**

```bash
cd .. && python -m brain_agent dashboard --port 3001
```

Open `http://localhost:3001`. Send a message. Verify:
- Particles flow from `_input` → thalamus → amygdala → ... along curved paths
- Region nodes flash when particles arrive
- Connections light up during activity
- Response appears in chat panel
- Memory flow bar updates (staging should decrease after consolidation)

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/components/SignalParticles.tsx dashboard/src/components/BrainScene.tsx dashboard/src/stores/brainState.ts dashboard/src/hooks/useWebSocket.ts
git commit -m "feat: add particle signal flow system with instanced mesh rendering"
```

- [ ] **Step 8: Final build + rebuild dashboard dist**

```bash
cd dashboard && npm run build && cd ..
```

Verify `dashboard/dist/` is updated.

- [ ] **Step 9: Final commit**

```bash
git add dashboard/dist/
git commit -m "build: rebuild dashboard with brain model + particle flow"
```
