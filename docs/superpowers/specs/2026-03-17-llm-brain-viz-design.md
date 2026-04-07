# Brain Agent: LLM Integration + 3D Brain Visualization Overhaul

**Date:** 2026-03-17
**Status:** Approved
**Scope:** PFC LLM integration, consolidation fix, anatomical 3D brain model, particle signal flow

---

## 1. Problem Statement

Four issues block the brain-agent dashboard from being functional and visually faithful:

| # | Problem | Root Cause |
|---|---------|------------|
| 1 | No real agent response | PFC is a stub returning `"Processing: {text}"` — no LLM call |
| 2 | Memory stuck in staging | Consolidation only triggers on session switch |
| 3 | Brain looks nothing like a brain | 9 spheres + wireframe sphere outline |
| 4 | Connections don't show signal flow | 8 hardcoded straight lines, no direction/animation |

## 2. Approach

Backend-first, frontend follows. The pipeline must produce real data before visualization can be meaningful.

**Order:** LLM integration → Consolidation fix → Pipeline event emission → 3D brain model → Particle flow system

---

## 3. Section 1: PFC LLM Integration

### 3.1 PrefrontalCortex Changes

Inject `LLMProvider` into PFC. When a provider is present, call `llm_provider.chat()` for real responses. Without a provider, fall back to existing stub behavior (test compatibility).

```python
class PrefrontalCortex(BrainRegion):
    def __init__(self, llm_provider: LLMProvider | None = None):
        super().__init__(name="prefrontal_cortex", position=Vec3(0, 60, 20))
        self.llm_provider = llm_provider
        self.goal_stack: list[str] = []
```

On `EXTERNAL_INPUT`:
- Build system prompt reflecting current network mode and emotional state
- Include working memory contents as context
- Call `llm_provider.chat(messages)` → get LLM response text
- Wrap response in PLAN signal: `{"tool": "respond", "confidence": 0.8, "args": {"text": llm_response}}`

On `CONFLICT_DETECTED`:
- Re-plan with a "reconsider" system prompt

### 3.2 Pipeline Changes

`ProcessingPipeline.__init__` accepts `llm_provider` and `emitter` and passes them through:

```python
class ProcessingPipeline:
    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        emitter: DashboardEmitter | None = None,
    ):
        ...
        self.pfc = PrefrontalCortex(llm_provider=llm_provider)
        self._emitter = emitter
```

### 3.3 BrainAgent Changes

BrainAgent creates a `LiteLLMProvider` from config and passes it to the pipeline:

```python
class BrainAgent:
    def __init__(self, ...):
        ...
        self._llm_provider = LiteLLMProvider(
            model=self.config.agent.model,
            api_key=self.config.provider.api_key,
        )
        self.pipeline = ProcessingPipeline(llm_provider=self._llm_provider)
```

When `use_mock_embeddings=True` or API key is empty/None, pass `None` to keep stub behavior:

```python
if not self.config.provider.api_key or use_mock_embeddings:
    self._llm_provider = None
else:
    self._llm_provider = LiteLLMProvider(
        model=self.config.agent.model,
        api_key=self.config.provider.api_key,
    )
```

### 3.4 LiteLLMProvider Verification

Verify that `litellm_provider.py` implements `chat(messages) -> LLMResponse` correctly. The existing base class defines the interface; ensure LiteLLM integration works with `openai/gpt-4o-mini`.

### 3.5 Consolidation Fix

Current: consolidation triggers only on session switch when staging count exceeds threshold. The threshold is a module-level constant `STAGING_PRESSURE_THRESHOLD = 20` in `consolidation.py`, **not** wired to config.

**Changes:**

1. **Wire config into ConsolidationEngine:** Pass `consolidation_threshold` from `BrainAgentConfig.memory` into `ConsolidationEngine.__init__()` and use it instead of the hardcoded constant.

2. **Lower default threshold:** Change `BrainAgentConfig.memory.consolidation_threshold` default from 20 → 5.

3. **Add post-request consolidation in BrainAgent.process():** Insert after the `await self.memory.encode(...)` block (currently line 118 of `agent.py`), before returning the result:

```python
# After encoding — check consolidation pressure
if await self.memory.consolidation.should_consolidate():
    await self.memory.consolidate()
```

4. **Preserve existing session-switch consolidation** on lines 89-90 of `agent.py` as-is. The post-request check is a second trigger path.

---

## 4. Section 2: Pipeline Event Emission

### 4.1 Emitter Injection

`ProcessingPipeline` accepts an optional `DashboardEmitter`. Each pipeline stage emits events in real-time during processing rather than post-hoc in the server.

### 4.2 Stage-by-Stage Events

| Step | Region/Action | Events Emitted |
|------|--------------|----------------|
| 1 | thalamus.process() | `region_activation("thalamus", level)` + `signal_flow("_input"→thalamus)` |
| 2 | sensory.register() | `memory_flow(sensory, working, staging, semantic)` |
| 3 | amygdala.process() | `region_activation("amygdala")` + `signal_flow(thalamus→amygdala)` |
| 4 | salience.process() | `network_switch(from, to)` + `signal_flow(amygdala→salience_network)` |
| 5 | working_memory.load() | `memory_flow(...)` |
| 6 | pfc.process() | `region_activation("prefrontal_cortex")` + `signal_flow(salience_network→prefrontal_cortex)` |
| 7 | acc.process() | `region_activation("acc")` + `signal_flow(prefrontal_cortex→acc)` |
| 8 | basal_ganglia.process() | `region_activation("basal_ganglia")` + `signal_flow(acc→basal_ganglia)` |
| 9 | cerebellum.process() | `region_activation("cerebellum")` + `signal_flow(basal_ganglia→cerebellum)` |
| 10 | execute | `broadcast("action_executed")` |
| 11 | cerebellum (error) | `signal_flow(executor→cerebellum)` |
| 12 | hippocampus encode (*in agent.py, not pipeline*) | `region_activation("hippocampus")` + `memory_flow(...)` |
| 13 | GWT broadcast | `broadcast(result)` + `network_switch(ECN→DMN)` |

### 4.3 New Event Type: `signal_flow`

```json
{
  "type": "signal_flow",
  "payload": {
    "source": "thalamus",
    "target": "amygdala",
    "signal_type": "EXTERNAL_INPUT",
    "strength": 0.8
  }
}
```

This event triggers particle generation in the frontend.

**Special source `"_input"`:** Represents external user input. The frontend maps this to a spawn point slightly outside the brain model (e.g., front-top) so particles visibly enter the brain toward the thalamus. `"_input"` is not a region node — it's a virtual origin for the entry particle.

### 4.4 DashboardEmitter Addition

Add `signal_flow()` method to `DashboardEmitter`:

```python
async def signal_flow(self, source: str, target: str, signal_type: str, strength: float) -> None:
    await event_bus.emit("signal_flow", {
        "source": source, "target": target,
        "signal_type": signal_type, "strength": strength,
    })
```

### 4.5 Server Simplification

`server.py`'s `process_message()` currently emits events post-hoc (the `_region_attr_map` loop, manual broadcast, network switch). These move into the pipeline.

**What moves to pipeline:** region_activation per step, signal_flow, network_switch, broadcast.

**What stays in server (requires agent-level access):**
- `memory_flow` after `agent.process()` — needs `agent.memory.stats()`
- `neuromodulator` update — needs `pipeline.neuromodulators.snapshot()`
- Hippocampus `region_activation` — encoding happens in `agent.py`, not pipeline

**What gets removed:** The `_region_attr_map` loop, manual thalamus activation, manual DMN→ECN/ECN→DMN switches, manual PFC/ACC resets.

The server's `process_message()` becomes:
1. Emit initial `signal_flow("_input" → "thalamus")`
2. Call `agent.process(text)` — pipeline emits events internally
3. Emit hippocampus activation + memory_flow + neuromodulator update
4. Return JSON response

---

## 5. Section 3: 3D Brain Model + Translucent Rendering

### 5.1 Model Sourcing

Acquire a CC-licensed anatomical brain GLTF/GLB model from Sketchfab or similar. Requirements:
- Recognizable brain silhouette: cerebral hemispheres (with sulci/gyri), cerebellum, brainstem
- Reasonable polygon count (< 100k faces for web performance)
- Single mesh or separated into major parts (hemispheres, cerebellum, brainstem)

Place model at `dashboard/public/models/brain.glb`.

### 5.2 Translucent Outer Shell

Load model with `useGLTF` from `@react-three/drei`. Apply translucent material:

```tsx
<meshPhysicalMaterial
  color="#1a1a3e"
  transparent
  opacity={0.18}
  roughness={0.3}
  transmission={0.6}
  thickness={2.0}
  side={THREE.DoubleSide}
/>
```

The brain exterior should be barely visible — a ghostly shell revealing the glowing regions inside.

### 5.3 Interior Region Nodes

Keep sphere-based region nodes inside the brain model. Adjust positions to be anatomically correct relative to the GLTF model's bounding box:

| Region | Anatomical Position |
|--------|-------------------|
| prefrontal_cortex | Front-top of frontal lobe |
| acc | Medial frontal, above corpus callosum |
| amygdala | Medial temporal lobe, deep |
| basal_ganglia | Central, subcortical |
| cerebellum | Posterior-inferior |
| thalamus | Central core |
| hypothalamus | Below thalamus, small |
| hippocampus | Medial temporal, curved |
| salience_network | Anterior insula region |

Positions will be fine-tuned after model import by inspecting the model's coordinate space.

### 5.4 Lighting

- Increase ambient light for translucent material visibility
- Region emissive glow bleeds through the translucent shell
- Consider subtle bloom post-processing (`@react-three/postprocessing` EffectComposer + Bloom) for active regions

### 5.5 Camera

- Adjust initial camera position to frame the brain model properly
- OrbitControls preserved

---

## 6. Section 4: Particle Signal Flow System

### 6.1 Connection Topology

Replace 8 hardcoded connections with the full pipeline flow plus neuroscience-informed connections:

**Pipeline path (directed):**
```
user → thalamus → amygdala → salience_network → prefrontal_cortex →
acc → basal_ganglia → cerebellum → (execute) → hippocampus → (GWT broadcast)
```

**Additional anatomical connections:**
- thalamus ↔ hippocampus
- thalamus ↔ cerebellum
- prefrontal_cortex ↔ hypothalamus
- hippocampus ↔ prefrontal_cortex (consolidation/DMN)
- salience_network ↔ amygdala
- amygdala ↔ hippocampus (emotional memory encoding)
- hypothalamus → neuromodulators (conceptual)

### 6.2 Curved Connections

Replace straight lines with `CatmullRomCurve3`:

```tsx
const curve = new THREE.CatmullRomCurve3([
  new THREE.Vector3(...fromPos),
  new THREE.Vector3(...midpoint),  // offset for curve
  new THREE.Vector3(...toPos),
])
```

Midpoint offset calculated to avoid passing through other regions.

### 6.3 Particle System

On `signal_flow` event:
1. Create 1-5 particles (based on `strength`)
2. Each particle: small emissive sphere (radius ~0.5)
3. Color by signal_type:
   - `EXTERNAL_INPUT` → cyan `#06b6d4`
   - `PLAN` → blue `#3b82f6`
   - `ACTION_SELECTED` → green `#22c55e`
   - `EMOTIONAL_TAG` → red `#f43f5e`
   - `GWT_BROADCAST` → gold `#eab308`
   - default → white
4. Animate along curve: `progress` 0→1 over 0.5-1.5 seconds (faster = higher strength)
5. Stagger particle start times for trail effect
6. On arrival: flash target node (brief scale + emissive pulse)

### 6.4 Implementation

Use instanced mesh or `THREE.Points` for performance:

```tsx
function SignalParticles() {
  const particles = useBrainStore(s => s.activeParticles)

  useFrame((_, delta) => {
    // Update each particle's progress
    // Remove completed particles
  })

  return (
    <instancedMesh args={[geometry, material, MAX_PARTICLES]}>
      {/* Update instance matrices per frame */}
    </instancedMesh>
  )
}
```

Particle queue: events arriving rapidly are queued with slight delay between spawns to prevent visual chaos.

### 6.5 Idle State

- All connections visible at low opacity (`0.08`)
- In DMN mode: slow ambient particles on hippocampus↔prefrontal_cortex path (consolidation visualization)
- No particles during ECN idle — clean state

### 6.6 Frontend Store Extension

Add to `brainState.ts`:

```typescript
interface SignalFlowEvent {
  source: string
  target: string
  signal_type: string
  strength: number
}

// New state
signalFlows: SignalFlowEvent[]
addSignalFlow: (flow: SignalFlowEvent) => void
```

WebSocket handler adds `signal_flow` case to dispatch these events.

---

## 7. File Changes Summary

### Backend (Python)
| File | Change |
|------|--------|
| `brain_agent/regions/prefrontal.py` | Inject LLMProvider, call LLM on EXTERNAL_INPUT |
| `brain_agent/pipeline.py` | Accept llm_provider + emitter, emit events per stage |
| `brain_agent/agent.py` | Create LiteLLMProvider, pass to pipeline; consolidate after each request |
| `brain_agent/dashboard/emitter.py` | Add `signal_flow()` method |
| `brain_agent/dashboard/server.py` | Pass emitter to pipeline, simplify process_message() |
| `brain_agent/config/schema.py` | Change consolidation_threshold default 20→5 |
| `brain_agent/providers/litellm_provider.py` | Verify chat() implementation works |

### Frontend (TypeScript/React)
| File | Change |
|------|--------|
| `dashboard/src/components/BrainScene.tsx` | GLTF model, translucent shell, new positions, curved connections, particle system |
| `dashboard/src/stores/brainState.ts` | Add signalFlows state + handler |
| `dashboard/src/hooks/useWebSocket.ts` | Handle `signal_flow` event type |
| `dashboard/public/models/brain.glb` | New: anatomical brain model asset |

### Tests
| File | Change |
|------|--------|
| `tests/regions/test_prefrontal.py` | Test with mock LLMProvider |
| `tests/test_pipeline.py` | Test event emission during processing |

---

## 8. Implementation Notes

- **BrainScene.tsx** covers a lot of scope. During planning, split into subcomponents: `BrainModel.tsx` (GLTF + translucent shell), `SignalParticles.tsx` (particle system), `CurvedConnections.tsx` (edge curves).
- **Brain model fallback:** If no suitable CC-licensed GLTF model is found within polygon budget (< 100k faces), fall back to a procedurally generated brain shape using CatmullRom surfaces.
- **MAX_PARTICLES cap:** Limit to 150 simultaneous particles to prevent performance degradation during rapid event bursts.
- **CatmullRomCurve3 midpoints:** Use perpendicular offset from connection midpoint (offset = 30% of connection length, direction = cross product of connection vector and up vector) to create natural curves avoiding other regions.
