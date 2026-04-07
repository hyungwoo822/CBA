# Brain Agent — Design Specification

**Date:** 2026-03-16
**Status:** Draft
**Version:** 0.1.0

## 1. Overview

Brain Agent is an open-source AI agent framework that faithfully models the brain's cognitive architecture. It maps neuroscience research—memory systems, functional networks, and decision-making circuits—onto a practical LLM-based agent system.

**Core Principle:** When nanobot's structural patterns conflict with neuroscience research, neuroscience wins. All architectural decisions are grounded in published cognitive science papers.

### 1.1 Key Goals

- Faithful modeling of brain cognitive architecture (GWT, CLS, Triple Network)
- Neuroscience-grounded memory system (4-layer CLS-extended Atkinson-Shiffrin + Baddeley + memory dynamics)
- Event-driven temporal model adapted for sporadic agent interactions
- 3D brain visualization dashboard with real-time activity and information flow
- Open-source SDK: `pip install brain-agent`
- LLM-provider agnostic (OpenAI, Anthropic, local models, etc.)

### 1.2 Borrowed from Nanobot

- Provider abstraction (LLM-agnostic interface)
- Tool registry + schema validation
- Pydantic-based config system
- Workspace layout pattern (AGENTS.md, SOUL.md, etc.)
- Skill system structure

### 1.3 Replaced / New

- Memory system → Full neuroscience-faithful CLS hierarchy
- MessageBus → Thalamic Router (intelligent routing + gating)
- Agent loop → GWT Orchestrator with brain region processors
- New: Triple Network state machine (DMN/ECN/SN)
- New: 3D visualization dashboard (React + Three.js)

---

## 2. Architecture

### 2.1 High-Level Structure

```
┌─────────────────────────────────────────────┐
│          GLOBAL WORKSPACE (GWT)             │
│   Competition → Winner → Global Broadcast   │
│   Baars (1988), Dehaene (2011)              │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┼──────────────────────────┐
│         THALAMIC ROUTER (Intelligent Bus)    │
│   Filtering + Priority Routing + Gating      │
│   Sherman (2007), Halassa & Kastner (2017)   │
└──┬────┬────┬────┬────┬────┬────┬────────────┘
   │    │    │    │    │    │    │
  PFC  Hippo Amyg  BG  Cere ACC  SN
  LLM   DB  Score Rule Rule Rule Rule
```

### 2.2 Design Principle: PFC-Only LLM

Only the Prefrontal Cortex calls the LLM. All other brain regions are algorithm/rule-based. This keeps LLM costs practical while preserving the brain's distributed processing model.

Exception: Procedural memory's "cognitive stage" routes through PFC. Once a procedure reaches "autonomous stage," it executes without LLM.

---

## 3. Memory System

The most critical subsystem. Combines three theoretical frameworks:

1. **Atkinson-Shiffrin (1968), extended with CLS** — 4-layer pipeline: Sensory → Working Memory → Hippocampal Staging → Long-term. The hippocampal staging layer is the CLS extension (McClelland et al., 1995) that mediates between short-term and long-term storage.
2. **Baddeley (2000)** — Working memory with Central Executive, Phonological Loop, Visuospatial Sketchpad, Episodic Buffer
3. **Memory Dynamics** — Ebbinghaus forgetting, hippocampal consolidation, emotional tagging, spreading activation retrieval

### 3.1 Event-Driven Temporal Model

**Critical design decision:** The brain receives continuous analog input. An agent receives sporadic requests. Therefore, absolute time-based decay (ms/s/hours) is replaced with an interaction-based hybrid model.

```python
TemporalModel:
  interaction_count: int     # Total interactions (monotonic)
  session_id: str            # Current session (burst of related interactions)
  wall_clock: datetime       # Only for long-idle detection

  def distance(memory) -> float:
    event_gap = current_interaction - memory.last_interaction
    session_gap = count_sessions_since(memory.last_session)
    idle_factor = classify_idle(wall_clock - memory.last_wall_clock)
    return α * event_gap + β * session_gap + γ * idle_factor
```

**Decay is lazy-evaluated:** Forgetting is computed only when events occur (retrieval, encoding, consolidation), not via background wall-clock sweeps.

### 3.2 Layer 1: Sensory Buffer

| Property | Value | Source |
|----------|-------|--------|
| Lifecycle | Per-request cycle | Adapted from Sperling (1960) |
| Capacity | Unlimited | Matches brain sensory register |
| Decay | Flush on next request arrival | New input overwrites (like brain) |

On new request: flush previous cycle → register input → apply attention filter → pass attended items to Working Memory → discard rest.

### 3.3 Layer 2: Working Memory (Baddeley Model)

| Property | Value | Source |
|----------|-------|--------|
| Capacity | 4±1 chunks | Cowan (2001) |
| Decay | Displacement-based (not time-based) | Interference > time decay |
| Session boundary | Re-evaluate relevance to new session | Context-dependent retention |

Components:
- **Central Executive** — Attention allocation, task switching, goal maintenance. Maps to the GWT Orchestrator.
- **Phonological Loop** — Text/language buffer, ~4-5 items. Circular buffer for recent text.
- **Visuospatial Sketchpad** — Spatial/visual context buffer, ~3-4 objects.
- **Episodic Buffer** — Integrates modalities + LTM fragments into coherent episodes, 4 chunks max.

Rehearsal model:
- **Maintenance rehearsal** (same topic referenced again) → reset displacement counter
- **Elaborative rehearsal** (linked to existing LTM knowledge) → triggers hippocampal encoding

### 3.4 Layer 3: Hippocampal Staging

| Property | Value | Source |
|----------|-------|--------|
| Storage | SQLite (staging_memories table) | Local, no external deps |
| Encoding | One-shot, full context | McClelland et al. (1995) CLS fast learning |
| Role | Temporary index binding distributed representations | Teyler & DiScenna (1986) |

Schema:
```
id: UUID
timestamp: datetime
content: text
context_embedding: blob
entities: json          # who, what, where, when (Eichenbaum 2000)
emotional_tag: json     # {valence: -1~+1, arousal: 0~1}
source_modality: str
access_count: int
strength: float         # Initial = 1.0
consolidated: bool
last_interaction: int
last_session: str
```

Key operations:
- **Pattern Separation** (dentate gyrus): Create distinct embeddings for similar experiences
- **Pattern Completion** (CA3): Retrieve full memory from partial cue
- **Reconsolidation on retrieval** (Nader 2000): Each retrieval boosts strength by ~2.0x

### 3.5 Layer 4: Long-Term Memory

#### 3.5.1 Episodic Store
- **Storage:** SQLite (consolidated_episodes table)
- **Brain mapping:** Hippocampus + temporal cortex
- **Content:** Time-stamped, context-rich personal experiences
- **Schema links:** References to related semantic memories

#### 3.5.2 Semantic Store
- **Storage:** ChromaDB (vector embeddings) + SQLite (knowledge graph)
- **Brain mapping:** Temporal cortex (hub-and-spoke model, Patterson 2007)
- **Vector DB:** Decontextualized facts/concepts with embeddings
- **Knowledge Graph:** Entity relationships with weights (for spreading activation)
  - Example: "Python" --is_a→ "Language" (0.9), "Python" --used_for→ "AI" (0.7)

#### 3.5.3 Procedural Store
- **Storage:** SQLite (procedures table)
- **Brain mapping:** Basal ganglia + cerebellum
- **Three stages** (Fitts & Posner, 1967):
  - Cognitive: LLM reasons every time (slow)
  - Associative: Partial cache, LLM confirms (medium)
  - Autonomous: Fully cached, no LLM call (fast)
- **Power law of learning:** Performance = a × practice^(-b) (Newell & Rosenbloom, 1981)

### 3.6 Memory Dynamics

#### 3.6.1 Forgetting Engine

Ebbinghaus curve adapted for event-driven model:
```
R = e^(-d / S)
```
Where d = interaction distance (not wall-clock time), S = memory strength.

Forgetting triggers (lazy evaluation):
- On retrieval: compute decay for candidates
- On new encode: compute interference with similar existing memories
- On consolidation: global strength re-evaluation

Interference (Anderson et al., 1994):
- New memory with cosine similarity > 0.8 to existing → reduce existing memory strength
- Retrieval-induced forgetting: retrieving X suppresses competitors of X by 10-20%

#### 3.6.2 Consolidation Engine

Trigger conditions (event-based, not periodic):
1. **Staging pressure:** Unconsolidated count ≥ threshold (~20)
2. **Session end:** Conversation session closes (= "sleep" analog)
3. **Long idle:** Wall-clock idle exceeds threshold (only wall-clock usage)
4. **WM overflow:** Evicted-but-not-consolidated items accumulate
5. **Explicit:** Agent requests consolidation

Consolidation process:
1. Fetch unconsolidated memories from hippocampal staging
2. Priority by emotional arousal (McGaugh, 2004)
3. Link to existing semantic knowledge
4. Frequently-accessed memories get strength bonus (testing effect)
5. Transfer to episodic store
6. Extract repeated patterns → semantic store (episodic-to-semantic transition, Winocur & Moscovitch 2011)
7. Compile repeated action patterns → procedural store
8. Synaptic homeostasis (Tononi & Cirelli, 2003): scale all strengths × 0.95, prune below threshold

#### 3.6.3 Retrieval Engine

Multi-factor scoring (extended from Park et al., 2023):
```
score = α×recency + β×relevance + γ×importance + δ×frequency + ε×context_match
```
- recency: exp(-interaction_distance / decay_constant)
- relevance: cosine similarity of query and memory embeddings
- importance: emotional arousal score
- frequency: log(access_count + 1) — testing effect
- context_match: cosine similarity of current context vs. encoding context (Godden & Baddeley, 1975)

Default weights (tunable via config):
```
α = 0.25  # recency
β = 0.30  # relevance (highest — content match most important)
γ = 0.20  # importance (emotional salience)
δ = 0.10  # frequency (testing effect bonus)
ε = 0.15  # context match
```
Weights are normalized to sum to 1.0. These defaults emphasize relevance and recency, consistent with Park et al. (2023)'s Generative Agents findings.

Spreading activation (Collins & Loftus, 1975):
- Traverse semantic knowledge graph from top candidates
- Decay per hop: ~15%
- Max hops: 3
- Fan effect: more connections → lower per-connection activation

Reconstructive retrieval (Bartlett, 1932):
- Memory is reconstruction, not playback
- LLM assembles coherent memory from retrieved fragments + schemas

### 3.7 Redis-Compatible Extension Points

Default: all local (in-memory + SQLite + ChromaDB). Pseudo code provided for Redis backends:
- Sensory Buffer → Redis Stream (XADD/XREAD + TTL)
- Working Memory → Redis Hash + EXPIRE
- Hippocampal Staging → Redis Sorted Set (score = strength)
- GWT Broadcast → Redis Pub/Sub
- Enables distributed multi-agent shared memory

### 3.8 Key Parameters (Paper-Grounded)

| Parameter | Value | Source |
|-----------|-------|--------|
| Working memory capacity | 4±1 chunks | Cowan (2001) |
| Retrieval strength boost | ~2.0-2.5× per access | SM-2 algorithm, Wozniak (1990) |
| Emotional memory advantage | ~30-40% better retention | Cahill & McGaugh (1995) |
| Retrieval-induced forgetting | ~10-20% suppression | Anderson et al. (1994) |
| Spreading activation decay | ~15% per hop | Collins & Loftus (1975) |
| Context match bonus | ~30% | Godden & Baddeley (1975) |
| Homeostatic scaling factor | ~0.95 per consolidation | Tononi & Cirelli (2003) |
| Procedural automation stages | 3 (cognitive→associative→autonomous) | Fitts & Posner (1967) |

---

## 4. Brain Region Processors

### 4.1 Base Interface

```python
class BrainRegion(ABC):
    name: str
    anatomical_position: Vec3           # For 3D visualization
    network_membership: list[Network]   # DMN, ECN, SN
    activation_level: float             # 0.0-1.0 (visualization)

    async process(signal: Signal) -> Signal | None
    def can_handle(signal: Signal) -> float
    def compete_for_workspace(signal) -> float
    def on_broadcast(signal: Signal)
    def emit_activation(level: float)
```

### 4.2 Region Specifications

#### Prefrontal Cortex (PFC) — LLM-Powered
- **Network:** ECN
- **Function:** Planning, reasoning, goal maintenance, cognitive control
- **Implementation:** LLM calls
- **Hierarchical goal stack** (Koechlin, 2003): abstract → sub-goals → current step
- **Dual-process** (CLARION-inspired): Check procedural cache first (fast path), LLM only on cache miss (slow path)
- **Neural basis:** Miller & Cohen (2001) top-down bias signals

#### Anterior Cingulate Cortex (ACC) — Algorithm
- **Network:** SN
- **Function:** Conflict monitoring, error detection, strategy switching
- **Implementation:** Rule-based conflict scoring + error accumulation
- **Conflict threshold:** High conflict → request PFC to deliberate more (Botvinick, 2001)
- **Strategy switch:** Error accumulator exceeds threshold → trigger full re-planning (Holroyd & Coles, 2002)
- **Expected Value of Control** (Shenhav et al., 2013): cost-benefit of investing more compute

#### Amygdala — Algorithm
- **Network:** SN
- **Function:** Emotional tagging, priority evaluation, fast threat detection
- **Dual-speed evaluation** (LeDoux, 1996):
  - Fast path: pattern matching for errors, security threats, urgent keywords
  - Slow path: contextual significance evaluation
- **Output:** EmotionalTag {valence: -1 to +1, arousal: 0 to 1} (Russell, 1980 Circumplex Model)
- **Memory modulation:** High arousal → boosted encoding strength (McGaugh, 2004)

#### Basal Ganglia — Algorithm
- **Network:** ECN
- **Function:** Action selection via Go/NoGo gate
- **Default state:** Tonic inhibition on all actions (Mink, 1996)
- **Go pathway:** Evaluate reward expectation (positive outcomes)
- **NoGo pathway:** Evaluate risk (negative outcomes)
- **Winner-take-all:** Highest net (Go - NoGo) score wins, if above threshold
- **Habit formation** (Graybiel, 2008): Repeated successful sequences → compiled procedures

#### Cerebellum — Algorithm
- **Network:** ECN
- **Function:** Prediction, error correction, procedural automation
- **Forward models** (Ito, 2008): Predict outcomes before execution
- **Two feedback loops:**
  - Fast cerebellar loop: Minor errors → micro-adjustment, continue
  - Slow cortical loop: Major errors → escalate to ACC for strategy evaluation
- **Learning:** Update forward models from prediction errors

#### Thalamus — Algorithm
- **Network:** All (connected to every region)
- **Function:** Input preprocessing, sensory relay, gating
- **First-order relay:** External input normalization
- **Higher-order relay:** Inter-region routing (via Thalamic Router)
- **Two modes** (McCormick & Bal, 1997):
  - Tonic: Faithful relay during active processing
  - Burst: Novelty detection during idle

#### Hypothalamus — Algorithm
- **Network:** None
- **Function:** Resource monitoring, homeostasis, neuromodulator control
- **Monitors:** Pending requests, memory pressure, error rates, token budgets
- **Controls neuromodulatory parameters:**
  - Urgency (NE-equivalent): speed/depth tradeoff
  - Learning rate (ACh-equivalent): plasticity signal
  - Patience (5-HT-equivalent): temporal discounting
- **Triggers consolidation** when memory pressure is high

#### Salience Network — Algorithm
- **Network:** SN
- **Function:** DMN↔ECN switching, important stimulus detection
- **Salience computation:** signal emotional tag × novelty × goal relevance
- **Causal switching** (Sridharan et al., 2008): Anterior insula drives both DMN activation and ECN activation
- **Anti-correlation enforcement** (Fox et al., 2005): When ECN activates, DMN resources freed

### 4.3 User-Extensible Regions

All regions can be overridden:
```python
agent = BrainAgent(region_overrides={"amygdala": CustomAmygdala()})
```

---

## 5. Communication & Network

### 5.1 Signal Schema

The core data structure for all inter-region communication:

```python
@dataclass
class Signal:
    id: UUID
    type: SignalType              # See 5.6
    source: str                   # Region name that produced this signal
    targets: list[str] | None     # None = determined by routing table
    payload: dict                 # Type-specific content
    priority: float = 0.5         # 0.0-1.0, set by router
    emotional_tag: EmotionalTag | None = None
    interaction_id: int = 0       # Current interaction counter
    session_id: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class EmotionalTag:
    valence: float    # -1.0 (negative) to +1.0 (positive)
    arousal: float    # 0.0 (calm) to 1.0 (excited)
```

### 5.2 Session Lifecycle

A session groups related interactions into a coherent conversation context.

```python
Session:
  id: str                      # Unique session identifier
  start_interaction: int       # First interaction in this session
  last_interaction: int        # Most recent interaction
  last_wall_clock: datetime    # Timestamp of last activity
  topic_embedding: blob        # Rolling embedding of session topic

  # Session boundary detection (any one triggers new session):
  boundaries:
    explicit:     user calls session.close() or sends "/new"
    idle_timeout: wall_clock gap > IDLE_SESSION_THRESHOLD (default: 30 min)
    topic_drift:  cosine_sim(current_input, session.topic_embedding) < 0.3

  # On session close:
  on_close:
    - Trigger consolidation (= "sleep" analog)
    - Increment session counter for all memories touched in this session
    - Archive session metadata to SQLite
```

`count_sessions_since(memory)` counts the number of closed sessions between the memory's `last_session` and the current session. Session metadata is stored in SQLite (`sessions` table).

### 5.3 Thalamic Router

Replaces nanobot's MessageBus with an intelligent routing system.

**Default routing table:**

| Signal Type | ECN Mode Targets | DMN Mode Targets |
|-------------|-----------------|-----------------|
| external_input | thalamus → sensory_buffer → salience_network | thalamus → salience_network |
| plan | acc, basal_ganglia | suppressed |
| action_selected | cerebellum | suppressed |
| action_result | cerebellum, acc, hippocampus, amygdala | hippocampus, amygdala |
| conflict_detected | pfc | pfc (force ECN switch) |
| strategy_switch | pfc | pfc (force ECN switch) |
| prediction_error | acc (small), pfc (large) | suppressed |
| emotional_tag | hippocampus | hippocampus |
| consolidation_trigger | consolidation_engine | consolidation_engine |
| gwt_broadcast | ALL active regions | ALL active regions |
| resource_status | hypothalamus | hypothalamus |

**TRN gating rules:**
- In ECN mode: suppress consolidation signals, reduce DMN region routing
- In DMN mode: suppress plan/action/conflict signals (unless force-switch)
- High urgency: bypass gating for priority signals
- Routing table is extensible via config

**Priority routing:** `priority = base_priority × (1 + emotional_tag.arousal × 0.5) × neuromodulators.urgency`

**Visualization events:** Every routing action emits `{source, targets, signal_type, priority, label}` for the dashboard.

### 5.4 Global Workspace

Implementation of Baars (1988) / Dehaene (2011):

1. Specialist processors work independently (unconscious processing)
2. Processors submit results to a competition queue
3. **Competition mechanism:**
   - Event-triggered: competition round runs after each processing stage completes (not on a fixed timer)
   - Each submitted signal scored: `gwt_score = salience × 0.4 + arousal × 0.3 + goal_relevance × 0.3`
   - Highest score wins. Ties broken by: (a) priority, then (b) earlier submission
   - If no signal exceeds `IGNITION_THRESHOLD` (default: 0.3), no broadcast occurs — signals processed locally only
4. Winner's signal broadcast to ALL active regions via Thalamic Router
5. **Serialized broadcasting:** A single `asyncio.Lock` ensures one broadcast at a time. Subsequent competition rounds queue behind the current broadcast.
6. Broadcast is fire-and-forget: regions receive asynchronously, no acknowledgment required

### 5.5 Triple Network State Machine

Three modes. DMN and ECN are from Menon (2011) Triple Network Model. CREATIVE mode is a design extension inspired by Beaty et al. (2018) showing creative cognition involves dynamic DMN-ECN coupling:

| Mode | Active Regions | Behavior | Agent Lifecycle |
|------|---------------|----------|-----------------|
| DMN | Hippocampus, Consolidation | Background processing, self-reflection, memory consolidation | Idle / between requests |
| ECN | PFC, ACC, BG, Cerebellum, Thalamus | Goal-directed task processing | Active request handling |
| CREATIVE | PFC + Hippocampus + ACC | Divergent generation + convergent evaluation | Complex/novel problems |

SN (Salience Network) is not a mode but the **switcher** — it drives transitions between modes. Transitions: SN evaluates salience → activates target mode → suppresses other mode (anti-correlation, Fox et al. 2005).

### 5.6 Neuromodulatory Global Parameters

| Parameter | Brain Analog | Function | Range |
|-----------|-------------|----------|-------|
| urgency | Norepinephrine | Speed/depth tradeoff | 0.0-1.0 |
| learning_rate | Acetylcholine | Plasticity / learning signal | 0.0-1.0 |
| patience | Serotonin | Temporal discounting | 0.0-1.0 |
| reward_signal | Dopamine | Reward prediction error | -1.0 to +1.0 |

### 5.7 Signal Types

```
Forward flow (green):   plan, action_selected, action_result, encode, retrieve
Feedback flow (blue):   prediction_error, conflict_detected, strategy_switch
Emotional flow (pink):  emotional_tag
Broadcast (gold):       gwt_broadcast
System (grey):          consolidation_trigger, network_switch, resource_status
```

### 5.8 Concurrency Model

```
Main event loop: asyncio
Region processing: Sequential within a single request pipeline (not parallel)
  — Follows the anatomical processing flow (Section 6)
  — Each region awaits the previous region's output
GWT competition: Runs after each stage, async lock for broadcast serialization
Consolidation: Runs in separate asyncio task during DMN mode
  — Never concurrent with active request processing (DMN/ECN anti-correlation)
Dashboard WebSocket: Separate asyncio task, fire-and-forget event emission
  — Events batched at 60fps max to prevent flooding
Shared state protection:
  — Memory stores: SQLite WAL mode (concurrent reads, serialized writes)
  — Working memory: accessed only by main pipeline (no contention)
  — Neuromodulators: atomic float updates via asyncio (single writer: hypothalamus)
  — Activation levels: written by regions, read by dashboard (eventual consistency OK)
```

### 5.9 Embedding Strategy

```
Default model: all-MiniLM-L6-v2 (sentence-transformers)
  — 384 dimensions, ~80MB, fast on CPU (~5ms per embedding)
  — Configurable: users can swap to any sentence-transformers model via config
Embedding timing: Computed at encoding time, cached with the memory record
  — Never recomputed unless memory content changes (reconsolidation)
Pattern separation: Add small Gaussian noise (σ=0.01) to embeddings of
  similar inputs to ensure distinct representations in vector space
Cosine similarity thresholds (model-dependent, configurable):
  — Interference threshold: 0.85 (high similarity → interference)
  — Retrieval relevance threshold: 0.3 (minimum for candidate consideration)
  — Context match threshold: 0.5 (reasonable context similarity)
```

### 5.10 Error Handling

```
LLM provider error (PFC):
  — Retry with exponential backoff (from nanobot's chat_with_retry)
  — After max retries: fall back to procedural cache if available
  — If no cache: return error signal to ACC → strategy_switch

Region process() exception:
  — Catch per-region, log, continue pipeline
  — Failing region's output treated as None (skip that stage)
  — ACC notified via error signal (increments error accumulator)
  — Dashboard notified (region shown in error state)

Database errors (SQLite/ChromaDB):
  — SQLite: WAL mode + retry on SQLITE_BUSY (up to 3 attempts)
  — ChromaDB: fallback to SQLite FTS5 full-text search if vector search fails
  — Consolidation failure: memories remain in staging (retry next trigger)

Dashboard WebSocket disconnect:
  — Agent continues processing normally (dashboard is optional)
  — Events buffered in a bounded deque (last 1000 events)
  — Reconnecting client receives recent buffer

Token budget (Hypothalamus):
  — Configurable per-session and per-day token limits (default: unlimited)
  — At 80% budget: increase patience parameter (prefer cached procedures)
  — At 100% budget: PFC falls back to procedural-only mode (no LLM calls)
  — Budget tracked via provider response token counts
```

---

## 6. Processing Flow

Complete flow for a single request:

```
1. [User Request]
   → Thalamus: preprocess/normalize
   → SensoryBuffer: register (flush previous)
   → SalienceNetwork: evaluate → DMN→ECN switch

2. [ECN Active]
   → Amygdala: emotional tagging (valence/arousal)
   → WorkingMemory: load into 4±1 slots (displace if needed)
   → RetrievalEngine: fetch relevant memories (spreading activation)
   → PFC (LLM): plan with goal stack + context + memories
       ├─ Procedural cache hit → skip LLM (fast path)
       └─ Cache miss → LLM reasoning (slow path)

3. [Evaluation]
   → ACC: conflict check on plan
       ├─ Conflict detected → return to PFC (deliberate more)
       └─ No conflict → proceed
   → BasalGanglia: Go/NoGo gate
       ├─ NoGo → hold / re-plan
       └─ Go → proceed

4. [Execution]
   → Cerebellum: predict outcome
   → Execute action (tool call)
   → Cerebellum: compute prediction error
       ├─ Small error → micro-adjust, continue
       └─ Large error → ACC escalation

5. [Learning & Storage]
   → ACC: outcome evaluation (expected vs actual)
   → Hippocampus: encode episode
   → Amygdala: attach emotional tag to episode
   → GWT Broadcast: result to all regions
       ├─ PFC: update goal stack
       ├─ ProceduralStore: pattern match / learn
       └─ SemanticStore: knowledge update

6. [Wind Down]
   → SalienceNetwork: ECN→DMN switch
   → [DMN Mode]: consolidation, pruning, idle
```

---

## 7. Visualization (Dashboard)

### 7.1 Tech Stack
- **Frontend:** React + Three.js (React Three Fiber)
- **3D Model:** GLB/GLTF brain mesh with separate region meshes
- **Communication:** WebSocket from Python backend → React frontend
- **State Management:** Zustand

### 7.2 Visual Design

**Brain Model:**
- Anatomically accurate 3D brain (sagittal/rotatable)
- Each region = separate mesh with emissive material
- Activation level controls emissive intensity

**Activity States:**
| State | Visual |
|-------|--------|
| Active | Full color + glow + gentle pulse |
| High Activity | Saturated + strong pulse + expanded glow |
| Inactive | Grey, no glow, dashed outline, low opacity |

**Network Modes:**
| Mode | Visual |
|------|--------|
| DMN | Cool blue ambient, memory regions glow |
| ECN | Warm orange pulse, executive regions active |
| SN Switch | Red flash burst from center |
| GWT Broadcast | Gold ripple wave expanding outward |

**Information Flow (key feature):**
- Green lines: forward data flow with labeled info packets
- Blue lines: feedback/error flow
- Pink lines: emotional tagging
- Each connection shows a label of what data is being transmitted
- Animated dots flow along connections showing direction

**HUD Panels:**
- Top-right: Network mode, neuromodulatory parameters, active regions
- Bottom: Memory flow pipeline (Sensory → WM → Hippocampus → Semantic)
- Top-left: Live event stream log

### 7.3 WebSocket Events

```
region_activation:  { region, level, mode }
network_switch:     { from, to, trigger }
routing_event:      { source, targets, priority, label }
broadcast:          { content, origin }
memory_event:       { type, store, id }
memory_flow:        { sensory, working, staging, semantic }
neuromodulator:     { urgency, learning_rate, patience, reward }
```

---

## 8. SDK & Package Structure

### 8.1 Distribution

```
pip install brain-agent              # Core SDK
pip install brain-agent[dashboard]   # + visualization dashboard
```

### 8.2 Directory Structure

```
brain-agent/
├── brain_agent/
│   ├── __init__.py          # BrainAgent class export
│   ├── __main__.py          # python -m brain_agent
│   ├── core/                # GWT Orchestrator, Thalamic Router, Network Modes, Neuromodulators
│   ├── regions/             # Brain region processors (base + 9 regions)
│   ├── memory/              # CLS memory system (8 modules)
│   ├── providers/           # LLM providers (from nanobot)
│   ├── tools/               # Tool registry (from nanobot)
│   ├── config/              # Pydantic config (from nanobot)
│   ├── bus/                 # Events/signals
│   └── cli/                 # CLI commands
├── dashboard/               # React + Three.js (separate npm package)
│   ├── src/components/      # BrainScene, RegionMesh, ConnectionLines, etc.
│   ├── src/hooks/           # useWebSocket
│   ├── src/stores/          # Zustand brain state
│   └── public/brain.glb     # 3D brain model asset
├── docs/
├── tests/
└── pyproject.toml
```

### 8.3 User API

```python
from brain_agent import BrainAgent

# Basic usage
agent = BrainAgent(provider="anthropic", model="claude-sonnet-4-20250514")
response = await agent.process("Find the bug in auth module")

# Session with auto-consolidation
async with agent.session() as session:
    r1 = await session.send("Explain the project structure")
    r2 = await session.send("Write tests for auth")

# Memory access
agent.memory.retrieve("authentication bug", top_k=5)
agent.memory.store(content="Project uses FastAPI", emotional_tag={"valence":0.3, "arousal":0.2})
agent.memory.stats()

# Region customization
agent = BrainAgent(region_overrides={"amygdala": CustomAmygdala()})

# Dashboard
agent.start_dashboard(port=3000)
```

### 8.4 CLI

```bash
brain-agent run                    # Start agent
brain-agent dashboard --port 3000  # Start visualization
brain-agent memory stats           # Memory statistics
brain-agent config                 # Configuration management
```

---

## 9. Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Backend | Python 3.11+ | Standard for AI/neuroscience community |
| Config | Pydantic v2 | Validated config (from nanobot) |
| LLM | LiteLLM / Provider abstraction | Multi-provider support (from nanobot) |
| Episodic/Procedural DB | SQLite | Zero dependency, local |
| Semantic DB | ChromaDB | Embedded vector DB, pip installable |
| Embeddings | sentence-transformers (default) | Local, no API needed |
| Frontend | React + TypeScript | Component-based UI |
| 3D | Three.js / React Three Fiber | Most mature 3D web ecosystem |
| Real-time | WebSocket (FastAPI) | Low-latency event streaming |
| State | Zustand | Lightweight React state |

---

## 10. Neuroscience References

### Memory Systems
- Atkinson & Shiffrin (1968) — Multi-store model
- Baddeley & Hitch (1974), Baddeley (2000) — Working memory model
- Cowan (2001) — Working memory capacity: 4±1
- Peterson & Peterson (1959) — Short-term memory decay
- McClelland, McNaughton & O'Reilly (1995) — Complementary Learning Systems
- Teyler & DiScenna (1986) — Hippocampal memory indexing theory
- Eichenbaum (2000) — Relational memory theory

### Memory Dynamics
- Ebbinghaus (1885) — Forgetting curve: R = e^(-t/S)
- Nader et al. (2000) — Reconsolidation theory
- Anderson et al. (1994) — Retrieval-induced forgetting
- Collins & Loftus (1975) — Spreading activation
- Tononi & Cirelli (2003) — Synaptic homeostasis hypothesis
- Winocur & Moscovitch (2011) — Episodic-to-semantic transition
- Godden & Baddeley (1975) — Context-dependent memory

### Emotional Memory
- McGaugh (2004) — Amygdala modulation of consolidation
- LeDoux (1996) — Dual amygdala pathways (fast/slow)
- Russell (1980) — Circumplex model (valence/arousal)
- Cahill & McGaugh (1995) — Emotional memory advantage

### Brain Networks
- Baars (1988) — Global Workspace Theory
- Dehaene & Changeux (2011) — Global Neuronal Workspace
- Menon & Uddin (2010) — Salience Network as causal hub
- Menon (2011) — Triple Network Model
- Fox et al. (2005) — DMN/ECN anti-correlation
- Sridharan et al. (2008) — SN causal switching
- Cole et al. (2013) — Frontoparietal flexible hub

### Brain Regions
- Miller & Cohen (2001) — PFC top-down bias signals
- Koechlin et al. (2003) — PFC hierarchical control
- Botvinick et al. (2001) — ACC conflict monitoring
- Mink (1996) — Basal ganglia focused disinhibition
- Frank (2005) — Go/NoGo reinforcement learning
- Ito (2008) — Cerebellar forward models
- Sherman & Guillery (2006) — Thalamus as router
- Halassa & Kastner (2017) — Thalamic gating

### Cognitive Architectures (Surveyed)
- Anderson (2007) — ACT-R
- Laird (2012) — SOAR
- Sun (2016) — CLARION
- Franklin et al. (2016) — LIDA
- Packer et al. (2023) — MemGPT
- Park et al. (2023) — Generative Agents
- Sumers et al. (2023) — CoALA framework

### Procedural Learning
- Fitts & Posner (1967) — Three stages of motor learning
- Newell & Rosenbloom (1981) — Power law of learning
- Graybiel (2008) — Habit chunking in basal ganglia
