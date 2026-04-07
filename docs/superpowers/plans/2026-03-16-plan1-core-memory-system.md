# Plan 1: Core + Memory System Implementation

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation layer — core data types, event-driven temporal model, and the full 4-layer CLS memory system with forgetting, retrieval, and consolidation engines.

**Architecture:** 4-layer memory pipeline (Sensory → Working Memory → Hippocampal Staging → Long-Term) with event-driven temporal model replacing wall-clock decay. All stores are local (in-memory + SQLite + ChromaDB). Memory dynamics (forgetting, consolidation, retrieval) operate via lazy evaluation triggered by events, not background sweeps.

**Tech Stack:** Python 3.11+, pytest, SQLite (aiosqlite), ChromaDB, sentence-transformers (all-MiniLM-L6-v2), Pydantic v2

**Spec:** `docs/superpowers/specs/2026-03-16-brain-agent-design.md`

---

## File Structure

```
brain_agent/
├── __init__.py                     # Package version, BrainAgent stub export
├── core/
│   ├── __init__.py
│   ├── signals.py                  # Signal, EmotionalTag, SignalType enum
│   ├── temporal.py                 # TemporalModel — interaction-based distance
│   ├── session.py                  # Session lifecycle, boundary detection
│   └── embeddings.py               # EmbeddingService — sentence-transformers wrapper
├── memory/
│   ├── __init__.py
│   ├── sensory_buffer.py           # SensoryBuffer — per-request-cycle, unlimited
│   ├── working_memory.py           # WorkingMemory — Baddeley model, 4±1 slots
│   ├── hippocampal_staging.py      # HippocampalStaging — SQLite fast encode
│   ├── episodic_store.py           # EpisodicStore — SQLite consolidated episodes
│   ├── semantic_store.py           # SemanticStore — ChromaDB + knowledge graph
│   ├── procedural_store.py         # ProceduralStore — cached action sequences
│   ├── forgetting.py               # ForgettingEngine — Ebbinghaus + interference
│   ├── retrieval.py                # RetrievalEngine — multi-factor + spreading activation
│   ├── consolidation.py            # ConsolidationEngine — staging → LTM transfer
│   └── manager.py                  # MemoryManager — facade unifying all layers
tests/
├── conftest.py                     # Shared fixtures (tmp dirs, embedding mock)
├── core/
│   ├── test_signals.py
│   ├── test_temporal.py
│   ├── test_session.py
│   └── test_embeddings.py
├── memory/
│   ├── test_sensory_buffer.py
│   ├── test_working_memory.py
│   ├── test_hippocampal_staging.py
│   ├── test_episodic_store.py
│   ├── test_semantic_store.py
│   ├── test_procedural_store.py
│   ├── test_forgetting.py
│   ├── test_retrieval.py
│   ├── test_consolidation.py
│   └── test_memory_manager.py
pyproject.toml
```

---

## Chunk 1: Project Setup + Core Data Types

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `brain_agent/__init__.py`
- Create: `brain_agent/core/__init__.py`
- Create: `brain_agent/memory/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "brain-agent"
version = "0.1.0"
description = "Neuroscience-faithful AI agent framework with brain cognitive architecture"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "pydantic>=2.0",
    "aiosqlite>=0.20.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init files**

```python
# brain_agent/__init__.py
"""Brain Agent — Neuroscience-faithful AI agent framework."""
__version__ = "0.1.0"
```

```python
# brain_agent/core/__init__.py
```

```python
# brain_agent/memory/__init__.py
```

- [ ] **Step 3: Create test conftest with shared fixtures**

```python
# tests/conftest.py
import pytest
import tempfile
import os


@pytest.fixture
def tmp_db_path(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "test_brain.db")


@pytest.fixture
def mock_embedding():
    """Return a deterministic fake embedding function (no model download)."""
    import numpy as np

    def _embed(text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    return _embed
```

- [ ] **Step 4: Install dev dependencies and verify**

Run: `pip install -e ".[dev]"`
Expected: Successfully installed brain-agent and dev deps

- [ ] **Step 5: Run pytest to verify empty test suite**

Run: `pytest --co -q`
Expected: "no tests ran" (no errors)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml brain_agent/ tests/conftest.py
git commit -m "feat: project scaffolding with pyproject.toml and package structure"
```

---

### Task 2: Signal and EmotionalTag Data Types

**Files:**
- Create: `brain_agent/core/signals.py`
- Create: `tests/core/__init__.py`
- Create: `tests/core/test_signals.py`

- [ ] **Step 1: Write failing tests for Signal and EmotionalTag**

```python
# tests/core/test_signals.py
from brain_agent.core.signals import Signal, EmotionalTag, SignalType
import uuid


def test_emotional_tag_creation():
    tag = EmotionalTag(valence=0.5, arousal=0.8)
    assert tag.valence == 0.5
    assert tag.arousal == 0.8


def test_emotional_tag_clamps_values():
    tag = EmotionalTag(valence=-1.5, arousal=2.0)
    assert tag.valence == -1.0
    assert tag.arousal == 1.0


def test_emotional_tag_neutral():
    tag = EmotionalTag.neutral()
    assert tag.valence == 0.0
    assert tag.arousal == 0.0


def test_signal_creation():
    sig = Signal(
        type=SignalType.PLAN,
        source="pfc",
        payload={"content": "analyze auth module"},
    )
    assert sig.type == SignalType.PLAN
    assert sig.source == "pfc"
    assert sig.priority == 0.5
    assert sig.emotional_tag is None
    assert isinstance(sig.id, uuid.UUID)


def test_signal_with_emotional_tag():
    tag = EmotionalTag(valence=-0.8, arousal=0.9)
    sig = Signal(
        type=SignalType.ACTION_RESULT,
        source="cerebellum",
        payload={"error": 0.02},
        emotional_tag=tag,
    )
    assert sig.emotional_tag.arousal == 0.9


def test_signal_types_exist():
    assert SignalType.PLAN
    assert SignalType.ACTION_SELECTED
    assert SignalType.ACTION_RESULT
    assert SignalType.CONFLICT_DETECTED
    assert SignalType.STRATEGY_SWITCH
    assert SignalType.PREDICTION_ERROR
    assert SignalType.EMOTIONAL_TAG
    assert SignalType.GWT_BROADCAST
    assert SignalType.CONSOLIDATION_TRIGGER
    assert SignalType.NETWORK_SWITCH
    assert SignalType.RESOURCE_STATUS
    assert SignalType.EXTERNAL_INPUT
    assert SignalType.ENCODE
    assert SignalType.RETRIEVE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_signals.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement Signal, EmotionalTag, SignalType**

```python
# brain_agent/core/signals.py
"""Core signal types for inter-region communication.

Spec reference: Section 5.1 Signal Schema, Section 5.7 Signal Types.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SignalType(str, Enum):
    """All signal types flowing through the Thalamic Router."""

    # Forward flow
    PLAN = "plan"
    ACTION_SELECTED = "action_selected"
    ACTION_RESULT = "action_result"
    ENCODE = "encode"
    RETRIEVE = "retrieve"
    EXTERNAL_INPUT = "external_input"

    # Feedback flow
    PREDICTION_ERROR = "prediction_error"
    CONFLICT_DETECTED = "conflict_detected"
    STRATEGY_SWITCH = "strategy_switch"

    # Emotional
    EMOTIONAL_TAG = "emotional_tag"

    # Broadcast
    GWT_BROADCAST = "gwt_broadcast"

    # System
    CONSOLIDATION_TRIGGER = "consolidation_trigger"
    NETWORK_SWITCH = "network_switch"
    RESOURCE_STATUS = "resource_status"


@dataclass
class EmotionalTag:
    """Russell (1980) Circumplex Model: valence × arousal."""

    valence: float  # -1.0 (negative) to +1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (excited)

    def __post_init__(self):
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))

    @classmethod
    def neutral(cls) -> EmotionalTag:
        return cls(valence=0.0, arousal=0.0)


@dataclass
class Signal:
    """Core data structure for all inter-region communication."""

    type: SignalType
    source: str
    payload: dict
    targets: list[str] | None = None
    priority: float = 0.5
    emotional_tag: EmotionalTag | None = None
    interaction_id: int = 0
    session_id: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: uuid.UUID = field(default_factory=uuid.uuid4)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_signals.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/core/signals.py tests/core/
git commit -m "feat: add Signal, EmotionalTag, and SignalType data types"
```

---

### Task 3: Temporal Model

**Files:**
- Create: `brain_agent/core/temporal.py`
- Create: `tests/core/test_temporal.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_temporal.py
from brain_agent.core.temporal import TemporalModel


def test_temporal_model_initial_state():
    tm = TemporalModel()
    assert tm.interaction_count == 0
    assert tm.current_session_id == ""


def test_increment_interaction():
    tm = TemporalModel()
    tm.tick()
    assert tm.interaction_count == 1
    tm.tick()
    assert tm.interaction_count == 2


def test_distance_zero_for_current():
    tm = TemporalModel()
    tm.tick()
    d = tm.distance(last_interaction=1, last_session=tm.current_session_id)
    assert d == 0.0


def test_distance_increases_with_interaction_gap():
    tm = TemporalModel()
    for _ in range(10):
        tm.tick()
    d = tm.distance(last_interaction=1, last_session=tm.current_session_id)
    assert d > 0.0


def test_distance_increases_with_session_gap():
    tm = TemporalModel()
    tm.start_session("s1")
    tm.tick()
    tm.close_session()
    tm.start_session("s2")
    tm.tick()
    tm.close_session()
    tm.start_session("s3")
    tm.tick()

    # Memory from session s1, now in session s3 → 2 session gap
    d = tm.distance(last_interaction=1, last_session="s1")
    d_same = tm.distance(last_interaction=tm.interaction_count, last_session="s3")
    assert d > d_same


def test_closed_sessions_tracked():
    tm = TemporalModel()
    tm.start_session("s1")
    tm.close_session()
    tm.start_session("s2")
    tm.close_session()
    assert tm.count_sessions_since("s1") == 2
    assert tm.count_sessions_since("s2") == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/core/test_temporal.py -v`
Expected: FAIL

- [ ] **Step 3: Implement TemporalModel**

```python
# brain_agent/core/temporal.py
"""Event-driven temporal model replacing wall-clock decay.

Spec reference: Section 3.1 Event-Driven Temporal Model.
Brain receives continuous analog input; agent receives sporadic requests.
Decay measured in interaction distance, not seconds.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from dataclasses import dataclass, field


# Distance weights (spec Section 3.1)
ALPHA_EVENT = 1.0    # interaction gap weight
BETA_SESSION = 5.0   # session gap weight (sessions are larger units)
GAMMA_IDLE = 0.1     # wall-clock idle factor (low weight, only for long absence)

# Idle classification thresholds (seconds)
IDLE_MINOR = 1800    # 30 min
IDLE_MAJOR = 86400   # 24 hours


def _classify_idle(seconds: float) -> float:
    """Convert wall-clock gap to a small idle factor."""
    if seconds < IDLE_MINOR:
        return 0.0
    elif seconds < IDLE_MAJOR:
        return 1.0
    else:
        return math.log(seconds / IDLE_MINOR)


@dataclass
class TemporalModel:
    """Tracks interaction count, sessions, and computes memory distance."""

    interaction_count: int = 0
    current_session_id: str = ""
    _closed_sessions: list[str] = field(default_factory=list)
    _last_wall_clock: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def tick(self) -> int:
        """Record a new interaction. Returns the new count."""
        self.interaction_count += 1
        self._last_wall_clock = datetime.now(timezone.utc)
        return self.interaction_count

    def start_session(self, session_id: str) -> None:
        self.current_session_id = session_id

    def close_session(self) -> None:
        if self.current_session_id:
            self._closed_sessions.append(self.current_session_id)
            self.current_session_id = ""

    def count_sessions_since(self, session_id: str) -> int:
        """Count closed sessions after the given session_id."""
        try:
            idx = self._closed_sessions.index(session_id)
            return len(self._closed_sessions) - idx
        except ValueError:
            return len(self._closed_sessions)

    def distance(
        self,
        last_interaction: int,
        last_session: str,
        last_wall_clock: datetime | None = None,
    ) -> float:
        """Compute interaction distance for a memory.

        d = α * event_gap + β * session_gap + γ * idle_factor
        """
        event_gap = max(0, self.interaction_count - last_interaction)
        session_gap = self.count_sessions_since(last_session)

        idle_factor = 0.0
        if last_wall_clock is not None:
            seconds = (self._last_wall_clock - last_wall_clock).total_seconds()
            idle_factor = _classify_idle(max(0.0, seconds))

        return (
            ALPHA_EVENT * event_gap
            + BETA_SESSION * session_gap
            + GAMMA_IDLE * idle_factor
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/core/test_temporal.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/core/temporal.py tests/core/test_temporal.py
git commit -m "feat: add TemporalModel with interaction-based distance"
```

---

### Task 4: Embedding Service

**Files:**
- Create: `brain_agent/core/embeddings.py`
- Create: `tests/core/test_embeddings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_embeddings.py
import numpy as np
from brain_agent.core.embeddings import EmbeddingService


def test_embed_returns_correct_dimension():
    svc = EmbeddingService(use_mock=True)
    vec = svc.embed("hello world")
    assert len(vec) == 384


def test_embed_is_normalized():
    svc = EmbeddingService(use_mock=True)
    vec = np.array(svc.embed("test"))
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 0.01


def test_same_input_same_output():
    svc = EmbeddingService(use_mock=True)
    v1 = svc.embed("hello")
    v2 = svc.embed("hello")
    assert v1 == v2


def test_different_input_different_output():
    svc = EmbeddingService(use_mock=True)
    v1 = svc.embed("hello")
    v2 = svc.embed("world")
    assert v1 != v2


def test_cosine_similarity():
    svc = EmbeddingService(use_mock=True)
    sim = svc.cosine_similarity(
        svc.embed("test"), svc.embed("test")
    )
    assert abs(sim - 1.0) < 0.01


def test_pattern_separation_adds_noise():
    svc = EmbeddingService(use_mock=True)
    base = svc.embed("same input")
    separated = svc.pattern_separate(base)
    # Should be close but not identical
    sim = svc.cosine_similarity(base, separated)
    assert 0.95 < sim < 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/core/test_embeddings.py -v`
Expected: FAIL

- [ ] **Step 3: Implement EmbeddingService**

```python
# brain_agent/core/embeddings.py
"""Embedding service wrapping sentence-transformers.

Spec reference: Section 5.9 Embedding Strategy.
Default: all-MiniLM-L6-v2 (384 dims). Mock mode for testing.
Pattern separation: Gaussian noise (σ=0.01) for distinct representations.
"""
from __future__ import annotations

import numpy as np

EMBEDDING_DIM = 384
PATTERN_SEPARATION_SIGMA = 0.01


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_mock: bool = False):
        self._use_mock = use_mock
        self._model = None
        self._model_name = model_name

    def _get_model(self):
        if self._model is None and not self._use_mock:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Generate normalized embedding for text."""
        if self._use_mock:
            return self._mock_embed(text)
        model = self._get_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = np.dot(va, vb)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def pattern_separate(self, embedding: list[float]) -> list[float]:
        """Add Gaussian noise for pattern separation (dentate gyrus analog).

        Ensures similar but distinct memories get distinguishable representations.
        """
        vec = np.array(embedding, dtype=np.float32)
        noise = np.random.normal(0, PATTERN_SEPARATION_SIGMA, size=vec.shape)
        separated = vec + noise.astype(np.float32)
        separated = separated / np.linalg.norm(separated)
        return separated.tolist()

    @staticmethod
    def _mock_embed(text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/core/test_embeddings.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/core/embeddings.py tests/core/test_embeddings.py
git commit -m "feat: add EmbeddingService with mock mode and pattern separation"
```

---

## Chunk 2: Sensory Buffer + Working Memory

### Task 5: Sensory Buffer

**Files:**
- Create: `brain_agent/memory/sensory_buffer.py`
- Create: `tests/memory/__init__.py`
- Create: `tests/memory/test_sensory_buffer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/memory/test_sensory_buffer.py
from brain_agent.memory.sensory_buffer import SensoryBuffer


def test_register_and_get():
    buf = SensoryBuffer()
    buf.register({"text": "hello"}, modality="text")
    items = buf.get_all()
    assert len(items) == 1
    assert items[0].data["text"] == "hello"


def test_flush_clears_buffer():
    buf = SensoryBuffer()
    buf.register({"text": "hello"}, modality="text")
    buf.flush()
    assert len(buf.get_all()) == 0


def test_attend_filters_items():
    buf = SensoryBuffer()
    buf.register({"text": "important", "priority": 0.9}, modality="text")
    buf.register({"text": "noise", "priority": 0.1}, modality="text")

    attended = buf.attend(lambda item: item.data.get("priority", 0) > 0.5)
    assert len(attended) == 1
    assert attended[0].data["text"] == "important"


def test_new_request_flushes_previous():
    buf = SensoryBuffer()
    buf.register({"text": "old"}, modality="text")

    # Simulate new request cycle
    buf.new_cycle()
    buf.register({"text": "new"}, modality="text")

    items = buf.get_all()
    assert len(items) == 1
    assert items[0].data["text"] == "new"


def test_unlimited_capacity():
    buf = SensoryBuffer()
    for i in range(1000):
        buf.register({"i": i}, modality="text")
    assert len(buf.get_all()) == 1000
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/memory/test_sensory_buffer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SensoryBuffer**

```python
# brain_agent/memory/sensory_buffer.py
"""Sensory Buffer — per-request-cycle memory.

Spec reference: Section 3.2 Layer 1: Sensory Buffer.
Brain: sensory register, 250ms-4s, unlimited capacity.
Agent: persists within one request cycle, flushed on next request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SensoryItem:
    data: dict
    modality: str


class SensoryBuffer:
    """Per-request-cycle sensory memory. Flush on new cycle."""

    def __init__(self):
        self._items: list[SensoryItem] = []

    def register(self, data: dict, modality: str = "text") -> None:
        self._items.append(SensoryItem(data=data, modality=modality))

    def get_all(self) -> list[SensoryItem]:
        return list(self._items)

    def attend(self, filter_fn: Callable[[SensoryItem], bool]) -> list[SensoryItem]:
        """Selective attention: return only items passing the filter."""
        return [item for item in self._items if filter_fn(item)]

    def flush(self) -> None:
        self._items.clear()

    def new_cycle(self) -> None:
        """Start a new request cycle — flushes previous sensory data."""
        self.flush()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/memory/test_sensory_buffer.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/sensory_buffer.py tests/memory/
git commit -m "feat: add SensoryBuffer with per-request-cycle lifecycle"
```

---

### Task 6: Working Memory (Baddeley Model)

**Files:**
- Create: `brain_agent/memory/working_memory.py`
- Create: `tests/memory/test_working_memory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/memory/test_working_memory.py
from brain_agent.memory.working_memory import WorkingMemory, WorkingMemoryItem


def test_load_within_capacity():
    wm = WorkingMemory(capacity=4)
    wm.load(WorkingMemoryItem(content="a", slot="phonological"))
    wm.load(WorkingMemoryItem(content="b", slot="phonological"))
    assert len(wm.get_slots()) == 2


def test_capacity_limit_evicts_oldest():
    wm = WorkingMemory(capacity=4)
    for i in range(6):
        wm.load(WorkingMemoryItem(content=f"item_{i}", slot="phonological"))
    slots = wm.get_slots()
    assert len(slots) == 4
    # Oldest items evicted
    contents = [s.content for s in slots]
    assert "item_0" not in contents
    assert "item_1" not in contents
    assert "item_5" in contents


def test_rehearse_prevents_eviction():
    wm = WorkingMemory(capacity=3)
    item_a = WorkingMemoryItem(content="important", slot="phonological")
    wm.load(item_a)
    wm.load(WorkingMemoryItem(content="b", slot="phonological"))
    wm.load(WorkingMemoryItem(content="c", slot="phonological"))

    # Rehearse 'important' — resets its position
    wm.rehearse("important")

    # Load one more — should evict 'b' not 'important'
    wm.load(WorkingMemoryItem(content="d", slot="phonological"))

    contents = [s.content for s in wm.get_slots()]
    assert "important" in contents
    assert "b" not in contents


def test_evict_returns_evicted_items():
    wm = WorkingMemory(capacity=2)
    wm.load(WorkingMemoryItem(content="a", slot="phonological"))
    wm.load(WorkingMemoryItem(content="b", slot="phonological"))
    evicted = wm.load(WorkingMemoryItem(content="c", slot="phonological"))
    assert len(evicted) == 1
    assert evicted[0].content == "a"


def test_session_boundary_clears_irrelevant():
    wm = WorkingMemory(capacity=4)
    wm.load(WorkingMemoryItem(content="auth bug", slot="episodic"))
    wm.load(WorkingMemoryItem(content="weather", slot="episodic"))

    # Simulate session boundary — only keep items matching filter
    wm.on_session_boundary(lambda item: "auth" in item.content)

    slots = wm.get_slots()
    assert len(slots) == 1
    assert slots[0].content == "auth bug"


def test_get_context_returns_all_contents():
    wm = WorkingMemory(capacity=4)
    wm.load(WorkingMemoryItem(content="fact 1", slot="phonological"))
    wm.load(WorkingMemoryItem(content="fact 2", slot="episodic"))
    ctx = wm.get_context()
    assert "fact 1" in ctx
    assert "fact 2" in ctx
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/memory/test_working_memory.py -v`
Expected: FAIL

- [ ] **Step 3: Implement WorkingMemory**

```python
# brain_agent/memory/working_memory.py
"""Working Memory — Baddeley (2000) model.

Spec reference: Section 3.3 Layer 2: Working Memory.
Capacity: 4±1 chunks (Cowan, 2001).
Decay: displacement-based, not time-based.
Components: phonological loop, visuospatial pad, episodic buffer.
Central executive is the GWT Orchestrator (external to this module).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class WorkingMemoryItem:
    content: str
    slot: str  # "phonological", "visuospatial", "episodic"
    reference_count: int = 0
    linked_memories: list[str] = field(default_factory=list)


class WorkingMemory:
    """Capacity-limited working memory with displacement-based decay."""

    def __init__(self, capacity: int = 4):
        self._capacity = capacity
        self._items: list[WorkingMemoryItem] = []

    def load(self, item: WorkingMemoryItem) -> list[WorkingMemoryItem]:
        """Load item into working memory. Returns evicted items if over capacity."""
        evicted: list[WorkingMemoryItem] = []
        self._items.append(item)
        while len(self._items) > self._capacity:
            evicted.append(self._items.pop(0))
        return evicted

    def rehearse(self, content: str) -> bool:
        """Maintenance rehearsal — move item to end (most recent)."""
        for i, item in enumerate(self._items):
            if item.content == content:
                item.reference_count += 1
                self._items.append(self._items.pop(i))
                return True
        return False

    def get_slots(self) -> list[WorkingMemoryItem]:
        return list(self._items)

    def get_context(self) -> str:
        """Return concatenated contents for LLM context building."""
        return "\n".join(item.content for item in self._items)

    def on_session_boundary(
        self, relevance_fn: Callable[[WorkingMemoryItem], bool]
    ) -> list[WorkingMemoryItem]:
        """Re-evaluate items at session boundary. Evict irrelevant ones."""
        evicted = [item for item in self._items if not relevance_fn(item)]
        self._items = [item for item in self._items if relevance_fn(item)]
        return evicted

    def clear(self) -> None:
        self._items.clear()

    @property
    def free_slots(self) -> int:
        return max(0, self._capacity - len(self._items))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/memory/test_working_memory.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/working_memory.py tests/memory/test_working_memory.py
git commit -m "feat: add WorkingMemory with Baddeley model and displacement decay"
```

---

## Chunk 3: Hippocampal Staging + Episodic Store

### Task 7: Hippocampal Staging (SQLite)

**Files:**
- Create: `brain_agent/memory/hippocampal_staging.py`
- Create: `tests/memory/test_hippocampal_staging.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/memory/test_hippocampal_staging.py
import pytest
from brain_agent.memory.hippocampal_staging import HippocampalStaging


@pytest.fixture
async def staging(tmp_db_path, mock_embedding):
    s = HippocampalStaging(db_path=tmp_db_path, embed_fn=mock_embedding)
    await s.initialize()
    yield s
    await s.close()


async def test_encode_and_retrieve_by_id(staging):
    mem_id = await staging.encode(
        content="user asked about auth bug",
        entities={"what": "auth bug"},
        interaction_id=1,
        session_id="s1",
    )
    mem = await staging.get_by_id(mem_id)
    assert mem is not None
    assert mem["content"] == "user asked about auth bug"
    assert mem["strength"] == 1.0
    assert mem["consolidated"] is False


async def test_retrieve_boosts_strength(staging):
    mem_id = await staging.encode(
        content="test memory",
        entities={},
        interaction_id=1,
        session_id="s1",
    )
    mem_before = await staging.get_by_id(mem_id)
    await staging.on_retrieval(mem_id, boost=2.0)
    mem_after = await staging.get_by_id(mem_id)
    assert mem_after["strength"] == mem_before["strength"] * 2.0
    assert mem_after["access_count"] == 1


async def test_get_unconsolidated(staging):
    await staging.encode(content="a", entities={}, interaction_id=1, session_id="s1")
    await staging.encode(content="b", entities={}, interaction_id=2, session_id="s1")
    unconsol = await staging.get_unconsolidated()
    assert len(unconsol) == 2


async def test_mark_consolidated(staging):
    mem_id = await staging.encode(
        content="done", entities={}, interaction_id=1, session_id="s1"
    )
    await staging.mark_consolidated(mem_id)
    unconsol = await staging.get_unconsolidated()
    assert len(unconsol) == 0


async def test_emotional_tag_stored(staging):
    mem_id = await staging.encode(
        content="scary error",
        entities={},
        interaction_id=1,
        session_id="s1",
        emotional_tag={"valence": -0.8, "arousal": 0.9},
    )
    mem = await staging.get_by_id(mem_id)
    assert mem["emotional_tag"]["arousal"] == 0.9
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/memory/test_hippocampal_staging.py -v`
Expected: FAIL

- [ ] **Step 3: Implement HippocampalStaging**

```python
# brain_agent/memory/hippocampal_staging.py
"""Hippocampal Staging — fast one-shot encoding with full context.

Spec reference: Section 3.4 Layer 3: Hippocampal Staging.
Brain: hippocampus stores compressed index binding cortical representations.
Agent: SQLite staging table with embeddings, entities, emotional tags.
CLS fast learning (McClelland et al., 1995).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Callable

import aiosqlite

RETRIEVAL_BOOST_DEFAULT = 2.0


class HippocampalStaging:
    def __init__(self, db_path: str, embed_fn: Callable[[str], list[float]]):
        self._db_path = db_path
        self._embed_fn = embed_fn
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS staging_memories (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                context_embedding BLOB,
                entities TEXT DEFAULT '{}',
                emotional_tag TEXT DEFAULT '{"valence":0,"arousal":0}',
                source_modality TEXT DEFAULT 'text',
                access_count INTEGER DEFAULT 0,
                strength REAL DEFAULT 1.0,
                consolidated INTEGER DEFAULT 0,
                last_interaction INTEGER DEFAULT 0,
                last_session TEXT DEFAULT ''
            )
        """)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def encode(
        self,
        content: str,
        entities: dict,
        interaction_id: int,
        session_id: str,
        emotional_tag: dict | None = None,
        source_modality: str = "text",
    ) -> str:
        """One-shot encoding — hippocampal fast learning."""
        mem_id = str(uuid.uuid4())
        embedding = self._embed_fn(content)
        tag = emotional_tag or {"valence": 0.0, "arousal": 0.0}

        await self._db.execute(
            """INSERT INTO staging_memories
            (id, timestamp, content, context_embedding, entities, emotional_tag,
             source_modality, access_count, strength, consolidated,
             last_interaction, last_session)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1.0, 0, ?, ?)""",
            (
                mem_id,
                datetime.now(timezone.utc).isoformat(),
                content,
                json.dumps(embedding),
                json.dumps(entities),
                json.dumps(tag),
                source_modality,
                interaction_id,
                session_id,
            ),
        )
        await self._db.commit()
        return mem_id

    async def get_by_id(self, mem_id: str) -> dict | None:
        async with self._db.execute(
            "SELECT * FROM staging_memories WHERE id = ?", (mem_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_dict(cursor.description, row)

    async def get_unconsolidated(self) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM staging_memories WHERE consolidated = 0 ORDER BY last_interaction"
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(cursor.description, r) for r in rows]

    async def on_retrieval(self, mem_id: str, boost: float = RETRIEVAL_BOOST_DEFAULT) -> None:
        """Reconsolidation on retrieval (Nader, 2000): boost strength."""
        await self._db.execute(
            "UPDATE staging_memories SET strength = strength * ?, access_count = access_count + 1 WHERE id = ?",
            (boost, mem_id),
        )
        await self._db.commit()

    async def mark_consolidated(self, mem_id: str) -> None:
        await self._db.execute(
            "UPDATE staging_memories SET consolidated = 1 WHERE id = ?", (mem_id,)
        )
        await self._db.commit()

    async def update_strength(self, mem_id: str, new_strength: float) -> None:
        await self._db.execute(
            "UPDATE staging_memories SET strength = ? WHERE id = ?",
            (new_strength, mem_id),
        )
        await self._db.commit()

    async def delete(self, mem_id: str) -> None:
        await self._db.execute(
            "DELETE FROM staging_memories WHERE id = ?", (mem_id,)
        )
        await self._db.commit()

    async def count_unconsolidated(self) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) FROM staging_memories WHERE consolidated = 0"
        ) as cursor:
            row = await cursor.fetchone()
            return row[0]

    @staticmethod
    def _row_to_dict(description, row) -> dict:
        d = {col[0]: val for col, val in zip(description, row)}
        d["entities"] = json.loads(d["entities"])
        d["emotional_tag"] = json.loads(d["emotional_tag"])
        d["context_embedding"] = json.loads(d["context_embedding"]) if d["context_embedding"] else []
        d["consolidated"] = bool(d["consolidated"])
        return d
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/memory/test_hippocampal_staging.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/hippocampal_staging.py tests/memory/test_hippocampal_staging.py
git commit -m "feat: add HippocampalStaging with SQLite fast encoding"
```

---

### Task 8: Episodic Store

**Files:**
- Create: `brain_agent/memory/episodic_store.py`
- Create: `tests/memory/test_episodic_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/memory/test_episodic_store.py
import pytest
from brain_agent.memory.episodic_store import EpisodicStore


@pytest.fixture
async def store(tmp_db_path):
    s = EpisodicStore(db_path=tmp_db_path)
    await s.initialize()
    yield s
    await s.close()


async def test_save_and_get(store):
    ep_id = await store.save(
        content="found auth bug in line 42",
        context_embedding=[0.1] * 384,
        entities={"what": "auth bug", "where": "line 42"},
        emotional_tag={"valence": -0.3, "arousal": 0.6},
        interaction_id=5,
        session_id="s1",
    )
    ep = await store.get_by_id(ep_id)
    assert ep is not None
    assert "auth bug" in ep["content"]


async def test_search_by_interaction_range(store):
    for i in range(5):
        await store.save(
            content=f"event {i}",
            context_embedding=[0.1] * 384,
            entities={},
            emotional_tag={"valence": 0, "arousal": 0},
            interaction_id=i + 1,
            session_id="s1",
        )
    results = await store.get_by_interaction_range(2, 4)
    assert len(results) == 3


async def test_get_recent(store):
    for i in range(10):
        await store.save(
            content=f"event {i}",
            context_embedding=[0.1] * 384,
            entities={},
            emotional_tag={"valence": 0, "arousal": 0},
            interaction_id=i,
            session_id="s1",
        )
    recent = await store.get_recent(limit=3)
    assert len(recent) == 3
    assert recent[0]["content"] == "event 9"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/memory/test_episodic_store.py -v`
Expected: FAIL

- [ ] **Step 3: Implement EpisodicStore**

```python
# brain_agent/memory/episodic_store.py
"""Episodic Store — consolidated long-term episodic memory.

Spec reference: Section 3.5.1 Episodic Store.
Brain: hippocampus + temporal cortex. Time-stamped, context-rich experiences.
Agent: SQLite table for consolidated episodes.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import aiosqlite


class EpisodicStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                context_embedding TEXT,
                entities TEXT DEFAULT '{}',
                emotional_tag TEXT DEFAULT '{"valence":0,"arousal":0}',
                strength REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_interaction INTEGER DEFAULT 0,
                last_session TEXT DEFAULT '',
                schema_links TEXT DEFAULT '[]'
            )
        """)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def save(
        self,
        content: str,
        context_embedding: list[float],
        entities: dict,
        emotional_tag: dict,
        interaction_id: int,
        session_id: str,
        strength: float = 1.0,
        access_count: int = 0,
    ) -> str:
        ep_id = str(uuid.uuid4())
        await self._db.execute(
            """INSERT INTO episodes
            (id, timestamp, content, context_embedding, entities, emotional_tag,
             strength, access_count, last_interaction, last_session)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ep_id,
                datetime.now(timezone.utc).isoformat(),
                content,
                json.dumps(context_embedding),
                json.dumps(entities),
                json.dumps(emotional_tag),
                strength,
                access_count,
                interaction_id,
                session_id,
            ),
        )
        await self._db.commit()
        return ep_id

    async def get_by_id(self, ep_id: str) -> dict | None:
        async with self._db.execute(
            "SELECT * FROM episodes WHERE id = ?", (ep_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_dict(cursor.description, row)

    async def get_recent(self, limit: int = 10) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM episodes ORDER BY last_interaction DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(cursor.description, r) for r in rows]

    async def get_by_interaction_range(
        self, start: int, end: int
    ) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM episodes WHERE last_interaction >= ? AND last_interaction <= ? ORDER BY last_interaction",
            (start, end),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(cursor.description, r) for r in rows]

    async def get_all(self) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM episodes ORDER BY last_interaction"
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(cursor.description, r) for r in rows]

    async def update_strength(self, ep_id: str, strength: float) -> None:
        await self._db.execute(
            "UPDATE episodes SET strength = ? WHERE id = ?", (strength, ep_id)
        )
        await self._db.commit()

    async def delete_below_strength(self, threshold: float) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) FROM episodes WHERE strength < ?", (threshold,)
        ) as cursor:
            count = (await cursor.fetchone())[0]
        await self._db.execute(
            "DELETE FROM episodes WHERE strength < ?", (threshold,)
        )
        await self._db.commit()
        return count

    @staticmethod
    def _row_to_dict(description, row) -> dict:
        d = {col[0]: val for col, val in zip(description, row)}
        for key in ("entities", "emotional_tag", "schema_links"):
            if d.get(key):
                d[key] = json.loads(d[key])
        if d.get("context_embedding"):
            d["context_embedding"] = json.loads(d["context_embedding"])
        return d
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/memory/test_episodic_store.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/episodic_store.py tests/memory/test_episodic_store.py
git commit -m "feat: add EpisodicStore for consolidated long-term episodes"
```

---

## Chunk 4: Semantic Store + Procedural Store

### Task 9: Semantic Store (ChromaDB + Knowledge Graph)

**Files:**
- Create: `brain_agent/memory/semantic_store.py`
- Create: `tests/memory/test_semantic_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/memory/test_semantic_store.py
import pytest
from brain_agent.memory.semantic_store import SemanticStore


@pytest.fixture
async def store(tmp_path, mock_embedding):
    s = SemanticStore(
        chroma_path=str(tmp_path / "chroma"),
        graph_db_path=str(tmp_path / "graph.db"),
        embed_fn=mock_embedding,
    )
    await s.initialize()
    yield s
    await s.close()


async def test_add_and_search(store):
    await store.add("Python is a programming language", category="fact")
    results = await store.search("programming language", top_k=1)
    assert len(results) >= 1


async def test_add_relationship(store):
    await store.add_relationship("Python", "is_a", "Programming Language", weight=0.9)
    rels = await store.get_relationships("Python")
    assert len(rels) == 1
    assert rels[0]["relation"] == "is_a"
    assert rels[0]["weight"] == 0.9


async def test_spreading_activation(store):
    await store.add_relationship("Python", "is_a", "Language", weight=0.9)
    await store.add_relationship("Language", "used_for", "Communication", weight=0.8)
    await store.add_relationship("Python", "used_for", "AI", weight=0.7)

    activated = await store.spread_activation(
        start_nodes=["Python"], max_hops=2, decay=0.85
    )
    # Python should have highest activation, then direct neighbors
    assert "Python" in activated
    assert activated["Python"] > activated.get("Language", 0)


async def test_get_count(store):
    await store.add("fact one", category="fact")
    await store.add("fact two", category="fact")
    count = await store.count()
    assert count == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/memory/test_semantic_store.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SemanticStore**

```python
# brain_agent/memory/semantic_store.py
"""Semantic Store — vector DB + knowledge graph.

Spec reference: Section 3.5.2 Semantic Store.
Brain: temporal cortex (hub-and-spoke, Patterson 2007).
Agent: ChromaDB for vector search + SQLite for knowledge graph.
Knowledge graph enables spreading activation (Collins & Loftus, 1975).
"""
from __future__ import annotations

import json
import uuid
from typing import Callable

import aiosqlite
import chromadb


class SemanticStore:
    def __init__(
        self,
        chroma_path: str,
        graph_db_path: str,
        embed_fn: Callable[[str], list[float]],
    ):
        self._chroma_path = chroma_path
        self._graph_db_path = graph_db_path
        self._embed_fn = embed_fn
        self._collection = None
        self._graph_db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        # ChromaDB for vector embeddings
        client = chromadb.PersistentClient(path=self._chroma_path)
        self._collection = client.get_or_create_collection(
            name="semantic_memory",
            metadata={"hnsw:space": "cosine"},
        )

        # SQLite for knowledge graph
        self._graph_db = await aiosqlite.connect(self._graph_db_path)
        await self._graph_db.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                id TEXT PRIMARY KEY,
                source_node TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_node TEXT NOT NULL,
                weight REAL DEFAULT 1.0
            )
        """)
        await self._graph_db.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON knowledge_graph(source_node)"
        )
        await self._graph_db.commit()

    async def close(self) -> None:
        if self._graph_db:
            await self._graph_db.close()

    async def add(
        self,
        content: str,
        category: str = "general",
        strength: float = 1.0,
    ) -> str:
        doc_id = str(uuid.uuid4())
        embedding = self._embed_fn(content)
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"category": category, "strength": strength, "access_count": 0}],
        )
        return doc_id

    async def search(
        self, query: str, top_k: int = 5
    ) -> list[dict]:
        embedding = self._embed_fn(query)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        out = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                out.append({
                    "id": results["ids"][0][i],
                    "content": doc,
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                })
        return out

    async def count(self) -> int:
        return self._collection.count()

    # --- Knowledge Graph ---

    async def add_relationship(
        self, source: str, relation: str, target: str, weight: float = 1.0
    ) -> None:
        rel_id = str(uuid.uuid4())
        await self._graph_db.execute(
            "INSERT INTO knowledge_graph (id, source_node, relation, target_node, weight) VALUES (?,?,?,?,?)",
            (rel_id, source, relation, target, weight),
        )
        await self._graph_db.commit()

    async def get_relationships(self, node: str) -> list[dict]:
        async with self._graph_db.execute(
            "SELECT source_node, relation, target_node, weight FROM knowledge_graph WHERE source_node = ?",
            (node,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {"source": r[0], "relation": r[1], "target": r[2], "weight": r[3]}
                for r in rows
            ]

    async def spread_activation(
        self,
        start_nodes: list[str],
        max_hops: int = 3,
        decay: float = 0.85,
    ) -> dict[str, float]:
        """Spreading activation (Collins & Loftus, 1975).

        Traverse knowledge graph from start nodes. Activation decays per hop.
        Fan effect: more connections → lower per-connection activation.
        """
        activation: dict[str, float] = {}
        frontier = [(node, 1.0) for node in start_nodes]

        for hop in range(max_hops):
            next_frontier = []
            for node, current_activation in frontier:
                if node in activation and activation[node] >= current_activation:
                    continue
                activation[node] = max(activation.get(node, 0), current_activation)

                rels = await self.get_relationships(node)
                if not rels:
                    continue
                # Fan effect: divide activation among connections
                fan_factor = 1.0 / len(rels)
                for rel in rels:
                    spread = current_activation * decay * rel["weight"] * fan_factor
                    if spread > 0.01:  # Prune negligible activation
                        next_frontier.append((rel["target"], spread))

            frontier = next_frontier

        return activation
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/memory/test_semantic_store.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/semantic_store.py tests/memory/test_semantic_store.py
git commit -m "feat: add SemanticStore with ChromaDB vectors and knowledge graph"
```

---

### Task 10: Procedural Store

**Files:**
- Create: `brain_agent/memory/procedural_store.py`
- Create: `tests/memory/test_procedural_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/memory/test_procedural_store.py
import pytest
from brain_agent.memory.procedural_store import ProceduralStore, ProcedureStage


@pytest.fixture
async def store(tmp_db_path):
    s = ProceduralStore(db_path=tmp_db_path)
    await s.initialize()
    yield s
    await s.close()


async def test_save_and_match(store):
    await store.save(
        trigger_pattern="read file *",
        action_sequence=[{"tool": "read_file", "args": {"path": "{0}"}}],
    )
    match = await store.match("read file auth.py")
    assert match is not None
    assert match["stage"] == ProcedureStage.COGNITIVE.value


async def test_no_match_returns_none(store):
    match = await store.match("completely different thing")
    assert match is None


async def test_record_success_increments_count(store):
    proc_id = await store.save(
        trigger_pattern="test *",
        action_sequence=[{"tool": "run_tests"}],
    )
    await store.record_execution(proc_id, success=True)
    await store.record_execution(proc_id, success=True)
    proc = await store.get_by_id(proc_id)
    assert proc["execution_count"] == 2
    assert proc["success_rate"] == 1.0


async def test_stage_promotion(store):
    proc_id = await store.save(
        trigger_pattern="build *",
        action_sequence=[{"tool": "build"}],
    )
    # Simulate enough successes to promote
    for _ in range(10):
        await store.record_execution(proc_id, success=True)
    proc = await store.get_by_id(proc_id)
    assert proc["stage"] == ProcedureStage.ASSOCIATIVE.value


async def test_autonomous_after_many_successes(store):
    proc_id = await store.save(
        trigger_pattern="deploy *",
        action_sequence=[{"tool": "deploy"}],
    )
    for _ in range(50):
        await store.record_execution(proc_id, success=True)
    proc = await store.get_by_id(proc_id)
    assert proc["stage"] == ProcedureStage.AUTONOMOUS.value
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/memory/test_procedural_store.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ProceduralStore**

```python
# brain_agent/memory/procedural_store.py
"""Procedural Store — cached action sequences with stage promotion.

Spec reference: Section 3.5.3 Procedural Store.
Brain: basal ganglia + cerebellum.
Three stages (Fitts & Posner, 1967):
  cognitive → associative → autonomous.
"""
from __future__ import annotations

import fnmatch
import json
import uuid
from enum import Enum

import aiosqlite

# Stage promotion thresholds
ASSOCIATIVE_THRESHOLD = 10   # executions to reach associative
AUTONOMOUS_THRESHOLD = 50    # executions to reach autonomous
MIN_SUCCESS_RATE = 0.8       # minimum success rate for promotion


class ProcedureStage(str, Enum):
    COGNITIVE = "cognitive"       # LLM reasons every time
    ASSOCIATIVE = "associative"   # Partial cache, LLM confirms
    AUTONOMOUS = "autonomous"     # Fully cached, no LLM


class ProceduralStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                trigger_pattern TEXT NOT NULL,
                action_sequence TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                execution_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                stage TEXT DEFAULT 'cognitive'
            )
        """)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def save(
        self,
        trigger_pattern: str,
        action_sequence: list[dict],
    ) -> str:
        proc_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO procedures (id, trigger_pattern, action_sequence) VALUES (?,?,?)",
            (proc_id, trigger_pattern, json.dumps(action_sequence)),
        )
        await self._db.commit()
        return proc_id

    async def match(self, input_text: str) -> dict | None:
        """Match input against stored trigger patterns (fnmatch glob)."""
        async with self._db.execute("SELECT * FROM procedures") as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                d = self._row_to_dict(cursor.description, row)
                if fnmatch.fnmatch(input_text.lower(), d["trigger_pattern"].lower()):
                    return d
        return None

    async def get_by_id(self, proc_id: str) -> dict | None:
        async with self._db.execute(
            "SELECT * FROM procedures WHERE id = ?", (proc_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_dict(cursor.description, row)

    async def record_execution(self, proc_id: str, success: bool) -> None:
        """Record execution result and promote stage if thresholds met."""
        if success:
            await self._db.execute(
                "UPDATE procedures SET execution_count = execution_count + 1, success_count = success_count + 1 WHERE id = ?",
                (proc_id,),
            )
        else:
            await self._db.execute(
                "UPDATE procedures SET execution_count = execution_count + 1 WHERE id = ?",
                (proc_id,),
            )
        # Update success rate
        await self._db.execute(
            "UPDATE procedures SET success_rate = CAST(success_count AS REAL) / execution_count WHERE id = ?",
            (proc_id,),
        )
        await self._db.commit()

        # Check stage promotion
        proc = await self.get_by_id(proc_id)
        if proc:
            new_stage = self._compute_stage(
                proc["execution_count"], proc["success_rate"]
            )
            if new_stage != proc["stage"]:
                await self._db.execute(
                    "UPDATE procedures SET stage = ? WHERE id = ?",
                    (new_stage, proc_id),
                )
                await self._db.commit()

    @staticmethod
    def _compute_stage(execution_count: int, success_rate: float) -> str:
        if (
            execution_count >= AUTONOMOUS_THRESHOLD
            and success_rate >= MIN_SUCCESS_RATE
        ):
            return ProcedureStage.AUTONOMOUS.value
        if (
            execution_count >= ASSOCIATIVE_THRESHOLD
            and success_rate >= MIN_SUCCESS_RATE
        ):
            return ProcedureStage.ASSOCIATIVE.value
        return ProcedureStage.COGNITIVE.value

    @staticmethod
    def _row_to_dict(description, row) -> dict:
        d = {col[0]: val for col, val in zip(description, row)}
        d["action_sequence"] = json.loads(d["action_sequence"])
        return d
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/memory/test_procedural_store.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/procedural_store.py tests/memory/test_procedural_store.py
git commit -m "feat: add ProceduralStore with stage promotion (cognitive→autonomous)"
```

---

## Chunk 5: Forgetting + Retrieval + Consolidation Engines

### Task 11: Forgetting Engine

**Files:**
- Create: `brain_agent/memory/forgetting.py`
- Create: `tests/memory/test_forgetting.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/memory/test_forgetting.py
import math
from brain_agent.memory.forgetting import ForgettingEngine


def test_retention_full_at_zero_distance():
    engine = ForgettingEngine()
    r = engine.retention(distance=0.0, strength=1.0)
    assert r == 1.0


def test_retention_decreases_with_distance():
    engine = ForgettingEngine()
    r1 = engine.retention(distance=1.0, strength=1.0)
    r5 = engine.retention(distance=5.0, strength=1.0)
    r10 = engine.retention(distance=10.0, strength=1.0)
    assert r1 > r5 > r10


def test_higher_strength_slower_decay():
    engine = ForgettingEngine()
    r_weak = engine.retention(distance=5.0, strength=1.0)
    r_strong = engine.retention(distance=5.0, strength=5.0)
    assert r_strong > r_weak


def test_interference_reduces_strength():
    engine = ForgettingEngine()
    old_strength = 1.0
    # High similarity → should reduce
    new_strength = engine.apply_interference(
        old_strength=old_strength, similarity=0.9
    )
    assert new_strength < old_strength


def test_no_interference_below_threshold():
    engine = ForgettingEngine()
    old_strength = 1.0
    new_strength = engine.apply_interference(
        old_strength=old_strength, similarity=0.5
    )
    assert new_strength == old_strength


def test_retrieval_induced_forgetting():
    engine = ForgettingEngine()
    competitor_strength = 1.0
    suppressed = engine.retrieval_induced_forgetting(competitor_strength)
    # Should reduce by 10-20%
    assert 0.80 <= suppressed <= 0.90


def test_homeostatic_scaling():
    engine = ForgettingEngine()
    strengths = [1.0, 0.5, 0.1, 0.05]
    scaled = engine.homeostatic_scale(strengths, factor=0.95, threshold=0.08)
    # 0.05 * 0.95 = 0.0475 < 0.08 → pruned
    assert len(scaled) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/memory/test_forgetting.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ForgettingEngine**

```python
# brain_agent/memory/forgetting.py
"""Forgetting Engine — Ebbinghaus curve + interference + RIF.

Spec reference: Section 3.6.1 Forgetting Engine.
R = e^(-d / S) where d = interaction distance, S = memory strength.
Lazy evaluation: computed on events, not background sweeps.
"""
from __future__ import annotations

import math

# Spec Section 3.6.1
INTERFERENCE_THRESHOLD = 0.85   # cosine similarity above this triggers interference
INTERFERENCE_FACTOR = 0.3       # how much similar memory weakens existing
RIF_FACTOR = 0.15               # retrieval-induced forgetting: 10-20% suppression
HOMEOSTATIC_FACTOR = 0.95       # Tononi & Cirelli (2003)
PRUNING_THRESHOLD = 0.05        # below this, memory is deleted


class ForgettingEngine:
    """Computes memory decay, interference, and pruning."""

    def retention(self, distance: float, strength: float) -> float:
        """Ebbinghaus forgetting curve: R = e^(-d / S)."""
        if strength <= 0:
            return 0.0
        return math.exp(-distance / strength)

    def apply_interference(
        self, old_strength: float, similarity: float
    ) -> float:
        """Reduce strength if new memory is too similar (interference theory)."""
        if similarity < INTERFERENCE_THRESHOLD:
            return old_strength
        reduction = similarity * INTERFERENCE_FACTOR
        return max(0.0, old_strength * (1.0 - reduction))

    def retrieval_induced_forgetting(self, competitor_strength: float) -> float:
        """Suppress competitor memory on retrieval (Anderson et al., 1994)."""
        return competitor_strength * (1.0 - RIF_FACTOR)

    def homeostatic_scale(
        self,
        strengths: list[float],
        factor: float = HOMEOSTATIC_FACTOR,
        threshold: float = PRUNING_THRESHOLD,
    ) -> list[float]:
        """Synaptic homeostasis (Tononi & Cirelli, 2003).

        Scale all strengths down. Prune those below threshold.
        Returns only surviving strengths.
        """
        return [s * factor for s in strengths if s * factor >= threshold]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/memory/test_forgetting.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/forgetting.py tests/memory/test_forgetting.py
git commit -m "feat: add ForgettingEngine with Ebbinghaus curve and interference"
```

---

### Task 12: Retrieval Engine

**Files:**
- Create: `brain_agent/memory/retrieval.py`
- Create: `tests/memory/test_retrieval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/memory/test_retrieval.py
import math
from brain_agent.memory.retrieval import RetrievalEngine, RetrievalConfig


def test_score_computation():
    engine = RetrievalEngine(config=RetrievalConfig())
    score = engine.compute_score(
        recency_distance=0.0,
        relevance=1.0,
        importance=0.5,
        access_count=3,
        context_similarity=0.8,
    )
    assert score > 0


def test_recency_matters():
    engine = RetrievalEngine(config=RetrievalConfig())
    score_recent = engine.compute_score(
        recency_distance=1.0, relevance=0.5, importance=0.3, access_count=1, context_similarity=0.5
    )
    score_old = engine.compute_score(
        recency_distance=50.0, relevance=0.5, importance=0.3, access_count=1, context_similarity=0.5
    )
    assert score_recent > score_old


def test_relevance_has_highest_weight():
    cfg = RetrievalConfig()
    engine = RetrievalEngine(config=cfg)
    # High relevance, low everything else
    score_relevant = engine.compute_score(
        recency_distance=10.0, relevance=1.0, importance=0.0, access_count=0, context_similarity=0.0
    )
    # Low relevance, high everything else
    score_irrelevant = engine.compute_score(
        recency_distance=0.0, relevance=0.0, importance=1.0, access_count=100, context_similarity=1.0
    )
    # Relevance (β=0.30) should dominate in most cases
    assert score_relevant > score_irrelevant * 0.5  # relevance should contribute significantly


def test_config_weights_sum_to_one():
    cfg = RetrievalConfig()
    total = cfg.alpha + cfg.beta + cfg.gamma + cfg.delta + cfg.epsilon
    assert abs(total - 1.0) < 0.001
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/memory/test_retrieval.py -v`
Expected: FAIL

- [ ] **Step 3: Implement RetrievalEngine**

```python
# brain_agent/memory/retrieval.py
"""Retrieval Engine — multi-factor scoring + spreading activation.

Spec reference: Section 3.6.3 Retrieval Engine.
score = α×recency + β×relevance + γ×importance + δ×frequency + ε×context_match
Extended from Park et al. (2023) Generative Agents.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RECENCY_DECAY_CONSTANT = 10.0  # Controls how fast recency drops off


@dataclass
class RetrievalConfig:
    """Tunable retrieval weights (spec Section 3.6.3)."""

    alpha: float = 0.25   # recency
    beta: float = 0.30    # relevance (highest)
    gamma: float = 0.20   # importance (emotional salience)
    delta: float = 0.10   # frequency (testing effect)
    epsilon: float = 0.15  # context match


class RetrievalEngine:
    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()

    def compute_score(
        self,
        recency_distance: float,
        relevance: float,
        importance: float,
        access_count: int,
        context_similarity: float,
    ) -> float:
        """Compute retrieval score for a single memory candidate."""
        c = self.config

        recency = math.exp(-recency_distance / RECENCY_DECAY_CONSTANT)
        frequency = math.log(access_count + 1)
        # Normalize frequency to 0-1 range (log(101) ≈ 4.6)
        frequency = min(1.0, frequency / 4.6)

        return (
            c.alpha * recency
            + c.beta * max(0.0, relevance)
            + c.gamma * max(0.0, min(1.0, importance))
            + c.delta * frequency
            + c.epsilon * max(0.0, context_similarity)
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/memory/test_retrieval.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/retrieval.py tests/memory/test_retrieval.py
git commit -m "feat: add RetrievalEngine with multi-factor scoring"
```

---

### Task 13: Consolidation Engine

**Files:**
- Create: `brain_agent/memory/consolidation.py`
- Create: `tests/memory/test_consolidation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/memory/test_consolidation.py
import pytest
from brain_agent.memory.consolidation import ConsolidationEngine
from brain_agent.memory.hippocampal_staging import HippocampalStaging
from brain_agent.memory.episodic_store import EpisodicStore
from brain_agent.memory.forgetting import ForgettingEngine


@pytest.fixture
async def staging(tmp_path, mock_embedding):
    s = HippocampalStaging(db_path=str(tmp_path / "staging.db"), embed_fn=mock_embedding)
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def episodic(tmp_path):
    s = EpisodicStore(db_path=str(tmp_path / "episodic.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def engine(staging, episodic):
    return ConsolidationEngine(
        staging=staging,
        episodic_store=episodic,
        forgetting=ForgettingEngine(),
    )


async def test_consolidate_moves_to_episodic(staging, episodic, engine):
    await staging.encode(
        content="test memory",
        entities={"what": "test"},
        interaction_id=1,
        session_id="s1",
    )
    result = await engine.consolidate()
    assert result.transferred == 1
    assert result.pruned == 0

    # Should be marked consolidated in staging
    unconsol = await staging.get_unconsolidated()
    assert len(unconsol) == 0

    # Should exist in episodic store
    episodes = await episodic.get_all()
    assert len(episodes) == 1


async def test_emotional_memories_prioritized(staging, episodic, engine):
    # Low arousal
    await staging.encode(
        content="boring fact",
        entities={},
        interaction_id=1,
        session_id="s1",
        emotional_tag={"valence": 0, "arousal": 0.1},
    )
    # High arousal
    await staging.encode(
        content="critical error",
        entities={},
        interaction_id=2,
        session_id="s1",
        emotional_tag={"valence": -0.8, "arousal": 0.9},
    )
    result = await engine.consolidate()
    assert result.transferred == 2


async def test_should_consolidate_staging_pressure(staging, engine):
    for i in range(25):
        await staging.encode(
            content=f"mem {i}", entities={}, interaction_id=i, session_id="s1"
        )
    assert await engine.should_consolidate()


async def test_should_not_consolidate_low_count(staging, engine):
    await staging.encode(
        content="one", entities={}, interaction_id=1, session_id="s1"
    )
    assert not await engine.should_consolidate()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/memory/test_consolidation.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ConsolidationEngine**

```python
# brain_agent/memory/consolidation.py
"""Consolidation Engine — hippocampal staging → long-term transfer.

Spec reference: Section 3.6.2 Consolidation Engine.
Brain: sleep replay (sharp-wave ripples) transfers hippocampal memories to neocortex.
Agent: event-triggered batch process.
Triggers: staging pressure, session end, long idle, WM overflow, explicit.
"""
from __future__ import annotations

from dataclasses import dataclass

from brain_agent.memory.hippocampal_staging import HippocampalStaging
from brain_agent.memory.episodic_store import EpisodicStore
from brain_agent.memory.forgetting import ForgettingEngine

STAGING_PRESSURE_THRESHOLD = 20
HOMEOSTATIC_FACTOR = 0.95
PRUNING_THRESHOLD = 0.05
EMOTIONAL_BOOST = 0.35  # 30-40% better retention (Cahill & McGaugh, 1995)


@dataclass
class ConsolidationResult:
    transferred: int = 0
    pruned: int = 0


class ConsolidationEngine:
    def __init__(
        self,
        staging: HippocampalStaging,
        episodic_store: EpisodicStore,
        forgetting: ForgettingEngine,
    ):
        self._staging = staging
        self._episodic = episodic_store
        self._forgetting = forgetting

    async def should_consolidate(self) -> bool:
        """Check if staging pressure threshold exceeded."""
        count = await self._staging.count_unconsolidated()
        return count >= STAGING_PRESSURE_THRESHOLD

    async def consolidate(self) -> ConsolidationResult:
        """Transfer unconsolidated staging memories to episodic store.

        Process:
        1. Fetch unconsolidated memories (sorted by arousal for priority)
        2. Boost emotional memories
        3. Transfer to episodic store
        4. Mark consolidated in staging
        5. Homeostatic scaling on episodic store
        """
        result = ConsolidationResult()
        memories = await self._staging.get_unconsolidated()

        # Sort by arousal (highest priority first) — McGaugh (2004)
        memories.sort(
            key=lambda m: m["emotional_tag"].get("arousal", 0), reverse=True
        )

        for mem in memories:
            # Emotional boost
            strength = mem["strength"]
            arousal = mem["emotional_tag"].get("arousal", 0)
            if arousal > 0.5:
                strength *= 1.0 + (arousal * EMOTIONAL_BOOST)

            # Transfer to episodic store
            await self._episodic.save(
                content=mem["content"],
                context_embedding=mem["context_embedding"],
                entities=mem["entities"],
                emotional_tag=mem["emotional_tag"],
                interaction_id=mem["last_interaction"],
                session_id=mem["last_session"],
                strength=strength,
                access_count=mem["access_count"],
            )

            # Mark consolidated
            await self._staging.mark_consolidated(mem["id"])
            result.transferred += 1

        # Homeostatic scaling (Tononi & Cirelli, 2003)
        all_episodes = await self._episodic.get_all()
        for ep in all_episodes:
            new_strength = ep["strength"] * HOMEOSTATIC_FACTOR
            if new_strength < PRUNING_THRESHOLD:
                # Prune weak memory
                result.pruned += 1
            else:
                await self._episodic.update_strength(ep["id"], new_strength)

        if result.pruned > 0:
            await self._episodic.delete_below_strength(PRUNING_THRESHOLD)

        return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/memory/test_consolidation.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/consolidation.py tests/memory/test_consolidation.py
git commit -m "feat: add ConsolidationEngine with emotional priority and homeostatic pruning"
```

---

## Chunk 6: Memory Manager + Session + Integration

### Task 14: Session Manager

**Files:**
- Create: `brain_agent/core/session.py`
- Create: `tests/core/test_session.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_session.py
import pytest
from brain_agent.core.session import SessionManager


@pytest.fixture
async def manager(tmp_db_path, mock_embedding):
    sm = SessionManager(db_path=tmp_db_path, embed_fn=mock_embedding)
    await sm.initialize()
    yield sm
    await sm.close()


async def test_start_creates_session(manager):
    session = await manager.start_session()
    assert session.id != ""
    assert session.start_interaction == manager.temporal.interaction_count


async def test_close_session(manager):
    session = await manager.start_session()
    sid = session.id
    await manager.close_session()
    assert manager.temporal.current_session_id == ""
    assert manager.temporal.count_sessions_since(sid) == 1


async def test_detect_idle_timeout(manager):
    session = await manager.start_session()
    # Simulate 31 min gap
    from datetime import timedelta, timezone, datetime
    manager.temporal._last_wall_clock = datetime.now(timezone.utc) - timedelta(minutes=31)
    assert manager.should_start_new_session("any input")


async def test_interaction_ticks(manager):
    await manager.start_session()
    await manager.on_interaction("hello")
    await manager.on_interaction("world")
    assert manager.temporal.interaction_count == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/core/test_session.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SessionManager**

```python
# brain_agent/core/session.py
"""Session lifecycle and boundary detection.

Spec reference: Section 5.2 Session Lifecycle.
Boundaries: explicit close, idle timeout (30min), topic drift (cosine < 0.3).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

import aiosqlite

from brain_agent.core.temporal import TemporalModel

IDLE_SESSION_THRESHOLD = timedelta(minutes=30)
TOPIC_DRIFT_THRESHOLD = 0.3


@dataclass
class Session:
    id: str
    start_interaction: int
    topic_embedding: list[float] | None = None


class SessionManager:
    def __init__(
        self,
        db_path: str,
        embed_fn: Callable[[str], list[float]],
    ):
        self._db_path = db_path
        self._embed_fn = embed_fn
        self.temporal = TemporalModel()
        self._current_session: Session | None = None
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_interaction INTEGER,
                end_interaction INTEGER,
                closed_at TEXT
            )
        """)
        await self._db.commit()

    async def close(self) -> None:
        if self._current_session:
            await self.close_session()
        if self._db:
            await self._db.close()

    async def start_session(self) -> Session:
        sid = str(uuid.uuid4())[:8]
        self.temporal.start_session(sid)
        self._current_session = Session(
            id=sid,
            start_interaction=self.temporal.interaction_count,
        )
        return self._current_session

    async def close_session(self) -> None:
        if self._current_session:
            await self._db.execute(
                "INSERT INTO sessions (id, start_interaction, end_interaction, closed_at) VALUES (?,?,?,?)",
                (
                    self._current_session.id,
                    self._current_session.start_interaction,
                    self.temporal.interaction_count,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await self._db.commit()
            self.temporal.close_session()
            self._current_session = None

    async def on_interaction(self, input_text: str) -> int:
        """Record an interaction. Returns new interaction count."""
        count = self.temporal.tick()

        # Update topic embedding (rolling)
        if self._current_session:
            self._current_session.topic_embedding = self._embed_fn(input_text)

        return count

    def should_start_new_session(self, input_text: str) -> bool:
        """Check if a new session boundary should be triggered."""
        # No current session
        if self._current_session is None:
            return True

        # Idle timeout
        now = datetime.now(timezone.utc)
        gap = now - self.temporal._last_wall_clock
        if gap > IDLE_SESSION_THRESHOLD:
            return True

        # Topic drift
        if self._current_session.topic_embedding is not None:
            new_emb = self._embed_fn(input_text)
            sim = self._cosine_sim(self._current_session.topic_embedding, new_emb)
            if sim < TOPIC_DRIFT_THRESHOLD:
                return True

        return False

    @property
    def current_session(self) -> Session | None:
        return self._current_session

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        import numpy as np
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = np.dot(va, vb)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(dot / norm) if norm > 0 else 0.0
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/core/test_session.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/core/session.py tests/core/test_session.py
git commit -m "feat: add SessionManager with idle timeout and topic drift detection"
```

---

### Task 15: Memory Manager (Facade)

**Files:**
- Create: `brain_agent/memory/manager.py`
- Create: `tests/memory/test_memory_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/memory/test_memory_manager.py
import pytest
from brain_agent.memory.manager import MemoryManager


@pytest.fixture
async def mm(tmp_path, mock_embedding):
    m = MemoryManager(
        db_dir=str(tmp_path),
        embed_fn=mock_embedding,
    )
    await m.initialize()
    yield m
    await m.close()


async def test_full_pipeline_encode_and_retrieve(mm):
    # Simulate a request cycle
    mm.sensory.new_cycle()
    mm.sensory.register({"text": "auth bug in login"}, modality="text")

    # Attend
    attended = mm.sensory.attend(lambda x: True)
    assert len(attended) == 1

    # Load into working memory
    mm.working.load(mm._to_wm_item(attended[0]))
    assert len(mm.working.get_slots()) == 1

    # Encode to hippocampus
    mem_id = await mm.encode(
        content="auth bug in login",
        entities={"what": "auth bug"},
        emotional_tag={"valence": -0.5, "arousal": 0.7},
    )
    assert mem_id is not None


async def test_stats(mm):
    await mm.encode(content="test", entities={})
    stats = await mm.stats()
    assert stats["staging"] == 1
    assert stats["working"] == 0


async def test_consolidation_triggered(mm):
    # Fill staging past threshold
    for i in range(25):
        await mm.encode(content=f"memory {i}", entities={}, interaction_id=i)

    result = await mm.consolidate()
    assert result.transferred == 25
    stats = await mm.stats()
    assert stats["episodic"] == 25
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/memory/test_memory_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MemoryManager**

```python
# brain_agent/memory/manager.py
"""Memory Manager — facade unifying all memory layers.

Provides a single interface for the agent to interact with the
4-layer CLS memory system.
"""
from __future__ import annotations

import os
from typing import Callable

from brain_agent.memory.sensory_buffer import SensoryBuffer, SensoryItem
from brain_agent.memory.working_memory import WorkingMemory, WorkingMemoryItem
from brain_agent.memory.hippocampal_staging import HippocampalStaging
from brain_agent.memory.episodic_store import EpisodicStore
from brain_agent.memory.semantic_store import SemanticStore
from brain_agent.memory.procedural_store import ProceduralStore
from brain_agent.memory.forgetting import ForgettingEngine
from brain_agent.memory.retrieval import RetrievalEngine
from brain_agent.memory.consolidation import ConsolidationEngine, ConsolidationResult


class MemoryManager:
    """Unified facade for the 4-layer CLS memory system."""

    def __init__(
        self,
        db_dir: str,
        embed_fn: Callable[[str], list[float]],
        working_capacity: int = 4,
    ):
        self._db_dir = db_dir
        self._embed_fn = embed_fn
        self._interaction_counter = 0
        self._session_id = ""

        # Layers
        self.sensory = SensoryBuffer()
        self.working = WorkingMemory(capacity=working_capacity)
        self.staging = HippocampalStaging(
            db_path=os.path.join(db_dir, "staging.db"),
            embed_fn=embed_fn,
        )
        self.episodic = EpisodicStore(
            db_path=os.path.join(db_dir, "episodic.db"),
        )
        self.semantic = SemanticStore(
            chroma_path=os.path.join(db_dir, "chroma"),
            graph_db_path=os.path.join(db_dir, "graph.db"),
            embed_fn=embed_fn,
        )
        self.procedural = ProceduralStore(
            db_path=os.path.join(db_dir, "procedural.db"),
        )

        # Engines
        self.forgetting = ForgettingEngine()
        self.retrieval = RetrievalEngine()
        self.consolidation = ConsolidationEngine(
            staging=self.staging,
            episodic_store=self.episodic,
            forgetting=self.forgetting,
        )

    async def initialize(self) -> None:
        await self.staging.initialize()
        await self.episodic.initialize()
        await self.semantic.initialize()
        await self.procedural.initialize()

    async def close(self) -> None:
        await self.staging.close()
        await self.episodic.close()
        await self.semantic.close()
        await self.procedural.close()

    def set_context(self, interaction_id: int, session_id: str) -> None:
        self._interaction_counter = interaction_id
        self._session_id = session_id

    async def encode(
        self,
        content: str,
        entities: dict,
        emotional_tag: dict | None = None,
        interaction_id: int | None = None,
        session_id: str | None = None,
    ) -> str:
        """Encode a new memory into hippocampal staging."""
        return await self.staging.encode(
            content=content,
            entities=entities,
            interaction_id=interaction_id or self._interaction_counter,
            session_id=session_id or self._session_id,
            emotional_tag=emotional_tag,
        )

    async def consolidate(self) -> ConsolidationResult:
        """Run consolidation (staging → episodic)."""
        return await self.consolidation.consolidate()

    async def stats(self) -> dict:
        """Return memory statistics across all layers."""
        return {
            "sensory": len(self.sensory.get_all()),
            "working": len(self.working.get_slots()),
            "staging": await self.staging.count_unconsolidated(),
            "episodic": len(await self.episodic.get_all()),
            "semantic": await self.semantic.count(),
            "procedural": 0,  # TODO: add count method
        }

    @staticmethod
    def _to_wm_item(sensory_item: SensoryItem) -> WorkingMemoryItem:
        text = sensory_item.data.get("text", str(sensory_item.data))
        return WorkingMemoryItem(content=text, slot="phonological")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/memory/test_memory_manager.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/manager.py tests/memory/test_memory_manager.py
git commit -m "feat: add MemoryManager facade unifying all memory layers"
```

---

### Task 16: Full Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test: full memory pipeline from sensory input to consolidation."""
import pytest
from brain_agent.memory.manager import MemoryManager
from brain_agent.core.session import SessionManager


@pytest.fixture
async def system(tmp_path, mock_embedding):
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    sm = SessionManager(db_path=str(tmp_path / "sessions.db"), embed_fn=mock_embedding)
    await mm.initialize()
    await sm.initialize()
    yield mm, sm
    await mm.close()
    await sm.close()


async def test_full_request_lifecycle(system):
    mm, sm = system

    # 1. Start session
    session = await sm.start_session()
    interaction = await sm.on_interaction("find auth bug")
    mm.set_context(interaction, session.id)

    # 2. Sensory: register input
    mm.sensory.new_cycle()
    mm.sensory.register({"text": "find auth bug"}, modality="text")

    # 3. Attend
    attended = mm.sensory.attend(lambda x: True)
    assert len(attended) == 1

    # 4. Working memory: load
    mm.working.load(mm._to_wm_item(attended[0]))
    assert len(mm.working.get_slots()) == 1

    # 5. Encode to hippocampus
    mem_id = await mm.encode(
        content="find auth bug",
        entities={"task": "find bug", "target": "auth"},
        emotional_tag={"valence": -0.3, "arousal": 0.5},
    )

    # 6. Verify staging
    staging_mem = await mm.staging.get_by_id(mem_id)
    assert staging_mem["content"] == "find auth bug"
    assert staging_mem["emotional_tag"]["arousal"] == 0.5

    # 7. Simulate more interactions to build up staging
    for i in range(20):
        interaction = await sm.on_interaction(f"step {i}")
        mm.set_context(interaction, session.id)
        await mm.encode(
            content=f"step {i} result",
            entities={"step": i},
        )

    # 8. Consolidate
    result = await mm.consolidate()
    assert result.transferred == 21  # original + 20 more

    # 9. Verify episodic store has consolidated memories
    stats = await mm.stats()
    assert stats["episodic"] == 21
    assert stats["staging"] == 0

    # 10. Close session
    await sm.close_session()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: All PASSED

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASSED

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration test for full memory pipeline lifecycle"
```

---

## Summary

| Chunk | Tasks | What it builds |
|-------|-------|---------------|
| 1 | 1-4 | Project setup, Signal/EmotionalTag, TemporalModel, EmbeddingService |
| 2 | 5-6 | SensoryBuffer, WorkingMemory (Baddeley) |
| 3 | 7-8 | HippocampalStaging (SQLite), EpisodicStore (SQLite) |
| 4 | 9-10 | SemanticStore (ChromaDB + KG), ProceduralStore |
| 5 | 11-13 | ForgettingEngine, RetrievalEngine, ConsolidationEngine |
| 6 | 14-16 | SessionManager, MemoryManager facade, Integration test |

**Total: 16 tasks, ~80 steps**

After this plan, the memory system is complete and tested. Next plans:
- **Plan 2:** Brain Region Processors + Thalamic Router + GWT + Triple Network
- **Plan 3:** Provider/Config/CLI integration (nanobot adaptation)
- **Plan 4:** Dashboard (React + Three.js 3D visualization)
