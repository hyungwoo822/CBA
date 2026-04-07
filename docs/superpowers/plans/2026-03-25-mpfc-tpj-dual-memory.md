# mPFC + TPJ + Dual Memory Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add mPFC (self-model) and TPJ (user-model) brain regions that manage identity through the semantic knowledge graph, while keeping markdown schema files as an always-loaded "schema layer" that works alongside the query-based retrieval layer.

**Architecture:** Two new brain regions (mPFC, TPJ) store identity facts in the semantic_store knowledge graph with typed nodes (`self_model`, `user_model`). During Phase 3, these regions produce identity context that flows into PFC. During Phase 5 consolidation, new self/user facts are extracted and stored in both the knowledge graph AND the markdown schema files. The PFC receives identity from two complementary layers: (1) schema markdown files always loaded as background context, (2) identity-specific knowledge graph nodes retrieved every request regardless of query similarity.

**Tech Stack:** Python 3.11+, asyncio, aiosqlite, ChromaDB, pytest

**Academic Grounding:**
- Northoff et al. (2006): mPFC self-referential processing
- Frith & Frith (2006): TPJ and Theory of Mind
- Ghosh & Gilboa (2014): Schema theory — mPFC schemas guide encoding and retrieval
- van Kesteren et al. (2012): Schema-congruent consolidation
- Tse et al. (2007): Schemas accelerate hippocampal learning

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `brain_agent/regions/mpfc.py` | Medial PFC — self-referential processing, self-model management |
| `brain_agent/regions/tpj.py` | Temporoparietal Junction — Theory of Mind, user-model management |
| `tests/regions/test_mpfc.py` | mPFC unit tests |
| `tests/regions/test_tpj.py` | TPJ unit tests |
| `tests/test_identity_retrieval.py` | Integration tests for identity retrieval |

### Modified Files
| File | Changes |
|------|---------|
| `brain_agent/memory/semantic_store.py` | Add `add_identity_fact()`, `get_identity_facts()`, identity node schema |
| `brain_agent/memory/manager.py` | Add `retrieve_identity()` method |
| `brain_agent/pipeline.py` | Wire mPFC + TPJ into Phase 3; identity retrieval in Phase 6; pass to PFC |
| `brain_agent/regions/prefrontal.py` | Receive identity context from pipeline instead of reading files directly |
| `brain_agent/memory/consolidation.py` | Add identity fact extraction in Phase 3 (episodic→semantic) |
| `brain_agent/memory/narrative_consolidation.py` | Also write identity facts to knowledge graph |
| `brain_agent/core/router.py` | Add routing for mPFC, TPJ signal types |
| `brain_agent/core/signals.py` | Add SELF_REFERENCE, SOCIAL_COGNITION signal types |

---

### Task 1: Extend semantic_store with identity node types

**Files:**
- Modify: `brain_agent/memory/semantic_store.py`
- Test: `tests/memory/test_semantic_identity.py`

- [ ] **Step 1: Write failing tests for identity storage and retrieval**

```python
# tests/memory/test_semantic_identity.py
import pytest
from brain_agent.memory.semantic_store import SemanticStore

def mock_embed(text):
    import hashlib
    h = hashlib.md5(text.encode()).digest()
    return [float(b) / 255.0 for b in h] * 24  # 384-dim mock

@pytest.fixture
async def store(tmp_path):
    s = SemanticStore(
        chroma_path=str(tmp_path / "chroma"),
        graph_db_path=str(tmp_path / "graph.db"),
        embed_fn=mock_embed,
    )
    await s.initialize()
    yield s
    await s.close()

@pytest.mark.asyncio
async def test_add_identity_fact_self(store):
    await store.add_identity_fact(
        fact_type="self_model",
        key="personality",
        value="curious and warm",
        source="consolidation",
    )
    facts = await store.get_identity_facts("self_model")
    assert len(facts) == 1
    assert facts[0]["key"] == "personality"
    assert facts[0]["value"] == "curious and warm"

@pytest.mark.asyncio
async def test_add_identity_fact_user(store):
    await store.add_identity_fact(
        fact_type="user_model",
        key="name",
        value="진혁",
        source="conversation",
    )
    facts = await store.get_identity_facts("user_model")
    assert len(facts) == 1
    assert facts[0]["value"] == "진혁"

@pytest.mark.asyncio
async def test_update_existing_identity_fact(store):
    await store.add_identity_fact("user_model", "name", "Unknown", "init")
    await store.add_identity_fact("user_model", "name", "진혁", "conversation")
    facts = await store.get_identity_facts("user_model")
    assert len(facts) == 1
    assert facts[0]["value"] == "진혁"

@pytest.mark.asyncio
async def test_get_identity_facts_filters_by_type(store):
    await store.add_identity_fact("self_model", "personality", "curious", "init")
    await store.add_identity_fact("user_model", "name", "진혁", "conversation")
    self_facts = await store.get_identity_facts("self_model")
    user_facts = await store.get_identity_facts("user_model")
    assert len(self_facts) == 1
    assert len(user_facts) == 1
    assert self_facts[0]["key"] == "personality"
    assert user_facts[0]["key"] == "name"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/memory/test_semantic_identity.py -v`
Expected: FAIL — `add_identity_fact` and `get_identity_facts` not found

- [ ] **Step 3: Implement identity fact storage in semantic_store**

Add to `brain_agent/memory/semantic_store.py` after the existing `initialize()` method:

```python
# In initialize(), add new table:
await self._graph_db.execute("""
    CREATE TABLE IF NOT EXISTS identity_facts (
        id TEXT PRIMARY KEY,
        fact_type TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        source TEXT DEFAULT 'unknown',
        confidence REAL DEFAULT 1.0,
        updated_at TEXT DEFAULT '',
        UNIQUE(fact_type, key)
    )
""")

async def add_identity_fact(
    self, fact_type: str, key: str, value: str,
    source: str = "unknown", confidence: float = 1.0,
) -> None:
    """Store or update an identity fact (self_model or user_model).

    Uses UPSERT: if a fact with same (fact_type, key) exists, update it.
    This models how schemas evolve — new information overwrites old.

    References:
      - Ghosh & Gilboa (2014): Schema updating in mPFC
      - van Kesteren et al. (2012): Schema-congruent consolidation
    """
    from datetime import datetime, timezone
    import uuid
    if not self._graph_db:
        return
    await self._graph_db.execute(
        """INSERT INTO identity_facts (id, fact_type, key, value, source, confidence, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(fact_type, key) DO UPDATE SET
             value=excluded.value, source=excluded.source,
             confidence=excluded.confidence, updated_at=excluded.updated_at""",
        (str(uuid.uuid4()), fact_type, key, value, source, confidence,
         datetime.now(timezone.utc).isoformat()),
    )
    await self._graph_db.commit()

async def get_identity_facts(self, fact_type: str) -> list[dict]:
    """Retrieve all identity facts of a given type.

    Args:
        fact_type: 'self_model' (mPFC) or 'user_model' (TPJ)

    Returns: List of dicts with keys: key, value, source, confidence, updated_at
    """
    if not self._graph_db:
        return []
    rows = []
    async with self._graph_db.execute(
        "SELECT key, value, source, confidence, updated_at FROM identity_facts WHERE fact_type = ? ORDER BY key",
        (fact_type,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [
        {"key": r[0], "value": r[1], "source": r[2], "confidence": r[3], "updated_at": r[4]}
        for r in rows
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/memory/test_semantic_identity.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/semantic_store.py tests/memory/test_semantic_identity.py
git commit -m "feat(memory): add identity_facts table for self_model and user_model storage"
```

---

### Task 2: Create mPFC region (self-referential processing)

**Files:**
- Create: `brain_agent/regions/mpfc.py`
- Test: `tests/regions/test_mpfc.py`

- [ ] **Step 1: Write failing tests for mPFC**

```python
# tests/regions/test_mpfc.py
import pytest
from brain_agent.regions.mpfc import MedialPrefrontalCortex
from brain_agent.core.signals import Signal, SignalType

@pytest.fixture
def mpfc():
    return MedialPrefrontalCortex()

def test_mpfc_basic_properties(mpfc):
    assert mpfc.name == "medial_pfc"
    assert mpfc.hemisphere.value == "bilateral"

def test_mpfc_has_self_model(mpfc):
    model = mpfc.get_self_model()
    assert isinstance(model, dict)

@pytest.mark.asyncio
async def test_mpfc_loads_identity_from_file(mpfc):
    context = await mpfc.get_self_context()
    assert isinstance(context, str)
    assert len(context) > 0

@pytest.mark.asyncio
async def test_mpfc_merges_schema_and_graph(mpfc):
    """Schema (SOUL.md) + graph facts should both appear in context."""
    mpfc.update_from_graph_facts([
        {"key": "core_trait", "value": "deeply curious"},
    ])
    context = await mpfc.get_self_context()
    assert "curious" in context.lower()

@pytest.mark.asyncio
async def test_mpfc_process_returns_signal(mpfc):
    sig = Signal(type=SignalType.TEXT_INPUT, source="pipeline",
                 payload={"text": "hello"})
    result = await mpfc.process(sig)
    assert result is not None
    assert "self_context" in result.metadata
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/regions/test_mpfc.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement mPFC region**

```python
# brain_agent/regions/mpfc.py
"""Medial Prefrontal Cortex — Self-referential processing.

Brain mapping: Medial wall of prefrontal cortex (BA 10/32).
Part of the Default Mode Network (DMN).

AI function: Maintains the agent's self-model (identity, personality,
values) as a "schema" that is always available to guide processing.
The self-model is stored in two complementary layers:
  1. Schema file (SOUL.md) — always loaded, holistic narrative
  2. Knowledge graph (identity_facts) — structured facts, queryable

References:
  - Northoff et al. (2006): Self-referential processing in mPFC
  - Gusnard et al. (2001): mPFC and self-reflection
  - Ghosh & Gilboa (2014): Schema theory — mPFC schemas guide cognition
  - D'Argembeau et al. (2005): mPFC activates during self-relevant thinking
"""
from __future__ import annotations

import os

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal


class MedialPrefrontalCortex(BrainRegion):
    """Medial PFC — self-referential processing and self-model management.

    Maintains two complementary self-representations:
      - Schema layer: SOUL.md narrative (always loaded, holistic)
      - Graph layer: identity_facts (structured, updatable per-fact)

    During Phase 3, produces self_context for PFC integration.
    During Phase 5, extracts self-relevant facts from new memories.
    """

    def __init__(self) -> None:
        super().__init__(
            name="medial_pfc",
            position=Vec3(0, 45, 30),       # Medial frontal wall
            lobe=Lobe.FRONTAL,
            hemisphere=Hemisphere.BILATERAL,
        )
        self._schema_text: str = ""
        self._graph_facts: list[dict] = []
        self._load_schema()

    def _load_schema(self) -> None:
        """Load SOUL.md schema from data directory."""
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
        )
        path = os.path.join(data_dir, "SOUL.md")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._schema_text = f.read()
        except FileNotFoundError:
            self._schema_text = ""

    def get_self_model(self) -> dict:
        """Return current self-model as structured dict."""
        return {
            "schema": self._schema_text,
            "facts": list(self._graph_facts),
        }

    def update_from_graph_facts(self, facts: list[dict]) -> None:
        """Update graph-layer facts (called by pipeline after loading from DB)."""
        self._graph_facts = list(facts)

    async def get_self_context(self) -> str:
        """Build self-context string for PFC injection.

        Combines schema narrative + structured graph facts into a
        coherent self-representation. This is what PFC "knows about itself".
        """
        parts = []
        if self._schema_text:
            parts.append(self._schema_text)

        if self._graph_facts:
            fact_lines = [f"- {f['key']}: {f['value']}" for f in self._graph_facts]
            parts.append("## Self-Model Facts (Knowledge Graph)\n" + "\n".join(fact_lines))

        return "\n\n".join(parts) if parts else "No self-model established yet."

    async def process(self, signal: Signal) -> Signal | None:
        """Process signal through self-referential network.

        Activates during Phase 3 to provide self-context to downstream
        processing. Higher activation for self-relevant content.

        References:
          - D'Argembeau et al. (2005): mPFC activation for self-relevance
        """
        self_context = await self.get_self_context()
        signal.metadata["self_context"] = self_context

        # Activation scales with self-relevance (baseline 0.4 for any input)
        self.emit_activation(0.4)
        return signal
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/regions/test_mpfc.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/regions/mpfc.py tests/regions/test_mpfc.py
git commit -m "feat(regions): add mPFC self-referential region (Northoff 2006)"
```

---

### Task 3: Create TPJ region (Theory of Mind)

**Files:**
- Create: `brain_agent/regions/tpj.py`
- Test: `tests/regions/test_tpj.py`

- [ ] **Step 1: Write failing tests for TPJ**

```python
# tests/regions/test_tpj.py
import pytest
from brain_agent.regions.tpj import TemporoparietalJunction
from brain_agent.core.signals import Signal, SignalType

@pytest.fixture
def tpj():
    return TemporoparietalJunction()

def test_tpj_basic_properties(tpj):
    assert tpj.name == "tpj"
    assert tpj.hemisphere.value == "right"

def test_tpj_has_user_model(tpj):
    model = tpj.get_user_model()
    assert isinstance(model, dict)

@pytest.mark.asyncio
async def test_tpj_loads_profile_from_file(tpj):
    context = await tpj.get_user_context()
    assert isinstance(context, str)
    assert len(context) > 0

@pytest.mark.asyncio
async def test_tpj_merges_schema_and_graph(tpj):
    tpj.update_from_graph_facts([
        {"key": "name", "value": "진혁"},
        {"key": "language", "value": "Korean"},
    ])
    context = await tpj.get_user_context()
    assert "진혁" in context

@pytest.mark.asyncio
async def test_tpj_process_returns_signal(tpj):
    sig = Signal(type=SignalType.TEXT_INPUT, source="pipeline",
                 payload={"text": "hello"})
    result = await tpj.process(sig)
    assert result is not None
    assert "user_context" in result.metadata
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/regions/test_tpj.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement TPJ region**

```python
# brain_agent/regions/tpj.py
"""Temporoparietal Junction — Theory of Mind and user modeling.

Brain mapping: Junction of temporal and parietal cortices, right-dominant.

AI function: Maintains the agent's model of the user (their identity,
preferences, personality, relationship context). This is the neural
basis of "understanding another mind".

The user-model is stored in two complementary layers:
  1. Schema file (USER.md) — always loaded, holistic narrative
  2. Knowledge graph (identity_facts) — structured facts, queryable

References:
  - Frith & Frith (2006): Theory of Mind and the TPJ
  - Saxe & Kanwisher (2003): TPJ specialization for mental state attribution
  - Van Overwalle (2009): TPJ in social cognition
"""
from __future__ import annotations

import os

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal


class TemporoparietalJunction(BrainRegion):
    """TPJ — Theory of Mind and user-model management.

    Maintains two complementary user-representations:
      - Schema layer: USER.md narrative (always loaded, holistic)
      - Graph layer: identity_facts (structured, updatable per-fact)

    During Phase 3, produces user_context for PFC integration.
    During Phase 5, extracts user-relevant facts from new memories.
    """

    def __init__(self) -> None:
        super().__init__(
            name="tpj",
            position=Vec3(50, -35, 30),     # Right TPJ location
            lobe=Lobe.PARIETAL,
            hemisphere=Hemisphere.RIGHT,     # Right-dominant (Saxe 2003)
        )
        self._schema_text: str = ""
        self._graph_facts: list[dict] = []
        self._load_schema()

    def _load_schema(self) -> None:
        """Load USER.md schema from data directory."""
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
        )
        path = os.path.join(data_dir, "USER.md")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._schema_text = f.read()
        except FileNotFoundError:
            self._schema_text = ""

    def get_user_model(self) -> dict:
        return {
            "schema": self._schema_text,
            "facts": list(self._graph_facts),
        }

    def update_from_graph_facts(self, facts: list[dict]) -> None:
        self._graph_facts = list(facts)

    async def get_user_context(self) -> str:
        """Build user-context string for PFC injection."""
        parts = []
        if self._schema_text:
            parts.append(self._schema_text)

        if self._graph_facts:
            fact_lines = [f"- {f['key']}: {f['value']}" for f in self._graph_facts]
            parts.append("## User Facts (Knowledge Graph)\n" + "\n".join(fact_lines))

        return "\n\n".join(parts) if parts else "No user profile established yet."

    async def process(self, signal: Signal) -> Signal | None:
        """Process signal through Theory of Mind network.

        Activates during Phase 3 to provide user-context to downstream.
        Higher activation for socially relevant content.
        """
        user_context = await self.get_user_context()
        signal.metadata["user_context"] = user_context

        self.emit_activation(0.4)
        return signal
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/regions/test_tpj.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/regions/tpj.py tests/regions/test_tpj.py
git commit -m "feat(regions): add TPJ Theory of Mind region (Frith & Frith 2006)"
```

---

### Task 4: Add identity retrieval to MemoryManager

**Files:**
- Modify: `brain_agent/memory/manager.py`
- Test: `tests/memory/test_identity_retrieval.py`

- [ ] **Step 1: Write failing test for identity retrieval**

```python
# tests/memory/test_identity_retrieval.py
import pytest

@pytest.mark.asyncio
async def test_retrieve_identity_returns_both_models(tmp_path):
    """Identity retrieval should return self_model and user_model facts."""
    from brain_agent.memory.semantic_store import SemanticStore
    import hashlib

    def mock_embed(text):
        h = hashlib.md5(text.encode()).digest()
        return [float(b) / 255.0 for b in h] * 24

    store = SemanticStore(
        chroma_path=str(tmp_path / "chroma"),
        graph_db_path=str(tmp_path / "graph.db"),
        embed_fn=mock_embed,
    )
    await store.initialize()

    await store.add_identity_fact("self_model", "personality", "curious and warm")
    await store.add_identity_fact("user_model", "name", "진혁")
    await store.add_identity_fact("user_model", "language", "Korean")

    self_facts = await store.get_identity_facts("self_model")
    user_facts = await store.get_identity_facts("user_model")

    assert len(self_facts) == 1
    assert len(user_facts) == 2
    assert any(f["value"] == "진혁" for f in user_facts)

    await store.close()
```

- [ ] **Step 2: Run test to verify it passes** (uses already-implemented methods)

Run: `pytest tests/memory/test_identity_retrieval.py -v`
Expected: PASS (using Task 1's implementation)

- [ ] **Step 3: Add retrieve_identity() to MemoryManager**

Add to `brain_agent/memory/manager.py`:

```python
async def retrieve_identity(self) -> dict:
    """Retrieve identity facts regardless of query.

    Always returns self_model and user_model facts from the semantic
    knowledge graph. These are "schema" facts that are always relevant,
    unlike episodic memories which are query-dependent.

    References:
      - Ghosh & Gilboa (2014): Schemas are always active in mPFC
      - Frith & Frith (2006): User model always active in TPJ
    """
    self_facts = await self.semantic.get_identity_facts("self_model")
    user_facts = await self.semantic.get_identity_facts("user_model")
    return {
        "self_model": self_facts,
        "user_model": user_facts,
    }
```

- [ ] **Step 4: Run all memory tests**

Run: `pytest tests/memory/ -v --ignore=tests/memory/test_memory_manager.py -q`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add brain_agent/memory/manager.py tests/memory/test_identity_retrieval.py
git commit -m "feat(memory): add identity retrieval for self_model and user_model facts"
```

---

### Task 5: Wire mPFC + TPJ into pipeline Phase 3

**Files:**
- Modify: `brain_agent/pipeline.py`
- Modify: `brain_agent/core/signals.py` (add signal types)

- [ ] **Step 1: Add signal types**

In `brain_agent/core/signals.py`, add to SignalType enum:
```python
SELF_REFERENCE = "self_reference"
SOCIAL_COGNITION = "social_cognition"
```

- [ ] **Step 2: Add mPFC + TPJ to pipeline constructor**

In `brain_agent/pipeline.py` `__init__`, after pSTS initialization:
```python
from brain_agent.regions.mpfc import MedialPrefrontalCortex
from brain_agent.regions.tpj import TemporoparietalJunction

# ── Phase 3: Identity regions (schema layer) ────────────────────
self.mpfc = MedialPrefrontalCortex()             # Self-model (Northoff 2006)
self.tpj = TemporoparietalJunction()             # User-model (Frith & Frith 2006)
```

Add to `_all_regions()` list.

- [ ] **Step 3: Load identity facts from DB on startup**

In `restore_brain_state()`:
```python
# Load identity facts into mPFC and TPJ
identity = await self.memory.retrieve_identity()
self.mpfc.update_from_graph_facts(identity.get("self_model", []))
self.tpj.update_from_graph_facts(identity.get("user_model", []))
```

- [ ] **Step 4: Activate mPFC + TPJ in Phase 3 (after salience network)**

In `process_request()`, after the salience network section (~line 437), add:
```python
# ── 3f. mPFC: self-referential processing (Northoff 2006) ──
# Load latest identity facts from semantic store
identity = await self.memory.retrieve_identity()
self.mpfc.update_from_graph_facts(identity.get("self_model", []))
await self.mpfc.process(input_signal)
signals_count += 1
await self._emit("region_activation", "medial_pfc", self.mpfc.activation_level, "active")
await self._step(0.1)

# ── 3g. TPJ: Theory of Mind — user modeling (Frith & Frith 2006) ──
self.tpj.update_from_graph_facts(identity.get("user_model", []))
await self.tpj.process(input_signal)
signals_count += 1
await self._emit("region_activation", "tpj", self.tpj.activation_level, "active")
await self._step(0.1)
```

- [ ] **Step 5: Pass identity context in upstream_context to PFC**

Update the upstream_context dict to include identity:
```python
input_signal.metadata["upstream_context"] = {
    "input_modality": input_modality,
    "comprehension": comprehension,
    "amygdala_left": input_signal.metadata.get("amygdala_left", {}),
    "amygdala_right": input_signal.metadata.get("amygdala_right", {}),
    "neuromodulators": self.neuromodulators.snapshot(),
    # Identity from mPFC + TPJ (schema + graph layers)
    "self_context": input_signal.metadata.get("self_context", ""),
    "user_context": input_signal.metadata.get("user_context", ""),
}
```

- [ ] **Step 6: Update PFC _call_llm to use identity from upstream instead of file reading**

In `brain_agent/regions/prefrontal.py`, replace the file-loading section:
```python
# OLD: Load persistent identity files
# soul = self._load_file("SOUL.md")
# user_profile = self._load_file("USER.md")
# long_term_memory = self._load_file("memory/MEMORY.md")

# NEW: Receive identity from mPFC and TPJ regions (via upstream context)
soul = upstream.get("self_context", "")
user_profile = upstream.get("user_context", "")
long_term_memory = self._load_file("memory/MEMORY.md")  # Keep MEMORY.md as neocortical schema
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/core/ tests/regions/ -q --tb=short`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add brain_agent/pipeline.py brain_agent/regions/prefrontal.py brain_agent/core/signals.py
git commit -m "feat(pipeline): wire mPFC + TPJ into Phase 3, identity flows to PFC"
```

---

### Task 6: Identity-aware consolidation (Phase 5)

**Files:**
- Modify: `brain_agent/memory/narrative_consolidation.py`

- [ ] **Step 1: Update narrative consolidation to write identity facts to knowledge graph**

Modify the `narrative_consolidate()` function to also extract identity facts:

Update the LLM system prompt to also output identity facts:
```python
_CONSOLIDATION_SYSTEM_PROMPT = """...existing prompt...

Additionally, extract structured identity facts and return them in the JSON:
{
  "memory_update": "...",
  "user_update": "...",
  "identity_facts": {
    "self_model": [{"key": "...", "value": "..."}],
    "user_model": [{"key": "...", "value": "..."}]
  }
}

For self_model: personality traits, values, communication patterns that define this brain.
For user_model: name, preferences, role, interests, personality observations about the user.
Only include facts with real information — skip placeholders like "(not yet learned)".
"""
```

After getting the response, write identity facts to semantic store:
```python
identity_facts = data.get("identity_facts", {})
if semantic_store:
    for fact in identity_facts.get("self_model", []):
        if fact.get("key") and fact.get("value"):
            await semantic_store.add_identity_fact(
                "self_model", fact["key"], fact["value"], source="consolidation"
            )
    for fact in identity_facts.get("user_model", []):
        if fact.get("key") and fact.get("value"):
            await semantic_store.add_identity_fact(
                "user_model", fact["key"], fact["value"], source="consolidation"
            )
```

Update the function signature to accept semantic_store:
```python
async def narrative_consolidate(
    memories: list[dict],
    llm_provider: LLMProvider | None = None,
    semantic_store: SemanticStore | None = None,
) -> bool:
```

- [ ] **Step 2: Update pipeline to pass semantic_store to narrative consolidation**

In pipeline.py Phase 5 section:
```python
await narrative_consolidate(
    staging_memories, llm,
    semantic_store=self.memory.semantic,
)
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -q --tb=short --ignore=tests/memory/test_memory_manager.py -k "not procedural"`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add brain_agent/memory/narrative_consolidation.py brain_agent/pipeline.py
git commit -m "feat(consolidation): extract identity facts into knowledge graph during Phase 5"
```

---

### Task 7: Dashboard registration and documentation

**Files:**
- Modify: `brain_agent/dashboard/server.py` — register mPFC + TPJ in state endpoint
- Modify: `dashboard/src/stores/brainState.ts` — add regions
- Modify: `dashboard/src/constants/brainRegions.ts` — add region config
- Update: `docs/context-flow-architecture.md` and `docs/context-flow-architecture-ko.md`

- [ ] **Step 1: Add mPFC + TPJ to dashboard brain state store**

In `dashboard/src/stores/brainState.ts`, add to regions:
```typescript
medial_pfc: { level: 0, mode: 'inactive' },
tpj: { level: 0, mode: 'inactive' },
```

- [ ] **Step 2: Add to brainRegions.ts config**

Add appropriate colors and positions for mPFC and TPJ.

- [ ] **Step 3: Update docs with dual-layer architecture**

Update both EN and KR architecture docs to reflect:
- mPFC manages self-model (schema + graph)
- TPJ manages user-model (schema + graph)
- Schema layer: always loaded, holistic
- Graph layer: structured, queryable, updatable
- Both layers feed into PFC context

- [ ] **Step 4: Build dashboard**

Run: `cd dashboard && npm run build`

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/core/ tests/regions/ -q`
Expected: ALL PASS

- [ ] **Step 6: Commit and push**

```bash
git add -A
git commit -m "feat(brain): mPFC + TPJ regions with dual-layer identity architecture

New brain regions:
- mPFC (Northoff 2006): self-referential processing, self-model management
- TPJ (Frith & Frith 2006): Theory of Mind, user-model management

Dual memory architecture:
- Schema layer: SOUL.md + USER.md (always loaded, holistic background)
- Graph layer: identity_facts table (structured, queryable, per-fact updatable)
- Both layers feed into PFC context during Phase 3

Consolidation:
- Phase 5 extracts identity facts into knowledge graph
- Also updates markdown schema files
- Bidirectional: graph facts inform encoding, encoding updates graph

Academic grounding:
- Ghosh & Gilboa (2014): mPFC schemas guide encoding and retrieval
- van Kesteren et al. (2012): Schema-congruent consolidation
- Tse et al. (2007): Schemas accelerate hippocampal learning"

git push origin main
```

---

## Architecture Summary

```
┌─ Schema Layer (always loaded) ──────────────────────┐
│                                                      │
│  mPFC: SOUL.md + self_model graph facts              │
│  TPJ:  USER.md + user_model graph facts              │
│  Neocortex: MEMORY.md (long-term narrative)          │
│                                                      │
│  → Background context (no query needed)              │
│  → Updated by Phase 5 consolidation                  │
└──────────────────────────────────────────────────────┘
         ↕  mutual reinforcement  ↕
┌─ Retrieval Layer (query-dependent) ─────────────────┐
│                                                      │
│  Hippocampus → Episodic store (specific events)      │
│  ATL → Semantic store (facts, relations, vectors)    │
│  Basal Ganglia → Procedural store (strategies)       │
│                                                      │
│  → Foreground context (query similarity match)       │
│  → Feeds into schema updates during consolidation    │
└──────────────────────────────────────────────────────┘
         ↓  both layers  ↓
┌─ PFC LLM Context ──────────────────────────────────┐
│  1. Self-context (mPFC schema + graph)               │
│  2. User-context (TPJ schema + graph)                │
│  3. MEMORY.md (neocortical narrative)                │
│  4. Wernicke analysis                                │
│  5. Amygdala emotional state                         │
│  6. Neuromodulators → cognitive state                │
│  7. Retrieved memories (top-5 by relevance)          │
│  8. Network mode + goals                             │
└──────────────────────────────────────────────────────┘
```
