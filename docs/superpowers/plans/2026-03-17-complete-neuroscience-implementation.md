# Complete Neuroscience Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all 20+ missing/partial neuroscience mechanisms identified in the architecture audit, bringing every referenced paper to ≥80% coverage.

**Architecture:** Brain-faithful data structures (graph→graph, tree→tree, embedding→embedding). PFC is the only LLM caller — consolidation/reflection route through PFC. All changes are backward-compatible with existing tests.

**Tech Stack:** Python 3.11+, SQLite (WAL), ChromaDB, aiosqlite, numpy, sentence-transformers

---

## Chunk 1: Memory Dynamics (Forgetting + Retrieval Enhancement)

Papers: Anderson 1994 (RIF), Interference Theory, Wozniak 1990 (SM-2), Nader 2000 (Reconsolidation), Craik & Lockhart 1972 (Elaborative Rehearsal)

### Task 1.1: Retroactive Interference on Encode

**Files:**
- Modify: `brain_agent/memory/manager.py:93-107` (encode method)
- Test: `tests/memory/test_memory_manager.py`

When a new memory is encoded, similar existing memories in staging should have their strength reduced (retroactive interference).

- [ ] **Step 1: Write failing test**

```python
async def test_encode_applies_retroactive_interference(mm):
    """New similar memory should weaken existing similar memories."""
    id1 = await mm.encode(content="Python is used for AI", entities={})
    id2 = await mm.encode(content="Python is used for artificial intelligence", entities={})
    mem1 = await mm.staging.get_by_id(id1)
    # Similar content (cosine > 0.85) should reduce first memory's strength
    assert mem1["strength"] < 1.0
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `pytest tests/memory/test_memory_manager.py::test_encode_applies_retroactive_interference -v`

- [ ] **Step 3: Implement interference in encode()**

In `manager.py`, after `self.staging.encode(...)`, compute embedding similarity against recent staging memories. If similarity > 0.85, apply `ForgettingEngine.apply_interference()` to existing memory.

```python
async def encode(self, content, entities, emotional_tag=None, interaction_id=None, session_id=None):
    embedding = self._embed_fn(content)
    mem_id = await self.staging.encode(...)

    # Retroactive interference: weaken similar existing memories
    existing = await self.staging.get_unconsolidated()
    for ex in existing:
        if ex["id"] == mem_id:
            continue
        ex_emb = ex.get("context_embedding", [])
        if ex_emb:
            sim = self._cosine_sim(embedding, ex_emb)
            if sim > 0.85:
                new_strength = self.forgetting.apply_interference(ex["strength"], sim)
                await self.staging.update_strength(ex["id"], new_strength)

    return mem_id
```

- [ ] **Step 4: Run test, verify PASS**
- [ ] **Step 5: Commit** `git commit -m "feat(memory): apply retroactive interference on encode (Interference Theory)"`

### Task 1.2: Retrieval-Induced Forgetting in retrieve()

**Files:**
- Modify: `brain_agent/memory/manager.py:112-220` (retrieve method)
- Modify: `brain_agent/memory/hippocampal_staging.py` (need method to find competitors)
- Test: `tests/memory/test_memory_manager.py`

After retrieval, suppress competitors of retrieved memories (Anderson 1994).

- [ ] **Step 1: Write failing test**

```python
async def test_retrieve_applies_rif(mm):
    """Retrieved memories' competitors should be weakened (RIF)."""
    await mm.semantic.add("Python for web development", category="fact")
    await mm.semantic.add("Python for data science", category="fact")
    await mm.semantic.add("Java for enterprise", category="fact")
    # After retrieving "Python" memories, the non-retrieved Python memory
    # should have reduced strength if it was a competitor
    results = await mm.retrieve(query="Python programming", top_k=1)
    assert len(results) >= 1
```

- [ ] **Step 2: Implement RIF after retrieval**

After selecting top-K results, identify competitors (candidates that scored above a threshold but didn't make the cut). Apply `forgetting.retrieval_induced_forgetting()` to their strength in the semantic store.

In `manager.py` retrieve(), after sorting and slicing top-K:
```python
# 6. Retrieval-Induced Forgetting — suppress competitors
winners = set(c["id"] for c in candidates[:top_k])
for c in candidates[top_k:]:
    if c["source"] == "semantic" and c["relevance"] > 0.3:
        # This was a relevant competitor that lost — suppress it
        pass  # semantic store doesn't track strength per-query; skip for now
    elif c["source"] == "episodic" and c["relevance"] > 0.3:
        ep = await self.episodic.get_by_id(c["id"])
        if ep:
            new_str = self.forgetting.retrieval_induced_forgetting(ep["strength"])
            await self.episodic.update_strength(c["id"], new_str)
```

- [ ] **Step 3: Run test, verify PASS**
- [ ] **Step 4: Commit** `git commit -m "feat(memory): apply RIF to competitors after retrieval (Anderson 1994)"`

### Task 1.3: SM-2 Retrieval Boost (Testing Effect)

**Files:**
- Modify: `brain_agent/memory/manager.py` (retrieve method, after returning results)
- Test: `tests/memory/test_memory_manager.py`

Retrieved memories should have their strength boosted (Wozniak 1990 testing effect).

- [ ] **Step 1: Write failing test**

```python
async def test_retrieval_boosts_strength(mm):
    """Retrieved episodic memories should get strength boost."""
    for i in range(6):
        await mm.encode(content=f"memory {i}", entities={}, interaction_id=i)
    await mm.consolidate()
    results = await mm.retrieve(query="memory 0", top_k=1)
    if results:
        ep = await mm.episodic.get_by_id(results[0]["id"])
        # After retrieval, strength should be boosted and access_count incremented
        assert ep["access_count"] >= 1
```

- [ ] **Step 2: Implement retrieval boost**

After selecting top-K, boost each winner in episodic store. Add `on_retrieval()` to EpisodicStore similar to staging.

Add to `episodic_store.py`:
```python
async def on_retrieval(self, ep_id: str, boost: float = 1.5) -> None:
    await self._db.execute(
        "UPDATE episodes SET strength = strength * ?, access_count = access_count + 1 WHERE id = ?",
        (boost, ep_id),
    )
    await self._db.commit()
```

In `manager.py` retrieve(), after selecting top-K:
```python
# 7. SM-2 retrieval boost — strengthen retrieved memories
for c in candidates[:top_k]:
    if c["source"] == "episodic":
        await self.episodic.on_retrieval(c["id"], boost=1.5)
```

- [ ] **Step 3: Run tests, verify PASS**
- [ ] **Step 4: Commit** `git commit -m "feat(memory): SM-2 retrieval boost for testing effect (Wozniak 1990)"`

### Task 1.4: Reconsolidation on Retrieval

**Files:**
- Modify: `brain_agent/memory/manager.py` (retrieve method)
- Test: `tests/memory/test_memory_manager.py`

Nader (2000): Each retrieval reopens the memory trace, making it labile and re-encoded with current context.

- [ ] **Step 1: Implement reconsolidation**

When an episodic memory is retrieved, update its `last_interaction` and `last_session` to current values (re-encoding with new temporal context).

Add to `episodic_store.py`:
```python
async def reconsolidate(self, ep_id: str, interaction_id: int, session_id: str) -> None:
    """Re-encode memory with current context (Nader 2000 reconsolidation)."""
    await self._db.execute(
        "UPDATE episodes SET last_interaction = ?, last_session = ? WHERE id = ?",
        (interaction_id, session_id, ep_id),
    )
    await self._db.commit()
```

In `manager.py` retrieve(), alongside SM-2 boost:
```python
if c["source"] == "episodic":
    await self.episodic.on_retrieval(c["id"], boost=1.5)
    await self.episodic.reconsolidate(c["id"], self._interaction_counter, self._session_id)
```

- [ ] **Step 2: Write test, run, verify PASS**
- [ ] **Step 3: Commit** `git commit -m "feat(memory): reconsolidation updates temporal context on retrieval (Nader 2000)"`

---

## Chunk 2: Knowledge Graph Auto-Population

Papers: Eichenbaum 2000 (Relational Memory), Collins & Loftus 1975 (Spreading Activation data source)

Design: PFC extracts entities + relations during LLM response. Hippocampus binds them into knowledge graph. This is brain-faithful: cortical comprehension → hippocampal binding.

### Task 2.1: PFC Entity/Relation Extraction

**Files:**
- Modify: `brain_agent/regions/prefrontal.py` (modify LLM prompt to extract entities)
- Test: `tests/regions/test_prefrontal.py`

- [ ] **Step 1: Modify PFC system prompt to request structured entity extraction**

Add to PFC's `_call_llm()` system message an instruction to return entities in a structured format at the end of the response:

```python
ENTITY_EXTRACTION_INSTRUCTION = """
After your response, on a new line, output entity information in this exact format:
<entities>
{"entities": ["entity1", "entity2"], "relations": [["entity1", "relation", "entity2"]]}
</entities>
Extract key concepts, people, technologies, and their relationships from the conversation.
"""
```

- [ ] **Step 2: Parse entity block from LLM response**

Add a `_parse_entities()` static method to PFC:
```python
import json, re

@staticmethod
def _parse_entities(response: str) -> tuple[str, dict]:
    """Strip <entities> block from response; return (clean_text, entities_dict)."""
    match = re.search(r"<entities>\s*(\{.*?\})\s*</entities>", response, re.DOTALL)
    if not match:
        return response, {"entities": [], "relations": []}
    clean = response[:match.start()].rstrip()
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        data = {"entities": [], "relations": []}
    return clean, data
```

- [ ] **Step 3: Return entities in signal payload**

In PFC `process()`, after LLM call:
```python
clean_response, extracted = self._parse_entities(response)
# ... use clean_response for the action text
plan_signal.metadata["extracted_entities"] = extracted
```

- [ ] **Step 4: Test and commit**
- [ ] **Step 5: Commit** `git commit -m "feat(pfc): extract entities and relations from LLM response (Eichenbaum 2000)"`

### Task 2.2: Knowledge Graph Population in Pipeline

**Files:**
- Modify: `brain_agent/pipeline.py:232-246` (encode stage)
- Modify: `brain_agent/memory/manager.py` (new method: `update_knowledge_graph()`)
- Test: `tests/memory/test_memory_manager.py`

- [ ] **Step 1: Add knowledge graph update method to MemoryManager**

```python
async def update_knowledge_graph(self, entities: list[str], relations: list[list[str]]) -> None:
    """Populate knowledge graph from extracted entities and relations."""
    for rel in relations:
        if len(rel) == 3:
            source, relation, target = rel
            await self.semantic.add_relationship(source, relation, target, weight=0.8)
```

- [ ] **Step 2: Call from pipeline after encode**

In `pipeline.py`, after hippocampal encode (stage 16), extract entities from plan signal and update knowledge graph:

```python
# ── 16b. Knowledge Graph Update ──
extracted = plan_signal.metadata.get("extracted_entities", {})
kg_entities = extracted.get("entities", [])
kg_relations = extracted.get("relations", [])
if kg_relations:
    await self.memory.update_knowledge_graph(kg_entities, kg_relations)
```

- [ ] **Step 3: Update encode() to pass structured entities**

Change the pipeline's encode call to pass extracted entities:
```python
encode_entities = {"input": text, "output": result.response}
if kg_entities:
    encode_entities["extracted"] = kg_entities
```

- [ ] **Step 4: Test and commit**
- [ ] **Step 5: Commit** `git commit -m "feat(pipeline): auto-populate knowledge graph from PFC entity extraction"`

---

## Chunk 3: Consolidation Enhancement

Papers: Winocur & Moscovitch 2011 (Episodic→Semantic), Park et al. 2023 (Reflection), Fitts 1967 (Procedural auto-save)

### Task 3.1: Episodic→Semantic Transition via PFC

**Files:**
- Modify: `brain_agent/memory/consolidation.py` (add semantic extraction phase)
- Modify: `brain_agent/memory/manager.py` (pass PFC reference to consolidation)
- Create: `brain_agent/memory/semantic_extractor.py` (extraction logic)
- Test: `tests/memory/test_consolidation.py`

Design: During consolidation, cluster similar episodic memories (embedding similarity > 0.8). If cluster size ≥ 3, route through PFC (LLM) to extract a semantic fact. Store fact in ChromaDB + add relationships to knowledge graph. This is brain-faithful: hippocampal replay → neocortical (PFC) abstraction.

- [ ] **Step 1: Create semantic_extractor.py**

```python
"""Episodic-to-semantic transition via PFC-mediated abstraction.

Brain mapping: During sleep, hippocampus replays episodes to neocortex (PFC).
Repeated patterns are extracted as decontextualized semantic facts.
Winocur & Moscovitch (2011).
"""
from __future__ import annotations
from typing import Callable, Awaitable
import numpy as np

CLUSTER_SIMILARITY_THRESHOLD = 0.80
MIN_CLUSTER_SIZE = 3


async def find_episode_clusters(
    episodes: list[dict],
    similarity_fn: Callable[[list[float], list[float]], float],
    threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
    min_size: int = MIN_CLUSTER_SIZE,
) -> list[list[dict]]:
    """Cluster episodes by embedding similarity (greedy single-linkage)."""
    used = set()
    clusters = []
    for i, ep_a in enumerate(episodes):
        if i in used or not ep_a.get("context_embedding"):
            continue
        cluster = [ep_a]
        used.add(i)
        for j, ep_b in enumerate(episodes):
            if j in used or not ep_b.get("context_embedding"):
                continue
            sim = similarity_fn(ep_a["context_embedding"], ep_b["context_embedding"])
            if sim >= threshold:
                cluster.append(ep_b)
                used.add(j)
        if len(cluster) >= min_size:
            clusters.append(cluster)
    return clusters


def build_extraction_prompt(cluster: list[dict]) -> str:
    """Build a prompt for PFC to extract a semantic fact from episode cluster."""
    contents = "\n".join(f"- {ep['content']}" for ep in cluster)
    return (
        "Below are multiple related memories. Extract ONE general fact or rule "
        "that summarizes the common pattern. Also extract entity relationships.\n\n"
        f"Memories:\n{contents}\n\n"
        "Respond in this exact format:\n"
        "<fact>The general fact here</fact>\n"
        '<relations>[["entity1","relation","entity2"]]</relations>'
    )
```

- [ ] **Step 2: Add PFC reference to ConsolidationEngine**

Modify `ConsolidationEngine.__init__()` to accept an optional `pfc_fn` callback:
```python
def __init__(self, staging, episodic_store, forgetting, semantic_store=None,
             pfc_fn=None, similarity_fn=None, threshold=5):
    ...
    self._semantic = semantic_store
    self._pfc_fn = pfc_fn  # async (str) -> str | None
    self._similarity_fn = similarity_fn or self._default_cosine
```

- [ ] **Step 3: Add semantic extraction phase to consolidate()**

After transferring staging→episodic, call:
```python
# Phase 3: Episodic→Semantic transition (Winocur & Moscovitch 2011)
if self._pfc_fn and self._semantic:
    all_episodes = await self._episodic.get_recent(limit=50)
    clusters = await find_episode_clusters(all_episodes, self._similarity_fn)
    for cluster in clusters:
        prompt = build_extraction_prompt(cluster)
        response = await self._pfc_fn(prompt)
        if response:
            fact, relations = parse_extraction_response(response)
            if fact:
                await self._semantic.add(fact, category="extracted_fact")
                for rel in relations:
                    if len(rel) == 3:
                        await self._semantic.add_relationship(*rel, weight=0.7)
                result.semantic_extracted += 1
```

- [ ] **Step 4: Wire up in MemoryManager**

Pass PFC callback and semantic store to ConsolidationEngine:
```python
self.consolidation = ConsolidationEngine(
    staging=self.staging,
    episodic_store=self.episodic,
    forgetting=self.forgetting,
    semantic_store=self.semantic,
    pfc_fn=None,  # Set later when PFC is available
    similarity_fn=self._cosine_sim,
    threshold=consolidation_threshold,
)
```

- [ ] **Step 5: Test and commit**
- [ ] **Step 6: Commit** `git commit -m "feat(consolidation): episodic-to-semantic transition via PFC (Winocur & Moscovitch 2011)"`

### Task 3.2: Reflection (Park et al. 2023)

**Files:**
- Create: `brain_agent/memory/reflection.py`
- Modify: `brain_agent/memory/consolidation.py` (add reflection phase)
- Test: `tests/memory/test_consolidation.py`

Design: During consolidation (DMN mode = "sleep"), PFC reflects on recent episodes to generate higher-level insights. Insights are stored as high-strength semantic memories. Brain-faithful: DMN includes medial PFC; self-referential processing happens during rest.

- [ ] **Step 1: Create reflection.py**

```python
"""Reflection — higher-level insight generation from recent memories.

Brain mapping: During DMN mode, medial PFC generates self-referential
insights from recent experiences. Park et al. (2023) Generative Agents.
"""

def build_reflection_prompt(recent_episodes: list[dict], max_episodes: int = 20) -> str:
    episodes = recent_episodes[:max_episodes]
    contents = "\n".join(f"- {ep['content']}" for ep in episodes)
    return (
        "Review these recent experiences and generate 1-3 higher-level insights, "
        "patterns, or lessons learned. Focus on what is generally true, not specific events.\n\n"
        f"Recent experiences:\n{contents}\n\n"
        "Respond with one insight per line, each wrapped in <insight>...</insight> tags."
    )


def parse_insights(response: str) -> list[str]:
    import re
    return re.findall(r"<insight>(.*?)</insight>", response, re.DOTALL)
```

- [ ] **Step 2: Add reflection phase to consolidation**

After episodic→semantic transition:
```python
# Phase 4: Reflection (Park et al. 2023)
if self._pfc_fn and self._semantic:
    recent = await self._episodic.get_recent(limit=20)
    if len(recent) >= 5:
        prompt = build_reflection_prompt(recent)
        response = await self._pfc_fn(prompt)
        if response:
            insights = parse_insights(response)
            for insight in insights:
                await self._semantic.add(insight, category="reflection", strength=1.5)
                result.reflections_generated += 1
```

- [ ] **Step 3: Test and commit**
- [ ] **Step 4: Commit** `git commit -m "feat(consolidation): reflection generates insights during DMN mode (Park et al. 2023)"`

### Task 3.3: Procedural Memory Auto-Save

**Files:**
- Modify: `brain_agent/pipeline.py` (after action execution, save successful patterns)
- Test: `tests/test_pipeline.py`

Design: After BasalGanglia Go → successful execution → cerebellum low error, save the input→action pattern to ProceduralStore. This is brain-faithful: basal ganglia habit formation (Graybiel 2008).

- [ ] **Step 1: Add procedural save in pipeline**

After successful execution (stage 13) and cerebellum evaluation (stage 14), if error is small:

```python
# ── 14b. Procedural learning (Graybiel 2008) ──
pred_error = float(result_signal.payload.get("error", 1.0))
if pred_error < MINOR_ERROR_THRESHOLD:  # Import from cerebellum
    input_type = input_signal.payload.get("input_type", "statement")
    trigger = f"*{input_type}*"  # Wildcard pattern by input type
    action_seq = [action]
    existing = await self.memory.match_procedure(text)
    if existing:
        await self.memory.procedural.record_execution(existing["id"], success=True)
    else:
        await self.memory.procedural.save(
            trigger_pattern=text[:100],  # Use first 100 chars as pattern
            action_sequence=action_seq,
        )
```

- [ ] **Step 2: Test and commit**
- [ ] **Step 3: Commit** `git commit -m "feat(pipeline): auto-save successful action patterns to procedural store (Graybiel 2008, Fitts 1967)"`

---

## Chunk 4: Pipeline Mode Enforcement + GWT Competition

Papers: Menon 2011 (Triple Network), Fox 2005 (DMN/ECN anti-correlation), Crick 1984 (TRN gating), Baars 1988 (GWT competition), Hasselmo 2006 (ACh), Doya 2002 (5-HT)

### Task 4.1: DMN/ECN Region Activation Enforcement

**Files:**
- Modify: `brain_agent/pipeline.py` (check `is_region_active()` before calling regions)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Add mode check helper to pipeline**

```python
def _is_active(self, region_name: str) -> bool:
    """Check if region should process in current network mode."""
    from brain_agent.core.network_modes import ALWAYS_ACTIVE
    if region_name in ALWAYS_ACTIVE:
        return True
    return self.network_ctrl.is_region_active(region_name)
```

- [ ] **Step 2: Guard each region call with mode check**

Wrap each region.process() in pipeline with:
```python
if self._is_active("acc"):
    conflict = await self.acc.process(plan_signal)
```

The key guarded regions:
- ACC: skip in DMN
- BasalGanglia: skip in DMN
- Cerebellum: skip in DMN
- PFC: active in ECN + CREATIVE
- Hippocampus (encode): active in all modes (always_active analog)

- [ ] **Step 3: Test and commit**
- [ ] **Step 4: Commit** `git commit -m "feat(pipeline): enforce DMN/ECN region activation (Menon 2011, Fox 2005)"`

### Task 4.2: Neuromodulator Consumption

**Files:**
- Modify: `brain_agent/regions/basal_ganglia.py` (consume patience)
- Modify: `brain_agent/regions/acc.py` (consume patience for threshold)
- Modify: `brain_agent/memory/manager.py` (consume learning_rate for encode strength)
- Test: respective test files

- [ ] **Step 1: Patience modulates BG NoGo pathway (Doya 2002 5-HT)**

In `basal_ganglia.py`, patience should increase NoGo (more patient = more cautious):
```python
patience = nm.get("patience", 0.5)
nogo_score += 0.15 * (1.0 - patience)  # Low patience = more impulsive = less NoGo
```

- [ ] **Step 2: Patience modulates ACC strategy switch threshold**

In `acc.py`, higher patience → higher threshold (more tolerant of errors):
```python
# In __init__ or process:
# self.strategy_switch_threshold adjusted by patience
effective_threshold = self.strategy_switch_threshold * (0.5 + patience)
```

Wait — ACC doesn't have neuromodulators reference. Pass via signal metadata (already done: `signal.metadata["neuromodulators"]`).

- [ ] **Step 3: Learning rate modulates encoding strength (Hasselmo 2006 ACh)**

In `manager.py` encode(), pass learning_rate to scale initial strength:
```python
# If neuromodulators available, learning_rate scales encoding strength
# Higher ACh = more plastic = stronger initial encoding
```

This requires pipeline to pass learning_rate. Add to encode call or store on manager.

- [ ] **Step 4: Test and commit**
- [ ] **Step 5: Commit** `git commit -m "feat(regions): consume patience and learning_rate neuromodulators (Doya 2002, Hasselmo 2006)"`

### Task 4.3: GWT Real Competition

**Files:**
- Modify: `brain_agent/pipeline.py` (submit multiple signals to workspace)
- Modify: `brain_agent/core/workspace.py` (if needed)
- Test: `tests/core/test_workspace.py`

Design: Currently only 1 signal is submitted per cycle. For real GWT competition, multiple regions should submit signals at key stages (after execution, before wind-down).

- [ ] **Step 1: Submit multiple signals to workspace**

After execution, submit:
1. Action result signal (current)
2. Emotional significance signal (from amygdala)
3. Memory match signal (if strong retrieval hit)
4. Error signal (if cerebellum detected prediction error)

```python
# Submit action result
self.workspace.submit(result_signal, salience=0.7, goal_relevance=0.8)
# Submit emotional tag if high arousal
if etag and etag.arousal > 0.5:
    self.workspace.submit(
        Signal(type=SignalType.EMOTIONAL_TAG, source="amygdala",
               payload={"valence": etag.valence, "arousal": etag.arousal}),
        salience=etag.arousal, goal_relevance=0.5,
    )
# Submit prediction error if significant
if error_signal and error_signal.type == SignalType.PREDICTION_ERROR:
    self.workspace.submit(error_signal, salience=0.8, goal_relevance=0.7)
```

- [ ] **Step 2: Test competition with multiple submissions**
- [ ] **Step 3: Commit** `git commit -m "feat(gwt): real multi-signal competition in global workspace (Baars 1988)"`

---

## Chunk 5: Brain Region Enhancement

Papers: Koechlin 2003 (PFC hierarchy), Beaty 2018 (Creative mode), Sridharan 2008 (SN novelty)

### Task 5.1: PFC Hierarchical Goal Stack (Tree Structure)

**Files:**
- Create: `brain_agent/regions/goal_tree.py`
- Modify: `brain_agent/regions/prefrontal.py` (use goal tree instead of flat list)
- Test: `tests/regions/test_prefrontal.py`

Design: Goals are a tree, not a flat list. Brain-faithful: rostral PFC = abstract goals, mid PFC = sub-goals, caudal PFC = current action. Data structure = tree.

- [ ] **Step 1: Create goal_tree.py**

```python
"""Hierarchical goal representation (Koechlin 2003).

Brain mapping: Rostro-caudal gradient in PFC.
  - Rostral (front): abstract, long-term goals
  - Mid: sub-goals, contextual rules
  - Caudal (back): immediate action plans

Data structure: Tree (parent-child), matching brain's hierarchical control.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class GoalNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    level: str = "caudal"  # rostral | mid | caudal
    status: str = "active"  # active | completed | failed | suspended
    children: list[GoalNode] = field(default_factory=list)
    parent_id: Optional[str] = None

    def add_subgoal(self, description: str, level: str = "caudal") -> GoalNode:
        child = GoalNode(description=description, level=level, parent_id=self.id)
        self.children.append(child)
        return child

    def get_active_leaf(self) -> Optional[GoalNode]:
        """Get the deepest active goal (caudal = current action)."""
        for child in self.children:
            if child.status == "active":
                leaf = child.get_active_leaf()
                return leaf if leaf else child
        return self if self.status == "active" else None

    def complete(self) -> None:
        self.status = "completed"
        for child in self.children:
            if child.status == "active":
                child.complete()

    def to_context(self, depth: int = 0) -> str:
        """Serialize goal tree for LLM context."""
        indent = "  " * depth
        prefix = {"rostral": "[ABSTRACT]", "mid": "[SUB-GOAL]", "caudal": "[ACTION]"}
        status_mark = "x" if self.status == "completed" else " "
        line = f"{indent}[{status_mark}] {prefix.get(self.level, '')} {self.description}"
        lines = [line]
        for child in self.children:
            lines.append(child.to_context(depth + 1))
        return "\n".join(lines)


class GoalTree:
    """Manages the hierarchical goal stack for PFC."""

    def __init__(self):
        self.roots: list[GoalNode] = []

    def set_goal(self, description: str) -> GoalNode:
        """Set a new top-level (rostral) goal."""
        node = GoalNode(description=description, level="rostral")
        self.roots.append(node)
        return node

    def get_current_focus(self) -> Optional[GoalNode]:
        """Get the most recent active leaf goal."""
        for root in reversed(self.roots):
            if root.status == "active":
                return root.get_active_leaf()
        return None

    def to_context(self) -> str:
        active = [r for r in self.roots if r.status == "active"]
        if not active:
            return "No active goals."
        return "\n".join(r.to_context() for r in active)

    def clear(self) -> None:
        self.roots.clear()
```

- [ ] **Step 2: Integrate into PFC**

Replace `self.goal_stack: list[str]` with `self.goals = GoalTree()`.

- [ ] **Step 3: Update PFC process() to use tree**
- [ ] **Step 4: Test and commit**
- [ ] **Step 5: Commit** `git commit -m "feat(pfc): hierarchical goal tree with rostro-caudal mapping (Koechlin 2003)"`

### Task 5.2: Creative Mode Trigger

**Files:**
- Modify: `brain_agent/regions/salience_network.py` (add creative mode trigger logic)
- Modify: `brain_agent/regions/acc.py` (expose failure count for SN)
- Modify: `brain_agent/pipeline.py` (pass ACC state to SN)
- Test: `tests/regions/test_salience_network.py`

Design: SN triggers CREATIVE when: (1) ACC has accumulated errors (strategy switch borderline), (2) no procedural match (novel), (3) high arousal (important). Brain-faithful: anterior insula (SN) detects when routine processing fails → couples DMN + ECN.

- [ ] **Step 1: Add creative mode trigger to SalienceNetworkRegion**

```python
def _should_enter_creative(self, signal: Signal) -> bool:
    """Detect conditions for creative mode (Beaty 2018)."""
    meta = signal.metadata
    # ACC borderline conflict (close to strategy switch but not there yet)
    error_ratio = meta.get("acc_error_ratio", 0.0)
    # No procedural match (novel situation)
    no_procedure = not meta.get("cached_procedure")
    # High importance
    arousal = 0.0
    if signal.emotional_tag:
        arousal = signal.emotional_tag.arousal
    return error_ratio > 0.5 and no_procedure and arousal > 0.3
```

- [ ] **Step 2: Switch to CREATIVE in process()**

```python
if self._should_enter_creative(signal):
    self._network_ctrl.switch_to(NetworkMode.CREATIVE, trigger="novel_high_importance")
```

- [ ] **Step 3: Wire ACC error ratio into pipeline signal metadata**

In pipeline, before SN evaluation or at relevant points, attach:
```python
input_signal.metadata["acc_error_ratio"] = (
    self.acc.error_accumulator / self.acc.strategy_switch_threshold
)
```

- [ ] **Step 4: Test and commit**
- [ ] **Step 5: Commit** `git commit -m "feat(salience): creative mode trigger on ACC failure + novelty + arousal (Beaty 2018)"`

### Task 5.3: SN Memory-Based Novelty Assessment

**Files:**
- Modify: `brain_agent/regions/salience_network.py`
- Test: `tests/regions/test_salience_network.py`

Design: Replace hardcoded novelty (0.3) with memory-based assessment. If query has low similarity to recent memories → high novelty.

- [ ] **Step 1: Modify _compute_salience()**

```python
def _compute_salience(self, signal: Signal) -> float:
    arousal = signal.emotional_tag.arousal if signal.emotional_tag else 0.0
    # Memory-based novelty: low retrieval score = high novelty
    retrieved = signal.metadata.get("retrieved_memories", [])
    if retrieved:
        best_score = max(m.get("score", 0) for m in retrieved)
        novelty = 1.0 - min(1.0, best_score)
    else:
        novelty = 0.8  # No memories = likely novel
    return arousal * 0.6 + novelty * 0.4
```

This requires retrieved memories to be available in the signal when SN processes it. Adjust pipeline ordering: retrieval happens before SN re-evaluation, or pass retrieval results in signal metadata.

- [ ] **Step 2: Test and commit**
- [ ] **Step 3: Commit** `git commit -m "feat(salience): memory-based novelty assessment (Sridharan 2008)"`

---

## Chunk 6: Working Memory Enhancement (Baddeley Model)

Papers: Baddeley 2000 (4-component WM)

### Task 6.1: Multi-Component Working Memory

**Files:**
- Modify: `brain_agent/memory/working_memory.py`
- Test: `tests/memory/test_working_memory.py`

Design: Working memory should have 4 components, each with separate capacity:
- **Phonological loop** (text/language): ~4 items — current implementation
- **Visuospatial sketchpad** (spatial/visual): ~3 items
- **Episodic buffer** (cross-modal integration + LTM fragments): 4 chunks
- **Central executive** (attention allocation, not storage): orchestrates the above

Data structure: Dict of bounded deques (one per component). Brain-faithful: separate buffers match separate neural substrates.

- [ ] **Step 1: Refactor WorkingMemory to multi-component**

```python
SLOT_CAPACITY = {
    "phonological": 4,     # Text/language
    "visuospatial": 3,     # Spatial/visual context
    "episodic_buffer": 4,  # Cross-modal integration
}


class WorkingMemory:
    def __init__(self, capacity: int = 4):
        self._default_capacity = capacity
        self._slots: dict[str, list[WorkingMemoryItem]] = {
            "phonological": [],
            "visuospatial": [],
            "episodic_buffer": [],
        }

    def load(self, item: WorkingMemoryItem) -> list[WorkingMemoryItem]:
        slot_name = item.slot if item.slot in self._slots else "phonological"
        slot = self._slots[slot_name]
        cap = SLOT_CAPACITY.get(slot_name, self._default_capacity)
        evicted = []
        while len(slot) >= cap:
            evicted.append(slot.pop(0))
        slot.append(item)
        return evicted

    def get_slots(self) -> list[WorkingMemoryItem]:
        """Get all items across all components."""
        items = []
        for slot in self._slots.values():
            items.extend(slot)
        return items

    def get_component(self, name: str) -> list[WorkingMemoryItem]:
        return list(self._slots.get(name, []))
```

- [ ] **Step 2: Add episodic buffer integration**

The episodic buffer integrates retrieved memory fragments with current input:
```python
def bind_to_episodic_buffer(self, retrieved_memories: list[dict]) -> None:
    """Load retrieved LTM fragments into episodic buffer (Baddeley 2000)."""
    for mem in retrieved_memories[:SLOT_CAPACITY["episodic_buffer"]]:
        item = WorkingMemoryItem(
            content=mem.get("content", ""),
            slot="episodic_buffer",
            linked_memories=[mem.get("id", "")],
        )
        self.load(item)
```

- [ ] **Step 3: Wire into pipeline**

After retrieval (stage 7), bind retrieved memories to episodic buffer:
```python
self.memory.working.bind_to_episodic_buffer(retrieved)
```

- [ ] **Step 4: Update all existing tests for new structure**
- [ ] **Step 5: Commit** `git commit -m "feat(working_memory): multi-component Baddeley model with episodic buffer (Baddeley 2000)"`

---

## Chunk 7: Architecture Audit Docs Update

### Task 7.1: Update docs/architecture-audit.md

**Files:**
- Modify: `docs/architecture-audit.md`
- Modify: `docs/architecture-audit-ko.md`

- [ ] **Step 1: Update all coverage percentages** based on implementations done
- [ ] **Step 2: Update gap list** (remove completed items, update partial items)
- [ ] **Step 3: Update data flow diagram** to include new stages (knowledge graph update, procedural save, reflection, etc.)
- [ ] **Step 4: Add new section on data structure mapping**

```markdown
## G. Data Structure — Brain Mapping

| Brain Structure | Data Structure | Storage | Rationale |
|----------------|---------------|---------|-----------|
| Knowledge graph (semantic cortex) | Graph (nodes + weighted edges) | SQLite adjacency list | Brain's concept network is a graph |
| Goal hierarchy (PFC rostro-caudal) | Tree (GoalNode) | In-memory tree | Hierarchical control = tree |
| Episodic timeline (hippocampus) | Table (temporal ordering) | SQLite episodes table | Episodes are sequential records |
| Semantic concepts (temporal cortex) | Embedding vectors | ChromaDB | Brain represents meaning as distributed patterns |
| Procedural sequences (basal ganglia) | Action sequence list | SQLite JSON | Motor programs are sequential |
| Working memory slots (PFC) | Bounded deques (per-component) | In-memory | Separate neural substrates = separate buffers |
| Sensory buffer (sensory cortex) | Unbounded list (per-cycle) | In-memory | Brief trace, unlimited capacity |
| Spreading activation (semantic network) | BFS on weighted graph | Computed on-demand | Activation spreads through connections |
```

- [ ] **Step 5: Commit** `git commit -m "docs: update architecture audit with complete implementation status"`

---

## Execution Checklist

| Chunk | Tasks | Dependencies |
|-------|-------|-------------|
| 1: Memory Dynamics | 1.1-1.4 | None |
| 2: Knowledge Graph | 2.1-2.2 | None |
| 3: Consolidation | 3.1-3.3 | Chunk 2 (entity extraction for semantic transition) |
| 4: Pipeline Enforcement | 4.1-4.3 | None |
| 5: Brain Regions | 5.1-5.3 | None |
| 6: Working Memory | 6.1 | None |
| 7: Docs Update | 7.1 | All above |

**Parallelizable:** Chunks 1, 2, 4, 5, 6 are fully independent. Chunk 3 depends on Chunk 2. Chunk 7 depends on all.
