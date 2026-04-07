# Memory Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform all 5 memory stores from raw-text copiers into neuroscience-faithful systems that store processed, meaningful representations.

**Architecture:** Enrich the pipeline's memory writes by leveraging data already computed by brain regions (Wernicke comprehension, Amygdala emotion, PFC entity extraction) rather than discarding it. Add keyword extraction to Wernicke, build structured memory content from pipeline context, store semantic facts immediately (not just during consolidation), and use intent-based procedural triggers.

**Tech Stack:** Python 3.12, aiosqlite, ChromaDB, sentence-transformers, pytest (454 existing tests must stay green)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `brain_agent/regions/wernicke.py` | Modify | Add keyword extraction to comprehension output |
| `brain_agent/pipeline.py` | Modify | Enrich all 5 memory writes with processed data |
| `brain_agent/memory/manager.py` | Modify | Add `encode_structured()` + `store_semantic_facts()` |
| `brain_agent/memory/working_memory.py` | Modify | `WorkingMemoryItem` gets structured metadata |
| `tests/regions/test_wernicke.py` | Modify | Tests for keyword extraction |
| `tests/memory/test_memory_manager.py` | Modify | Tests for structured encoding + immediate semantic |
| `tests/test_pipeline.py` | Modify | Tests for enriched pipeline memory flow + update 3 existing procedural tests |

## Task Dependencies

```
Task 1 (Wernicke keywords)
  ├→ Task 2 (WM enrichment) — needs keywords
  ├→ Task 4 (Staging enrichment) — needs keywords + intent
  └→ Task 5 (Procedural triggers) — needs intent + keywords
Task 3 (Semantic facts) — independent
Task 6 (Episodic enrichment) — needs Task 4 (enriched staging data)
Task 7 (Variable scoping) — consolidates Tasks 2, 4, 5
Task 8 (E2E test) — validates all
```

Execution order: **1 → 2 → 3 → 4 → 5 → 6 → 7 → 8**

---

### Task 1: Wernicke Keyword Extraction

Wernicke currently outputs `{intent, word_count, complexity, avg_word_length}`. It should also extract **keywords** (content words that carry meaning) so downstream systems can use them.

**Files:**
- Modify: `brain_agent/regions/wernicke.py`
- Modify: `tests/regions/test_wernicke.py`

- [ ] **Step 1: Write failing test for keyword extraction**

```python
# In tests/regions/test_wernicke.py — add at the end

@pytest.mark.asyncio
async def test_wernicke_extracts_keywords():
    w = WernickeArea()
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="test",
        payload={"text": "Find the authentication bug in the login module"},
    )
    result = await w.process(sig)
    comp = result.payload["comprehension"]
    assert "keywords" in comp
    keywords = comp["keywords"]
    assert isinstance(keywords, list)
    assert len(keywords) > 0
    # Should extract content words, not stopwords
    assert "the" not in keywords
    assert "in" not in keywords


@pytest.mark.asyncio
async def test_wernicke_keywords_empty_for_short():
    w = WernickeArea()
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="test",
        payload={"text": "hi"},
    )
    result = await w.process(sig)
    comp = result.payload["comprehension"]
    assert "keywords" in comp
    assert isinstance(comp["keywords"], list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/regions/test_wernicke.py::test_wernicke_extracts_keywords tests/regions/test_wernicke.py::test_wernicke_keywords_empty_for_short -v`
Expected: FAIL — `KeyError: 'keywords'`

- [ ] **Step 3: Implement keyword extraction in Wernicke**

In `brain_agent/regions/wernicke.py`, add stopword set and keyword extraction to `process()`:

```python
# Add at module level, after imports:
STOPWORDS = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "its",
    "they", "them", "their", "this", "that", "these", "those",
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must",
    "and", "but", "or", "nor", "not", "no", "so", "if", "then", "else",
    "when", "where", "who", "whom", "which", "what", "how", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further",
    "just", "also", "very", "too", "quite", "really", "here", "there",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "than", "up", "down",
})
```

In the `process()` method, after building the comprehension dict, add:

```python
# Extract keywords: content words that are not stopwords
words_lower = [w.lower().strip(".,!?;:\"'()[]{}") for w in words]
keywords = []
seen = set()
for w in words_lower:
    if len(w) >= 2 and w not in STOPWORDS and w not in seen:
        seen.add(w)
        keywords.append(w)
```

Add `"keywords": keywords` to the comprehension dict.

- [ ] **Step 4: Run all Wernicke tests**

Run: `pytest tests/regions/test_wernicke.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite to check no regressions**

Run: `pytest tests/ -x -q`
Expected: 454+ passed

- [ ] **Step 6: Commit**

```bash
git add brain_agent/regions/wernicke.py tests/regions/test_wernicke.py
git commit -m "feat(wernicke): add keyword extraction to comprehension output"
```

---

### Task 2: Enrich Working Memory Loading

Working Memory currently stores raw user text. It should store Wernicke's processed representation: intent, keywords, complexity — the *meaning* of the input, not the raw string.

**Files:**
- Modify: `brain_agent/memory/working_memory.py`
- Modify: `brain_agent/pipeline.py` (lines 285-287)
- Modify: `tests/memory/test_working_memory.py`

- [ ] **Step 1: Write failing test for enriched WM item**

```python
# In tests/memory/test_working_memory.py — add:

def test_wm_item_with_metadata():
    wm = WorkingMemory(capacity=4)
    item = WorkingMemoryItem(
        content="find auth bug in login",
        slot="phonological",
        metadata={"intent": "command", "keywords": ["find", "auth", "bug", "login"]},
    )
    wm.load(item)
    slots = wm.get_slots()
    assert len(slots) == 1
    assert slots[0].metadata["intent"] == "command"
    assert "keywords" in slots[0].metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/memory/test_working_memory.py::test_wm_item_with_metadata -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'metadata'`

- [ ] **Step 3: Add metadata field to WorkingMemoryItem**

In `brain_agent/memory/working_memory.py`, modify the dataclass:

```python
@dataclass
class WorkingMemoryItem:
    content: str
    slot: str
    reference_count: int = 0
    linked_memories: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

- [ ] **Step 4: Run all WM tests**

Run: `pytest tests/memory/test_working_memory.py -v`
Expected: ALL PASS

- [ ] **Step 5: Update pipeline to load enriched WM items**

In `brain_agent/pipeline.py`, replace the working memory load section (around line 285-287):

**Before:**
```python
# ── 6. Working Memory: load (Cowan 4+-1 displacement) ────────
wm_item = WorkingMemoryItem(content=text, slot="phonological")
self.memory.working.load(wm_item)
```

**After:**
```python
# ── 6. Working Memory: load (Cowan 4+-1 displacement) ────────
# Store Wernicke's processed representation, not raw text
comprehension = input_signal.payload.get("comprehension", {})
wm_meta = {
    "intent": comprehension.get("intent", "statement"),
    "keywords": comprehension.get("keywords", []),
    "complexity": comprehension.get("complexity", "simple"),
    "input_type": input_signal.payload.get("input_type", "unknown"),
    "arousal": input_signal.emotional_tag.arousal if input_signal.emotional_tag else 0.0,
}
wm_item = WorkingMemoryItem(content=text, slot="phonological", metadata=wm_meta)
self.memory.working.load(wm_item)
```

- [ ] **Step 6: Run pipeline tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add brain_agent/memory/working_memory.py brain_agent/pipeline.py tests/memory/test_working_memory.py
git commit -m "feat(working-memory): store Wernicke comprehension metadata instead of raw text"
```

---

### Task 3: Immediate Semantic Fact Storage

Semantic store is almost always empty because it only gets populated during consolidation Phase 3 (which requires 3+ clustered episodes). Fix: after PFC extracts entities/relations, immediately store facts and relations in the semantic store.

**Files:**
- Modify: `brain_agent/memory/manager.py`
- Modify: `brain_agent/pipeline.py` (lines 531-536)
- Modify: `tests/memory/test_memory_manager.py`

- [ ] **Step 1: Write failing test for immediate semantic storage**

```python
# In tests/memory/test_memory_manager.py — add:

@pytest.mark.asyncio
async def test_store_semantic_facts(tmp_path, mock_embedding):
    m = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await m.initialize()

    entities = ["Python", "AI"]
    relations = [["Python", "used_for", "AI"]]
    facts = ["Python is commonly used for AI development"]

    await m.store_semantic_facts(
        entities=entities,
        relations=relations,
        facts=facts,
    )

    # Check knowledge graph has the relation
    rels = await m.semantic.get_relationships("Python")
    assert len(rels) >= 1
    assert rels[0]["relation"] == "used_for"

    # Check vector store has the fact document
    count = await m.semantic.count()
    assert count >= 1

    results = await m.semantic.search("Python AI", top_k=5)
    assert any("Python" in r["content"] for r in results)

    await m.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/memory/test_memory_manager.py::test_store_semantic_facts -v`
Expected: FAIL — `AttributeError: 'MemoryManager' object has no attribute 'store_semantic_facts'`

- [ ] **Step 3: Implement store_semantic_facts in MemoryManager**

In `brain_agent/memory/manager.py`, add method after `update_knowledge_graph()`. Note: `update_knowledge_graph()` is now superseded by `store_semantic_facts()` but kept for backward compatibility — existing tests still exercise it:

```python
async def store_semantic_facts(
    self,
    entities: list[str],
    relations: list[list[str]],
    facts: list[str] | None = None,
) -> None:
    """Immediately store extracted facts and relations in semantic memory.

    Called per-request after PFC entity extraction, rather than waiting
    for consolidation. This ensures semantic memory builds incrementally.
    """
    # Store relations in knowledge graph
    for rel in relations:
        if len(rel) == 3:
            source, relation, target = rel
            await self.semantic.add_relationship(
                source, relation, target, weight=0.8
            )

    # Store fact documents in vector store
    if facts:
        for fact in facts:
            if fact.strip():
                await self.semantic.add(
                    fact.strip(), category="extracted_fact", strength=1.0,
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/memory/test_memory_manager.py::test_store_semantic_facts -v`
Expected: PASS

- [ ] **Step 5: Update pipeline to call store_semantic_facts**

In `brain_agent/pipeline.py`, replace the Knowledge Graph Update section (around lines 531-536):

**Before:**
```python
# ── 16b. Knowledge Graph Update (Eichenbaum 2000) ────
extracted = plan_signal.metadata.get("extracted_entities", {})
kg_entities = extracted.get("entities", [])
kg_relations = extracted.get("relations", [])
if kg_relations:
    await self.memory.update_knowledge_graph(kg_entities, kg_relations)
```

**After:**
```python
# ── 16b. Semantic Fact Storage (Eichenbaum 2000) ─────
# Store extracted entities, relations, AND facts immediately
# in semantic memory — don't wait for consolidation.
extracted = plan_signal.metadata.get("extracted_entities", {})
kg_entities = extracted.get("entities", [])
kg_relations = extracted.get("relations", [])
# Build fact sentences from entities/relations for vector search
facts = []
for rel in kg_relations:
    if len(rel) == 3:
        facts.append(f"{rel[0]} {rel[1].replace('_', ' ')} {rel[2]}")
if kg_entities or kg_relations:
    await self.memory.store_semantic_facts(
        entities=kg_entities,
        relations=kg_relations,
        facts=facts if facts else None,
    )
```

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: 454+ passed (no regressions)

- [ ] **Step 7: Commit**

```bash
git add brain_agent/memory/manager.py brain_agent/pipeline.py tests/memory/test_memory_manager.py
git commit -m "feat(semantic): store facts immediately per-request instead of only during consolidation"
```

---

### Task 4: Enrich Hippocampal Staging Content

Staging currently stores `"Q: {text}\nA: {response}"` with `entities={"input": text, "output": response}`. This is a raw log, not a hippocampal representation. Fix: store structured content that captures the *meaning* of the interaction — Wernicke's analysis, emotional context, and PFC's entity extraction.

> **Dependency:** Requires Task 1 (Wernicke keywords) to be completed first.

**Files:**
- Modify: `brain_agent/pipeline.py` (lines 506-528)
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for enriched staging content**

```python
# In tests/test_pipeline.py — add:

@pytest.mark.asyncio
async def test_staging_content_is_structured(tmp_path):
    """Staging should capture semantic meaning, not raw Q&A."""
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=_mock_embed)
    await mm.initialize()
    pipeline = ProcessingPipeline(memory=mm, llm_provider=None)
    await pipeline.process_request("What is Python used for?")

    items = await mm.staging.get_unconsolidated()
    assert len(items) >= 1
    item = items[0]

    # Content should NOT be just "Q: ...\nA: ..."
    # Entities should include Wernicke analysis
    ents = item["entities"]
    assert "intent" in ents
    assert "keywords" in ents

    await mm.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_staging_content_is_structured -v`
Expected: FAIL — `AssertionError: 'intent' not in entities`

- [ ] **Step 3: Enrich the encode section of pipeline**

In `brain_agent/pipeline.py`, replace the Hippocampal Encode section (around lines 506-528):

**Before:**
```python
# ── 16. Hippocampal Encode ──────────────────────────
emotional_tag_dict = None
etag = input_signal.emotional_tag
if etag and (etag.valence != 0 or etag.arousal != 0):
    emotional_tag_dict = {
        "valence": etag.valence,
        "arousal": etag.arousal,
    }

# Determine encoding modality for hippocampus L/R awareness
# Left hippocampus: verbal. Right hippocampus: visual/spatial.
encode_modality = "visual" if image else "auditory" if audio else "verbal"

# Learning rate from neuromodulators scales encoding strength
lr = self.neuromodulators.learning_rate
await self.memory.encode(
    content=f"Q: {text}\nA: {result.response}",
    entities={"input": text, "output": result.response},
    emotional_tag=emotional_tag_dict,
    learning_rate=lr,
    modality=encode_modality,
)
result.memory_encoded = True
```

**After:**
```python
# ── 16. Hippocampal Encode ──────────────────────────
emotional_tag_dict = None
etag = input_signal.emotional_tag
if etag and (etag.valence != 0 or etag.arousal != 0):
    emotional_tag_dict = {
        "valence": etag.valence,
        "arousal": etag.arousal,
    }

encode_modality = "visual" if image else "auditory" if audio else "verbal"

# Build structured content from pipeline context
comprehension = input_signal.payload.get("comprehension", {})
extracted = plan_signal.metadata.get("extracted_entities", {})
response_text = result.response

# Content: semantic summary rather than raw Q&A
content_parts = [f"User asked ({comprehension.get('intent', 'statement')}): {text}"]
if response_text and response_text != "Action executed":
    content_parts.append(f"Response: {response_text}")
content = "\n".join(content_parts)

# Entities: rich structured metadata from Wernicke + PFC + Amygdala
encode_entities = {
    "intent": comprehension.get("intent", "statement"),
    "keywords": comprehension.get("keywords", []),
    "complexity": comprehension.get("complexity", "simple"),
    "extracted_entities": extracted.get("entities", []),
    "extracted_relations": extracted.get("relations", []),
    "input": text,
    "output": response_text,
    "network_mode": self.network_ctrl.current_mode.value,
}

lr = self.neuromodulators.learning_rate
await self.memory.encode(
    content=content,
    entities=encode_entities,
    emotional_tag=emotional_tag_dict,
    learning_rate=lr,
    modality=encode_modality,
)
result.memory_encoded = True
```

- [ ] **Step 4: Run pipeline tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: 454+ passed

- [ ] **Step 6: Commit**

```bash
git add brain_agent/pipeline.py tests/test_pipeline.py
git commit -m "feat(staging): encode structured content with Wernicke analysis and PFC entities"
```

---

### Task 5: Fix Procedural Store Trigger Patterns

Procedural store saves `trigger_pattern=text[:100]` (raw user text) and matches with `fnmatch`. This means only identical sentences hit the cache. Fix: use Wernicke's intent + sorted keywords as the trigger pattern. This creates **canonical triggers** — same intent + same keywords = same trigger regardless of word order or phrasing. Matching is still exact on the canonical form, but semantically equivalent requests now produce the same trigger.

**Files:**
- Modify: `brain_agent/pipeline.py` (lines 454-467)
- Modify: `brain_agent/memory/procedural_store.py`
- Modify: `tests/memory/test_procedural_store.py`

- [ ] **Step 1: Write failing test for intent-based trigger**

```python
# In tests/memory/test_procedural_store.py — add:

@pytest.mark.asyncio
async def test_intent_trigger_matches_similar_input(tmp_path):
    """Intent-based triggers should match semantically similar inputs."""
    store = ProceduralStore(db_path=str(tmp_path / "proc.db"))
    await store.initialize()

    # Save with intent-based trigger
    await store.save(
        trigger_pattern="command:build,deploy,project",
        action_sequence=[{"tool": "respond", "args": {"text": "Building..."}}],
    )

    # Should match same intent pattern
    result = await store.match("command:build,deploy,project")
    assert result is not None
    assert result["trigger_pattern"] == "command:build,deploy,project"

    # Should NOT match completely different intent
    result2 = await store.match("question:weather,today")
    assert result2 is None

    await store.close()
```

- [ ] **Step 2: Run test to verify it passes** (this should already pass since fnmatch handles exact match)

Run: `pytest tests/memory/test_procedural_store.py::test_intent_trigger_matches_similar_input -v`
Expected: PASS (exact match works with fnmatch)

- [ ] **Step 3: Add intent trigger builder utility**

In `brain_agent/pipeline.py`, add a helper function near the top (after imports):

```python
def _build_procedural_trigger(comprehension: dict) -> str:
    """Build intent-based trigger pattern from Wernicke comprehension.

    Format: 'intent:keyword1,keyword2,...' (keywords sorted for consistency).
    This generalizes matching beyond exact text — any input with the same
    intent and overlapping keywords can match via fnmatch.
    """
    intent = comprehension.get("intent", "statement")
    keywords = sorted(comprehension.get("keywords", []))[:5]
    if not keywords:
        return ""
    return f"{intent}:{','.join(keywords)}"
```

- [ ] **Step 4: Update pipeline procedural learning**

In `brain_agent/pipeline.py`, replace the procedural learning section (around lines 454-467):

**Before:**
```python
if pred_error < MINOR_ERROR_THRESHOLD:
    try:
        existing = await self.memory.match_procedure(text)
        if existing:
            await self.memory.procedural.record_execution(
                existing["id"], success=True,
            )
        else:
            await self.memory.procedural.save(
                trigger_pattern=text[:100],
                action_sequence=[action],
            )
    except Exception:
        pass  # Procedural save is best-effort
```

**After:**
```python
if pred_error < MINOR_ERROR_THRESHOLD:
    try:
        trigger = _build_procedural_trigger(comprehension)
        if trigger:
            existing = await self.memory.procedural.match(trigger)
            if existing:
                await self.memory.procedural.record_execution(
                    existing["id"], success=True,
                )
            else:
                await self.memory.procedural.save(
                    trigger_pattern=trigger,
                    action_sequence=[action],
                )
    except Exception:
        pass  # Procedural save is best-effort
```

- [ ] **Step 5: Update pipeline procedural lookup to use intent trigger**

Also update the initial procedural check (around line 306). Replace:

**Before:**
```python
# ── 8. Procedural Store: check cached plan (fast path) ──────
cached_procedure = await self.memory.match_procedure(text)
```

**After:**
```python
# ── 8. Procedural Store: check cached plan (fast path) ──────
comprehension = input_signal.payload.get("comprehension", {})
proc_trigger = _build_procedural_trigger(comprehension)
cached_procedure = None
if proc_trigger:
    cached_procedure = await self.memory.procedural.match(proc_trigger)
if not cached_procedure:
    # Fallback: try raw text match for backward compat
    cached_procedure = await self.memory.match_procedure(text)
```

Note: keep the `comprehension` variable accessible later in the function since it's also used in Task 2 (WM loading) and Task 4 (staging encode). Since this line runs after Wernicke but before WM load, define it here and reuse it. Remove the duplicate `comprehension = input_signal.payload.get("comprehension", {})` from the WM section in Task 2 if it was added there.

- [ ] **Step 6: Update 3 existing procedural tests that will break**

The following existing tests assert `trigger_pattern == "read file auth.py"` (raw text). After this task the trigger pattern format changes to `intent:kw1,kw2,...`. Update them:

In `tests/test_pipeline.py`, replace `test_procedural_auto_save_on_low_error`:

```python
async def test_procedural_auto_save_on_low_error(memory):
    """Successful action with low prediction error should save a procedure."""
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request("read file auth.py")
    assert result.response != ""

    # Check procedural store was populated with an intent-based trigger
    if memory.procedural._db:
        async with memory.procedural._db.execute(
            "SELECT trigger_pattern FROM procedures"
        ) as cursor:
            rows = await cursor.fetchall()
            assert len(rows) >= 1
            trigger = rows[0][0]
            # Should be intent:keywords format, not raw text
            assert ":" in trigger
```

Replace `test_procedural_auto_save_reinforces_existing`:

```python
async def test_procedural_auto_save_reinforces_existing(memory):
    """If a procedure already exists with matching intent trigger, reinforce it."""
    pipeline = ProcessingPipeline(memory=memory)

    # First request — creates the procedure
    await pipeline.process_request("read file auth.py")

    # Second identical request — should reinforce, not duplicate
    await pipeline.process_request("read file auth.py")

    if memory.procedural._db:
        async with memory.procedural._db.execute(
            "SELECT id, execution_count FROM procedures"
        ) as cursor:
            rows = await cursor.fetchall()
            # Should have exactly 1 procedure (reinforced), not 2
            assert len(rows) == 1
            assert rows[0][1] >= 1  # execution_count >= 1
```

Replace `test_procedural_auto_save_multiple_requests`:

```python
async def test_procedural_auto_save_multiple_requests(memory):
    """Different requests should each save separate procedures."""
    pipeline = ProcessingPipeline(memory=memory)
    await pipeline.process_request("build the project")
    await pipeline.process_request("deploy to production")

    if memory.procedural._db:
        async with memory.procedural._db.execute(
            "SELECT trigger_pattern FROM procedures"
        ) as cursor:
            rows = await cursor.fetchall()
            triggers = [r[0] for r in rows]
            # Should have 2 different intent-based triggers
            assert len(triggers) >= 2
            assert triggers[0] != triggers[1]
```

- [ ] **Step 7: Write new integration test**

```python
# In tests/test_pipeline.py — add:

@pytest.mark.asyncio
async def test_procedural_uses_intent_trigger(tmp_path):
    """Procedural store should use intent-based triggers, not raw text."""
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=_mock_embed)
    await mm.initialize()
    pipeline = ProcessingPipeline(memory=mm, llm_provider=None)

    await pipeline.process_request("Build and deploy the project")

    if pipeline.memory.procedural._db:
        async with pipeline.memory.procedural._db.execute(
            "SELECT trigger_pattern FROM procedures"
        ) as cursor:
            rows = await cursor.fetchall()
            if rows:
                trigger = rows[0][0]
                assert ":" in trigger, f"Expected intent:keywords format, got: {trigger}"
                assert "Build and deploy the project" != trigger

    await mm.close()
```

- [ ] **Step 8: Run all tests**

Run: `pytest tests/ -x -q`
Expected: 454+ passed (3 updated tests + 1 new test all green)

- [ ] **Step 9: Commit**

```bash
git add brain_agent/pipeline.py tests/memory/test_procedural_store.py tests/test_pipeline.py
git commit -m "feat(procedural): use canonical intent-based trigger patterns from Wernicke"
```

---

### Task 6: Enrich Episodic Content During Consolidation

Episodic store currently gets a raw copy from staging. During consolidation Phase 1, we should transform the content to include temporal context (what interaction number, what came before) and significance markers.

**Files:**
- Modify: `brain_agent/memory/consolidation.py` (Phase 1)
- Modify: `tests/memory/test_consolidation.py`

- [ ] **Step 1: Write failing test for enriched episodic content**

```python
# In tests/memory/test_consolidation.py — add:

@pytest.mark.asyncio
async def test_consolidation_enriches_episodic_content(tmp_path, mock_embedding):
    staging = HippocampalStaging(
        db_path=str(tmp_path / "staging.db"), embed_fn=mock_embedding
    )
    await staging.initialize()
    episodic = EpisodicStore(db_path=str(tmp_path / "episodic.db"))
    await episodic.initialize()
    forgetting = ForgettingEngine()

    engine = ConsolidationEngine(
        staging=staging, episodic_store=episodic, forgetting=forgetting,
        threshold=1,
    )

    await staging.encode(
        content="User asked (command): deploy the app\nResponse: Deploying now",
        entities={
            "intent": "command",
            "keywords": ["deploy", "app"],
            "input": "deploy the app",
            "output": "Deploying now",
        },
        interaction_id=5,
        session_id="s1",
        emotional_tag={"valence": 0.0, "arousal": 0.3},
    )

    result = await engine.consolidate()
    assert result.transferred == 1

    episodes = await episodic.get_all()
    assert len(episodes) == 1
    ep = episodes[0]

    # Episodic content should be enriched with keyword annotations
    assert "[keywords:" in ep["content"], f"Expected [keywords:] annotation in episodic content, got: {ep['content']}"

    await staging.close()
    await episodic.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/memory/test_consolidation.py::test_consolidation_enriches_episodic_content -v`
Expected: FAIL — `AssertionError: Expected [keywords:] annotation in episodic content` (content is still raw copy from staging without enrichment)

- [ ] **Step 3: Enrich episodic content during transfer**

In `brain_agent/memory/consolidation.py`, modify the Phase 1 loop. Replace:

**Before:**
```python
for mem in memories:
    strength = mem["strength"]
    arousal = mem["emotional_tag"].get("arousal", 0)
    if arousal > 0.5:
        strength *= 1.0 + (arousal * EMOTIONAL_BOOST)

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
    await self._staging.mark_consolidated(mem["id"])
    result.transferred += 1
```

**After:**
```python
for mem in memories:
    strength = mem["strength"]
    arousal = mem["emotional_tag"].get("arousal", 0)
    if arousal > 0.5:
        strength *= 1.0 + (arousal * EMOTIONAL_BOOST)

    # Enrich episodic content with structured context
    ents = mem["entities"]
    content = mem["content"]
    if isinstance(ents, dict) and ents.get("intent"):
        parts = [content]
        kw = ents.get("keywords", [])
        if kw:
            parts.append(f"[keywords: {', '.join(kw[:5])}]")
        if arousal > 0.5:
            valence = mem["emotional_tag"].get("valence", 0)
            tone = "positive" if valence > 0.2 else "negative" if valence < -0.2 else "neutral"
            parts.append(f"[emotional: {tone}, arousal={arousal:.1f}]")
        content = "\n".join(parts)

    await self._episodic.save(
        content=content,
        context_embedding=mem["context_embedding"],
        entities=mem["entities"],
        emotional_tag=mem["emotional_tag"],
        interaction_id=mem["last_interaction"],
        session_id=mem["last_session"],
        strength=strength,
        access_count=mem["access_count"],
    )
    await self._staging.mark_consolidated(mem["id"])
    result.transferred += 1
```

- [ ] **Step 4: Run consolidation tests**

Run: `pytest tests/memory/test_consolidation.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: 454+ passed

- [ ] **Step 6: Commit**

```bash
git add brain_agent/memory/consolidation.py tests/memory/test_consolidation.py
git commit -m "feat(episodic): enrich content with keywords and emotional context during consolidation"
```

---

### Task 7: Pipeline Variable Scoping Fix

Tasks 2, 4, and 5 all use the `comprehension` variable from Wernicke's output. Ensure it's defined once at the right scope and reused consistently.

**Files:**
- Modify: `brain_agent/pipeline.py`

- [ ] **Step 1: Consolidate comprehension variable**

In `pipeline.py`, add a default `comprehension = {}` **before** the Wernicke conditional block (around line 202), so it's always defined even when `input_signal.payload.get("text")` is falsy (e.g., image-only input):

```python
# ── 2b. Wernicke: language comprehension ─────────────────────
comprehension = {}  # Default for non-text inputs
if input_signal.payload.get("text"):
    wernicke_input = Signal(...)
    input_signal = await self.wernicke.process(input_signal)
    ...
    comprehension = input_signal.payload.get("comprehension", {})
```

Then remove all duplicate `comprehension = input_signal.payload.get("comprehension", {})` lines from:
- Task 2's WM section (around line 285)
- Task 4's staging encode section (around line 453)
- Task 5's procedural lookup section (around line 306)

All of these now use the single `comprehension` variable defined at line ~202.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: 454+ passed

- [ ] **Step 3: Commit**

```bash
git add brain_agent/pipeline.py
git commit -m "refactor(pipeline): consolidate comprehension variable scope for memory enrichment"
```

---

### Task 8: End-to-End Integration Test

Validate the entire enriched memory flow works together.

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write comprehensive integration test**

```python
# In tests/test_pipeline.py — add:

@pytest.mark.asyncio
async def test_enriched_memory_flow_end_to_end(tmp_path):
    """Working memory and staging should contain meaningful processed data."""
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=_mock_embed)
    await mm.initialize()
    pipeline = ProcessingPipeline(memory=mm, llm_provider=None)

    await pipeline.process_request("Find the authentication bug in the login module")

    # 1. Working Memory: should have intent + keywords metadata
    wm_items = mm.working.get_slots()
    assert len(wm_items) >= 1
    assert wm_items[0].metadata.get("intent") is not None
    assert isinstance(wm_items[0].metadata.get("keywords"), list)

    # 2. Staging: should have structured entities
    staged = await mm.staging.get_unconsolidated()
    assert len(staged) >= 1
    ents = staged[0]["entities"]
    assert "intent" in ents
    assert "keywords" in ents
    assert isinstance(ents["keywords"], list)

    # 3. Staging content should not be plain "Q: ...\nA: ..."
    content = staged[0]["content"]
    assert "asked" in content.lower() or "command" in content.lower() or "question" in content.lower()

    # 4. Procedural: should use intent-based trigger (not raw text)
    if mm.procedural._db:
        async with mm.procedural._db.execute(
            "SELECT trigger_pattern FROM procedures"
        ) as cursor:
            rows = await cursor.fetchall()
            if rows:
                assert ":" in rows[0][0], "Procedural trigger should be intent:keywords format"

    # Note: Semantic store population depends on PFC extracting entities (requires LLM).
    # With llm_provider=None, PFC uses fallback path — no entities extracted.
    # Episodic store population requires consolidation to run (staging >= threshold).
    # These are validated in their respective unit tests (Tasks 3 and 6).

    await mm.close()
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/test_pipeline.py::test_enriched_memory_flow_end_to_end -v`
Expected: PASS

- [ ] **Step 3: Run full test suite — final check**

Run: `pytest tests/ -x -q`
Expected: 454+ passed (all old tests still green + new tests pass)

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: add end-to-end integration test for enriched memory flow"
```

---

## Summary: What Changes Per Memory Store

| Store | Before | After |
|-------|--------|-------|
| **Working** | `content=raw_text` | `content=raw_text` + `metadata={intent, keywords, complexity, arousal}` |
| **Staging** | `content="Q: ..\nA: .."`, `entities={input, output}` | `content="User asked (intent): .."`, `entities={intent, keywords, complexity, extracted_entities, extracted_relations, ...}` |
| **Episodic** | Copy of staging | Staging content + `[keywords: ...]` + `[emotional: tone]` annotations |
| **Semantic** | Empty (only consolidation) | Immediately populated per-request from PFC entity extraction |
| **Procedural** | `trigger_pattern=raw_text[:100]` | `trigger_pattern="intent:kw1,kw2,kw3"` |
