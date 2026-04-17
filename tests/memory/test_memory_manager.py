import numpy as np
import pytest
from brain_agent.memory.manager import MemoryManager
from brain_agent.memory.contradictions_store import ContradictionsStore
from brain_agent.memory.open_questions_store import OpenQuestionsStore
from brain_agent.memory.workspace_store import WorkspaceStore, PERSONAL_WORKSPACE_ID
from brain_agent.memory.ontology_store import OntologyStore
from brain_agent.memory.ontology_seed import UNIVERSAL_WORKSPACE_ID


@pytest.fixture
async def mm(tmp_path, mock_embedding):
    m = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await m.initialize()
    yield m
    await m.close()


def _similar_embedding_fn():
    """Return an embed_fn that maps the *first word* to a deterministic
    direction, so texts sharing a first word have cosine > 0.99 while
    texts with different first words are nearly orthogonal.
    """
    _cache: dict[str, list[float]] = {}

    def _embed(text: str) -> list[float]:
        key = text.split()[0] if text.strip() else text
        if key not in _cache:
            rng = np.random.RandomState(hash(key) % 2**31)
            vec = rng.randn(384).astype(np.float32)
            vec /= np.linalg.norm(vec)
            _cache[key] = vec.tolist()
        # Add tiny noise so vectors are not *identical*
        base = np.array(_cache[key], dtype=np.float32)
        rng2 = np.random.RandomState(hash(text) % 2**31)
        noise = rng2.randn(384).astype(np.float32) * 1e-4
        vec = base + noise
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    return _embed


@pytest.fixture
async def mm_similar(tmp_path):
    """MemoryManager using an embed_fn that gives high similarity for
    texts sharing the same first word."""
    db_dir = tmp_path / "sim"
    db_dir.mkdir()
    embed_fn = _similar_embedding_fn()
    m = MemoryManager(db_dir=str(db_dir), embed_fn=embed_fn)
    await m.initialize()
    yield m
    await m.close()


async def test_full_pipeline_encode_and_retrieve(mm):
    mm.sensory.new_cycle()
    mm.sensory.register({"text": "auth bug in login"}, modality="text")
    attended = mm.sensory.attend(lambda x: True)
    assert len(attended) == 1

    mm.working.load(mm._to_wm_item(attended[0]))
    assert len(mm.working.get_slots()) == 1

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
    for i in range(25):
        await mm.encode(content=f"memory {i}", entities={}, interaction_id=i)
    result = await mm.consolidate()
    assert result.transferred == 25
    stats = await mm.stats()
    assert stats["episodic"] == 25


async def test_retrieve_with_spreading_activation(mm):
    """Spreading activation should surface graph-connected memories."""
    # Add semantic memories
    await mm.semantic.add("Python is great for AI", category="fact")
    await mm.semantic.add("Machine learning uses neural networks", category="fact")
    await mm.semantic.add("Communication skills are important", category="fact")

    # Build knowledge graph: Python -> AI -> neural_networks
    await mm.semantic.add_relationship("Python", "used_for", "AI", weight=0.9)
    await mm.semantic.add_relationship("AI", "includes", "neural_networks", weight=0.8)

    # Query "Python" — spreading activation should boost AI-related memories
    results = await mm.retrieve(query="Python programming", top_k=5)
    assert len(results) >= 1
    # All results should have activation_boost field (possibly 0.0)
    for r in results:
        assert "activation_boost" in r or "score" in r


async def test_retrieve_no_graph_graceful(mm):
    """Retrieve works normally when knowledge graph is empty."""
    await mm.semantic.add("test document about cats", category="fact")
    results = await mm.retrieve(query="cats", top_k=3)
    assert len(results) >= 1
    assert results[0]["source"] == "semantic"


# ------------------------------------------------------------------
# Task 1.1: Retroactive Interference on Encode
# ------------------------------------------------------------------


async def test_encode_applies_retroactive_interference(mm_similar):
    """New similar memory should weaken existing similar memories."""
    mm = mm_similar
    id1 = await mm.encode(content="Python is used for AI", entities={})
    mem1_before = await mm.staging.get_by_id(id1)
    assert mem1_before["strength"] == 1.0

    # Second encode with similar content (same first word -> high cosine)
    id2 = await mm.encode(
        content="Python is used for artificial intelligence", entities={},
    )
    mem1_after = await mm.staging.get_by_id(id1)
    # Interference should have reduced mem1's strength below 1.0
    assert mem1_after["strength"] < 1.0


async def test_encode_no_interference_for_dissimilar(mm_similar):
    """Dissimilar memories should NOT trigger retroactive interference."""
    mm = mm_similar
    id1 = await mm.encode(content="Python is used for AI", entities={})
    # Different first word -> nearly orthogonal embedding -> no interference
    id2 = await mm.encode(content="Java is used for enterprise", entities={})
    mem1 = await mm.staging.get_by_id(id1)
    assert mem1["strength"] == 1.0


# ------------------------------------------------------------------
# Task 1.2: Retrieval-Induced Forgetting
# ------------------------------------------------------------------


async def test_retrieve_applies_rif(mm_similar):
    """Non-retrieved episodic competitors should be weakened (RIF)."""
    mm = mm_similar
    # Encode several similar memories so they end up in episodic store
    for i in range(6):
        await mm.encode(
            content=f"Alpha memory about topic {i}",
            entities={},
            interaction_id=i,
        )
    await mm.consolidate()

    # All episodes should start with strength ~1.0 (modulo homeostatic)
    all_eps = await mm.episodic.get_all()
    assert len(all_eps) >= 6
    strengths_before = {ep["id"]: ep["strength"] for ep in all_eps}

    # Retrieve with top_k=1 — one winner, rest are competitors
    results = await mm.retrieve(query="Alpha memory about topic 0", top_k=1)
    assert len(results) >= 1

    # Check that at least one competitor (non-winner, relevance > 0.3)
    # had its strength reduced
    winner_id = results[0]["id"] if results[0]["source"] == "episodic" else None
    if winner_id:
        all_eps_after = await mm.episodic.get_all()
        suppressed = False
        for ep in all_eps_after:
            if ep["id"] != winner_id and ep["id"] in strengths_before:
                if ep["strength"] < strengths_before[ep["id"]]:
                    suppressed = True
                    break
        assert suppressed, "At least one competitor should be suppressed by RIF"


# ------------------------------------------------------------------
# Task 1.3: SM-2 Retrieval Boost
# ------------------------------------------------------------------


async def test_retrieval_boosts_episodic_strength(mm):
    """Retrieved episodic memories should get strength boost."""
    for i in range(6):
        await mm.encode(content=f"memory {i}", entities={}, interaction_id=i)
    await mm.consolidate()

    all_eps = await mm.episodic.get_all()
    assert len(all_eps) >= 6

    # Record access counts before retrieval
    counts_before = {ep["id"]: ep["access_count"] for ep in all_eps}

    results = await mm.retrieve(query="memory 0", top_k=1)
    assert len(results) >= 1

    if results[0]["source"] == "episodic":
        ep_after = await mm.episodic.get_by_id(results[0]["id"])
        # access_count should have been incremented
        assert ep_after["access_count"] > counts_before[results[0]["id"]]
        # strength should have been boosted (multiplied by 1.5)
        # Original strength was ~0.95 (after homeostatic scaling),
        # boosted should be ~1.425
        assert ep_after["strength"] > 1.0


# ------------------------------------------------------------------
# Task 1.4: Reconsolidation
# ------------------------------------------------------------------


async def test_retrieval_reconsolidates_episodic(mm):
    """Retrieved episodic memories should have temporal context updated."""
    for i in range(6):
        await mm.encode(content=f"memory {i}", entities={}, interaction_id=i)
    await mm.consolidate()

    # Set interaction slightly ahead so recency distance stays small
    # (large distances cause Ebbinghaus retention to drop to ~0)
    mm.set_context(interaction_id=7, session_id="session-new")

    results = await mm.retrieve(query="memory 0", top_k=1)
    assert len(results) >= 1

    if results[0]["source"] == "episodic":
        ep_after = await mm.episodic.get_by_id(results[0]["id"])
        # Reconsolidation should update temporal context
        assert ep_after["last_interaction"] == 7
        assert ep_after["last_session"] == "session-new"


# ------------------------------------------------------------------
# Task 2.2: Knowledge Graph Auto-Population
# ------------------------------------------------------------------


async def test_update_knowledge_graph_adds_relations(mm):
    """update_knowledge_graph should create graph edges for valid triples."""
    entities = ["Python", "AI"]
    relations = [["Python", "used_for", "AI"]]
    await mm.update_knowledge_graph(entities, relations)

    rels = await mm.semantic.get_relationships("Python")
    assert len(rels) == 1
    assert rels[0]["source"] == "Python"
    assert rels[0]["relation"] == "used_for"
    assert rels[0]["target"] == "AI"
    assert rels[0]["weight"] == pytest.approx(0.8)


async def test_update_knowledge_graph_multiple_relations(mm):
    """Multiple relations should all be added."""
    relations = [
        ["Python", "used_for", "AI"],
        ["AI", "includes", "deep_learning"],
        ["deep_learning", "uses", "neural_networks"],
    ]
    await mm.update_knowledge_graph([], relations)

    rels_python = await mm.semantic.get_relationships("Python")
    rels_ai = await mm.semantic.get_relationships("AI")
    rels_dl = await mm.semantic.get_relationships("deep_learning")
    assert len(rels_python) == 1  # source of used_for
    assert len(rels_ai) == 2      # target of used_for + source of includes (bidirectional)
    assert len(rels_dl) == 2       # target of includes + source of uses (bidirectional)


async def test_update_knowledge_graph_skips_invalid_triples(mm):
    """Relations with < 3 elements should be silently skipped."""
    relations = [
        ["Python", "used_for"],      # Too short — skipped
        ["Python", "used_for", "AI"], # Valid
        ["a", "b", "c", "d"],        # 4 elements, d not float → weight=0.8, still valid
    ]
    await mm.update_knowledge_graph([], relations)

    rels_python = await mm.semantic.get_relationships("Python")
    rels_a = await mm.semantic.get_relationships("a")
    assert len(rels_python) == 1
    assert len(rels_a) == 1


async def test_update_knowledge_graph_empty(mm):
    """Empty relations list should be a no-op (no error)."""
    await mm.update_knowledge_graph([], [])
    # Should not raise


# ------------------------------------------------------------------
# Task 4.2: Learning rate modulates encoding strength (Hasselmo 2006)
# ------------------------------------------------------------------


async def test_encode_with_high_learning_rate(mm):
    """High learning_rate (ACh) should increase initial encoding strength
    above 1.0 (Hasselmo 2006)."""
    mem_id = await mm.encode(
        content="important fact to remember",
        entities={},
        learning_rate=1.0,
    )
    mem = await mm.staging.get_by_id(mem_id)
    # learning_rate=1.0 => strength = 1.0 + 0.5*(1.0-0.5) = 1.25
    assert mem["strength"] == pytest.approx(1.25, abs=0.01)


async def test_encode_with_low_learning_rate(mm):
    """Low learning_rate (ACh) should decrease initial encoding strength
    below 1.0 (Hasselmo 2006)."""
    mem_id = await mm.encode(
        content="less important fact",
        entities={},
        learning_rate=0.0,
    )
    mem = await mm.staging.get_by_id(mem_id)
    # learning_rate=0.0 => strength = 1.0 + 0.5*(0.0-0.5) = 0.75
    assert mem["strength"] == pytest.approx(0.75, abs=0.01)


async def test_encode_with_default_learning_rate(mm):
    """Default learning_rate (0.5) should give strength = 1.0."""
    mem_id = await mm.encode(
        content="normal fact",
        entities={},
        learning_rate=0.5,
    )
    mem = await mm.staging.get_by_id(mem_id)
    # learning_rate=0.5 => strength = 1.0 + 0.5*(0.5-0.5) = 1.0
    assert mem["strength"] == pytest.approx(1.0, abs=0.01)


async def test_encode_without_learning_rate(mm):
    """Without learning_rate, strength should remain at default 1.0."""
    mem_id = await mm.encode(
        content="standard encoding",
        entities={},
    )
    mem = await mm.staging.get_by_id(mem_id)
    assert mem["strength"] == pytest.approx(1.0, abs=0.01)


# ------------------------------------------------------------------
# Task 3: Immediate Semantic Fact Storage
# ------------------------------------------------------------------


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

    # Check vector store has entities + fact documents
    # 2 entities ("Python", "AI") + 1 fact = 3 documents
    count = await m.semantic.count()
    assert count >= 3, f"Expected >=3 docs (2 entities + 1 fact), got {count}"

    results = await m.semantic.search("Python AI", top_k=5)
    assert any("Python" in r["content"] for r in results)

    await m.close()


# ------------------------------------------------------------------
# Task 3: 5-element relation handling with category
# ------------------------------------------------------------------


async def test_store_semantic_facts_with_category(mm):
    """store_semantic_facts should pass category to add_relationship."""
    relations = [["user", "like", "coffee", 0.9, "PREFERENCE"]]
    await mm.store_semantic_facts(entities=["user", "coffee"], relations=relations)
    rels = await mm.semantic.get_relationships("user")
    assert len(rels) == 1
    assert rels[0]["category"] == "PREFERENCE"
    assert rels[0]["weight"] == pytest.approx(0.9)


async def test_store_semantic_facts_upsert(mm):
    """Duplicate relations should UPSERT, not create duplicates."""
    rel = [["user", "like", "coffee", 0.8, "PREFERENCE"]]
    await mm.store_semantic_facts(entities=[], relations=rel)
    await mm.store_semantic_facts(entities=[], relations=rel)
    rels = await mm.semantic.get_relationships("user")
    assert len(rels) == 1
    assert rels[0]["occurrence_count"] == 2


async def test_update_knowledge_graph_with_category(mm):
    """update_knowledge_graph should handle 5-element relations."""
    relations = [["python", "use_for", "ai", 0.95, "ACTION"]]
    await mm.update_knowledge_graph([], relations)
    rels = await mm.semantic.get_relationships("python")
    assert rels[0]["category"] == "ACTION"


async def test_memory_manager_registers_workspace_store(tmp_path, mock_embedding):
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm.initialize()
    try:
        assert isinstance(mm.workspace, WorkspaceStore)
        personal = await mm.workspace.get_workspace(PERSONAL_WORKSPACE_ID)
        assert personal is not None
    finally:
        await mm.close()


async def test_memory_manager_registers_ontology_store_with_universal_seed(
    tmp_path, mock_embedding
):
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm.initialize()
    try:
        assert isinstance(mm.ontology, OntologyStore)
        universal_nodes = await mm.ontology.get_node_types(UNIVERSAL_WORKSPACE_ID)
        assert len(universal_nodes) == 7
        universal_relations = await mm.ontology.get_relation_types(
            UNIVERSAL_WORKSPACE_ID
        )
        assert len(universal_relations) == 10
    finally:
        await mm.close()


async def test_memory_manager_initialize_idempotent(tmp_path, mock_embedding):
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm.initialize()
    try:
        nodes_first = await mm.ontology.get_node_types(UNIVERSAL_WORKSPACE_ID)
    finally:
        await mm.close()

    mm2 = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm2.initialize()
    try:
        nodes_second = await mm2.ontology.get_node_types(UNIVERSAL_WORKSPACE_ID)
    finally:
        await mm2.close()

    assert len(nodes_first) == len(nodes_second) == 7


async def test_memory_manager_registers_contradictions_store(memory_manager):
    assert isinstance(memory_manager.contradictions, ContradictionsStore)
    row = await memory_manager.contradictions.detect(
        workspace_id="personal",
        subject="Alice",
        key_or_relation="prefers_coffee",
        value_a="yes",
        value_b="no",
        value_a_confidence="EXTRACTED",
        value_b_confidence="EXTRACTED",
    )
    assert row["severity"] == "severe"
    openings = await memory_manager.contradictions.list_open("personal")
    assert len(openings) == 1


async def test_memory_manager_registers_open_questions_store(memory_manager):
    assert isinstance(memory_manager.open_questions, OpenQuestionsStore)
    q = await memory_manager.open_questions.add_question(
        workspace_id="personal",
        question="Which region?",
        raised_by="unknown_fact",
        severity="severe",
    )
    assert q["blocking"] == 1
    blocking = await memory_manager.open_questions.list_blocking("personal")
    assert len(blocking) == 1


async def test_memory_manager_phase2_stores_survive_reopen(tmp_path, mock_embedding):
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm.initialize()
    try:
        await mm.contradictions.detect(
            workspace_id="personal",
            subject="Alice",
            key_or_relation="k",
            value_a="x",
            value_b="y",
            value_a_confidence="EXTRACTED",
            value_b_confidence="EXTRACTED",
        )
        await mm.open_questions.add_question(
            workspace_id="personal",
            question="Q",
            raised_by="user",
            severity="severe",
        )
    finally:
        await mm.close()

    mm2 = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embedding)
    await mm2.initialize()
    try:
        assert len(await mm2.contradictions.list_open("personal")) == 1
        assert await mm2.open_questions.count_blocking("personal") == 1
    finally:
        await mm2.close()
