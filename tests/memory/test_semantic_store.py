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
    await store.add_relationship("Python", "is_a", "Programming Language", weight=0.9, category="ATTRIBUTE")
    rels = await store.get_relationships("Python")
    assert len(rels) == 1
    assert rels[0]["relation"] == "is_a"
    assert rels[0]["weight"] == pytest.approx(0.9)
    assert rels[0]["category"] == "ATTRIBUTE"


async def test_spreading_activation(store):
    await store.add_relationship("Python", "is_a", "Language", weight=0.9)
    await store.add_relationship("Language", "used_for", "Communication", weight=0.8)
    await store.add_relationship("Python", "used_for", "AI", weight=0.7)
    activated = await store.spread_activation(
        start_nodes=["Python"], max_hops=2, decay=0.85
    )
    assert "Python" in activated
    assert activated["Python"] > activated.get("Language", 0)


async def test_get_count(store):
    await store.add("fact one", category="fact")
    await store.add("fact two", category="fact")
    count = await store.count()
    assert count == 2


async def test_add_relationship_upsert(store):
    """Duplicate triple should UPSERT: increment count, bump weight."""
    await store.add_relationship("coffee", "drink", "americano", weight=0.7, category="PREFERENCE")
    await store.add_relationship("coffee", "drink", "americano", weight=0.8, category="PREFERENCE")
    rels = await store.get_relationships("coffee")
    assert len(rels) == 1  # deduplicated
    # MAX(0.8, 0.7) + 0.1/(1 + 1*0.5) ≈ 0.867 (diminishing-return formula)
    assert rels[0]["weight"] == pytest.approx(0.867, abs=0.01)
    assert rels[0]["occurrence_count"] == 2
    assert rels[0]["category"] == "PREFERENCE"


async def test_add_relationship_category_upgrade(store):
    """Non-GENERAL category should overwrite GENERAL."""
    await store.add_relationship("user", "like", "python", weight=0.8, category="GENERAL")
    await store.add_relationship("user", "like", "python", weight=0.9, category="PREFERENCE")
    rels = await store.get_relationships("user")
    assert rels[0]["category"] == "PREFERENCE"


async def test_add_relationship_weight_cap(store):
    """Weight should cap at 1.0 after many UPSERTs."""
    for _ in range(30):
        await store.add_relationship("a", "relate", "b", weight=0.9)
    rels = await store.get_relationships("a")
    assert rels[0]["weight"] <= 1.0


async def test_bidirectional_get_relationships(store):
    """get_relationships should find node as source OR target."""
    await store.add_relationship("coffee", "contain", "caffeine", weight=0.9, category="ATTRIBUTE")
    rels = await store.get_relationships("caffeine")
    assert len(rels) == 1
    assert rels[0]["source"] == "coffee"


async def test_spread_activation_bidirectional(store):
    """Spread activation should traverse both directions."""
    await store.add_relationship("coffee", "contain", "caffeine", weight=0.9, category="ATTRIBUTE")
    await store.add_relationship("tea", "contain", "caffeine", weight=0.8, category="ATTRIBUTE")
    activated = await store.spread_activation(start_nodes=["caffeine"], max_hops=2, decay=0.85)
    assert "coffee" in activated
    assert "tea" in activated
