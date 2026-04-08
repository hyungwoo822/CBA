import pytest
import networkx as nx
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


async def test_add_relationship_with_confidence(store):
    await store.add_relationship(
        "attention", "implements", "transformer",
        weight=0.9, category="CAUSAL", confidence="EXTRACTED",
    )
    rels = await store.get_relationships("attention")
    assert len(rels) == 1
    assert rels[0]["confidence"] == "EXTRACTED"


async def test_add_relationship_default_confidence(store):
    await store.add_relationship("a", "relates", "b", weight=0.5)
    rels = await store.get_relationships("a")
    assert rels[0]["confidence"] == "INFERRED"


async def test_add_relationship_ambiguous(store):
    await store.add_relationship(
        "x", "maybe_causes", "y", weight=0.3, confidence="AMBIGUOUS",
    )
    rels = await store.get_relationships("x")
    assert rels[0]["confidence"] == "AMBIGUOUS"


async def test_export_as_networkx(store):
    await store.add_relationship("Python", "is_a", "Language", weight=0.9)
    await store.add_relationship("Language", "used_for", "Communication", weight=0.8)
    G = await store.export_as_networkx()
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G.nodes["Python"]["label"] == "Python"


async def test_export_empty_graph(store):
    G = await store.export_as_networkx()
    assert G.number_of_nodes() == 0


async def test_cluster_knowledge(store):
    await store.add_relationship("a", "r1", "b", weight=0.9)
    await store.add_relationship("b", "r2", "c", weight=0.8)
    await store.add_relationship("a", "r3", "c", weight=0.7)
    comms = await store.cluster_knowledge()
    assert isinstance(comms, dict)
    all_nodes = {n for nodes in comms.values() for n in nodes}
    assert "a" in all_nodes


async def test_find_hub_concepts(store):
    for i in range(5):
        await store.add_relationship("hub", f"r{i}", f"leaf_{i}", weight=0.8)
    hubs = await store.find_hub_concepts(top_n=3)
    assert len(hubs) >= 1
    assert hubs[0]["id"] == "hub"


async def test_find_bridges(store):
    await store.add_relationship("a", "r1", "b", weight=0.9)
    await store.add_relationship("b", "r2", "c", weight=0.8)
    await store.add_relationship("a", "r3", "c", weight=0.7)
    await store.add_relationship("d", "r4", "e", weight=0.9)
    await store.add_relationship("e", "r5", "f", weight=0.8)
    await store.add_relationship("d", "r6", "f", weight=0.7)
    await store.add_relationship("c", "bridge", "d", weight=0.5)
    bridges = await store.find_bridges(top_n=3)
    assert isinstance(bridges, list)


async def test_spread_activation_community_bonus(store):
    """Community members should get a small activation boost."""
    # Create a tight cluster a-b-c and a separate node d
    await store.add_relationship("a", "r1", "b", weight=0.9)
    await store.add_relationship("b", "r2", "c", weight=0.8)
    await store.add_relationship("a", "r3", "c", weight=0.7)
    await store.add_relationship("c", "r4", "d", weight=0.5)  # bridge to d
    activated = await store.spread_activation(["a"], max_hops=2, decay=0.85)
    # b and c should be activated (same community as a)
    assert "b" in activated
    assert "c" in activated


async def test_spread_activation_with_assemblies(store):
    """Assembly members should be co-activated."""
    await store.add_relationship("X", "r", "Y", weight=0.9)
    await store.add_hyperedge(["X", "Z", "W"], "test_assembly", strength=0.8)
    activated = await store.spread_activation(["X"], max_hops=2, decay=0.85)
    # Z and W should get activation from assembly co-activation
    assert "Z" in activated or "W" in activated


async def test_prune_weak_edges(store):
    await store.add_relationship("a", "r", "b", weight=0.05)
    await store.add_relationship("c", "r", "d", weight=0.5)
    pruned = await store.prune_weak_edges(min_weight=0.1)
    assert pruned == 1
    rels_a = await store.get_relationships("a")
    assert len(rels_a) == 0
    rels_c = await store.get_relationships("c")
    assert len(rels_c) == 1


async def test_decay_edge_weights(store):
    await store.add_relationship("a", "r", "b", weight=1.0)
    affected = await store.decay_edge_weights(factor=0.9)
    assert affected >= 1
    rels = await store.get_relationships("a")
    assert rels[0]["weight"] == pytest.approx(0.9, abs=0.01)


async def test_prune_after_decay(store):
    """Decay + prune should remove edges that fall below threshold."""
    await store.add_relationship("weak", "r", "node", weight=0.12)
    await store.decay_edge_weights(factor=0.8)  # 0.12 * 0.8 = 0.096
    pruned = await store.prune_weak_edges(min_weight=0.1)
    assert pruned == 1
