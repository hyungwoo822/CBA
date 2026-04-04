"""Tests for KG origin tracking and cloning score."""
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


@pytest.mark.asyncio
async def test_origin_stored_on_relationship(store):
    await store.add_relationship("user", "like", "coffee", origin="user_input")
    rels = await store.get_relationships("user")
    assert any(r.get("origin") == "user_input" for r in rels)


@pytest.mark.asyncio
async def test_origin_default_unknown(store):
    await store.add_relationship("cat", "is", "animal")
    rels = await store.get_relationships("cat")
    assert any(r.get("origin") == "unknown" for r in rels)


@pytest.mark.asyncio
async def test_same_triple_different_origins(store):
    """Same triple from user and agent should create two separate rows."""
    await store.add_relationship("user", "like", "coffee", origin="user_input")
    await store.add_relationship("user", "like", "coffee", origin="agent_response")
    rels = await store.get_relationships("user")
    # Should have two entries: one per origin
    coffee_rels = [r for r in rels if r["target"] == "coffee" and r["relation"] == "like"]
    assert len(coffee_rels) == 2
    origins = {r["origin"] for r in coffee_rels}
    assert origins == {"user_input", "agent_response"}


@pytest.mark.asyncio
async def test_cloning_score_empty(store):
    result = await store.compute_cloning_score()
    assert result["cloning_score"] == 0.0
    assert result["user_graph_size"] == 0


@pytest.mark.asyncio
async def test_cloning_score_identical_graphs(store):
    """Same triples from both sources -> high cloning score."""
    await store.add_relationship("user", "like", "coffee", origin="user_input")
    await store.add_relationship("coffee", "is", "beverage", origin="user_input")
    # Agent produces same knowledge
    await store.add_relationship("user", "like", "coffee", origin="agent_response")
    await store.add_relationship("coffee", "is", "beverage", origin="agent_response")
    result = await store.compute_cloning_score()
    # raw_recall should be high (identical graphs); maturity is low with few edges
    assert result["raw_recall"] > 0.8
    assert result["maturity"] < 0.5  # few edges = low maturity
    assert result["cloning_score"] > 0.1  # maturity-weighted


@pytest.mark.asyncio
async def test_cloning_score_disjoint_graphs(store):
    """Completely different triples -> low cloning score."""
    await store.add_relationship("user", "like", "coffee", origin="user_input")
    await store.add_relationship("agent", "know", "python", origin="agent_response")
    result = await store.compute_cloning_score()
    assert result["cloning_score"] < 0.5


@pytest.mark.asyncio
async def test_cloning_score_partial_overlap(store):
    """Some shared concepts -> moderate cloning score."""
    await store.add_relationship("user", "like", "coffee", origin="user_input")
    await store.add_relationship("user", "work", "office", origin="user_input")
    await store.add_relationship("user", "like", "coffee", origin="agent_response")
    await store.add_relationship("user", "enjoy", "music", origin="agent_response")
    result = await store.compute_cloning_score()
    assert result["raw_recall"] > 0.2
    assert result.get("node_recall", result.get("node_overlap", 0)) > 0  # "user" and "coffee" shared


@pytest.mark.asyncio
async def test_cloning_score_only_user(store):
    """Only user triples, no agent -> score 0."""
    await store.add_relationship("user", "like", "coffee", origin="user_input")
    result = await store.compute_cloning_score()
    assert result["cloning_score"] == 0.0
    assert result["user_graph_size"] == 1
    assert result["agent_graph_size"] == 0


@pytest.mark.asyncio
async def test_cloning_score_only_agent(store):
    """Only agent triples, no user -> score 0."""
    await store.add_relationship("agent", "know", "python", origin="agent_response")
    result = await store.compute_cloning_score()
    assert result["cloning_score"] == 0.0
    assert result["user_graph_size"] == 0
    assert result["agent_graph_size"] == 1


@pytest.mark.asyncio
async def test_cloning_score_directional(store):
    """Agent having extra knowledge shouldn't lower score (recall, not Jaccard)."""
    await store.add_relationship("user", "like", "coffee", origin="user_input")
    await store.add_relationship("user", "like", "coffee", origin="agent_response")
    # Agent knows extra stuff user didn't mention
    await store.add_relationship("coffee", "contain", "caffeine", origin="agent_response")
    await store.add_relationship("caffeine", "stimulate", "brain", origin="agent_response")
    result = await store.compute_cloning_score()
    # raw_recall should still be high: agent knows what user knows (+ extra)
    assert result["raw_recall"] > 0.7


@pytest.mark.asyncio
async def test_fuzzy_reversed_triple(store):
    """Reversed direction triples should fuzzy-match (A,rel,B) ↔ (B,rel,A)."""
    await store.add_relationship("coffee", "spill", "user", origin="user_input")
    await store.add_relationship("user", "spill", "coffee", origin="agent_response")
    result = await store.compute_cloning_score()
    # Should be high — same fact, just reversed
    assert result["edge_recall"] >= 0.6
    assert result["raw_recall"] > 0.5


@pytest.mark.asyncio
async def test_fuzzy_substring_node(store):
    """Substring nodes should fuzzy-match: 'extroverted' ≈ 'extroverted personality'."""
    await store.add_relationship("user", "be", "extroverted", origin="user_input")
    await store.add_relationship("user", "have", "extroverted personality", origin="agent_response")
    result = await store.compute_cloning_score()
    # "extroverted" is substring of "extroverted personality" → node match
    assert result["node_recall"] > 0.5


@pytest.mark.asyncio
async def test_fuzzy_both_directions(store):
    """Real-world scenario: user and agent express same facts differently."""
    # User says facts
    await store.add_relationship("user", "drink", "coffee", origin="user_input")
    await store.add_relationship("coffee", "spill", "user", origin="user_input")
    await store.add_relationship("user", "identify_as", "extroverted", origin="user_input")
    # Agent understands but expresses differently
    await store.add_relationship("user", "spill", "coffee", origin="agent_response")  # reversed
    await store.add_relationship("user", "have", "extroverted personality", origin="agent_response")  # substring
    await store.add_relationship("coffee", "cause", "surprise", origin="agent_response")  # different
    result = await store.compute_cloning_score()
    # raw_recall should be moderate-to-high: agent understands most of what user said
    assert result["raw_recall"] > 0.35
