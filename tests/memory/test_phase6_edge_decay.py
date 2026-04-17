"""Tests for Phase 6 SemanticStore edge decay/pruning awareness."""

import pytest

from brain_agent.memory.semantic_store import SemanticStore


@pytest.fixture
async def semantic(tmp_path, mock_embedding):
    store = SemanticStore(
        chroma_path=str(tmp_path / "chroma"),
        graph_db_path=str(tmp_path / "graph.db"),
        embed_fn=mock_embedding,
    )
    await store.initialize()
    yield store
    await store.close()


async def test_decay_edge_weights_skips_never_decay(semantic):
    await semantic.add_relationship(
        "alice",
        "owns",
        "billing_service",
        weight=0.8,
        never_decay=1,
        importance_score=0.5,
    )
    await semantic.add_relationship(
        "alice",
        "likes",
        "coffee",
        weight=0.8,
        never_decay=0,
        importance_score=0.5,
    )
    await semantic.decay_edge_weights(factor=0.5)
    edges = await semantic.get_edges()
    protected = next(e for e in edges if e["relation"] == "owns")
    regular = next(e for e in edges if e["relation"] == "likes")
    assert protected["weight"] == pytest.approx(0.8, abs=1e-6)
    assert regular["weight"] < 0.8


async def test_decay_edge_weights_importance_reduces_decay(semantic):
    await semantic.add_relationship(
        "a",
        "r",
        "b",
        weight=1.0,
        importance_score=0.0,
    )
    await semantic.add_relationship(
        "c",
        "r",
        "d",
        weight=1.0,
        importance_score=1.0,
    )
    await semantic.decay_edge_weights(factor=0.5)
    edges = await semantic.get_edges()
    low = next(e for e in edges if e["source_node"] == "a")
    high = next(e for e in edges if e["source_node"] == "c")
    assert low["weight"] == pytest.approx(0.5, abs=1e-6)
    assert high["weight"] == pytest.approx(0.75, abs=1e-6)


async def test_decay_edge_weights_workspace_filter(semantic):
    await semantic.add_relationship(
        "a",
        "r",
        "b",
        weight=1.0,
        workspace_id="personal",
    )
    await semantic.add_relationship(
        "c",
        "r",
        "d",
        weight=1.0,
        workspace_id="biz",
    )
    await semantic.decay_edge_weights(factor=0.5, workspace_id="personal")
    edges = await semantic.get_edges()
    personal = next(e for e in edges if e["source_node"] == "a")
    biz = next(e for e in edges if e["source_node"] == "c")
    assert personal["weight"] < 1.0
    assert biz["weight"] == pytest.approx(1.0, abs=1e-6)


async def test_prune_weak_edges_skips_never_decay(semantic):
    await semantic.add_relationship(
        "a",
        "r",
        "b",
        weight=0.0,
        never_decay=1,
    )
    await semantic.add_relationship(
        "c",
        "r",
        "d",
        weight=0.0,
        never_decay=0,
    )
    pruned = await semantic.prune_weak_edges(min_weight=0.1)
    edges = await semantic.get_edges()
    relations = {(e["source_node"], e["target_node"]) for e in edges}
    assert ("a", "b") in relations
    assert ("c", "d") not in relations
    assert pruned == 1


async def test_prune_weak_edges_workspace_filter(semantic):
    await semantic.add_relationship(
        "a",
        "r",
        "b",
        weight=0.01,
        workspace_id="personal",
    )
    await semantic.add_relationship(
        "c",
        "r",
        "d",
        weight=0.01,
        workspace_id="biz",
    )
    pruned = await semantic.prune_weak_edges(min_weight=0.1, workspace_id="biz")
    edges = await semantic.get_edges()
    personal_keep = any(e["source_node"] == "a" for e in edges)
    biz_gone = not any(e["source_node"] == "c" for e in edges)
    assert personal_keep and biz_gone
    assert pruned == 1
