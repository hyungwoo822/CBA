"""Tests for brain_agent.memory.graph_analysis.

TDD: these tests are written before the implementation to drive development.
"""
from __future__ import annotations

import networkx as nx
import pytest

from brain_agent.memory.graph_analysis import (
    cluster_graph,
    cohesion_score,
    cohesion_scores,
    hub_concepts,
    surprising_connections,
    graph_diff,
)


# ---------------------------------------------------------------------------
# Helper graph factories (module-level functions used as fixtures)
# ---------------------------------------------------------------------------


def _triangle_graph() -> nx.Graph:
    """Three nodes a, b, c fully connected."""
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c"])
    G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
    return G


def _two_cluster_graph() -> nx.Graph:
    """Two triangles (a,b,c) and (d,e,f) connected by a single bridge c-d."""
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c", "d", "e", "f"])
    G.add_edges_from([
        ("a", "b"), ("b", "c"), ("a", "c"),   # first triangle
        ("d", "e"), ("e", "f"), ("d", "f"),   # second triangle
        ("c", "d"),                            # bridge
    ])
    return G


def _star_graph() -> nx.Graph:
    """Hub 'center' connected to leaf_0 through leaf_4."""
    G = nx.Graph()
    hub = "center"
    leaves = [f"leaf_{i}" for i in range(5)]
    G.add_node(hub)
    for leaf in leaves:
        G.add_node(leaf)
        G.add_edge(hub, leaf)
    return G


def _bridged_graph() -> nx.Graph:
    """Two triangles with labels and confidence attributes on nodes and edges."""
    G = _two_cluster_graph()
    # Add node labels
    for node in G.nodes():
        G.nodes[node]["label"] = f"label_{node}"
    # Add edge attributes: confidence and relation
    confidence_map = {
        ("a", "b"): "EXTRACTED",
        ("b", "c"): "INFERRED",
        ("a", "c"): "AMBIGUOUS",
        ("d", "e"): "EXTRACTED",
        ("e", "f"): "EXTRACTED",
        ("d", "f"): "INFERRED",
        ("c", "d"): "AMBIGUOUS",   # bridge gets highest surprise score
    }
    relation_map = {
        ("a", "b"): "knows",
        ("b", "c"): "related_to",
        ("a", "c"): "similar_to",
        ("d", "e"): "knows",
        ("e", "f"): "related_to",
        ("d", "f"): "similar_to",
        ("c", "d"): "bridges",
    }
    for (u, v), conf in confidence_map.items():
        if G.has_edge(u, v):
            G[u][v]["confidence"] = conf
            G[u][v]["relation"] = relation_map[(u, v)]
    return G


# ---------------------------------------------------------------------------
# Clustering tests
# ---------------------------------------------------------------------------


def test_cluster_empty_graph():
    """Empty graph should return an empty dict."""
    G = nx.Graph()
    result = cluster_graph(G)
    assert result == {}


def test_cluster_no_edges():
    """Graph with nodes but no edges — each node gets its own community."""
    G = nx.Graph()
    G.add_nodes_from(["x", "y", "z"])
    result = cluster_graph(G)
    assert isinstance(result, dict)
    assert len(result) == 3
    all_nodes = {n for nodes in result.values() for n in nodes}
    assert all_nodes == {"x", "y", "z"}


def test_cluster_triangle():
    """Triangle graph — all 3 nodes present in the resulting communities."""
    G = _triangle_graph()
    result = cluster_graph(G)
    assert isinstance(result, dict)
    all_nodes = {n for nodes in result.values() for n in nodes}
    assert all_nodes == {"a", "b", "c"}


def test_cluster_two_clusters():
    """Two-cluster graph — at least 2 communities, all 6 nodes covered."""
    G = _two_cluster_graph()
    result = cluster_graph(G)
    assert isinstance(result, dict)
    assert len(result) >= 2
    all_nodes = {n for nodes in result.values() for n in nodes}
    assert all_nodes == {"a", "b", "c", "d", "e", "f"}


# ---------------------------------------------------------------------------
# Cohesion tests
# ---------------------------------------------------------------------------


def test_cohesion_score_triangle():
    """Fully connected triangle → cohesion = 1.0."""
    G = _triangle_graph()
    score = cohesion_score(G, ["a", "b", "c"])
    assert score == pytest.approx(1.0)


def test_cohesion_score_partial():
    """Triangle where only 2 of 3 possible edges exist → cohesion ≈ 0.6667."""
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c"])
    G.add_edges_from([("a", "b"), ("b", "c")])  # missing a-c
    score = cohesion_score(G, ["a", "b", "c"])
    assert score == pytest.approx(2 / 3, abs=1e-4)


def test_cohesion_score_single_node():
    """Single-node community → cohesion = 1.0."""
    G = _triangle_graph()
    score = cohesion_score(G, ["a"])
    assert score == 1.0


def test_cohesion_scores_all():
    """All cohesion scores should be between 0 and 1 inclusive."""
    G = _two_cluster_graph()
    communities = cluster_graph(G)
    scores = cohesion_scores(G, communities)
    assert set(scores.keys()) == set(communities.keys())
    for cid, score in scores.items():
        assert 0.0 <= score <= 1.0, f"Community {cid} has score {score} out of range"


# ---------------------------------------------------------------------------
# Hub concept tests
# ---------------------------------------------------------------------------


def test_hub_concepts_star():
    """In a star graph, 'center' should be the top hub with 5 edges."""
    G = _star_graph()
    hubs = hub_concepts(G, top_n=10)
    assert len(hubs) > 0
    top = hubs[0]
    assert top["id"] == "center"
    assert top["edges"] == 5


def test_hub_concepts_empty():
    """Empty graph → no hubs."""
    G = nx.Graph()
    hubs = hub_concepts(G)
    assert hubs == []


def test_hub_concepts_skips_degree_zero():
    """Nodes with degree 0 should not appear in hub_concepts output."""
    G = nx.Graph()
    G.add_nodes_from(["isolated_a", "isolated_b"])
    G.add_node("connected")
    G.add_node("connected2")
    G.add_edge("connected", "connected2")
    hubs = hub_concepts(G, top_n=10)
    hub_ids = {h["id"] for h in hubs}
    assert "isolated_a" not in hub_ids
    assert "isolated_b" not in hub_ids


# ---------------------------------------------------------------------------
# Surprising connections tests
# ---------------------------------------------------------------------------


def test_surprising_connections_bridged():
    """Bridge edge c-d should appear in surprising connections for bridged graph."""
    G = _bridged_graph()
    communities = cluster_graph(G)
    surprises = surprising_connections(G, communities, top_n=5)
    assert isinstance(surprises, list)
    assert len(surprises) > 0
    # Verify structure
    for item in surprises:
        assert "source" in item
        assert "target" in item
        assert "confidence" in item
        assert "relation" in item
        assert "why" in item
    # Bridge c-d should be discovered (may be first or within top results)
    bridge_found = any(
        (s["source"] == "c" and s["target"] == "d")
        or (s["source"] == "d" and s["target"] == "c")
        for s in surprises
    )
    assert bridge_found, f"Bridge c-d not found in: {surprises}"


def test_surprising_connections_no_communities():
    """When no communities provided, falls back to edge betweenness centrality."""
    G = _two_cluster_graph()
    surprises = surprising_connections(G, {}, top_n=3)
    assert isinstance(surprises, list)
    assert len(surprises) > 0
    for item in surprises:
        assert "source" in item
        assert "target" in item
        assert "confidence" in item
        assert "relation" in item
        assert "why" in item


def test_surprising_connections_returns_top_n():
    """surprising_connections should return at most top_n results."""
    G = _bridged_graph()
    communities = cluster_graph(G)
    surprises = surprising_connections(G, communities, top_n=2)
    assert len(surprises) <= 2


# ---------------------------------------------------------------------------
# Graph diff (neuroplasticity) tests
# ---------------------------------------------------------------------------


def test_graph_diff_no_changes():
    G = _triangle_graph()
    diff = graph_diff(G, G)
    assert diff["summary"] == "no changes"
    assert diff["ltp"] == []
    assert diff["ltd"] == []


def test_graph_diff_new_node():
    G_old = _triangle_graph()
    G_new = _triangle_graph()
    G_new.add_node("d", label="d")
    G_new.add_edge("c", "d")
    diff = graph_diff(G_old, G_new)
    assert len(diff["ltp"]) == 1
    assert diff["ltp"][0]["id"] == "d"
    assert len(diff["new_synapses"]) == 1


def test_graph_diff_removed_node():
    G_old = _triangle_graph()
    G_old.add_node("d", label="d")
    G_new = _triangle_graph()
    diff = graph_diff(G_old, G_new)
    assert len(diff["ltd"]) == 1
    assert diff["ltd"][0]["id"] == "d"


def test_graph_diff_summary():
    G_old = nx.Graph()
    G_old.add_edges_from([("a", "b"), ("b", "c")])
    G_new = nx.Graph()
    G_new.add_edges_from([("a", "b"), ("c", "d")])
    diff = graph_diff(G_old, G_new)
    assert "no changes" not in diff["summary"]
