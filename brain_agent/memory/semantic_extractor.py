"""Episodic-to-semantic transition via PFC-mediated abstraction.

Brain mapping: During sleep, hippocampus replays episodes to neocortex (PFC).
Repeated patterns are extracted as decontextualized semantic facts.
Winocur & Moscovitch (2011).
"""
from __future__ import annotations

import json
import re
from typing import Callable

CLUSTER_SIMILARITY_THRESHOLD = 0.80
MIN_CLUSTER_SIZE = 3


def find_episode_clusters(
    episodes: list[dict],
    similarity_fn: Callable[[list[float], list[float]], float],
    threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
    min_size: int = MIN_CLUSTER_SIZE,
) -> list[list[dict]]:
    """Cluster episodes by embedding similarity (greedy single-linkage).

    Parameters
    ----------
    episodes : list[dict]
        Episodes with ``context_embedding`` field.
    similarity_fn : callable
        ``(vec_a, vec_b) -> float`` returning cosine similarity in [0,1].
    threshold : float
        Minimum similarity to join a cluster (default 0.80).
    min_size : int
        Minimum cluster size to keep (default 3).

    Returns
    -------
    list[list[dict]]
        Clusters of related episodes that meet the ``min_size`` requirement.
    """
    used: set[int] = set()
    clusters: list[list[dict]] = []

    for i, ep_a in enumerate(episodes):
        if i in used or not ep_a.get("context_embedding"):
            continue
        cluster = [ep_a]
        used.add(i)
        for j, ep_b in enumerate(episodes):
            if j in used or not ep_b.get("context_embedding"):
                continue
            sim = similarity_fn(
                ep_a["context_embedding"], ep_b["context_embedding"]
            )
            if sim >= threshold:
                cluster.append(ep_b)
                used.add(j)
        if len(cluster) >= min_size:
            clusters.append(cluster)

    return clusters


def find_episode_clusters_leiden(
    episodes: list[dict],
    similarity_fn: Callable[[list[float], list[float]], float],
    threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
    min_size: int = MIN_CLUSTER_SIZE,
) -> list[list[dict]]:
    """Cluster episodes using Leiden community detection on similarity graph.

    Neuroscience: systems consolidation (Frankland & Bontempi 2005).
    Falls back to greedy single-linkage if networkx is not available.
    """
    if len(episodes) < min_size:
        return []

    valid = [(i, ep) for i, ep in enumerate(episodes) if ep.get("context_embedding")]
    if len(valid) < min_size:
        return find_episode_clusters(episodes, similarity_fn, threshold, min_size)

    try:
        import networkx as nx
        from brain_agent.memory.graph_analysis import cluster_graph
    except ImportError:
        return find_episode_clusters(episodes, similarity_fn, threshold, min_size)

    G = nx.Graph()
    for i, ep in valid:
        G.add_node(i)

    for idx_a in range(len(valid)):
        i_a, ep_a = valid[idx_a]
        for idx_b in range(idx_a + 1, len(valid)):
            i_b, ep_b = valid[idx_b]
            sim = similarity_fn(ep_a["context_embedding"], ep_b["context_embedding"])
            if sim >= threshold:
                G.add_edge(i_a, i_b, weight=sim)

    comms = cluster_graph(G)
    clusters = []
    for nodes in comms.values():
        cluster = [episodes[n] for n in nodes]
        if len(cluster) >= min_size:
            clusters.append(cluster)

    if not clusters:
        return find_episode_clusters(episodes, similarity_fn, threshold, min_size)

    return clusters


def build_extraction_prompt(cluster: list[dict]) -> str:
    """Build a prompt for PFC to extract a semantic fact from an episode cluster."""
    contents = "\n".join(f"- {ep['content']}" for ep in cluster)
    return (
        "Below are multiple related memories. Extract ONE general fact or rule "
        "that summarizes the common pattern. Also extract entity relationships.\n\n"
        f"Memories:\n{contents}\n\n"
        "Respond in this exact format:\n"
        "<fact>The general fact here</fact>\n"
        '<relations>[["subject", "relation", "object", confidence, "CATEGORY"]]</relations>\n\n'
        "Rules:\n"
        "- subject/object: English lowercase nouns (e.g., \"coffee\", \"stress\", \"health\")\n"
        "- relation: English verb infinitive (e.g., \"cause\", \"relieve\", \"contain\", \"require\")\n"
        "- confidence: 0.0-1.0 (how certain this relation is)\n"
        "- category: PREFERENCE | ACTION | ATTRIBUTE | SOCIAL | CAUSAL | SPATIAL | TEMPORAL | IDENTITY | GENERAL\n"
        "- Even if memories are in Korean/other language, always extract in English.\n"
        "- Focus on CONCEPT-to-CONCEPT relations (e.g., cigarette→relieve→stress), not just user→verb→object."
    )


def parse_extraction_response(response: str) -> tuple[str, list[list[str]]]:
    """Parse ``<fact>`` and ``<relations>`` tags from a PFC extraction response.

    Returns
    -------
    tuple[str, list[list[str]]]
        ``(fact_text, list_of_relation_triples)``
        Returns ``("", [])`` when tags are missing or malformed.
    """
    fact = ""
    relations: list[list[str]] = []

    fact_match = re.search(r"<fact>(.*?)</fact>", response, re.DOTALL)
    if fact_match:
        fact = fact_match.group(1).strip()

    rel_match = re.search(r"<relations>(.*?)</relations>", response, re.DOTALL)
    if rel_match:
        try:
            parsed = json.loads(rel_match.group(1).strip())
            if isinstance(parsed, list):
                relations = [r for r in parsed if isinstance(r, list) and len(r) >= 3]
        except (json.JSONDecodeError, TypeError):
            pass

    return fact, relations
