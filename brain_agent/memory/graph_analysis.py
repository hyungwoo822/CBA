"""Graph analysis for brain_agent semantic memory.

Provides Leiden community detection, hub identification, cross-community
bridge detection, cohesion scoring, and graph diffing.

Workspace filtering is handled before this module by
SemanticStore.export_as_networkx(workspace_id=...). These functions operate
on the already-filtered graph they receive.

Neuroscience grounding:
- Community detection = cortical columns (Mountcastle 1997)
- Hub nodes = rich-club organization (van den Heuvel & Sporns 2011)
- Bridges = long-range cortical projections
- Cohesion = intra-region functional connectivity
"""
from __future__ import annotations

import networkx as nx

_MAX_COMMUNITY_FRACTION = 0.25  # communities larger than 25% of graph get split
_MIN_SPLIT_SIZE = 10            # only split if community has at least this many nodes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _partition(G: nx.Graph) -> dict[str, int]:
    """Run community detection. Returns {node_id: community_id}.

    Tries Leiden (graspologic) first — best quality.
    Falls back to Louvain (built into networkx) if graspologic is not installed.
    """
    try:
        from graspologic.partition import leiden  # type: ignore[import-untyped]
        return leiden(G)
    except ImportError:
        pass

    # Fallback: networkx Louvain (available since networkx 2.7).
    # max_level=10 and threshold=1e-4 prevent indefinite hangs on large sparse
    # graphs while producing equivalent community quality to the defaults on
    # typical corpora.
    communities = nx.community.louvain_communities(G, seed=42, max_level=10, threshold=1e-4)
    return {node: cid for cid, nodes in enumerate(communities) for node in nodes}


def _split_community(G: nx.Graph, nodes: list[str]) -> list[list[str]]:
    """Run a second Leiden pass on an oversized community subgraph.

    Returns [[n] for n in sorted(nodes)] if the subgraph has no edges.
    """
    subgraph = G.subgraph(nodes)
    if subgraph.number_of_edges() == 0:
        return [[n] for n in sorted(nodes)]
    try:
        sub_partition = _partition(subgraph)
        sub_communities: dict[int, list[str]] = {}
        for node, cid in sub_partition.items():
            sub_communities.setdefault(cid, []).append(node)
        if len(sub_communities) <= 1:
            return [sorted(nodes)]
        return [sorted(v) for v in sub_communities.values()]
    except Exception:
        return [sorted(nodes)]


def _node_community_map(communities: dict[int, list[str]]) -> dict[str, int]:
    """Invert communities dict: {node_id: community_id}."""
    return {node: cid for cid, nodes in communities.items() for node in nodes}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_graph(G: nx.Graph) -> dict[int, list[str]]:
    """Main clustering function. Returns {community_id: [sorted node_ids]}.

    Handles:
    - Empty graph → {}
    - Graph with no edges → each node is its own community
    - Isolates are placed in their own single-node communities
    - Oversized communities (>25% of graph, min 10 nodes) are recursively split

    Communities are sorted by size descending and re-indexed from 0.
    """
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_edges() == 0:
        return {i: [n] for i, n in enumerate(sorted(G.nodes()))}

    # Leiden/Louvain warn and drop isolates — handle them separately
    isolates = [n for n in G.nodes() if G.degree(n) == 0]
    connected_nodes = [n for n in G.nodes() if G.degree(n) > 0]
    connected = G.subgraph(connected_nodes)

    raw: dict[int, list[str]] = {}
    if connected.number_of_nodes() > 0:
        partition = _partition(connected)
        for node, cid in partition.items():
            raw.setdefault(cid, []).append(node)

    # Each isolate becomes its own single-node community
    next_cid = max(raw.keys(), default=-1) + 1
    for node in isolates:
        raw[next_cid] = [node]
        next_cid += 1

    # Split oversized communities
    max_size = max(_MIN_SPLIT_SIZE, int(G.number_of_nodes() * _MAX_COMMUNITY_FRACTION))
    final_communities: list[list[str]] = []
    for nodes in raw.values():
        if len(nodes) > max_size:
            final_communities.extend(_split_community(G, nodes))
        else:
            final_communities.append(nodes)

    # Re-index by size descending for deterministic ordering
    final_communities.sort(key=len, reverse=True)
    return {i: sorted(nodes) for i, nodes in enumerate(final_communities)}


def cohesion_score(G: nx.Graph, community_nodes: list[str]) -> float:
    """Ratio of actual to maximum possible intra-community edges.

    Returns 1.0 for a single-node community.
    Result is rounded to 4 decimal places.
    """
    n = len(community_nodes)
    if n <= 1:
        return 1.0
    subgraph = G.subgraph(community_nodes)
    actual = subgraph.number_of_edges()
    possible = n * (n - 1) / 2
    return round(actual / possible, 4) if possible > 0 else 0.0


def cohesion_scores(G: nx.Graph, communities: dict[int, list[str]]) -> dict[int, float]:
    """Compute cohesion_score for every community. Returns {community_id: score}."""
    return {cid: cohesion_score(G, nodes) for cid, nodes in communities.items()}


def hub_concepts(G: nx.Graph, top_n: int = 10) -> list[dict]:
    """Return the most-connected nodes sorted by degree descending.

    Nodes with degree 0 are skipped.
    Each entry: {"id": node_id, "label": label, "edges": degree}.
    """
    if G.number_of_nodes() == 0:
        return []
    ranked = sorted(
        ((node, deg) for node, deg in G.degree() if deg > 0),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {
            "id": node,
            "label": G.nodes[node].get("label", node),
            "edges": deg,
        }
        for node, deg in ranked[:top_n]
    ]


def surprising_connections(
    G: nx.Graph,
    communities: dict[int, list[str]],
    top_n: int = 5,
) -> list[dict]:
    """Cross-community edges ranked by surprise score.

    If no communities are provided (empty dict), falls back to edge
    betweenness centrality to rank edges by structural importance.

    Surprise score factors:
    - Confidence bonus: AMBIGUOUS=3, INFERRED=2, EXTRACTED=1
    - Cross-community bonus: +1 if source and target in different communities
    - Peripheral-to-hub bonus: +1 if min_degree <= 2 and max_degree >= 5

    Each entry: {"source", "target", "confidence", "relation", "why"}.
    """
    _CONF_BONUS: dict[str, int] = {"AMBIGUOUS": 3, "INFERRED": 2, "EXTRACTED": 1}

    if G.number_of_edges() == 0:
        return []

    # Fallback: no communities — rank by edge betweenness centrality
    if not communities:
        betweenness = nx.edge_betweenness_centrality(G)
        ranked_edges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        results: list[dict] = []
        for (u, v), _ in ranked_edges[:top_n]:
            edata = G[u][v]
            results.append({
                "source": u,
                "target": v,
                "confidence": edata.get("confidence", "UNKNOWN"),
                "relation": edata.get("relation", ""),
                "why": "high betweenness centrality (structural bridge)",
            })
        return results

    node_to_community = _node_community_map(communities)

    scored: list[tuple[float, str, str, dict]] = []
    for u, v, edata in G.edges(data=True):
        score = 0.0

        # Confidence bonus
        conf = edata.get("confidence", "EXTRACTED")
        score += _CONF_BONUS.get(conf, 1)

        # Cross-community bonus
        cu = node_to_community.get(u)
        cv = node_to_community.get(v)
        is_cross = (cu is not None and cv is not None and cu != cv)
        if is_cross:
            score += 1

        # Peripheral-to-hub bonus
        deg_u = G.degree(u)
        deg_v = G.degree(v)
        min_deg = min(deg_u, deg_v)
        max_deg = max(deg_u, deg_v)
        if min_deg <= 2 and max_deg >= 5:
            score += 1

        why_parts: list[str] = []
        if is_cross:
            why_parts.append(f"cross-community bridge (community {cu} → {cv})")
        if conf in ("AMBIGUOUS", "INFERRED"):
            why_parts.append(f"{conf.lower()} confidence")
        if min_deg <= 2 and max_deg >= 5:
            why_parts.append("peripheral-to-hub connection")
        if not why_parts:
            why_parts.append("notable connection")

        scored.append((score, u, v, edata))

    # Sort by score descending, then by source/target for determinism
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))

    results = []
    for score, u, v, edata in scored[:top_n]:
        conf = edata.get("confidence", "UNKNOWN")
        cu = node_to_community.get(u)
        cv = node_to_community.get(v)
        is_cross = (cu is not None and cv is not None and cu != cv)

        why_parts = []
        if is_cross:
            why_parts.append(f"cross-community bridge (community {cu} → {cv})")
        if conf in ("AMBIGUOUS", "INFERRED"):
            why_parts.append(f"{conf.lower()} confidence")
        deg_u = G.degree(u)
        deg_v = G.degree(v)
        if min(deg_u, deg_v) <= 2 and max(deg_u, deg_v) >= 5:
            why_parts.append("peripheral-to-hub connection")
        if not why_parts:
            why_parts.append("notable connection")

        results.append({
            "source": u,
            "target": v,
            "confidence": conf,
            "relation": edata.get("relation", ""),
            "why": "; ".join(why_parts),
        })
    return results


def graph_diff(G_old: nx.Graph, G_new: nx.Graph) -> dict:
    """Compare two graph snapshots. Returns neuroplasticity-classified changes.

    Neuroscience mapping:
    - ltp (new nodes): Long-Term Potentiation — new synaptic connections
    - ltd (removed nodes): Long-Term Depression — weakened/lost connections
    - new_synapses (new edges): synaptogenesis
    - pruned_synapses (removed edges): synaptic pruning (Huttenlocher 1979)
    """
    old_nodes = set(G_old.nodes())
    new_nodes = set(G_new.nodes())

    ltp = [
        {"id": n, "label": G_new.nodes[n].get("label", n)}
        for n in sorted(new_nodes - old_nodes)
    ]
    ltd = [
        {"id": n, "label": G_old.nodes[n].get("label", n)}
        for n in sorted(old_nodes - new_nodes)
    ]

    def _edge_key(u: str, v: str, data: dict) -> tuple:
        return (min(u, v), max(u, v), data.get("relation", ""))

    old_edges = {_edge_key(u, v, d) for u, v, d in G_old.edges(data=True)}
    new_edges = {_edge_key(u, v, d) for u, v, d in G_new.edges(data=True)}

    new_synapses = []
    for u, v, d in G_new.edges(data=True):
        if _edge_key(u, v, d) in (new_edges - old_edges):
            new_synapses.append({"source": u, "target": v, "relation": d.get("relation", "")})

    pruned_synapses = []
    for u, v, d in G_old.edges(data=True):
        if _edge_key(u, v, d) in (old_edges - new_edges):
            pruned_synapses.append({"source": u, "target": v, "relation": d.get("relation", "")})

    parts = []
    if ltp:
        parts.append(f"{len(ltp)} ltp (new concepts)")
    if ltd:
        parts.append(f"{len(ltd)} ltd (lost concepts)")
    if new_synapses:
        parts.append(f"{len(new_synapses)} new synapses")
    if pruned_synapses:
        parts.append(f"{len(pruned_synapses)} pruned synapses")
    summary = ", ".join(parts) if parts else "no changes"

    return {
        "ltp": ltp,
        "ltd": ltd,
        "new_synapses": new_synapses,
        "pruned_synapses": pruned_synapses,
        "summary": summary,
    }


def assembly_coactivation(
    active_nodes: list[str],
    assemblies: list[dict],
    decay: float = 0.7,
) -> dict[str, float]:
    """Co-activate assembly members when any member is active.

    Neuroscience: Hebb's Cell Assembly (1949). Neurons that fire
    together wire together — activating one member triggers the ensemble.

    Args:
        active_nodes: Currently active concept nodes.
        assemblies: List of dicts with "members" (list[str]) and "strength" (float).
        decay: Activation spread factor (0-1).

    Returns:
        Dict mapping inactive member nodes to their co-activation level.
    """
    activation: dict[str, float] = {}
    active_set = set(active_nodes)
    for assembly in assemblies:
        members = set(assembly.get("members", []))
        overlap = members & active_set
        if not overlap:
            continue
        strength = assembly.get("strength", 1.0)
        spread = strength * decay * (len(overlap) / len(members))
        for member in members - active_set:
            activation[member] = max(activation.get(member, 0.0), spread)
    return activation
