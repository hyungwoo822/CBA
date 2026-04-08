"""MCP tools for querying brain_agent's knowledge graph.

Neuroscience: metacognition (Flavell 1979) — the prefrontal cortex
monitors its own cognitive processes. These tools let the agent
query its own semantic memory as structured graph operations.
"""
from __future__ import annotations

from typing import Any


class KnowledgeGraphTools:
    """Stateless tool definitions for knowledge graph queries.

    Each method takes a SemanticStore instance and query parameters.
    Designed to be called by the agent's tool execution pipeline
    or exposed via MCP server.
    """

    @staticmethod
    async def query_graph(semantic_store: Any, query: str, top_k: int = 5) -> dict:
        """Search semantic memory by text similarity."""
        results = await semantic_store.search(query, top_k=top_k)
        return {"results": results, "count": len(results)}

    @staticmethod
    async def get_neighbors(semantic_store: Any, node: str) -> dict:
        """Get all relationships for a concept node."""
        rels = await semantic_store.get_relationships(node)
        return {"node": node, "neighbors": rels, "count": len(rels)}

    @staticmethod
    async def list_communities(semantic_store: Any) -> dict:
        """List concept communities with cohesion scores."""
        comms = await semantic_store.cluster_knowledge()
        G = await semantic_store.export_as_networkx()
        from brain_agent.memory.graph_analysis import cohesion_scores
        scores = cohesion_scores(G, comms)
        return {
            "communities": {
                str(cid): {"members": nodes, "cohesion": scores.get(cid, 0.0)}
                for cid, nodes in comms.items()
            },
            "count": len(comms),
        }

    @staticmethod
    async def find_hubs(semantic_store: Any, top_n: int = 5) -> dict:
        """Find most-connected hub concepts."""
        hubs = await semantic_store.find_hub_concepts(top_n=top_n)
        return {"hubs": hubs, "count": len(hubs)}

    @staticmethod
    async def find_bridges(semantic_store: Any, top_n: int = 3) -> dict:
        """Find surprising cross-community connections."""
        bridges = await semantic_store.find_bridges(top_n=top_n)
        return {"bridges": bridges, "count": len(bridges)}

    @staticmethod
    async def get_assemblies(semantic_store: Any, node: str) -> dict:
        """Get cell assemblies containing a concept node."""
        assemblies = await semantic_store.get_assemblies_for_node(node)
        return {"node": node, "assemblies": assemblies, "count": len(assemblies)}
