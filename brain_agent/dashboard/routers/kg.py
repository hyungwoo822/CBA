"""Knowledge graph visualization endpoint with workspace filters."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query


def build_router(state: dict) -> APIRouter:
    router = APIRouter(prefix="/api/memory", tags=["knowledge-graph"])

    def _agent():
        agent = state.get("agent")
        if agent is None or not getattr(agent, "_initialized", False):
            raise HTTPException(503, "agent not initialized")
        return agent

    @router.get("/knowledge-graph")
    async def get_knowledge_graph(
        workspace_id: str | None = Query(None),
        include_cross_refs: bool = Query(False),
    ):
        agent = _agent()
        try:
            graph = await agent.memory.semantic.export_as_networkx(
                workspace_id=workspace_id,
                include_cross_refs=include_cross_refs,
            )
            from brain_agent.memory.graph_analysis import (
                cluster_graph,
                cohesion_scores,
                hub_concepts,
            )

            communities = cluster_graph(graph)
            scores = cohesion_scores(graph, communities)
            hubs = hub_concepts(graph, top_n=10)
            nodes = [
                {
                    "id": node_id,
                    "label": graph.nodes[node_id].get("label", node_id),
                    "community": next(
                        (
                            community_id
                            for community_id, members in communities.items()
                            if node_id in members
                        ),
                        -1,
                    ),
                    "workspace_id": graph.nodes[node_id].get("workspace_id"),
                    "source_ref": graph.nodes[node_id].get("source_ref"),
                    "importance_score": graph.nodes[node_id].get("importance_score", 0.0),
                    "never_decay": bool(graph.nodes[node_id].get("never_decay")),
                }
                for node_id in graph.nodes()
            ]
            edges = [
                {
                    "id": data.get("id"),
                    "source": source,
                    "target": target,
                    "relation": data.get("relation", ""),
                    "confidence": data.get("confidence", "PROVISIONAL"),
                    "weight": data.get("weight", 0.5),
                    "workspace_id": data.get("workspace_id"),
                    "target_workspace_id": data.get("target_workspace_id"),
                    "source_ref": data.get("source_ref"),
                    "cross_ref": bool(
                        data.get("target_workspace_id")
                        and data.get("target_workspace_id") != data.get("workspace_id")
                    ),
                    "importance_score": data.get("importance_score", 0.0),
                    "never_decay": bool(data.get("never_decay")),
                }
                for source, target, data in graph.edges(data=True)
            ]
            return {
                "nodes": nodes,
                "edges": edges,
                "communities": {
                    str(community_id): {
                        "members": members,
                        "cohesion": scores.get(community_id, 0.0),
                    }
                    for community_id, members in communities.items()
                },
                "hubs": hubs,
            }
        except Exception as exc:
            raise HTTPException(500, f"knowledge graph export failed: {exc}") from exc

    return router
