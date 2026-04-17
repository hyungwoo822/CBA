from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

import aiosqlite
import chromadb
import networkx as nx


class SemanticStore:
    """ChromaDB-backed semantic memory with a SQLite knowledge graph."""

    def __init__(
        self,
        chroma_path: str,
        graph_db_path: str,
        embed_fn: Callable[[str], list[float]],
    ):
        self._chroma_path = chroma_path
        self._graph_db_path = graph_db_path
        self._embed_fn = embed_fn
        self._collection = None
        self._graph_db: aiosqlite.Connection | None = None
        self._workspace_store = None

    async def initialize(self):
        client = chromadb.PersistentClient(path=self._chroma_path)
        self._collection = client.get_or_create_collection(
            name="semantic_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self._graph_db = await aiosqlite.connect(self._graph_db_path)
        await self._graph_db.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_graph (
                id TEXT PRIMARY KEY,
                source_node TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_node TEXT NOT NULL,
                category TEXT DEFAULT 'GENERAL',
                confidence TEXT DEFAULT 'INFERRED',
                weight REAL DEFAULT 0.5,
                occurrence_count INTEGER DEFAULT 1,
                first_seen TEXT NOT NULL DEFAULT '',
                last_seen TEXT NOT NULL DEFAULT '',
                origin TEXT DEFAULT 'unknown',
                workspace_id TEXT NOT NULL DEFAULT 'personal',
                target_workspace_id TEXT,
                source_ref TEXT,
                valid_from TEXT,
                valid_to TEXT,
                superseded_by TEXT,
                type_id TEXT,
                epistemic_source TEXT DEFAULT 'asserted',
                importance_score REAL DEFAULT 0.5,
                never_decay INTEGER DEFAULT 0,
                UNIQUE(workspace_id, source_node, relation, target_node, origin)
            )"""
        )
        await self._graph_db.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON knowledge_graph(source_node)"
        )
        await self._graph_db.execute(
            "CREATE INDEX IF NOT EXISTS idx_target ON knowledge_graph(target_node)"
        )
        # Migration for existing DBs: add new columns silently if missing
        for col, defn in [
            ("category", "TEXT DEFAULT 'GENERAL'"),
            ("confidence", "TEXT DEFAULT 'INFERRED'"),
            ("occurrence_count", "INTEGER DEFAULT 1"),
            ("first_seen", "TEXT NOT NULL DEFAULT ''"),
            ("last_seen", "TEXT NOT NULL DEFAULT ''"),
            ("origin", "TEXT DEFAULT 'unknown'"),
            ("workspace_id", "TEXT DEFAULT 'personal'"),
            ("target_workspace_id", "TEXT"),
            ("source_ref", "TEXT"),
            ("valid_from", "TEXT"),
            ("valid_to", "TEXT"),
            ("superseded_by", "TEXT"),
            ("type_id", "TEXT"),
            ("epistemic_source", "TEXT DEFAULT 'asserted'"),
            ("importance_score", "REAL DEFAULT 0.5"),
            ("never_decay", "INTEGER DEFAULT 0"),
        ]:
            try:
                await self._graph_db.execute(
                    f"ALTER TABLE knowledge_graph ADD COLUMN {col} {defn}"
                )
            except Exception:
                pass  # column already exists
        await self._graph_db.execute(
            "UPDATE knowledge_graph SET workspace_id = 'personal' "
            "WHERE workspace_id IS NULL"
        )
        await self._graph_db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_workspace "
            "ON knowledge_graph(workspace_id)"
        )
        await self._graph_db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_workspace_source "
            "ON knowledge_graph(workspace_id, source_node)"
        )
        await self._graph_db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_workspace_target "
            "ON knowledge_graph(workspace_id, target_node)"
        )
        await self._graph_db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_target_workspace "
            "ON knowledge_graph(target_workspace_id)"
        )
        await self._graph_db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_never_decay "
            "ON knowledge_graph(workspace_id, never_decay)"
        )
        # Migration: add UNIQUE index if missing (for DBs created before UNIQUE constraint)
        try:
            await self._graph_db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_kg_unique_workspace_triple_origin "
                "ON knowledge_graph(workspace_id, source_node, relation, "
                "target_node, origin)"
            )
        except Exception:
            pass
        # Drop old triple-only unique index if it exists.
        try:
            await self._graph_db.execute(
                "DROP INDEX IF EXISTS idx_kg_unique_triple_origin"
            )
        except Exception:
            pass
        try:
            await self._graph_db.execute("DROP INDEX IF EXISTS idx_kg_unique_triple")
        except Exception:
            pass
        # Migration: remove pre-normalization non-English entries
        # (KG overhaul enforces English-only; old Korean data is noise)
        try:
            await self._graph_db.execute(
                "DELETE FROM knowledge_graph WHERE "
                "source_node != LOWER(source_node) OR target_node != LOWER(target_node) OR "
                "UNICODE(source_node) > 127 OR UNICODE(target_node) > 127 OR "
                "UNICODE(relation) > 127"
            )
            await self._graph_db.commit()
        except Exception:
            pass

        await self._graph_db.execute(
            """CREATE TABLE IF NOT EXISTS identity_facts (
                id TEXT PRIMARY KEY,
                fact_type TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT DEFAULT 'unknown',
                confidence REAL DEFAULT 1.0,
                updated_at TEXT DEFAULT '',
                UNIQUE(fact_type, key)
            )"""
        )
        await self._graph_db.execute(
            """CREATE TABLE IF NOT EXISTS hyperedges(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL UNIQUE,
                members TEXT NOT NULL,
                category TEXT DEFAULT 'GENERAL',
                strength REAL DEFAULT 1.0,
                created_at TEXT
            )"""
        )
        await self._graph_db.commit()

    async def close(self):
        if self._graph_db:
            await self._graph_db.close()

    # ------------------------------------------------------------------
    # Identity facts (mPFC self-model / TPJ user-model)
    # Ref: Ghosh & Gilboa (2014) schema updating in mPFC
    # ------------------------------------------------------------------

    async def add_identity_fact(
        self,
        fact_type: str,
        key: str,
        value: str,
        source: str = "unknown",
        confidence: float = 1.0,
    ) -> None:
        """Store or update an identity fact (UPSERT on fact_type+key)."""
        now = datetime.now(timezone.utc).isoformat()
        await self._graph_db.execute(
            """INSERT INTO identity_facts (id, fact_type, key, value, source, confidence, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(fact_type, key) DO UPDATE SET
                   value = excluded.value,
                   source = excluded.source,
                   confidence = excluded.confidence,
                   updated_at = excluded.updated_at""",
            (str(uuid.uuid4()), fact_type, key, value, source, confidence, now),
        )
        await self._graph_db.commit()

    async def get_identity_facts(self, fact_type: str) -> list[dict]:
        """Return all identity facts of the given type, ordered by key."""
        async with self._graph_db.execute(
            "SELECT key, value, source, confidence, updated_at "
            "FROM identity_facts WHERE fact_type = ? ORDER BY key",
            (fact_type,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "key": r[0],
                    "value": r[1],
                    "source": r[2],
                    "confidence": r[3],
                    "updated_at": r[4],
                }
                for r in rows
            ]

    async def add(
        self,
        content: str,
        category: str = "general",
        strength: float = 1.0,
        workspace_id: str = "personal",
    ) -> str:
        doc_id = str(uuid.uuid4())
        embedding = self._embed_fn(content)
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[
                {
                    "category": category,
                    "strength": strength,
                    "access_count": 0,
                    "workspace_id": workspace_id,
                }
            ],
        )
        return doc_id

    async def search(
        self,
        query: str,
        top_k: int = 5,
        workspace_id: str | None = None,
    ) -> list[dict]:
        embedding = self._embed_fn(query)
        kwargs: dict = {"query_embeddings": [embedding], "n_results": top_k}
        if workspace_id is not None and workspace_id != "personal":
            kwargs["where"] = {"workspace_id": workspace_id}
        elif workspace_id == "personal":
            # Legacy Chroma docs lack workspace_id. Query broadly, then treat
            # missing metadata as personal in Python.
            count = max(top_k, self._collection.count())
            kwargs["n_results"] = count or top_k
        results = self._collection.query(**kwargs)
        out = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = (
                    results["metadatas"][0][i]
                    if results.get("metadatas")
                    else {}
                )
                if meta is not None and "workspace_id" not in meta:
                    meta = {**meta, "workspace_id": "personal"}
                meta = meta or {"workspace_id": "personal"}
                if workspace_id == "personal" and meta.get("workspace_id") != "personal":
                    continue
                out.append(
                    {
                        "id": results["ids"][0][i],
                        "content": doc,
                        "distance": (
                            results["distances"][0][i]
                            if results.get("distances")
                            else None
                        ),
                        "metadata": meta,
                    }
                )
                if len(out) >= top_k:
                    break
        return out

    async def count(self) -> int:
        return self._collection.count()

    async def add_relationship(
        self,
        source: str,
        relation: str,
        target: str,
        weight: float = 0.5,
        category: str = "GENERAL",
        confidence: str = "INFERRED",
        origin: str = "unknown",
        workspace_id: str = "personal",
        target_workspace_id: str | None = None,
        source_ref: str | None = None,
        valid_from: str | None = None,
        epistemic_source: str = "asserted",
        importance_score: float = 0.5,
        never_decay: bool = False,
    ):
        now = datetime.now(timezone.utc).isoformat()
        await self._graph_db.execute(
            """INSERT INTO knowledge_graph
               (id, source_node, relation, target_node, category, confidence, weight,
                occurrence_count, first_seen, last_seen, origin,
                workspace_id, target_workspace_id, source_ref, valid_from,
                epistemic_source, importance_score, never_decay)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(workspace_id, source_node, relation, target_node, origin)
               DO UPDATE SET
                   weight = MIN(1.0, MAX(excluded.weight, knowledge_graph.weight)
                            + 0.1 / (1.0 + knowledge_graph.occurrence_count * 0.5)),
                   occurrence_count = knowledge_graph.occurrence_count + 1,
                   last_seen = excluded.last_seen,
                   category = CASE
                       WHEN excluded.category != 'GENERAL' THEN excluded.category
                       ELSE knowledge_graph.category
                   END,
                   confidence = excluded.confidence,
                   target_workspace_id = excluded.target_workspace_id,
                   source_ref = COALESCE(excluded.source_ref, knowledge_graph.source_ref),
                   valid_from = COALESCE(excluded.valid_from, knowledge_graph.valid_from),
                   epistemic_source = excluded.epistemic_source,
                   importance_score = MAX(
                       COALESCE(knowledge_graph.importance_score, 0.5),
                       COALESCE(excluded.importance_score, 0.5)
                   ),
                   never_decay = MAX(
                       COALESCE(knowledge_graph.never_decay, 0),
                       COALESCE(excluded.never_decay, 0)
                   )""",
            (
                str(uuid.uuid4()),
                source,
                relation,
                target,
                category,
                confidence,
                weight,
                now,
                now,
                origin,
                workspace_id,
                target_workspace_id,
                source_ref,
                valid_from,
                epistemic_source,
                importance_score,
                1 if never_decay else 0,
            ),
        )
        await self._graph_db.commit()

    async def get_relationships(
        self,
        node: str,
        workspace_id: str | None = "personal",
    ) -> list[dict]:
        select = (
            "SELECT id, source_node, relation, target_node, weight, category, "
            "occurrence_count, origin, confidence, workspace_id, "
            "target_workspace_id, source_ref, valid_from, valid_to, "
            "superseded_by, epistemic_source, importance_score, never_decay "
            "FROM knowledge_graph "
        )
        if workspace_id is None:
            sql = select + "WHERE source_node = ? OR target_node = ?"
            args: tuple = (node, node)
        else:
            sql = (
                select
                + "WHERE (source_node = ? OR target_node = ?) AND workspace_id = ?"
            )
            args = (node, node, workspace_id)
        async with self._graph_db.execute(sql, args) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "source": r[1],
                "relation": r[2],
                "target": r[3],
                "weight": r[4],
                "category": r[5],
                "occurrence_count": r[6],
                "origin": r[7],
                "confidence": r[8],
                "workspace_id": r[9],
                "target_workspace_id": r[10],
                "source_ref": r[11],
                "valid_from": r[12],
                "valid_to": r[13],
                "superseded_by": r[14],
                "epistemic_source": r[15],
                "importance_score": r[16],
                "never_decay": r[17],
            }
            for r in rows
        ]

    async def get_edges(
        self,
        workspace_id: str | None = None,
    ) -> list[dict]:
        """Return knowledge graph edges with raw source/target column names."""
        select = (
            "SELECT id, source_node, relation, target_node, weight, category, "
            "occurrence_count, origin, confidence, workspace_id, "
            "target_workspace_id, source_ref, valid_from, valid_to, "
            "superseded_by, epistemic_source, importance_score, never_decay "
            "FROM knowledge_graph "
        )
        if workspace_id is None:
            sql = select + "ORDER BY source_node, relation, target_node"
            args: tuple = ()
        else:
            sql = (
                select
                + "WHERE workspace_id = ? ORDER BY source_node, relation, target_node"
            )
            args = (workspace_id,)
        async with self._graph_db.execute(sql, args) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "source_node": r[1],
                "relation": r[2],
                "target_node": r[3],
                "weight": r[4],
                "category": r[5],
                "occurrence_count": r[6],
                "origin": r[7],
                "confidence": r[8],
                "workspace_id": r[9],
                "target_workspace_id": r[10],
                "source_ref": r[11],
                "valid_from": r[12],
                "valid_to": r[13],
                "superseded_by": r[14],
                "epistemic_source": r[15],
                "importance_score": r[16],
                "never_decay": r[17],
            }
            for r in rows
        ]

    @staticmethod
    def _fuzzy_node_match(user_node: str, agent_nodes: set[str]) -> bool:
        """Check if a user node has a fuzzy match in agent nodes.

        Matches on: exact, substring containment, or high token overlap.
        e.g., "extroverted" matches "extroverted personality"
              "coffee" matches "coffee"
        """
        if user_node in agent_nodes:
            return True
        for an in agent_nodes:
            # Substring: one contains the other
            if user_node in an or an in user_node:
                return True
            # Token overlap: >50% of words shared
            u_tokens = set(user_node.split())
            a_tokens = set(an.split())
            if u_tokens and a_tokens:
                overlap = len(u_tokens & a_tokens)
                if overlap / min(len(u_tokens), len(a_tokens)) > 0.5:
                    return True
        return False

    @staticmethod
    def _fuzzy_edge_match(
        user_edge: tuple[str, str, str],
        agent_edges: set[tuple[str, str, str]],
        agent_nodes: set[str],
    ) -> float:
        """Score how well a user edge matches agent edges (0.0-1.0).

        Checks:
          1.0 — exact triple match
          0.8 — reversed direction (A,rel,B) matches (B,rel',A) with same nodes
          0.6 — same nodes involved in any agent edge (structural match)
          0.0 — no match
        """
        s, r, t = user_edge
        # Exact match
        if user_edge in agent_edges:
            return 1.0
        # Reversed direction: (coffee,spill,user) ↔ (user,spill,coffee)
        for as_, ar, at in agent_edges:
            # Same nodes, any relation
            if (s == at and t == as_) or (s == as_ and t == at):
                return 0.8
            # Fuzzy node match + reversed
            s_in = (s in as_ or as_ in s or s in at or at in s)
            t_in = (t in as_ or as_ in t or t in at or at in t)
            if s_in and t_in:
                return 0.6
        return 0.0

    async def compute_cloning_score(self) -> dict:
        """Compute how well the agent understands the user's mental model.

        Compares user_input triples against the agent's full understanding:
          - agent_about_user: direct user facts (full weight)
          - agent_knowledge: concept relations, included when at least one
            node overlaps with the user graph (contextual understanding)

        This prevents a structural ceiling where concept-concept edges in
        the user graph (e.g., 참깨라면→도파민) can never be matched because
        the agent stores them as agent_knowledge.

        References:
          - Park et al. (2023): Generative agents — belief alignment
        """
        if not self._graph_db:
            return {"cloning_score": 0.0, "user_graph_size": 0, "agent_graph_size": 0}

        try:
            # User facts
            async with self._graph_db.execute(
                "SELECT source_node, relation, target_node FROM knowledge_graph WHERE origin = 'user_input'"
            ) as cur:
                user_rows = await cur.fetchall()

            # Agent's direct user understanding
            async with self._graph_db.execute(
                "SELECT source_node, relation, target_node FROM knowledge_graph "
                "WHERE origin IN ('agent_about_user', 'agent_response')"
            ) as cur:
                agent_direct_rows = await cur.fetchall()

            # Agent's concept knowledge (will be filtered to user-relevant)
            async with self._graph_db.execute(
                "SELECT source_node, relation, target_node FROM knowledge_graph "
                "WHERE origin = 'agent_knowledge'"
            ) as cur:
                agent_knowledge_rows = await cur.fetchall()

            if not user_rows:
                return {"cloning_score": 0.0, "user_graph_size": 0,
                        "agent_graph_size": len(agent_direct_rows),
                        "node_recall": 0.0, "edge_recall": 0.0, "relation_recall": 0.0}

            # Extract user node set for context filtering
            user_nodes: set[str] = set()
            for s, _r, t in user_rows:
                user_nodes.update([s, t])

            # Combine agent_about_user (multi-entity) + agent_knowledge
            # where at least one node overlaps with the user graph.
            # agent_about_user now contains multi-entity triples (e.g.,
            # grandmother→visit→user), not just user→X→Y flat links.
            agent_rows = list(agent_direct_rows)
            for s, r, t in agent_knowledge_rows:
                if (self._fuzzy_node_match(s, user_nodes)
                        or self._fuzzy_node_match(t, user_nodes)):
                    agent_rows.append((s, r, t))

            agent_nodes: set[str] = set()
            for s, _r, t in agent_rows:
                agent_nodes.update([s, t])

            user_edges = {(s, r, t) for s, r, t in user_rows}
            agent_edges = {(s, r, t) for s, r, t in agent_rows}

            user_rels = {r for _, r, _ in user_rows}
            agent_rels = {r for _, r, _ in agent_rows}

            # Fuzzy node recall
            matched_nodes = sum(
                1 for un in user_nodes
                if self._fuzzy_node_match(un, agent_nodes)
            )
            node_recall = matched_nodes / len(user_nodes) if user_nodes else 0.0

            # Fuzzy edge recall (weighted: exact=1.0, reversed=0.8, structural=0.6)
            edge_scores = [
                self._fuzzy_edge_match(ue, agent_edges, agent_nodes)
                for ue in user_edges
            ]
            edge_recall = sum(edge_scores) / len(edge_scores) if edge_scores else 0.0

            # Relation recall (exact — relation types are small set)
            relation_recall = len(user_rels & agent_rels) / len(user_rels) if user_rels else 0.0

            # Weighted recall composite (existing logic)
            raw_recall = (
                node_recall * 0.45 +       # Same concepts?
                relation_recall * 0.25 +   # Same relationship types?
                edge_recall * 0.30         # Same triples (fuzzy)?
            )

            # Learning curve maturity: cloning requires sufficient data.
            # Even perfect recall on 2 edges doesn't mean you've cloned someone.
            # maturity = 1 - e^(-total_edges / growth_constant)
            import math
            GROWTH_CONSTANT = 15
            total_edges = len(user_rows) + len(agent_rows)
            maturity = 1.0 - math.exp(-total_edges / GROWTH_CONSTANT)

            cloning_score = raw_recall * maturity

            return {
                "cloning_score": round(cloning_score, 4),
                "raw_recall": round(raw_recall, 4),
                "maturity": round(maturity, 4),
                "node_recall": round(node_recall, 4),
                "edge_recall": round(edge_recall, 4),
                "relation_recall": round(relation_recall, 4),
                "user_graph_size": len(user_rows),
                "agent_graph_size": len(agent_rows),
            }
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Cloning score computation failed: %s", e)
            return {"cloning_score": 0.0, "user_graph_size": 0, "agent_graph_size": 0}

    async def spread_activation(
        self,
        start_nodes: list[str],
        max_hops: int = 3,
        decay: float = 0.85,
        community_bonus: float = 0.15,
    ) -> dict[str, float]:
        """Spread activation through knowledge graph with community awareness.

        Neuroscience: spreading activation (Collins & Loftus 1975) enhanced
        with cortical column affinity — nodes in the same community as start
        nodes receive a small activation bonus, reflecting intra-region
        facilitation in the brain.

        Args:
            start_nodes: Seed concept nodes.
            max_hops: Maximum traversal depth.
            decay: Per-hop decay factor.
            community_bonus: Extra activation for same-community neighbors.
        """
        # Pre-compute community membership for bonus
        community_map: dict[str, int] = {}
        start_communities: set[int] = set()
        try:
            comms = await self.cluster_knowledge()
            for cid, members in comms.items():
                for m in members:
                    community_map[m] = cid
            start_communities = {community_map[n] for n in start_nodes if n in community_map}
        except Exception:
            pass  # graceful degradation: no community bonus if clustering fails

        # Assembly co-activation: boost from hyperedges
        assembly_boost: dict[str, float] = {}
        try:
            from brain_agent.memory.graph_analysis import assembly_coactivation
            assemblies = await self.get_hyperedges()
            if assemblies:
                assembly_boost = assembly_coactivation(start_nodes, assemblies)
        except Exception:
            pass

        activation: dict[str, float] = {}
        # Seed assembly co-activated nodes into frontier
        frontier = [(n, 1.0) for n in start_nodes]
        for node, boost in assembly_boost.items():
            if boost > 0.01:
                frontier.append((node, boost))

        for _ in range(max_hops):
            next_frontier = []
            for node, act in frontier:
                if node in activation and activation[node] >= act:
                    continue
                activation[node] = max(activation.get(node, 0), act)
                rels = await self.get_relationships(node)
                if not rels:
                    continue
                fan = 1.0 / len(rels)
                for r in rels:
                    neighbor = r["target"] if r["source"] == node else r["source"]
                    spread = act * decay * r["weight"] * fan
                    # Community bonus: same community as start nodes
                    if community_bonus > 0 and community_map.get(neighbor) in start_communities:
                        spread *= (1.0 + community_bonus)
                    if spread > 0.01:
                        next_frontier.append((neighbor, spread))
            frontier = next_frontier
        return activation

    # ------------------------------------------------------------------
    # NetworkX bridge methods (graph_analysis integration)
    # ------------------------------------------------------------------

    async def export_as_networkx(
        self,
        workspace_id: str | None = None,
        include_cross_refs: bool = True,
    ) -> nx.Graph:
        """Export the knowledge_graph table as a NetworkX Graph.

        Each source and target becomes a node with a ``label`` attribute.
        Each row becomes an undirected edge carrying relation, weight,
        category, and confidence attributes.
        """
        G = nx.Graph()
        if workspace_id is None:
            sql = (
                "SELECT source_node, relation, target_node, weight, category, "
                "confidence FROM knowledge_graph"
            )
            args: tuple = ()
        elif include_cross_refs:
            sql = (
                "SELECT source_node, relation, target_node, weight, category, "
                "confidence FROM knowledge_graph "
                "WHERE workspace_id = ? OR target_workspace_id = ?"
            )
            args = (workspace_id, workspace_id)
        else:
            sql = (
                "SELECT source_node, relation, target_node, weight, category, "
                "confidence FROM knowledge_graph WHERE workspace_id = ?"
            )
            args = (workspace_id,)
        async with self._graph_db.execute(sql, args) as cursor:
            rows = await cursor.fetchall()
        for source, relation, target, weight, category, confidence in rows:
            if source not in G:
                G.add_node(source, label=source)
            if target not in G:
                G.add_node(target, label=target)
            G.add_edge(
                source,
                target,
                relation=relation,
                weight=weight,
                category=category,
                confidence=confidence,
            )
        return G

    async def cluster_knowledge(self) -> dict[int, list[str]]:
        """Cluster the knowledge graph into communities.

        Returns a mapping of community id to list of node ids.
        """
        from brain_agent.memory.graph_analysis import cluster_graph
        G = await self.export_as_networkx()
        return cluster_graph(G)

    async def find_hub_concepts(self, top_n: int = 10) -> list[dict]:
        """Return the top-N most-connected concepts in the knowledge graph.

        Each entry: {"id": node_id, "label": label, "edges": degree}.
        """
        from brain_agent.memory.graph_analysis import hub_concepts
        G = await self.export_as_networkx()
        return hub_concepts(G, top_n)

    async def find_bridges(self, top_n: int = 5) -> list[dict]:
        """Return the most surprising cross-community edges.

        Clusters the graph first, then scores edges by structural
        surprise (cross-community, confidence, peripheral-to-hub).
        """
        from brain_agent.memory.graph_analysis import cluster_graph, surprising_connections
        G = await self.export_as_networkx()
        communities = cluster_graph(G)
        return surprising_connections(G, communities, top_n)

    # ------------------------------------------------------------------
    # Hyperedge / Cell Assembly methods (Hebb, 1949)
    # ------------------------------------------------------------------

    async def add_hyperedge(
        self, members: list[str], label: str,
        category: str = "GENERAL", strength: float = 1.0,
    ) -> None:
        """Add a cell assembly (hyperedge) connecting 3+ concepts.

        Neuroscience: Hebb's Cell Assembly (1949) — neurons that fire
        together wire together. Hyperedges represent group-level
        activation patterns beyond pairwise connections.
        """
        import json
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        members_json = json.dumps(sorted(members))
        async with aiosqlite.connect(self._graph_db_path) as db:
            await db.execute(
                """INSERT INTO hyperedges (label, members, category, strength, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(label) DO UPDATE SET
                     members = excluded.members,
                     category = excluded.category,
                     strength = excluded.strength
                """,
                (label, members_json, category, strength, now),
            )
            await db.commit()

    async def get_hyperedges(self) -> list[dict]:
        """Return all cell assemblies."""
        import json
        async with aiosqlite.connect(self._graph_db_path) as db:
            cursor = await db.execute(
                "SELECT label, members, category, strength, created_at FROM hyperedges"
            )
            rows = await cursor.fetchall()
        return [
            {
                "label": r[0],
                "members": json.loads(r[1]),
                "category": r[2],
                "strength": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    async def get_assemblies_for_node(self, node: str) -> list[dict]:
        """Return all cell assemblies containing a given node."""
        all_edges = await self.get_hyperedges()
        return [e for e in all_edges if node in e["members"]]

    # ------------------------------------------------------------------
    # Synaptic pruning & homeostatic scaling (Huttenlocher 1979;
    # Tononi & Cirelli 2003)
    # ------------------------------------------------------------------

    async def prune_weak_edges(
        self,
        min_weight: float = 0.1,
        workspace_id: str | None = None,
    ) -> int:
        """Remove knowledge graph edges below the weight threshold.

        Neuroscience: synaptic pruning (Huttenlocher 1979) — weak
        connections are eliminated to maintain efficient network topology.
        Runs during homeostatic scaling in consolidation.

        Returns the number of pruned edges.
        """
        if workspace_id is not None:
            async with self._graph_db.execute(
                "SELECT COUNT(*) FROM knowledge_graph "
                "WHERE workspace_id = ? AND weight < ? "
                "AND COALESCE(never_decay, 0) = 0",
                (workspace_id, min_weight),
            ) as cursor:
                count = (await cursor.fetchone())[0]
            if count > 0:
                await self._graph_db.execute(
                    "DELETE FROM knowledge_graph "
                    "WHERE workspace_id = ? AND weight < ? "
                    "AND COALESCE(never_decay, 0) = 0",
                    (workspace_id, min_weight),
                )
                await self._graph_db.commit()
            return count

        ws_store = getattr(self, "_workspace_store", None)
        if ws_store is not None:
            total = 0
            for ws in await ws_store.list_workspaces():
                if ws.get("decay_policy") == "none":
                    continue
                total += await self.prune_weak_edges(
                    min_weight=min_weight,
                    workspace_id=ws["id"],
                )
            return total

        async with self._graph_db.execute(
            "SELECT COUNT(*) FROM knowledge_graph "
            "WHERE weight < ? AND COALESCE(never_decay, 0) = 0",
            (min_weight,),
        ) as cursor:
            count = (await cursor.fetchone())[0]
        if count > 0:
            await self._graph_db.execute(
                "DELETE FROM knowledge_graph "
                "WHERE weight < ? AND COALESCE(never_decay, 0) = 0",
                (min_weight,),
            )
            await self._graph_db.commit()
        return count

    async def decay_edge_weights(
        self,
        factor: float = 0.95,
        workspace_id: str | None = None,
    ) -> int:
        """Apply homeostatic scaling to all knowledge graph edges.

        Neuroscience: synaptic homeostasis (Tononi & Cirelli 2003).
        Scales all weights down by factor, simulating global synaptic
        downscaling during consolidation ("sleep").

        Returns the number of affected edges.
        """
        update_sql = (
            "UPDATE knowledge_graph "
            "SET weight = weight * "
            "(1.0 - (1.0 - ?) * "
            "(1.0 - COALESCE(importance_score, 0.5) * 0.5)) "
        )
        if workspace_id is not None:
            cursor = await self._graph_db.execute(
                update_sql
                + "WHERE workspace_id = ? AND COALESCE(never_decay, 0) = 0",
                (factor, workspace_id),
            )
            await self._graph_db.commit()
            return cursor.rowcount or 0

        ws_store = getattr(self, "_workspace_store", None)
        if ws_store is not None:
            total = 0
            for ws in await ws_store.list_workspaces():
                policy = ws.get("decay_policy", "normal")
                if policy == "none":
                    continue
                ws_factor = 0.99 if policy == "slow" else factor
                cursor = await self._graph_db.execute(
                    update_sql
                    + "WHERE workspace_id = ? AND COALESCE(never_decay, 0) = 0",
                    (ws_factor, ws["id"]),
                )
                total += cursor.rowcount or 0
            await self._graph_db.commit()
            return total

        cursor = await self._graph_db.execute(
            update_sql + "WHERE COALESCE(never_decay, 0) = 0",
            (factor,),
        )
        await self._graph_db.commit()
        return cursor.rowcount or 0

    async def mark_superseded(self, edge_id: str, valid_to: str) -> None:
        """Mark a knowledge graph edge as no longer current."""
        await self._graph_db.execute(
            "UPDATE knowledge_graph SET valid_to = ? WHERE id = ?",
            (valid_to, edge_id),
        )
        await self._graph_db.commit()
