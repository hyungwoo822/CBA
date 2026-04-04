from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

import aiosqlite
import chromadb


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
                weight REAL DEFAULT 0.5,
                occurrence_count INTEGER DEFAULT 1,
                first_seen TEXT NOT NULL DEFAULT '',
                last_seen TEXT NOT NULL DEFAULT '',
                origin TEXT DEFAULT 'unknown',
                UNIQUE(source_node, relation, target_node, origin)
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
            ("occurrence_count", "INTEGER DEFAULT 1"),
            ("first_seen", "TEXT NOT NULL DEFAULT ''"),
            ("last_seen", "TEXT NOT NULL DEFAULT ''"),
            ("origin", "TEXT DEFAULT 'unknown'"),
        ]:
            try:
                await self._graph_db.execute(
                    f"ALTER TABLE knowledge_graph ADD COLUMN {col} {defn}"
                )
            except Exception:
                pass  # column already exists
        # Migration: add UNIQUE index if missing (for DBs created before UNIQUE constraint)
        try:
            await self._graph_db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_unique_triple_origin "
                "ON knowledge_graph(source_node, relation, target_node, origin)"
            )
        except Exception:
            pass
        # Drop old triple-only unique index if it exists (replaced by triple+origin)
        try:
            await self._graph_db.execute(
                "DROP INDEX IF EXISTS idx_kg_unique_triple"
            )
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
                }
            ],
        )
        return doc_id

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = self._embed_fn(query)
        results = self._collection.query(
            query_embeddings=[embedding], n_results=top_k
        )
        out = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                out.append(
                    {
                        "id": results["ids"][0][i],
                        "content": doc,
                        "distance": (
                            results["distances"][0][i]
                            if results.get("distances")
                            else None
                        ),
                        "metadata": (
                            results["metadatas"][0][i]
                            if results.get("metadatas")
                            else {}
                        ),
                    }
                )
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
        origin: str = "unknown",
    ):
        now = datetime.now(timezone.utc).isoformat()
        await self._graph_db.execute(
            """INSERT INTO knowledge_graph
               (id, source_node, relation, target_node, category, weight,
                occurrence_count, first_seen, last_seen, origin)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
               ON CONFLICT(source_node, relation, target_node, origin) DO UPDATE SET
                   weight = MIN(1.0, MAX(excluded.weight, knowledge_graph.weight)
                            + 0.1 / (1.0 + knowledge_graph.occurrence_count * 0.5)),
                   occurrence_count = knowledge_graph.occurrence_count + 1,
                   last_seen = excluded.last_seen,
                   category = CASE
                       WHEN excluded.category != 'GENERAL' THEN excluded.category
                       ELSE knowledge_graph.category
                   END""",
            (str(uuid.uuid4()), source, relation, target, category, weight, now, now, origin),
        )
        await self._graph_db.commit()

    async def get_relationships(self, node: str) -> list[dict]:
        async with self._graph_db.execute(
            "SELECT source_node, relation, target_node, weight, category, occurrence_count, origin "
            "FROM knowledge_graph WHERE source_node = ? OR target_node = ?",
            (node, node),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "source": r[0],
                    "relation": r[1],
                    "target": r[2],
                    "weight": r[3],
                    "category": r[4],
                    "occurrence_count": r[5],
                    "origin": r[6] if len(r) > 6 else "unknown",
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

            # Also include identity_facts as virtual edges (they store
            # multi-entity facts like "grandmother visit user" as values)
            try:
                id_facts = await self.get_identity_facts("user_model")
                for fact in id_facts:
                    val = fact.get("value", "")
                    parts = val.split(" ", 2)
                    if len(parts) >= 3:
                        agent_rows.append((parts[0], parts[1], parts[2]))
                    elif len(parts) == 2:
                        agent_rows.append(("user", parts[0], parts[1]))
            except Exception:
                pass

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
    ) -> dict[str, float]:
        activation: dict[str, float] = {}
        frontier = [(n, 1.0) for n in start_nodes]
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
                    if spread > 0.01:
                        next_frontier.append((neighbor, spread))
            frontier = next_frontier
        return activation
