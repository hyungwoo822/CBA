from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Callable

import aiosqlite

RETRIEVAL_BOOST_DEFAULT = 2.0


class HippocampalStaging:
    """Short-term memory staging area modeled after the hippocampus.

    Stores newly encoded memories in SQLite before consolidation into
    long-term episodic / semantic stores.
    """

    def __init__(self, db_path: str, embed_fn: Callable[[str], list[float]]):
        self._db_path = db_path
        self._embed_fn = embed_fn
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS staging_memories (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                context_embedding BLOB,
                entities TEXT DEFAULT '{}',
                emotional_tag TEXT DEFAULT '{"valence":0,"arousal":0}',
                source_modality TEXT DEFAULT 'text',
                access_count INTEGER DEFAULT 0,
                strength REAL DEFAULT 1.0,
                consolidated INTEGER DEFAULT 0,
                last_interaction INTEGER DEFAULT 0,
                last_session TEXT DEFAULT '',
                workspace_id TEXT DEFAULT 'personal'
            )"""
        )
        try:
            await self._db.execute(
                "ALTER TABLE staging_memories "
                "ADD COLUMN workspace_id TEXT DEFAULT 'personal'"
            )
        except Exception:
            pass
        await self._db.execute(
            "UPDATE staging_memories SET workspace_id = 'personal' "
            "WHERE workspace_id IS NULL"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_staging_workspace "
            "ON staging_memories(workspace_id, consolidated)"
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS staging_edges (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL DEFAULT 'personal',
                source_node TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_node TEXT NOT NULL,
                interaction_id INTEGER NOT NULL DEFAULT 0,
                session_id TEXT NOT NULL DEFAULT '',
                importance_score REAL DEFAULT 0.5,
                never_decay INTEGER DEFAULT 0,
                consolidated INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_staging_edges_workspace "
            "ON staging_edges(workspace_id, consolidated)"
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def encode(
        self,
        content: str,
        entities: dict,
        interaction_id: int,
        session_id: str,
        emotional_tag: dict | None = None,
        source_modality: str = "text",
        workspace_id: str = "personal",
    ) -> str:
        mem_id = str(uuid.uuid4())
        embedding = self._embed_fn(content)
        tag = emotional_tag or {"valence": 0.0, "arousal": 0.0}

        # Pattern separation (DG, Yassa & Stark 2011): detect near-duplicates
        separated_from = None
        if self._db:
            try:
                async with self._db.execute(
                    "SELECT id, context_embedding FROM staging_memories "
                    "WHERE consolidated = 0 AND workspace_id = ? "
                    "ORDER BY last_interaction DESC LIMIT 10",
                    (workspace_id,),
                ) as cur:
                    rows = await cur.fetchall()
                for row in rows:
                    if row[1]:
                        import json as _json
                        existing_emb = _json.loads(row[1]) if isinstance(row[1], str) else row[1]
                        if existing_emb and embedding:
                            sim = self._cosine_sim(embedding, existing_emb)
                            if sim > 0.85:
                                separated_from = row[0]
                                break
            except Exception:
                pass  # Separation check is best-effort

        if separated_from:
            if isinstance(entities, dict):
                entities = dict(entities)
                entities["separated_from"] = separated_from

        await self._db.execute(
            "INSERT INTO staging_memories "
            "(id, timestamp, content, context_embedding, entities, emotional_tag, "
            "source_modality, access_count, strength, consolidated, "
            "last_interaction, last_session, workspace_id) "
            "VALUES (?,?,?,?,?,?,?,0,1.0,0,?,?,?)",
            (
                mem_id,
                datetime.now(timezone.utc).isoformat(),
                content,
                json.dumps(embedding),
                json.dumps(entities),
                json.dumps(tag),
                source_modality,
                interaction_id,
                session_id,
                workspace_id,
            ),
        )
        await self._db.commit()
        return mem_id

    async def on_retrieval(
        self, mem_id: str, boost: float = RETRIEVAL_BOOST_DEFAULT
    ) -> None:
        await self._db.execute(
            "UPDATE staging_memories "
            "SET strength = strength * ?, access_count = access_count + 1 "
            "WHERE id = ?",
            (boost, mem_id),
        )
        await self._db.commit()

    async def mark_consolidated(self, mem_id: str) -> None:
        await self._db.execute(
            "UPDATE staging_memories SET consolidated = 1 WHERE id = ?",
            (mem_id,),
        )
        await self._db.commit()

    async def update_strength(self, mem_id: str, new_strength: float) -> None:
        await self._db.execute(
            "UPDATE staging_memories SET strength = ? WHERE id = ?",
            (new_strength, mem_id),
        )
        await self._db.commit()

    async def delete(self, mem_id: str) -> None:
        await self._db.execute(
            "DELETE FROM staging_memories WHERE id = ?", (mem_id,)
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_by_id(self, mem_id: str) -> dict | None:
        async with self._db.execute(
            "SELECT * FROM staging_memories WHERE id = ?", (mem_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_dict(cursor.description, row) if row else None

    async def get_unconsolidated(
        self, workspace_id: str | None = "personal"
    ) -> list[dict]:
        if workspace_id is None:
            sql = (
                "SELECT * FROM staging_memories WHERE consolidated = 0 "
                "ORDER BY last_interaction"
            )
            args: tuple = ()
        else:
            sql = (
                "SELECT * FROM staging_memories "
                "WHERE consolidated = 0 AND workspace_id = ? "
                "ORDER BY last_interaction"
            )
            args = (workspace_id,)
        async with self._db.execute(sql, args) as cursor:
            return [
                self._row_to_dict(cursor.description, r)
                for r in await cursor.fetchall()
            ]

    async def count_unconsolidated(
        self, workspace_id: str | None = "personal"
    ) -> int:
        if workspace_id is None:
            sql = "SELECT COUNT(*) FROM staging_memories WHERE consolidated = 0"
            args: tuple = ()
        else:
            sql = (
                "SELECT COUNT(*) FROM staging_memories "
                "WHERE consolidated = 0 AND workspace_id = ?"
            )
            args = (workspace_id,)
        async with self._db.execute(sql, args) as cursor:
            return (await cursor.fetchone())[0]

    # ------------------------------------------------------------------
    # Edge staging
    # ------------------------------------------------------------------

    async def encode_edge(
        self,
        source: str,
        relation: str,
        target: str,
        interaction_id: int,
        session_id: str,
        workspace_id: str = "personal",
        importance_score: float = 0.5,
        never_decay: bool = False,
    ) -> str:
        eid = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO staging_edges "
            "(id, workspace_id, source_node, relation, target_node, "
            "interaction_id, session_id, importance_score, never_decay, "
            "consolidated, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (
                eid,
                workspace_id,
                source,
                relation,
                target,
                interaction_id,
                session_id,
                importance_score,
                1 if never_decay else 0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self._db.commit()
        return eid

    async def get_unconsolidated_edges(
        self, workspace_id: str | None = "personal"
    ) -> list[dict]:
        if workspace_id is None:
            sql = (
                "SELECT * FROM staging_edges WHERE consolidated = 0 "
                "ORDER BY created_at"
            )
            args: tuple = ()
        else:
            sql = (
                "SELECT * FROM staging_edges "
                "WHERE consolidated = 0 AND workspace_id = ? "
                "ORDER BY created_at"
            )
            args = (workspace_id,)
        async with self._db.execute(sql, args) as cursor:
            desc = cursor.description
            return [
                {col[0]: val for col, val in zip(desc, row)}
                for row in await cursor.fetchall()
            ]

    async def reinforce(
        self, mem_id: str, boost: float = RETRIEVAL_BOOST_DEFAULT
    ) -> None:
        await self._db.execute(
            "UPDATE staging_memories "
            "SET strength = strength * ?, access_count = access_count + 1 "
            "WHERE id = ?",
            (boost, mem_id),
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        if not a or not b:
            return 0.0
        min_len = min(len(a), len(b))
        dot = sum(a[i] * b[i] for i in range(min_len))
        norm_a = sum(x * x for x in a[:min_len]) ** 0.5
        norm_b = sum(x * x for x in b[:min_len]) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _row_to_dict(desc, row) -> dict:
        d = {col[0]: val for col, val in zip(desc, row)}
        d["entities"] = json.loads(d["entities"])
        d["emotional_tag"] = json.loads(d["emotional_tag"])
        d["context_embedding"] = (
            json.loads(d["context_embedding"]) if d["context_embedding"] else []
        )
        d["consolidated"] = bool(d["consolidated"])
        return d
