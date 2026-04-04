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
                last_session TEXT DEFAULT ''
            )"""
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
                    "WHERE consolidated = 0 ORDER BY last_interaction DESC LIMIT 10"
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
            "last_interaction, last_session) "
            "VALUES (?,?,?,?,?,?,?,0,1.0,0,?,?)",
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

    async def get_unconsolidated(self) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM staging_memories "
            "WHERE consolidated = 0 ORDER BY last_interaction"
        ) as cursor:
            return [
                self._row_to_dict(cursor.description, r)
                for r in await cursor.fetchall()
            ]

    async def count_unconsolidated(self) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) FROM staging_memories WHERE consolidated = 0"
        ) as cursor:
            return (await cursor.fetchone())[0]

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
