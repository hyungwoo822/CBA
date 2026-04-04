from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import aiosqlite


class EpisodicStore:
    """Long-term episodic memory store.

    Persists consolidated episodes (autobiographical events) in SQLite,
    supporting retrieval by recency, interaction range, and strength-based
    pruning.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                context_embedding TEXT,
                entities TEXT DEFAULT '{}',
                emotional_tag TEXT DEFAULT '{"valence":0,"arousal":0}',
                strength REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_interaction INTEGER DEFAULT 0,
                last_session TEXT DEFAULT '',
                schema_links TEXT DEFAULT '[]'
            )"""
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def save(
        self,
        content: str,
        context_embedding: list[float],
        entities: dict,
        emotional_tag: dict,
        interaction_id: int,
        session_id: str,
        strength: float = 1.0,
        access_count: int = 0,
    ) -> str:
        ep_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO episodes "
            "(id, timestamp, content, context_embedding, entities, "
            "emotional_tag, strength, access_count, last_interaction, last_session) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                ep_id,
                datetime.now(timezone.utc).isoformat(),
                content,
                json.dumps(context_embedding),
                json.dumps(entities),
                json.dumps(emotional_tag),
                strength,
                access_count,
                interaction_id,
                session_id,
            ),
        )
        await self._db.commit()
        return ep_id

    async def update_strength(self, ep_id: str, strength: float) -> None:
        await self._db.execute(
            "UPDATE episodes SET strength = ? WHERE id = ?",
            (strength, ep_id),
        )
        await self._db.commit()

    async def on_retrieval(self, ep_id: str, boost: float = 1.5) -> None:
        """SM-2 retrieval boost — strengthen retrieved memories (Wozniak 1990)."""
        await self._db.execute(
            "UPDATE episodes SET strength = strength * ?, access_count = access_count + 1 "
            "WHERE id = ?",
            (boost, ep_id),
        )
        await self._db.commit()

    async def reconsolidate(
        self, ep_id: str, interaction_id: int, session_id: str,
    ) -> None:
        """Re-encode memory with current context (Nader 2000 reconsolidation)."""
        await self._db.execute(
            "UPDATE episodes SET last_interaction = ?, last_session = ? WHERE id = ?",
            (interaction_id, session_id, ep_id),
        )
        await self._db.commit()

    async def delete_below_strength(self, threshold: float) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) FROM episodes WHERE strength < ?", (threshold,)
        ) as cursor:
            count = (await cursor.fetchone())[0]
        await self._db.execute(
            "DELETE FROM episodes WHERE strength < ?", (threshold,)
        )
        await self._db.commit()
        return count

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_by_id(self, ep_id: str) -> dict | None:
        async with self._db.execute(
            "SELECT * FROM episodes WHERE id = ?", (ep_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_dict(cursor.description, row) if row else None

    async def get_recent(self, limit: int = 10) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM episodes ORDER BY last_interaction DESC LIMIT ?",
            (limit,),
        ) as cursor:
            return [
                self._row_to_dict(cursor.description, r)
                for r in await cursor.fetchall()
            ]

    async def get_by_interaction_range(
        self, start: int, end: int
    ) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM episodes "
            "WHERE last_interaction >= ? AND last_interaction <= ? "
            "ORDER BY last_interaction",
            (start, end),
        ) as cursor:
            return [
                self._row_to_dict(cursor.description, r)
                for r in await cursor.fetchall()
            ]

    async def get_all(self) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM episodes ORDER BY last_interaction"
        ) as cursor:
            return [
                self._row_to_dict(cursor.description, r)
                for r in await cursor.fetchall()
            ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(desc, row) -> dict:
        d = {col[0]: val for col, val in zip(desc, row)}
        for key in ("entities", "emotional_tag", "schema_links"):
            if d.get(key):
                d[key] = json.loads(d[key])
        if d.get("context_embedding"):
            d["context_embedding"] = json.loads(d["context_embedding"])
        return d
