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
                schema_links TEXT DEFAULT '[]',
                workspace_id TEXT DEFAULT 'personal',
                source_id TEXT,
                event_type TEXT DEFAULT 'conversation_turn',
                actor TEXT,
                importance_score REAL DEFAULT 0.5,
                never_decay INTEGER DEFAULT 0
            )"""
        )
        for col, defn in [
            ("workspace_id", "TEXT DEFAULT 'personal'"),
            ("source_id", "TEXT"),
            ("event_type", "TEXT DEFAULT 'conversation_turn'"),
            ("actor", "TEXT"),
            ("importance_score", "REAL DEFAULT 0.5"),
            ("never_decay", "INTEGER DEFAULT 0"),
        ]:
            try:
                await self._db.execute(f"ALTER TABLE episodes ADD COLUMN {col} {defn}")
            except Exception:
                pass
        await self._db.execute(
            "UPDATE episodes SET workspace_id = 'personal' "
            "WHERE workspace_id IS NULL"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_workspace "
            "ON episodes(workspace_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_workspace_interaction "
            "ON episodes(workspace_id, last_interaction)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_never_decay "
            "ON episodes(workspace_id, never_decay)"
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
        workspace_id: str = "personal",
        source_id: str | None = None,
        event_type: str = "conversation_turn",
        actor: str | None = None,
        importance_score: float = 0.5,
        never_decay: bool = False,
    ) -> str:
        ep_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO episodes "
            "(id, timestamp, content, context_embedding, entities, "
            "emotional_tag, strength, access_count, last_interaction, last_session, "
            "workspace_id, source_id, event_type, actor, importance_score, never_decay) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                workspace_id,
                source_id,
                event_type,
                actor,
                importance_score,
                1 if never_decay else 0,
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
        cursor = await self._db.execute(
            "DELETE FROM episodes WHERE strength < ? AND never_decay = 0",
            (threshold,),
        )
        await self._db.commit()
        return cursor.rowcount or 0

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_by_id(self, ep_id: str) -> dict | None:
        async with self._db.execute(
            "SELECT * FROM episodes WHERE id = ?", (ep_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_dict(cursor.description, row) if row else None

    async def get_recent(
        self,
        limit: int = 10,
        workspace_id: str | None = "personal",
    ) -> list[dict]:
        if workspace_id is None:
            sql = "SELECT * FROM episodes ORDER BY last_interaction DESC LIMIT ?"
            args: tuple = (limit,)
        else:
            sql = (
                "SELECT * FROM episodes WHERE workspace_id = ? "
                "ORDER BY last_interaction DESC LIMIT ?"
            )
            args = (workspace_id, limit)
        async with self._db.execute(sql, args) as cursor:
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
