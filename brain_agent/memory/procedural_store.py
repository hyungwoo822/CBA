from __future__ import annotations

import fnmatch
import json
import uuid
from enum import Enum

import aiosqlite

ASSOCIATIVE_THRESHOLD = 10
AUTONOMOUS_THRESHOLD = 50
MIN_SUCCESS_RATE = 0.8


class ProcedureStage(str, Enum):
    COGNITIVE = "cognitive"
    ASSOCIATIVE = "associative"
    AUTONOMOUS = "autonomous"


class ProceduralStore:
    """SQLite-backed store for learned procedures with stage promotion."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self):
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                trigger_pattern TEXT NOT NULL,
                action_sequence TEXT NOT NULL,
                strategy TEXT DEFAULT '',
                success_rate REAL DEFAULT 0.0,
                execution_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                stage TEXT DEFAULT 'cognitive'
            )"""
        )
        # Migrate: add strategy column if missing (existing DBs)
        try:
            await self._db.execute("ALTER TABLE procedures ADD COLUMN strategy TEXT DEFAULT ''")
        except Exception:
            pass  # Column already exists
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    async def save(
        self,
        trigger_pattern: str,
        action_sequence: list[dict],
        strategy: str = "",
    ) -> str:
        proc_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO procedures (id, trigger_pattern, action_sequence, strategy) VALUES (?, ?, ?, ?)",
            (proc_id, trigger_pattern, json.dumps(action_sequence), strategy),
        )
        await self._db.commit()
        return proc_id

    async def get_all(self) -> list[dict]:
        """Return all stored procedures."""
        async with self._db.execute("SELECT * FROM procedures") as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(cursor.description, r) for r in rows]

    async def match(self, input_text: str) -> dict | None:
        async with self._db.execute("SELECT * FROM procedures") as cursor:
            for row in await cursor.fetchall():
                d = self._row_to_dict(cursor.description, row)
                if fnmatch.fnmatch(input_text.lower(), d["trigger_pattern"].lower()):
                    return d
        return None

    async def get_by_id(self, proc_id: str) -> dict | None:
        async with self._db.execute(
            "SELECT * FROM procedures WHERE id = ?", (proc_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_dict(cursor.description, row) if row else None

    async def record_execution(self, proc_id: str, success: bool):
        if success:
            await self._db.execute(
                "UPDATE procedures SET execution_count = execution_count + 1, "
                "success_count = success_count + 1 WHERE id = ?",
                (proc_id,),
            )
        else:
            await self._db.execute(
                "UPDATE procedures SET execution_count = execution_count + 1 WHERE id = ?",
                (proc_id,),
            )
        await self._db.execute(
            "UPDATE procedures SET success_rate = CAST(success_count AS REAL) / execution_count "
            "WHERE id = ?",
            (proc_id,),
        )
        await self._db.commit()

        proc = await self.get_by_id(proc_id)
        if proc:
            new_stage = self._compute_stage(proc["execution_count"], proc["success_rate"])
            if new_stage != proc["stage"]:
                await self._db.execute(
                    "UPDATE procedures SET stage = ? WHERE id = ?",
                    (new_stage, proc_id),
                )
                await self._db.commit()

    @staticmethod
    def _compute_stage(count: int, rate: float) -> str:
        if count >= AUTONOMOUS_THRESHOLD and rate >= MIN_SUCCESS_RATE:
            return ProcedureStage.AUTONOMOUS.value
        if count >= ASSOCIATIVE_THRESHOLD and rate >= MIN_SUCCESS_RATE:
            return ProcedureStage.ASSOCIATIVE.value
        return ProcedureStage.COGNITIVE.value

    @staticmethod
    def _row_to_dict(desc, row) -> dict:
        d = {col[0]: val for col, val in zip(desc, row)}
        d["action_sequence"] = json.loads(d["action_sequence"])
        return d
