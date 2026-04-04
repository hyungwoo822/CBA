"""Brain State Persistence — saves and restores neuromodulator levels
and region activation levels across sessions.

Allows the agent's internal state to persist between conversations,
modeling the brain's continuous state rather than resetting each session.

The brain doesn't "reset to baseline" between conversations — chronic stress
keeps cortisol elevated, repeated rewards build dopamine sensitivity, etc.
This module persists that continuity.

References:
  - Grace (2000): Tonic dopamine maintains baseline readiness
  - McEwen (2007): Allostatic load — cumulative cortisol effects persist
  - Aston-Jones & Cohen (2005): LC-NE tonic mode persists across tasks
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import aiosqlite

logger = logging.getLogger(__name__)


class BrainStateStore:
    """SQLite-backed persistence for neuromodulator and activation state."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS brain_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                dopamine REAL DEFAULT 0.5,
                norepinephrine REAL DEFAULT 0.5,
                serotonin REAL DEFAULT 0.5,
                acetylcholine REAL DEFAULT 0.5,
                cortisol REAL DEFAULT 0.5,
                epinephrine REAL DEFAULT 0.5,
                gaba REAL DEFAULT 0.5,
                region_activations TEXT DEFAULT '{}',
                interaction_count INTEGER DEFAULT 0,
                last_session_id TEXT DEFAULT '',
                updated_at TEXT DEFAULT ''
            )
        """)
        # Neuromodulator history — time series for dashboard graphs
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS neuromodulator_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dopamine REAL, norepinephrine REAL, serotonin REAL,
                acetylcholine REAL, cortisol REAL, epinephrine REAL,
                gaba REAL DEFAULT 0.5,
                timestamp TEXT NOT NULL
            )
        """)
        # Ensure exactly one row exists (singleton pattern)
        async with self._db.execute("SELECT COUNT(*) FROM brain_state") as cur:
            count = (await cur.fetchone())[0]
        if count == 0:
            await self._db.execute(
                "INSERT INTO brain_state (id) VALUES (1)"
            )
            await self._db.commit()

        # Migration: add gaba column to existing DBs
        try:
            await self._db.execute("ALTER TABLE brain_state ADD COLUMN gaba REAL DEFAULT 0.5")
            await self._db.commit()
        except Exception:
            pass
        try:
            await self._db.execute("ALTER TABLE neuromodulator_history ADD COLUMN gaba REAL DEFAULT 0.5")
            await self._db.commit()
        except Exception:
            pass

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def load_neuromodulators(self) -> dict:
        """Load persisted neuromodulator state.

        Returns dict with keys: dopamine, norepinephrine, serotonin,
        acetylcholine, cortisol, epinephrine.
        Returns default (0.5) values if no state saved.
        """
        if not self._db:
            return self._defaults()
        try:
            async with self._db.execute(
                "SELECT dopamine, norepinephrine, serotonin, acetylcholine, "
                "cortisol, epinephrine, gaba FROM brain_state WHERE id = 1"
            ) as cur:
                row = await cur.fetchone()
                if row:
                    return {
                        "dopamine": row[0],
                        "norepinephrine": row[1],
                        "serotonin": row[2],
                        "acetylcholine": row[3],
                        "cortisol": row[4],
                        "epinephrine": row[5],
                        "gaba": row[6] if len(row) > 6 else 0.5,
                    }
        except Exception as e:
            logger.warning("Failed to load neuromodulator state: %s", e)
        return self._defaults()

    async def save_neuromodulators(self, state: dict) -> None:
        """Persist current neuromodulator state to DB + append to history."""
        if not self._db:
            return
        try:
            ts = datetime.now(timezone.utc).isoformat()
            # Update current state (singleton row)
            await self._db.execute(
                "UPDATE brain_state SET "
                "dopamine=?, norepinephrine=?, serotonin=?, "
                "acetylcholine=?, cortisol=?, epinephrine=?, "
                "gaba=?, updated_at=? WHERE id = 1",
                (
                    state.get("dopamine", 0.5),
                    state.get("norepinephrine", 0.5),
                    state.get("serotonin", 0.5),
                    state.get("acetylcholine", 0.5),
                    state.get("cortisol", 0.5),
                    state.get("epinephrine", 0.5),
                    state.get("gaba", 0.5),
                    ts,
                ),
            )
            # Append to history time series
            await self._db.execute(
                "INSERT INTO neuromodulator_history "
                "(dopamine, norepinephrine, serotonin, acetylcholine, cortisol, epinephrine, gaba, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    state.get("dopamine", 0.5),
                    state.get("norepinephrine", 0.5),
                    state.get("serotonin", 0.5),
                    state.get("acetylcholine", 0.5),
                    state.get("cortisol", 0.5),
                    state.get("epinephrine", 0.5),
                    state.get("gaba", 0.5),
                    ts,
                ),
            )
            await self._db.commit()
        except Exception as e:
            logger.warning("Failed to save neuromodulator state: %s", e)

    async def get_neuromodulator_history(self, limit: int = 100) -> list[dict]:
        """Get recent neuromodulator history for dashboard graphs."""
        if not self._db:
            return []
        try:
            async with self._db.execute(
                "SELECT dopamine, norepinephrine, serotonin, acetylcholine, "
                "cortisol, epinephrine, gaba, timestamp FROM neuromodulator_history "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
            return [
                {
                    "dopamine": r[0], "norepinephrine": r[1], "serotonin": r[2],
                    "acetylcholine": r[3], "cortisol": r[4], "epinephrine": r[5],
                    "gaba": r[6] if len(r) > 7 else 0.5,
                    "timestamp": r[7] if len(r) > 7 else r[6],
                }
                for r in reversed(rows)
            ]
        except Exception as e:
            logger.warning("Failed to load neuromodulator history: %s", e)
            return []

    async def load_region_activations(self) -> dict[str, float]:
        """Load persisted region activation levels."""
        if not self._db:
            return {}
        try:
            async with self._db.execute(
                "SELECT region_activations FROM brain_state WHERE id = 1"
            ) as cur:
                row = await cur.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
        except Exception as e:
            logger.warning("Failed to load region activations: %s", e)
        return {}

    async def save_region_activations(self, activations: dict[str, float]) -> None:
        """Persist region activation levels."""
        if not self._db:
            return
        try:
            await self._db.execute(
                "UPDATE brain_state SET region_activations=?, updated_at=? WHERE id = 1",
                (json.dumps(activations), datetime.now(timezone.utc).isoformat()),
            )
            await self._db.commit()
        except Exception as e:
            logger.warning("Failed to save region activations: %s", e)

    async def update_interaction_count(self, count: int, session_id: str) -> None:
        """Update session metadata."""
        if not self._db:
            return
        try:
            await self._db.execute(
                "UPDATE brain_state SET interaction_count=?, last_session_id=?, "
                "updated_at=? WHERE id = 1",
                (count, session_id, datetime.now(timezone.utc).isoformat()),
            )
            await self._db.commit()
        except Exception as e:
            logger.warning("Failed to update interaction count: %s", e)

    async def load_interaction_count(self) -> tuple[int, str]:
        """Load interaction count and last session ID."""
        if not self._db:
            return 0, ""
        try:
            async with self._db.execute(
                "SELECT interaction_count, last_session_id FROM brain_state WHERE id = 1"
            ) as cur:
                row = await cur.fetchone()
                if row:
                    return row[0], row[1]
        except Exception as e:
            logger.warning("Failed to load interaction count: %s", e)
        return 0, ""

    @staticmethod
    def _defaults() -> dict:
        return {
            "dopamine": 0.5,
            "norepinephrine": 0.5,
            "serotonin": 0.5,
            "acetylcholine": 0.5,
            "cortisol": 0.5,
            "epinephrine": 0.5,
            "gaba": 0.5,
        }
