"""Create schema_version table in brain_state.db."""
from __future__ import annotations

import aiosqlite


MIGRATION_ID = "m000_init_schema_version"


async def apply(brain_state_db: str, data_dir: str) -> None:
    async with aiosqlite.connect(brain_state_db) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                migration_id TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        await db.commit()
