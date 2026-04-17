"""Add deprecated columns for template upgrade soft-deletes."""
from __future__ import annotations

import os

import aiosqlite


MIGRATION_ID = "m002_template_upgrade_columns"


async def _has_column(db: aiosqlite.Connection, table: str, column: str) -> bool:
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        rows = await cur.fetchall()
    return any(row[1] == column for row in rows)


async def _has_table(db: aiosqlite.Connection, table: str) -> bool:
    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ) as cur:
        return await cur.fetchone() is not None


async def apply(brain_state_db: str, data_dir: str) -> None:
    """Apply to ontology.db under data_dir, if that database already exists."""
    ontology_db = os.path.join(data_dir, "ontology.db")
    if not os.path.exists(ontology_db):
        return
    async with aiosqlite.connect(ontology_db) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        for table in ("node_types", "relation_types"):
            if not await _has_table(db, table):
                continue
            if not await _has_column(db, table, "deprecated"):
                await db.execute(
                    f"ALTER TABLE {table} ADD COLUMN deprecated "
                    "INTEGER NOT NULL DEFAULT 0"
                )
        await db.commit()
