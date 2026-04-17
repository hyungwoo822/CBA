"""Add workspace, provenance, temporal, and decay-policy columns."""
from __future__ import annotations

import logging
import os

import aiosqlite

logger = logging.getLogger(__name__)

MIGRATION_ID = "m001_workspace_columns"

_GRAPH_COLUMNS = [
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
]
_EPISODES_COLUMNS = [
    ("workspace_id", "TEXT DEFAULT 'personal'"),
    ("source_id", "TEXT"),
    ("event_type", "TEXT DEFAULT 'conversation_turn'"),
    ("actor", "TEXT"),
    ("importance_score", "REAL DEFAULT 0.5"),
    ("never_decay", "INTEGER DEFAULT 0"),
]
_PROCEDURES_COLUMNS = [
    ("workspace_id", "TEXT DEFAULT 'personal'"),
    ("trigger_embedding", "TEXT"),
    ("applicable_scope", "TEXT DEFAULT '{}'"),
    ("source_id", "TEXT"),
]
_STAGING_COLUMNS = [
    ("workspace_id", "TEXT DEFAULT 'personal'"),
]

_GRAPH_INDICES = [
    ("idx_kg_workspace", "knowledge_graph(workspace_id)"),
    ("idx_kg_workspace_source", "knowledge_graph(workspace_id, source_node)"),
    ("idx_kg_workspace_target", "knowledge_graph(workspace_id, target_node)"),
    ("idx_kg_target_workspace", "knowledge_graph(target_workspace_id)"),
    ("idx_kg_never_decay", "knowledge_graph(workspace_id, never_decay)"),
    (
        "idx_kg_unique_workspace_triple_origin",
        "knowledge_graph(workspace_id, source_node, relation, target_node, origin)",
    ),
]
_EPISODES_INDICES = [
    ("idx_episodes_workspace", "episodes(workspace_id)"),
    ("idx_episodes_workspace_interaction", "episodes(workspace_id, last_interaction)"),
    ("idx_episodes_never_decay", "episodes(workspace_id, never_decay)"),
]
_PROCEDURES_INDICES = [
    ("idx_procedures_workspace", "procedures(workspace_id)"),
]
_STAGING_INDICES = [
    ("idx_staging_workspace", "staging_memories(workspace_id, consolidated)"),
]


async def _table_exists(db: aiosqlite.Connection, table: str) -> bool:
    async with db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table,)
    ) as cur:
        return await cur.fetchone() is not None


async def _columns(db: aiosqlite.Connection, table: str) -> set[str]:
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        return {row[1] for row in await cur.fetchall()}


async def _add_column(
    db: aiosqlite.Connection, table: str, col: str, defn: str
) -> None:
    try:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
    except aiosqlite.OperationalError as exc:
        msg = str(exc).lower()
        if "duplicate column name" in msg or "no such table" in msg:
            logger.debug("Skipping ALTER %s.%s: %s", table, col, msg)
            return
        raise


async def _add_index(db: aiosqlite.Connection, name: str, spec: str) -> None:
    unique = "UNIQUE " if name.startswith("idx_kg_unique_") else ""
    try:
        await db.execute(f"CREATE {unique}INDEX IF NOT EXISTS {name} ON {spec}")
    except aiosqlite.OperationalError as exc:
        msg = str(exc).lower()
        if "no such table" in msg:
            logger.debug("Skipping INDEX %s: %s", name, msg)
            return
        raise


async def _backfill_workspace(db: aiosqlite.Connection, table: str) -> None:
    try:
        await db.execute(
            f"UPDATE {table} SET workspace_id = 'personal' "
            "WHERE workspace_id IS NULL"
        )
    except aiosqlite.OperationalError as exc:
        msg = str(exc).lower()
        if "no such table" in msg or "no such column" in msg:
            logger.debug("Skipping workspace backfill for %s: %s", table, msg)
            return
        raise


async def _rebuild_knowledge_graph_if_needed(db: aiosqlite.Connection) -> None:
    """Replace the old triple-only UNIQUE table constraint with workspace scope."""
    if not await _table_exists(db, "knowledge_graph"):
        return
    async with db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='knowledge_graph'"
    ) as cur:
        row = await cur.fetchone()
    ddl = row[0] or ""
    normalized = " ".join(ddl.lower().split())
    if "unique(workspace_id, source_node, relation, target_node, origin)" in normalized:
        return

    await db.execute("DROP INDEX IF EXISTS idx_kg_unique_triple_origin")
    await db.execute(
        """
        CREATE TABLE knowledge_graph_new (
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
        )
        """
    )
    cols = await _columns(db, "knowledge_graph")
    copy_cols = [
        "id",
        "source_node",
        "relation",
        "target_node",
        "category",
        "confidence",
        "weight",
        "occurrence_count",
        "first_seen",
        "last_seen",
        "origin",
        "workspace_id",
        "target_workspace_id",
        "source_ref",
        "valid_from",
        "valid_to",
        "superseded_by",
        "type_id",
        "epistemic_source",
        "importance_score",
        "never_decay",
    ]
    available = [col for col in copy_cols if col in cols]
    joined = ", ".join(available)
    await db.execute(
        f"INSERT INTO knowledge_graph_new ({joined}) "
        f"SELECT {joined} FROM knowledge_graph"
    )
    await db.execute("DROP TABLE knowledge_graph")
    await db.execute("ALTER TABLE knowledge_graph_new RENAME TO knowledge_graph")


async def _apply_to_db(
    db_path: str,
    table: str,
    columns: list[tuple[str, str]],
    indices: list[tuple[str, str]],
) -> None:
    if not os.path.exists(db_path):
        logger.debug("Legacy DB %s does not exist; skipping", db_path)
        return
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        if not await _table_exists(db, table):
            await db.commit()
            return
        for col, defn in columns:
            await _add_column(db, table, col, defn)
        await _backfill_workspace(db, table)
        if table == "knowledge_graph":
            await _rebuild_knowledge_graph_if_needed(db)
        for name, spec in indices:
            await _add_index(db, name, spec)
        await db.commit()


async def apply(brain_state_db: str, data_dir: str) -> None:
    """Apply m001 to legacy store databases under data_dir."""
    await _apply_to_db(
        os.path.join(data_dir, "graph.db"),
        "knowledge_graph",
        _GRAPH_COLUMNS,
        _GRAPH_INDICES,
    )
    await _apply_to_db(
        os.path.join(data_dir, "episodic.db"),
        "episodes",
        _EPISODES_COLUMNS,
        _EPISODES_INDICES,
    )
    await _apply_to_db(
        os.path.join(data_dir, "procedural.db"),
        "procedures",
        _PROCEDURES_COLUMNS,
        _PROCEDURES_INDICES,
    )
    await _apply_to_db(
        os.path.join(data_dir, "staging.db"),
        "staging_memories",
        _STAGING_COLUMNS,
        _STAGING_INDICES,
    )
    logger.info("m001_workspace_columns applied")
