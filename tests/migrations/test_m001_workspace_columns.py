"""Tests for m001_workspace_columns."""
import os

import aiosqlite
import pytest

from brain_agent.migrations.steps import m001_workspace_columns as m001


async def _create_legacy_graph_db(path: str) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """CREATE TABLE knowledge_graph (
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
                UNIQUE(source_node, relation, target_node, origin)
            )"""
        )
        await db.execute(
            "INSERT INTO knowledge_graph (id, source_node, relation, target_node, "
            "first_seen, last_seen) VALUES ('r1', 'alice', 'likes', 'coffee', "
            "'2026-01-01', '2026-01-01')"
        )
        await db.execute(
            "INSERT INTO knowledge_graph (id, source_node, relation, target_node, "
            "first_seen, last_seen) VALUES ('r2', 'bob', 'uses', 'vim', "
            "'2026-01-01', '2026-01-01')"
        )
        await db.commit()


async def _create_legacy_episodic_db(path: str) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """CREATE TABLE episodes (
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
        await db.execute(
            "INSERT INTO episodes (id, timestamp, content) "
            "VALUES ('e1', '2026-01-01', 'an episode')"
        )
        await db.commit()


async def _create_legacy_procedural_db(path: str) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """CREATE TABLE procedures (
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
        await db.execute(
            "INSERT INTO procedures (id, trigger_pattern, action_sequence) "
            "VALUES ('p1', '*hello*', '[]')"
        )
        await db.commit()


async def _create_legacy_staging_db(path: str) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """CREATE TABLE staging_memories (
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
        await db.execute(
            "INSERT INTO staging_memories (id, timestamp, content) "
            "VALUES ('s1', '2026-01-01', 'staged')"
        )
        await db.commit()


@pytest.fixture
async def legacy_data_dir(tmp_path):
    await _create_legacy_graph_db(str(tmp_path / "graph.db"))
    await _create_legacy_episodic_db(str(tmp_path / "episodic.db"))
    await _create_legacy_procedural_db(str(tmp_path / "procedural.db"))
    await _create_legacy_staging_db(str(tmp_path / "staging.db"))
    return str(tmp_path)


async def _count(db_path: str, table: str) -> int:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(f"SELECT COUNT(*) FROM {table}") as cur:
            return (await cur.fetchone())[0]


async def _columns(db_path: str, table: str) -> set[str]:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(f"PRAGMA table_info({table})") as cur:
            return {r[1] for r in await cur.fetchall()}


async def _indices(db_path: str) -> set[str]:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ) as cur:
            return {r[0] for r in await cur.fetchall() if r[0]}


def test_migration_id_matches_filename():
    assert m001.MIGRATION_ID == "m001_workspace_columns"


async def test_apply_adds_knowledge_graph_columns(legacy_data_dir, tmp_path):
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=legacy_data_dir,
    )
    cols = await _columns(os.path.join(legacy_data_dir, "graph.db"), "knowledge_graph")
    for col in (
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
    ):
        assert col in cols


async def test_apply_adds_episodes_columns(legacy_data_dir, tmp_path):
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=legacy_data_dir,
    )
    cols = await _columns(os.path.join(legacy_data_dir, "episodic.db"), "episodes")
    for col in (
        "workspace_id",
        "source_id",
        "event_type",
        "actor",
        "importance_score",
        "never_decay",
    ):
        assert col in cols


async def test_apply_adds_procedures_columns(legacy_data_dir, tmp_path):
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=legacy_data_dir,
    )
    cols = await _columns(os.path.join(legacy_data_dir, "procedural.db"), "procedures")
    for col in ("workspace_id", "trigger_embedding", "applicable_scope", "source_id"):
        assert col in cols


async def test_apply_adds_staging_workspace_column(legacy_data_dir, tmp_path):
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=legacy_data_dir,
    )
    cols = await _columns(
        os.path.join(legacy_data_dir, "staging.db"), "staging_memories"
    )
    assert "workspace_id" in cols


async def test_apply_backfills_workspace_id_to_personal(legacy_data_dir, tmp_path):
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=legacy_data_dir,
    )
    for db_name, table in (
        ("graph.db", "knowledge_graph"),
        ("episodic.db", "episodes"),
        ("procedural.db", "procedures"),
        ("staging.db", "staging_memories"),
    ):
        async with aiosqlite.connect(os.path.join(legacy_data_dir, db_name)) as db:
            async with db.execute(f"SELECT workspace_id FROM {table}") as cur:
                rows = await cur.fetchall()
        for row in rows:
            assert row[0] == "personal"


async def test_apply_preserves_row_counts(legacy_data_dir, tmp_path):
    before = {
        "knowledge_graph": await _count(
            os.path.join(legacy_data_dir, "graph.db"), "knowledge_graph"
        ),
        "episodes": await _count(
            os.path.join(legacy_data_dir, "episodic.db"), "episodes"
        ),
        "procedures": await _count(
            os.path.join(legacy_data_dir, "procedural.db"), "procedures"
        ),
        "staging_memories": await _count(
            os.path.join(legacy_data_dir, "staging.db"), "staging_memories"
        ),
    }
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=legacy_data_dir,
    )
    after = {
        "knowledge_graph": await _count(
            os.path.join(legacy_data_dir, "graph.db"), "knowledge_graph"
        ),
        "episodes": await _count(
            os.path.join(legacy_data_dir, "episodic.db"), "episodes"
        ),
        "procedures": await _count(
            os.path.join(legacy_data_dir, "procedural.db"), "procedures"
        ),
        "staging_memories": await _count(
            os.path.join(legacy_data_dir, "staging.db"), "staging_memories"
        ),
    }
    assert before == after


async def test_apply_creates_indices(legacy_data_dir, tmp_path):
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=legacy_data_dir,
    )
    graph_idx = await _indices(os.path.join(legacy_data_dir, "graph.db"))
    for idx in (
        "idx_kg_workspace",
        "idx_kg_workspace_source",
        "idx_kg_workspace_target",
        "idx_kg_target_workspace",
        "idx_kg_never_decay",
    ):
        assert idx in graph_idx
    ep_idx = await _indices(os.path.join(legacy_data_dir, "episodic.db"))
    for idx in (
        "idx_episodes_workspace",
        "idx_episodes_workspace_interaction",
        "idx_episodes_never_decay",
    ):
        assert idx in ep_idx
    proc_idx = await _indices(os.path.join(legacy_data_dir, "procedural.db"))
    assert "idx_procedures_workspace" in proc_idx


async def test_apply_is_idempotent(legacy_data_dir, tmp_path):
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=legacy_data_dir,
    )
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=legacy_data_dir,
    )
    cols = await _columns(os.path.join(legacy_data_dir, "graph.db"), "knowledge_graph")
    assert "workspace_id" in cols
    assert (
        await _count(os.path.join(legacy_data_dir, "graph.db"), "knowledge_graph")
        == 2
    )


async def test_apply_handles_missing_db_files(tmp_path):
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=str(tmp_path),
    )
