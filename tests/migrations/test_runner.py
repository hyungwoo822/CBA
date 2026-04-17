"""Tests for MigrationRunner."""
import aiosqlite
import pytest

from brain_agent.migrations import MigrationRunner


@pytest.fixture
async def runner(tmp_path):
    brain_state_db = str(tmp_path / "brain_state.db")
    runner = MigrationRunner(brain_state_db=brain_state_db, data_dir=str(tmp_path))
    yield runner


async def test_bootstrap_creates_schema_version_table(runner, tmp_path):
    await runner.apply_pending()
    async with aiosqlite.connect(str(tmp_path / "brain_state.db")) as db:
        cur = await db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='schema_version'"
        )
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == "schema_version"


async def test_m000_recorded_after_first_apply(runner, tmp_path):
    await runner.apply_pending()
    async with aiosqlite.connect(str(tmp_path / "brain_state.db")) as db:
        cur = await db.execute(
            "SELECT migration_id FROM schema_version ORDER BY migration_id"
        )
        rows = [row[0] for row in await cur.fetchall()]
    assert "m000_init_schema_version" in rows


async def test_apply_pending_is_idempotent(runner, tmp_path):
    await runner.apply_pending()
    await runner.apply_pending()
    async with aiosqlite.connect(str(tmp_path / "brain_state.db")) as db:
        cur = await db.execute("SELECT migration_id FROM schema_version")
        rows = [row[0] for row in await cur.fetchall()]
    assert len(rows) == len(set(rows))
    assert "m000_init_schema_version" in rows
    assert "m001_workspace_columns" in rows
