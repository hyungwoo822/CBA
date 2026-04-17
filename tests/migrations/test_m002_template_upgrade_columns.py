"""Tests for m002_template_upgrade_columns migration."""
import aiosqlite
import pytest

from brain_agent.memory.ontology_store import OntologyStore
from brain_agent.migrations import MigrationRunner


@pytest.fixture
async def runner_and_ontology(tmp_path):
    ontology_db = str(tmp_path / "ontology.db")
    store = OntologyStore(db_path=ontology_db)
    await store.initialize()
    await store.close()
    runner = MigrationRunner(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=str(tmp_path),
    )
    yield runner, ontology_db


async def test_m002_adds_deprecated_columns(runner_and_ontology):
    runner, ontology_db = runner_and_ontology
    await runner.apply_pending()
    async with aiosqlite.connect(ontology_db) as db:
        cur = await db.execute("PRAGMA table_info(node_types)")
        node_cols = {row[1] for row in await cur.fetchall()}
        cur = await db.execute("PRAGMA table_info(relation_types)")
        rel_cols = {row[1] for row in await cur.fetchall()}
    assert "deprecated" in node_cols
    assert "deprecated" in rel_cols


async def test_m002_default_deprecated_zero(runner_and_ontology):
    runner, ontology_db = runner_and_ontology
    await runner.apply_pending()
    store = OntologyStore(db_path=ontology_db)
    await store.initialize()
    await store.seed_universal()
    async with aiosqlite.connect(ontology_db) as db:
        cur = await db.execute(
            "SELECT deprecated FROM node_types WHERE name = 'Entity'"
        )
        row = await cur.fetchone()
    assert row[0] == 0
    await store.close()


async def test_m002_idempotent(runner_and_ontology):
    runner, _ = runner_and_ontology
    await runner.apply_pending()
    await runner.apply_pending()
