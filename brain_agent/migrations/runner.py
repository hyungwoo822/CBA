"""Schema migration runner for memory store databases."""
from __future__ import annotations

import importlib
import logging
import pkgutil
from datetime import datetime, timezone

import aiosqlite

logger = logging.getLogger(__name__)


class MigrationRunner:
    """Discover and apply unapplied migrations in filename order."""

    def __init__(self, brain_state_db: str, data_dir: str):
        self._brain_state_db = brain_state_db
        self._data_dir = data_dir

    async def apply_pending(self) -> list[str]:
        """Apply all pending migrations and return applied migration ids."""
        from brain_agent.migrations.steps import m000_init_schema_version as m000

        await m000.apply(self._brain_state_db, self._data_dir)
        await self._record_if_absent(m000.MIGRATION_ID)

        applied: list[str] = []
        existing_ids = await self._list_applied()

        from brain_agent.migrations import steps as steps_pkg

        module_names = sorted(
            name
            for _, name, _ in pkgutil.iter_modules(steps_pkg.__path__)
            if name.startswith("m") and name != "m000_init_schema_version"
        )
        for module_name in module_names:
            module = importlib.import_module(
                f"brain_agent.migrations.steps.{module_name}"
            )
            migration_id = getattr(module, "MIGRATION_ID", module_name)
            if migration_id in existing_ids:
                continue
            logger.info("Applying migration %s", migration_id)
            await module.apply(self._brain_state_db, self._data_dir)
            await self._record_if_absent(migration_id)
            existing_ids.add(migration_id)
            applied.append(migration_id)
        return applied

    async def _list_applied(self) -> set[str]:
        async with aiosqlite.connect(self._brain_state_db) as db:
            cur = await db.execute("SELECT migration_id FROM schema_version")
            return {row[0] for row in await cur.fetchall()}

    async def _record_if_absent(self, migration_id: str) -> None:
        async with aiosqlite.connect(self._brain_state_db) as db:
            await db.execute(
                "INSERT OR IGNORE INTO schema_version "
                "(migration_id, applied_at) VALUES (?, ?)",
                (migration_id, datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()


async def apply_pending(brain_state_db: str, data_dir: str) -> list[str]:
    """Convenience wrapper around MigrationRunner.apply_pending()."""
    runner = MigrationRunner(brain_state_db=brain_state_db, data_dir=data_dir)
    return await runner.apply_pending()
