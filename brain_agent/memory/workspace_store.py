"""Workspace registry for workspace-scoped memory storage."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import aiosqlite


PERSONAL_WORKSPACE_ID = "personal"
PERSONAL_WORKSPACE_NAME = "Personal Knowledge"


class WorkspaceStore:
    """SQLite-backed registry of workspaces and session bindings."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT DEFAULT '',
                decay_policy TEXT NOT NULL DEFAULT 'normal'
                    CHECK (decay_policy IN ('none', 'slow', 'normal')),
                template_id TEXT,
                template_version TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_session (
                session_id TEXT PRIMARY KEY,
                current_workspace_id TEXT NOT NULL,
                set_at TEXT NOT NULL
            )
            """
        )
        await self._db.commit()
        await self.get_or_create_personal()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def create_workspace(
        self,
        name: str,
        description: str = "",
        decay_policy: str = "normal",
        template_id: str | None = None,
        template_version: str | None = None,
    ) -> dict:
        assert self._db is not None
        ws_id = str(uuid.uuid4())
        now = _now()
        try:
            await self._db.execute(
                """
                INSERT INTO workspaces
                (id, name, description, decay_policy, template_id,
                 template_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ws_id,
                    name,
                    description,
                    decay_policy,
                    template_id,
                    template_version,
                    now,
                    now,
                ),
            )
            await self._db.commit()
        except aiosqlite.IntegrityError as exc:
            raise ValueError(f"Workspace '{name}' already exists") from exc
        created = await self.get_workspace(ws_id)
        assert created is not None
        return created

    async def get_workspace(self, name_or_id: str) -> dict | None:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM workspaces WHERE id = ? OR name = ?",
            (name_or_id, name_or_id),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def list_workspaces(self) -> list[dict]:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM workspaces ORDER BY created_at"
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def update_workspace(self, ws_id: str, **fields) -> None:
        assert self._db is not None
        allowed = {
            "name",
            "description",
            "decay_policy",
            "template_id",
            "template_version",
        }
        cleaned = {key: value for key, value in fields.items() if key in allowed}
        if not cleaned:
            return
        cleaned["updated_at"] = _now()
        set_clause = ", ".join(f"{key} = ?" for key in cleaned)
        values = list(cleaned.values()) + [ws_id]
        await self._db.execute(
            f"UPDATE workspaces SET {set_clause} WHERE id = ?", values
        )
        await self._db.commit()

    async def delete_workspace(self, ws_id: str) -> None:
        if ws_id == PERSONAL_WORKSPACE_ID:
            raise ValueError("Cannot delete personal workspace")
        assert self._db is not None
        await self._db.execute("DELETE FROM workspaces WHERE id = ?", (ws_id,))
        await self._db.commit()

    async def get_or_create_personal(self) -> dict:
        existing = await self.get_workspace(PERSONAL_WORKSPACE_ID)
        if existing:
            return existing
        assert self._db is not None
        now = _now()
        await self._db.execute(
            """
            INSERT OR IGNORE INTO workspaces
            (id, name, description, decay_policy, created_at, updated_at)
            VALUES (?, ?, '', 'normal', ?, ?)
            """,
            (PERSONAL_WORKSPACE_ID, PERSONAL_WORKSPACE_NAME, now, now),
        )
        await self._db.commit()
        personal = await self.get_workspace(PERSONAL_WORKSPACE_ID)
        assert personal is not None
        return personal

    async def set_session_workspace(self, session_id: str, workspace_id: str) -> None:
        if await self.get_workspace(workspace_id) is None:
            raise ValueError(f"Workspace not found: {workspace_id}")
        assert self._db is not None
        now = _now()
        await self._db.execute(
            """
            INSERT INTO workspace_session (session_id, current_workspace_id, set_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                current_workspace_id = excluded.current_workspace_id,
                set_at = excluded.set_at
            """,
            (session_id, workspace_id, now),
        )
        await self._db.commit()

    async def get_session_workspace(self, session_id: str) -> str:
        """Return the bound workspace id, or personal if the session is unbound."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT current_workspace_id FROM workspace_session WHERE session_id = ?",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else PERSONAL_WORKSPACE_ID

    async def get_last_workspace(self) -> str:
        """Return the most recently bound workspace id, or personal."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT current_workspace_id FROM workspace_session "
            "ORDER BY set_at DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else PERSONAL_WORKSPACE_ID


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
