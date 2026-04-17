"""Workspace-scoped queue of unresolved questions."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import aiosqlite

logger = logging.getLogger(__name__)


RAISED_BY_VALUES = ("ambiguity_detector", "unknown_fact", "contradiction", "user")
SEVERITY_VALUES = ("minor", "moderate", "severe")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OpenQuestionsStore:
    """SQLite-backed queue of questions that still need answers."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS open_questions (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                question TEXT NOT NULL,
                raised_by TEXT NOT NULL
                    CHECK (raised_by IN
                        ('ambiguity_detector', 'unknown_fact', 'contradiction', 'user')),
                context_node TEXT DEFAULT '',
                context_input TEXT DEFAULT '',
                severity TEXT NOT NULL DEFAULT 'moderate'
                    CHECK (severity IN ('minor', 'moderate', 'severe')),
                blocking INTEGER NOT NULL DEFAULT 0
                    CHECK (blocking IN (0, 1)),
                asked_at TEXT NOT NULL,
                answered_at TEXT,
                answer TEXT,
                answer_source TEXT
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_open_questions_workspace "
            "ON open_questions(workspace_id, severity)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_open_questions_blocking "
            "ON open_questions(workspace_id, blocking)"
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def add_question(
        self,
        workspace_id: str,
        question: str,
        raised_by: str,
        severity: str = "moderate",
        context_node: str = "",
        context_input: str = "",
    ) -> dict:
        """Register a question; severe questions are automatically blocking."""
        assert self._db is not None
        blocking = 1 if severity == "severe" else 0
        new_id = str(uuid.uuid4())
        now = _now()
        await self._db.execute(
            """
            INSERT INTO open_questions
            (id, workspace_id, question, raised_by, context_node, context_input,
             severity, blocking, asked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                workspace_id,
                question,
                raised_by,
                context_node,
                context_input,
                severity,
                blocking,
                now,
            ),
        )
        await self._db.commit()
        created = await self._get_by_id(new_id)
        assert created is not None
        return created

    async def answer_question(
        self,
        question_id: str,
        answer: str,
        answer_source: str = "",
    ) -> dict:
        """Answer a question and remove it from unanswered/blocking queries."""
        assert self._db is not None
        existing = await self._get_by_id(question_id)
        if existing is None:
            raise ValueError(f"Question not found: {question_id}")
        await self._db.execute(
            """
            UPDATE open_questions
            SET answer = ?, answer_source = ?, answered_at = ?
            WHERE id = ?
            """,
            (answer, answer_source, _now(), question_id),
        )
        await self._db.commit()
        answered = await self._get_by_id(question_id)
        assert answered is not None
        return answered

    async def get_question(self, question_id: str) -> dict | None:
        """Return a question by id."""
        return await self._get_by_id(question_id)

    async def list_unanswered(self, workspace_id: str) -> list[dict]:
        """Return unanswered questions for a workspace."""
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM open_questions "
            "WHERE workspace_id = ? AND answered_at IS NULL "
            "ORDER BY asked_at DESC",
            (workspace_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def list_blocking(self, workspace_id: str) -> list[dict]:
        """Return unanswered blocking questions for a workspace."""
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM open_questions "
            "WHERE workspace_id = ? AND blocking = 1 AND answered_at IS NULL "
            "ORDER BY asked_at DESC",
            (workspace_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def list_by_severity(self, workspace_id: str, severity: str) -> list[dict]:
        """Return questions in a workspace by severity, answered or unanswered."""
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM open_questions "
            "WHERE workspace_id = ? AND severity = ? "
            "ORDER BY asked_at DESC",
            (workspace_id, severity),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def count_blocking(self, workspace_id: str) -> int:
        """Count unanswered blocking questions for a workspace."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT COUNT(*) FROM open_questions "
            "WHERE workspace_id = ? AND blocking = 1 AND answered_at IS NULL",
            (workspace_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def _get_by_id(self, question_id: str) -> dict | None:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM open_questions WHERE id = ?", (question_id,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None
