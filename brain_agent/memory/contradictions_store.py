"""Workspace-scoped conflict registry with severity auto-compute."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Iterable

import aiosqlite

logger = logging.getLogger(__name__)


SEVERE_RELATIONS = frozenset({"supersedes", "contradicts"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_severity(
    value_a_confidence: str,
    value_b_confidence: str,
    subject_node: str,
    core_node_set: Iterable[str] | None,
    key_or_relation: str,
) -> str:
    """Compute severity from confidence tiers, hub membership, and relation kind."""
    core = frozenset(core_node_set) if core_node_set else frozenset()

    a_extracted = value_a_confidence == "EXTRACTED"
    b_extracted = value_b_confidence == "EXTRACTED"
    a_inferred = "INFERRED" in (value_a_confidence or "")
    b_inferred = "INFERRED" in (value_b_confidence or "")

    if a_extracted and b_extracted:
        return "severe"
    if subject_node in core:
        return "severe"
    if key_or_relation in SEVERE_RELATIONS:
        return "severe"

    if a_inferred and b_inferred:
        return "minor"

    return "moderate"


class ContradictionsStore:
    """SQLite-backed registry of detected contradictions."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS contradictions (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                subject_node TEXT NOT NULL,
                key_or_relation TEXT NOT NULL,
                value_a TEXT NOT NULL,
                value_a_source TEXT DEFAULT '',
                value_a_confidence TEXT DEFAULT 'INFERRED',
                value_b TEXT NOT NULL,
                value_b_source TEXT DEFAULT '',
                value_b_confidence TEXT DEFAULT 'INFERRED',
                severity TEXT NOT NULL DEFAULT 'moderate'
                    CHECK (severity IN ('minor', 'moderate', 'severe')),
                status TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open', 'resolved', 'dismissed')),
                detected_at TEXT NOT NULL,
                resolved_at TEXT,
                resolved_by TEXT,
                resolution TEXT,
                resolution_confidence TEXT
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_contradictions_workspace "
            "ON contradictions(workspace_id, status)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_contradictions_subject "
            "ON contradictions(workspace_id, subject_node, status)"
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def detect(
        self,
        workspace_id: str,
        subject: str,
        key_or_relation: str,
        value_a: str,
        value_b: str,
        value_a_source: str = "",
        value_b_source: str = "",
        value_a_confidence: str = "INFERRED",
        value_b_confidence: str = "INFERRED",
        core_node_set: Iterable[str] | None = None,
    ) -> dict:
        """Register a new contradiction with auto-computed severity."""
        assert self._db is not None
        severity = _compute_severity(
            value_a_confidence=value_a_confidence,
            value_b_confidence=value_b_confidence,
            subject_node=subject,
            core_node_set=core_node_set,
            key_or_relation=key_or_relation,
        )
        new_id = str(uuid.uuid4())
        now = _now()
        await self._db.execute(
            """
            INSERT INTO contradictions
            (id, workspace_id, subject_node, key_or_relation,
             value_a, value_a_source, value_a_confidence,
             value_b, value_b_source, value_b_confidence,
             severity, status, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            (
                new_id,
                workspace_id,
                subject,
                key_or_relation,
                value_a,
                value_a_source,
                value_a_confidence,
                value_b,
                value_b_source,
                value_b_confidence,
                severity,
                now,
            ),
        )
        await self._db.commit()
        created = await self._get_by_id(new_id)
        assert created is not None
        return created

    async def resolve(
        self,
        contradiction_id: str,
        resolution: str,
        resolved_by: str = "user",
        resolution_confidence: str = "EXTRACTED",
    ) -> dict:
        """Mark a contradiction resolved with the selected resolution."""
        assert self._db is not None
        existing = await self._get_by_id(contradiction_id)
        if existing is None:
            raise ValueError(f"Contradiction not found: {contradiction_id}")
        await self._db.execute(
            """
            UPDATE contradictions
            SET status = 'resolved',
                resolution = ?,
                resolved_by = ?,
                resolution_confidence = ?,
                resolved_at = ?
            WHERE id = ?
            """,
            (resolution, resolved_by, resolution_confidence, _now(), contradiction_id),
        )
        await self._db.commit()
        resolved = await self._get_by_id(contradiction_id)
        assert resolved is not None
        return resolved

    async def dismiss(self, contradiction_id: str) -> dict:
        """Dismiss a contradiction without choosing a resolution."""
        assert self._db is not None
        existing = await self._get_by_id(contradiction_id)
        if existing is None:
            raise ValueError(f"Contradiction not found: {contradiction_id}")
        await self._db.execute(
            "UPDATE contradictions SET status = 'dismissed', resolved_at = ? "
            "WHERE id = ?",
            (_now(), contradiction_id),
        )
        await self._db.commit()
        dismissed = await self._get_by_id(contradiction_id)
        assert dismissed is not None
        return dismissed

    async def get_contradiction(self, contradiction_id: str) -> dict | None:
        """Return a contradiction by id."""
        return await self._get_by_id(contradiction_id)

    async def list_open(self, workspace_id: str) -> list[dict]:
        """Return open contradictions for a workspace."""
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM contradictions "
            "WHERE workspace_id = ? AND status = 'open' "
            "ORDER BY detected_at DESC",
            (workspace_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def list_by_severity(self, workspace_id: str, severity: str) -> list[dict]:
        """Return contradictions in a workspace by severity, regardless of status."""
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM contradictions "
            "WHERE workspace_id = ? AND severity = ? "
            "ORDER BY detected_at DESC",
            (workspace_id, severity),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def get_for_subject(self, workspace_id: str, subject: str) -> list[dict]:
        """Return open contradictions for a single subject."""
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM contradictions "
            "WHERE workspace_id = ? AND subject_node = ? AND status = 'open' "
            "ORDER BY detected_at DESC",
            (workspace_id, subject),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def get_for_subject_batch(
        self, workspace_id: str, subject_ids: list[str]
    ) -> dict[str, list[dict]]:
        """Return open contradictions keyed by every requested subject id."""
        if not subject_ids:
            return {}
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        placeholders = ",".join("?" * len(subject_ids))
        sql = (
            "SELECT * FROM contradictions "
            "WHERE workspace_id = ? AND status = 'open' "
            f"AND subject_node IN ({placeholders}) "
            "ORDER BY detected_at DESC"
        )
        args: tuple = (workspace_id, *subject_ids)
        async with self._db.execute(sql, args) as cur:
            rows = await cur.fetchall()

        out: dict[str, list[dict]] = {subject: [] for subject in subject_ids}
        for row in rows:
            item = dict(row)
            out[item["subject_node"]].append(item)
        return out

    async def _get_by_id(self, contradiction_id: str) -> dict | None:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM contradictions WHERE id = ?", (contradiction_id,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None
