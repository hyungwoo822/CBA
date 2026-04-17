"""Workspace-scoped ontology registry with confidence lifecycle."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import aiosqlite

from brain_agent.memory.ontology_seed import (
    UNIVERSAL_NODE_TYPES,
    UNIVERSAL_RELATION_TYPES,
    UNIVERSAL_WORKSPACE_ID,
)


CONFIDENCE_TIERS = ("PROVISIONAL", "STABLE", "CANONICAL", "USER_GROUND_TRUTH")
PROMOTION_THRESHOLD_N = 3


def _tier_rank(tier: str) -> int:
    if tier not in CONFIDENCE_TIERS:
        raise ValueError(f"Unknown confidence tier: {tier}")
    return CONFIDENCE_TIERS.index(tier)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OntologyStore:
    """SQLite-backed node/relation type registry plus proposal queue."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS node_types (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                parent_type_id TEXT,
                schema TEXT NOT NULL DEFAULT '{}',
                decay_override TEXT,
                confidence TEXT NOT NULL DEFAULT 'PROVISIONAL'
                    CHECK (confidence IN (
                        'PROVISIONAL', 'STABLE', 'CANONICAL',
                        'USER_GROUND_TRUTH'
                    )),
                occurrence_count INTEGER NOT NULL DEFAULT 1,
                source_snippet TEXT DEFAULT '',
                source_id TEXT DEFAULT 'seed',
                created_at TEXT NOT NULL,
                promoted_at TEXT,
                UNIQUE(workspace_id, name)
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS relation_types (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                domain_type_id TEXT,
                range_type_id TEXT,
                transitive INTEGER NOT NULL DEFAULT 0,
                symmetric INTEGER NOT NULL DEFAULT 0,
                inverse_of TEXT,
                confidence TEXT NOT NULL DEFAULT 'PROVISIONAL'
                    CHECK (confidence IN (
                        'PROVISIONAL', 'STABLE', 'CANONICAL',
                        'USER_GROUND_TRUTH'
                    )),
                occurrence_count INTEGER NOT NULL DEFAULT 1,
                source_snippet TEXT DEFAULT '',
                source_id TEXT DEFAULT 'seed',
                created_at TEXT NOT NULL,
                promoted_at TEXT,
                UNIQUE(workspace_id, name)
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_ontology_proposals (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('node_type', 'relation_type')),
                proposed_name TEXT NOT NULL,
                definition TEXT NOT NULL DEFAULT '{}',
                proposed_by TEXT NOT NULL DEFAULT 'llm:extractor',
                confidence TEXT NOT NULL,
                source_input TEXT DEFAULT '',
                proposed_at TEXT NOT NULL,
                approved_by TEXT,
                approved_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected')),
                UNIQUE(workspace_id, kind, proposed_name)
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_node_types_workspace "
            "ON node_types(workspace_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_types_workspace "
            "ON relation_types(workspace_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_node_types_confidence "
            "ON node_types(workspace_id, confidence)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_types_confidence "
            "ON relation_types(workspace_id, confidence)"
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def seed_universal(self, workspace_id: str = UNIVERSAL_WORKSPACE_ID) -> None:
        """Insert universal ontology rows idempotently."""
        assert self._db is not None
        now = _now()
        name_to_id: dict[str, str] = {}

        for node_type in UNIVERSAL_NODE_TYPES:
            existing = await self._get_node_type_by_name(
                workspace_id, node_type["name"]
            )
            if existing:
                name_to_id[node_type["name"]] = existing["id"]
                continue
            new_id = str(uuid.uuid4())
            await self._db.execute(
                """
                INSERT OR IGNORE INTO node_types
                (id, workspace_id, name, schema, confidence, occurrence_count,
                 source_id, created_at, promoted_at)
                VALUES (?, ?, ?, ?, 'CANONICAL', 1, 'seed', ?, ?)
                """,
                (
                    new_id,
                    workspace_id,
                    node_type["name"],
                    json.dumps(node_type["schema"]),
                    now,
                    now,
                ),
            )
            name_to_id[node_type["name"]] = new_id
        await self._db.commit()

        for node_type in UNIVERSAL_NODE_TYPES:
            parent_name = node_type.get("parent")
            if parent_name is None:
                continue
            await self._db.execute(
                "UPDATE node_types SET parent_type_id = ? WHERE id = ?",
                (name_to_id[parent_name], name_to_id[node_type["name"]]),
            )
        await self._db.commit()

        relation_name_to_id: dict[str, str] = {}
        for relation_type in UNIVERSAL_RELATION_TYPES:
            existing = await self._get_relation_type_by_name(
                workspace_id, relation_type["name"]
            )
            if existing:
                relation_name_to_id[relation_type["name"]] = existing["id"]
                continue
            new_id = str(uuid.uuid4())
            domain_id = (
                name_to_id.get(relation_type["domain"])
                if relation_type.get("domain")
                else None
            )
            range_id = (
                name_to_id.get(relation_type["range"])
                if relation_type.get("range")
                else None
            )
            await self._db.execute(
                """
                INSERT OR IGNORE INTO relation_types
                (id, workspace_id, name, domain_type_id, range_type_id,
                 transitive, symmetric, confidence, occurrence_count,
                 source_id, created_at, promoted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'CANONICAL', 1, 'seed', ?, ?)
                """,
                (
                    new_id,
                    workspace_id,
                    relation_type["name"],
                    domain_id,
                    range_id,
                    int(relation_type.get("transitive", False)),
                    int(relation_type.get("symmetric", False)),
                    now,
                    now,
                ),
            )
            relation_name_to_id[relation_type["name"]] = new_id
        await self._db.commit()

        for relation_type in UNIVERSAL_RELATION_TYPES:
            inverse_name = relation_type.get("inverse_of")
            if inverse_name is None:
                continue
            inverse_id = relation_name_to_id.get(inverse_name)
            self_id = relation_name_to_id.get(relation_type["name"])
            if inverse_id and self_id:
                await self._db.execute(
                    "UPDATE relation_types SET inverse_of = ? WHERE id = ?",
                    (inverse_id, self_id),
                )
        await self._db.commit()

    async def get_node_types(self, workspace_id: str) -> list[dict]:
        """Return universal and workspace-local node types."""
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        if workspace_id == UNIVERSAL_WORKSPACE_ID:
            sql = (
                "SELECT * FROM node_types WHERE workspace_id = ? "
                "ORDER BY name"
            )
            args: tuple = (UNIVERSAL_WORKSPACE_ID,)
        else:
            sql = (
                "SELECT * FROM node_types WHERE workspace_id IN (?, ?) "
                "ORDER BY CASE WHEN workspace_id = ? THEN 0 ELSE 1 END, name"
            )
            args = (UNIVERSAL_WORKSPACE_ID, workspace_id, UNIVERSAL_WORKSPACE_ID)
        async with self._db.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    async def get_relation_types(self, workspace_id: str) -> list[dict]:
        """Return universal and workspace-local relation types."""
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        if workspace_id == UNIVERSAL_WORKSPACE_ID:
            sql = (
                "SELECT * FROM relation_types WHERE workspace_id = ? "
                "ORDER BY name"
            )
            args: tuple = (UNIVERSAL_WORKSPACE_ID,)
        else:
            sql = (
                "SELECT * FROM relation_types WHERE workspace_id IN (?, ?) "
                "ORDER BY CASE WHEN workspace_id = ? THEN 0 ELSE 1 END, name"
            )
            args = (UNIVERSAL_WORKSPACE_ID, workspace_id, UNIVERSAL_WORKSPACE_ID)
        async with self._db.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    async def register_node_type(
        self,
        workspace_id: str,
        name: str,
        parent_name: str | None = None,
        schema: dict | None = None,
        decay_override: str | None = None,
        source_id: str = "llm:extractor",
        source_snippet: str = "",
    ) -> dict:
        existing = await self._get_node_type_by_name(workspace_id, name)
        if existing:
            return existing

        parent_type_id: str | None = None
        if parent_name is not None:
            parent = await self.resolve_node_type(workspace_id, parent_name)
            if parent is None:
                raise ValueError(f"Parent type not found: {parent_name}")
            parent_type_id = parent["id"]

        assert self._db is not None
        now = _now()
        await self._db.execute(
            """
            INSERT INTO node_types
            (id, workspace_id, name, parent_type_id, schema, decay_override,
             confidence, occurrence_count, source_id, source_snippet,
             created_at, promoted_at)
            VALUES (?, ?, ?, ?, ?, ?, 'PROVISIONAL', 1, ?, ?, ?, NULL)
            """,
            (
                str(uuid.uuid4()),
                workspace_id,
                name,
                parent_type_id,
                json.dumps(schema or {}),
                decay_override,
                source_id,
                source_snippet,
                now,
            ),
        )
        await self._db.commit()
        created = await self._get_node_type_by_name(workspace_id, name)
        assert created is not None
        return created

    async def register_relation_type(
        self,
        workspace_id: str,
        name: str,
        domain_type: str | None = None,
        range_type: str | None = None,
        transitive: bool = False,
        symmetric: bool = False,
        source_id: str = "llm:extractor",
        source_snippet: str = "",
    ) -> dict:
        existing = await self._get_relation_type_by_name(workspace_id, name)
        if existing:
            return existing

        domain_id: str | None = None
        range_id: str | None = None
        if domain_type is not None:
            domain = await self.resolve_node_type(workspace_id, domain_type)
            if domain is None:
                raise ValueError(f"Domain type not found: {domain_type}")
            domain_id = domain["id"]
        if range_type is not None:
            range_node = await self.resolve_node_type(workspace_id, range_type)
            if range_node is None:
                raise ValueError(f"Range type not found: {range_type}")
            range_id = range_node["id"]

        assert self._db is not None
        now = _now()
        await self._db.execute(
            """
            INSERT INTO relation_types
            (id, workspace_id, name, domain_type_id, range_type_id,
             transitive, symmetric, confidence, occurrence_count,
             source_id, source_snippet, created_at, promoted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PROVISIONAL', 1, ?, ?, ?, NULL)
            """,
            (
                str(uuid.uuid4()),
                workspace_id,
                name,
                domain_id,
                range_id,
                int(transitive),
                int(symmetric),
                source_id,
                source_snippet,
                now,
            ),
        )
        await self._db.commit()
        created = await self._get_relation_type_by_name(workspace_id, name)
        assert created is not None
        return created

    async def resolve_node_type(self, workspace_id: str, name: str) -> dict | None:
        """Resolve local node type first, then universal fallback."""
        local = await self._get_node_type_by_name(workspace_id, name)
        if local:
            return local
        if workspace_id == UNIVERSAL_WORKSPACE_ID:
            return None
        return await self._get_node_type_by_name(UNIVERSAL_WORKSPACE_ID, name)

    async def resolve_relation_type(self, workspace_id: str, name: str) -> dict | None:
        """Resolve local relation type first, then universal fallback."""
        local = await self._get_relation_type_by_name(workspace_id, name)
        if local:
            return local
        if workspace_id == UNIVERSAL_WORKSPACE_ID:
            return None
        return await self._get_relation_type_by_name(UNIVERSAL_WORKSPACE_ID, name)

    async def increment_occurrence(self, type_id: str) -> dict:
        """Increment occurrence_count and auto-promote PROVISIONAL to STABLE."""
        assert self._db is not None
        table = await self._locate_table(type_id)
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            f"SELECT * FROM {table} WHERE id = ?", (type_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise ValueError(f"Type not found: {type_id}")

        new_count = row["occurrence_count"] + 1
        new_confidence = row["confidence"]
        promoted_at = row["promoted_at"]
        promoted = False
        if new_confidence == "PROVISIONAL" and new_count >= PROMOTION_THRESHOLD_N:
            new_confidence = "STABLE"
            promoted_at = _now()
            promoted = True

        await self._db.execute(
            f"UPDATE {table} SET occurrence_count = ?, confidence = ?, "
            "promoted_at = ? WHERE id = ?",
            (new_count, new_confidence, promoted_at, type_id),
        )
        await self._db.commit()
        return {
            "confidence": new_confidence,
            "occurrence_count": new_count,
            "promoted": promoted,
        }

    async def promote_confidence(
        self, type_id: str, to_level: str, promoted_by: str = "system"
    ) -> None:
        """Promote a node or relation type to a higher confidence tier."""
        _tier_rank(to_level)
        assert self._db is not None
        table = await self._locate_table(type_id)
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            f"SELECT confidence FROM {table} WHERE id = ?", (type_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise ValueError(f"Type not found: {type_id}")
        current = row["confidence"]
        if _tier_rank(to_level) < _tier_rank(current):
            raise ValueError(f"Cannot promote backwards: {current} -> {to_level}")
        await self._db.execute(
            f"UPDATE {table} SET confidence = ?, promoted_at = ? WHERE id = ?",
            (to_level, _now(), type_id),
        )
        await self._db.commit()

    async def resolve_type_or_fallback(
        self,
        workspace_id: str,
        name: str,
        min_confidence: str = "PROVISIONAL",
    ) -> dict:
        """Resolve a node type or fall back to universal Concept."""
        _tier_rank(min_confidence)
        candidate = await self.resolve_node_type(workspace_id, name)
        if candidate and _tier_rank(candidate["confidence"]) >= _tier_rank(
            min_confidence
        ):
            return candidate
        concept = await self._get_node_type_by_name(
            UNIVERSAL_WORKSPACE_ID, "Concept"
        )
        if concept is None:
            raise RuntimeError("Universal seed missing")
        return concept

    async def propose_node_type(
        self,
        workspace_id: str,
        name: str,
        definition: dict,
        confidence: str,
        source_input: str = "",
        proposed_by: str = "llm:extractor",
    ) -> dict:
        return await self._propose(
            workspace_id,
            "node_type",
            name,
            definition,
            confidence,
            source_input,
            proposed_by,
        )

    async def propose_relation_type(
        self,
        workspace_id: str,
        name: str,
        definition: dict,
        confidence: str,
        source_input: str = "",
        proposed_by: str = "llm:extractor",
    ) -> dict:
        return await self._propose(
            workspace_id,
            "relation_type",
            name,
            definition,
            confidence,
            source_input,
            proposed_by,
        )

    async def approve_proposal(
        self, proposal_id: str, approved_by: str = "user"
    ) -> dict:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM pending_ontology_proposals WHERE id = ?",
            (proposal_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise ValueError(f"Proposal not found: {proposal_id}")
        if row["status"] != "pending":
            raise ValueError(
                f"Proposal {proposal_id} is not pending (status={row['status']})"
            )

        definition = json.loads(row["definition"]) if row["definition"] else {}
        workspace_id = row["workspace_id"]
        if row["kind"] == "node_type":
            registered = await self.register_node_type(
                workspace_id=workspace_id,
                name=row["proposed_name"],
                parent_name=definition.get("parent"),
                schema=definition.get("schema", definition),
                source_id=f"proposal:{proposal_id}",
                source_snippet=row["source_input"] or "",
            )
        else:
            registered = await self.register_relation_type(
                workspace_id=workspace_id,
                name=row["proposed_name"],
                domain_type=definition.get("domain"),
                range_type=definition.get("range"),
                transitive=bool(definition.get("transitive", False)),
                symmetric=bool(definition.get("symmetric", False)),
                source_id=f"proposal:{proposal_id}",
                source_snippet=row["source_input"] or "",
            )
        await self.promote_confidence(
            registered["id"], "CANONICAL", promoted_by=approved_by
        )
        await self._db.execute(
            """
            UPDATE pending_ontology_proposals
            SET status = 'approved', approved_by = ?, approved_at = ?
            WHERE id = ?
            """,
            (approved_by, _now(), proposal_id),
        )
        await self._db.commit()
        async with self._db.execute(
            "SELECT * FROM pending_ontology_proposals WHERE id = ?",
            (proposal_id,),
        ) as cur:
            updated = await cur.fetchone()
        return dict(updated)

    async def reject_proposal(self, proposal_id: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE pending_ontology_proposals SET status = 'rejected' WHERE id = ?",
            (proposal_id,),
        )
        await self._db.commit()

    async def list_pending(self, workspace_id: str) -> list[dict]:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            """
            SELECT * FROM pending_ontology_proposals
            WHERE workspace_id = ? AND status = 'pending'
            ORDER BY proposed_at
            """,
            (workspace_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def resolve_parent_chain(self, type_id: str) -> list[dict]:
        """Return type, parent, grandparent, and so on while detecting cycles."""
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        chain: list[dict] = []
        seen: set[str] = set()
        current_id: str | None = type_id
        while current_id is not None:
            if current_id in seen:
                raise ValueError(f"Cycle detected in type hierarchy at {current_id}")
            seen.add(current_id)
            async with self._db.execute(
                "SELECT * FROM node_types WHERE id = ?", (current_id,)
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                break
            chain.append(self._row_to_dict(row))
            current_id = row["parent_type_id"]
        return chain

    async def validate_node_properties(
        self, type_id: str, properties: dict
    ) -> tuple[bool, list[str]]:
        """Validate required properties from the node type schema."""
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT schema FROM node_types WHERE id = ?", (type_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return False, [f"Type not found: {type_id}"]
        try:
            schema = json.loads(row["schema"]) if row["schema"] else {}
        except json.JSONDecodeError:
            schema = {}
        required = schema.get("required", []) or []
        missing = [prop for prop in required if prop not in properties]
        if missing:
            return False, [f"missing required property: {prop}" for prop in missing]
        return True, []

    async def _propose(
        self,
        workspace_id: str,
        kind: str,
        name: str,
        definition: dict,
        confidence: str,
        source_input: str,
        proposed_by: str,
    ) -> dict:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            """
            SELECT * FROM pending_ontology_proposals
            WHERE workspace_id = ? AND kind = ? AND proposed_name = ?
            """,
            (workspace_id, kind, name),
        ) as cur:
            existing = await cur.fetchone()
        if existing is not None:
            return dict(existing)

        proposal_id = str(uuid.uuid4())
        await self._db.execute(
            """
            INSERT INTO pending_ontology_proposals
            (id, workspace_id, kind, proposed_name, definition,
             proposed_by, confidence, source_input, proposed_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                proposal_id,
                workspace_id,
                kind,
                name,
                json.dumps(definition),
                proposed_by,
                confidence,
                source_input,
                _now(),
            ),
        )
        await self._db.commit()
        async with self._db.execute(
            "SELECT * FROM pending_ontology_proposals WHERE id = ?",
            (proposal_id,),
        ) as cur:
            row = await cur.fetchone()
        return dict(row)

    async def _locate_table(self, type_id: str) -> str:
        assert self._db is not None
        async with self._db.execute(
            """
            SELECT 'node_types' FROM node_types WHERE id = ?
            UNION ALL
            SELECT 'relation_types' FROM relation_types WHERE id = ?
            """,
            (type_id, type_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise ValueError(f"Type not found: {type_id}")
        return row[0]

    async def _get_node_type_by_name(
        self, workspace_id: str, name: str
    ) -> dict | None:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM node_types WHERE workspace_id = ? AND name = ?",
            (workspace_id, name),
        ) as cur:
            row = await cur.fetchone()
        return self._row_to_dict(row) if row else None

    async def _get_relation_type_by_name(
        self, workspace_id: str, name: str
    ) -> dict | None:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM relation_types WHERE workspace_id = ? AND name = ?",
            (workspace_id, name),
        ) as cur:
            row = await cur.fetchone()
        return self._row_to_dict(row) if row else None

    @staticmethod
    def _row_to_dict(row) -> dict:
        data = dict(row)
        if "schema" in data and isinstance(data["schema"], str):
            try:
                data["schema"] = json.loads(data["schema"])
            except json.JSONDecodeError:
                data["schema"] = {}
        return data
