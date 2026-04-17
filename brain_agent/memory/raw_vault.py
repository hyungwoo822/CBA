"""Content-addressed lossless storage for original inputs."""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from datetime import datetime, timezone

import aiosqlite

logger = logging.getLogger(__name__)

VAULT_SIZE_THRESHOLD = 10 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_id_for_sha256(sha256: str) -> str:
    """Return the deterministic source id for a SHA256 hex digest."""
    if len(sha256) < 24:
        raise ValueError(f"sha256 too short: {sha256}")
    return "src_" + sha256[:24]


def _sha256_of_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class RawVault:
    """SQLite-backed source registry plus on-disk blob vault."""

    def __init__(self, db_path: str, data_dir: str):
        self._db_path = db_path
        self._data_dir = data_dir
        self._vault_dir = os.path.join(data_dir, "vault")
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        os.makedirs(self._vault_dir, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                workspace_id TEXT,
                kind TEXT NOT NULL,
                uri TEXT,
                sha256 TEXT,
                vault_path TEXT,
                mime_type TEXT,
                original_filename TEXT,
                extracted_text TEXT,
                byte_size INTEGER,
                integrity_valid INTEGER DEFAULT 1,
                last_verified TEXT,
                ingested_at TEXT
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_workspace "
            "ON sources(workspace_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_sha256 ON sources(sha256)"
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _vault_path_for(self, sha256: str) -> str:
        return os.path.join(self._vault_dir, sha256[:2], sha256)

    async def ingest(
        self,
        workspace_id: str,
        kind: str,
        data: bytes | None = None,
        path: str | None = None,
        mime: str = "",
        filename: str = "",
        extracted_text: str = "",
        uri: str | None = None,
    ) -> dict:
        """Ingest an original input and return its source row.

        `data` and `path` are mutually exclusive raw byte sources. When neither
        is supplied, `extracted_text` is treated as the original text payload.
        If data/path is supplied, `extracted_text` is stored only as metadata.
        """
        if data is not None and path is not None:
            raise ValueError("ingest() accepts only one of: data= or path=")
        if data is None and path is None and not extracted_text:
            raise ValueError(
                "ingest() requires one of: data=, path=, or extracted_text="
            )
        assert self._db is not None

        if data is not None:
            raw_bytes = data
            byte_size = len(raw_bytes)
            sha256 = _sha256_of_bytes(raw_bytes)
            source_uri = uri
        elif path is not None:
            raw_bytes = None
            byte_size = os.path.getsize(path)
            sha256 = _sha256_of_file(path)
            source_uri = uri or path
        else:
            raw_bytes = extracted_text.encode("utf-8")
            byte_size = len(raw_bytes)
            sha256 = _sha256_of_bytes(raw_bytes)
            source_uri = uri

        src_id = source_id_for_sha256(sha256)
        existing = await self.get_source(src_id)
        if existing is not None:
            return existing

        vault_path: str | None = None
        if byte_size < VAULT_SIZE_THRESHOLD:
            vault_path = self._vault_path_for(sha256)
            os.makedirs(os.path.dirname(vault_path), exist_ok=True)
            if raw_bytes is not None:
                with open(vault_path, "wb") as fh:
                    fh.write(raw_bytes)
            else:
                assert path is not None
                with open(path, "rb") as src_fh, open(vault_path, "wb") as dst_fh:
                    shutil.copyfileobj(src_fh, dst_fh, length=1024 * 1024)

        now = _now()
        await self._db.execute(
            """
            INSERT INTO sources
            (id, workspace_id, kind, uri, sha256, vault_path, mime_type,
             original_filename, extracted_text, byte_size, integrity_valid,
             last_verified, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                src_id,
                workspace_id,
                kind,
                source_uri,
                sha256,
                vault_path,
                mime,
                filename,
                extracted_text,
                byte_size,
                now,
                now,
            ),
        )
        await self._db.commit()
        source = await self.get_source(src_id)
        assert source is not None
        return source

    async def get_source(self, source_id: str) -> dict | None:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def list_sources(self, workspace_id: str | None = "personal") -> list[dict]:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        if workspace_id is None:
            sql = "SELECT * FROM sources ORDER BY ingested_at"
            args: tuple = ()
        else:
            sql = "SELECT * FROM sources WHERE workspace_id = ? ORDER BY ingested_at"
            args = (workspace_id,)
        async with self._db.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def find_by_sha256(self, sha256: str) -> dict | None:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM sources WHERE sha256 = ?", (sha256,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def get_raw_bytes(self, source_id: str) -> bytes | None:
        """Return the original bytes for a source, if still reachable."""
        src = await self.get_source(source_id)
        if src is None:
            return None
        if src["vault_path"]:
            try:
                with open(src["vault_path"], "rb") as fh:
                    return fh.read()
            except OSError as exc:
                logger.warning("Vault read failed for %s: %s", source_id, exc)
                return None
        if src["uri"]:
            try:
                with open(src["uri"], "rb") as fh:
                    return fh.read()
            except OSError as exc:
                logger.warning(
                    "Pointer read failed for %s (uri=%s): %s",
                    source_id,
                    src["uri"],
                    exc,
                )
                return None
        return None

    async def verify_integrity(self, source_id: str) -> bool:
        """Re-hash stored bytes, update integrity_valid, and return validity."""
        src = await self.get_source(source_id)
        if src is None:
            return False

        actual: str | None = None
        try:
            if src["vault_path"] and os.path.exists(src["vault_path"]):
                actual = _sha256_of_file(src["vault_path"])
            elif src["uri"] and os.path.exists(src["uri"]):
                actual = _sha256_of_file(src["uri"])
        except OSError as exc:
            logger.warning("Integrity hash failed for %s: %s", source_id, exc)

        valid = actual is not None and actual == src["sha256"]
        assert self._db is not None
        await self._db.execute(
            "UPDATE sources SET integrity_valid = ?, last_verified = ? WHERE id = ?",
            (1 if valid else 0, _now(), source_id),
        )
        await self._db.commit()
        return valid
