from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

import aiosqlite
import numpy as np

from brain_agent.core.temporal import TemporalModel

IDLE_SESSION_THRESHOLD = timedelta(minutes=30)
TOPIC_DRIFT_THRESHOLD = 0.3


@dataclass
class Session:
    id: str
    start_interaction: int
    topic_embedding: list[float] | None = None


class SessionManager:
    def __init__(self, db_path: str, embed_fn: Callable[[str], list[float]]):
        self._db_path = db_path
        self._embed_fn = embed_fn
        self.temporal = TemporalModel()
        self._current_session: Session | None = None
        self._db: aiosqlite.Connection | None = None

    async def initialize(self):
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_interaction INTEGER,
                end_interaction INTEGER,
                closed_at TEXT
            )"""
        )
        await self._db.commit()

    async def close(self):
        if self._current_session:
            await self.close_session()
        if self._db:
            await self._db.close()

    async def start_session(self) -> Session:
        sid = str(uuid.uuid4())[:8]
        self.temporal.start_session(sid)
        self._current_session = Session(
            id=sid, start_interaction=self.temporal.interaction_count
        )
        return self._current_session

    async def close_session(self) -> None:
        if self._current_session:
            await self._db.execute(
                "INSERT INTO sessions (id, start_interaction, end_interaction, closed_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    self._current_session.id,
                    self._current_session.start_interaction,
                    self.temporal.interaction_count,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await self._db.commit()
            self.temporal.close_session()
            self._current_session = None

    async def on_interaction(self, input_text: str) -> int:
        count = self.temporal.tick()
        if self._current_session:
            self._current_session.topic_embedding = self._embed_fn(input_text)
        return count

    def should_start_new_session(self, input_text: str) -> bool:
        if self._current_session is None:
            return True
        now = datetime.now(timezone.utc)
        if now - self.temporal._last_wall_clock > IDLE_SESSION_THRESHOLD:
            return True
        if self._current_session.topic_embedding is not None:
            new_emb = self._embed_fn(input_text)
            sim = self._cosine_sim(self._current_session.topic_embedding, new_emb)
            if sim < TOPIC_DRIFT_THRESHOLD:
                return True
        return False

    @property
    def current_session(self) -> Session | None:
        return self._current_session

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = np.dot(va, vb)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(dot / norm) if norm > 0 else 0.0
