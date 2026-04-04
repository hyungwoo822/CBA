from __future__ import annotations
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field

ALPHA_EVENT = 1.0
BETA_SESSION = 5.0
GAMMA_IDLE = 0.1
IDLE_MINOR = 1800
IDLE_MAJOR = 86400


def _classify_idle(seconds: float) -> float:
    if seconds < IDLE_MINOR:
        return 0.0
    elif seconds < IDLE_MAJOR:
        return 1.0
    else:
        return math.log(seconds / IDLE_MINOR)


@dataclass
class TemporalModel:
    interaction_count: int = 0
    current_session_id: str = ""
    _closed_sessions: list[str] = field(default_factory=list)
    _last_wall_clock: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def tick(self) -> int:
        self.interaction_count += 1
        self._last_wall_clock = datetime.now(timezone.utc)
        return self.interaction_count

    def start_session(self, session_id: str) -> None:
        self.current_session_id = session_id

    def close_session(self) -> None:
        if self.current_session_id:
            self._closed_sessions.append(self.current_session_id)
            self.current_session_id = ""

    def count_sessions_since(self, session_id: str) -> int:
        try:
            idx = self._closed_sessions.index(session_id)
            return len(self._closed_sessions) - idx
        except ValueError:
            return len(self._closed_sessions)

    def distance(self, last_interaction: int, last_session: str, last_wall_clock: datetime | None = None) -> float:
        event_gap = max(0, self.interaction_count - last_interaction)
        session_gap = self.count_sessions_since(last_session)
        idle_factor = 0.0
        if last_wall_clock is not None:
            seconds = (self._last_wall_clock - last_wall_clock).total_seconds()
            idle_factor = _classify_idle(max(0.0, seconds))
        return ALPHA_EVENT * event_gap + BETA_SESSION * session_gap + GAMMA_IDLE * idle_factor
