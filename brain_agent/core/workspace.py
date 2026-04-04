from __future__ import annotations
from dataclasses import dataclass
from brain_agent.core.signals import Signal

DEFAULT_IGNITION_THRESHOLD = 0.3


@dataclass
class _Candidate:
    signal: Signal
    salience: float
    goal_relevance: float
    score: float = 0.0


class GlobalWorkspace:
    def __init__(self, ignition_threshold: float = DEFAULT_IGNITION_THRESHOLD):
        self._threshold = ignition_threshold
        self._candidates: list[_Candidate] = []

    def submit(self, signal: Signal, salience: float = 0.5, goal_relevance: float = 0.5):
        score = self._compute_score(signal, salience, goal_relevance)
        self._candidates.append(_Candidate(
            signal=signal, salience=salience,
            goal_relevance=goal_relevance, score=score,
        ))

    def compete(self) -> Signal | None:
        if not self._candidates:
            return None
        self._candidates.sort(key=lambda c: (-c.score, c.signal.timestamp))
        winner = self._candidates[0]
        self._candidates.clear()
        return winner.signal if winner.score >= self._threshold else None

    def _compute_score(self, signal: Signal, salience: float, goal_relevance: float) -> float:
        arousal = signal.emotional_tag.arousal if signal.emotional_tag else 0.0
        return salience * 0.4 + arousal * 0.3 + goal_relevance * 0.3
