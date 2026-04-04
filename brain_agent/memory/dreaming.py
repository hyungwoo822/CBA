"""Dreaming Engine — recall-based memory promotion to MEMORY.md.

Inspired by openclaw's short-term-promotion system. During "dreaming" phases,
the engine analyzes which memories have been recalled frequently and promotes
high-value candidates to long-term storage (MEMORY.md).

This implements the neuroscience concept of memory consolidation during sleep:
memories that are actively retrieved (recalled) during waking hours are
selectively strengthened during offline periods.

Architecture:
  1. RecallTracker — logs every retrieval hit (query, score, source)
  2. DreamingEngine — periodically scores recall entries and promotes
     high-scoring candidates to MEMORY.md via narrative consolidation

References:
  - Diekelmann & Born (2010): Memory consolidation during sleep
  - Rasch & Born (2013): About sleep's role in memory
  - openclaw/extensions/memory-core/src/short-term-promotion.ts
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
DAY_SECONDS = 86_400
DEFAULT_RECENCY_HALF_LIFE_DAYS = 14
MAX_RECALL_DAYS = 16
MAX_QUERY_HASHES = 32

# Scoring weights (openclaw defaults, 6 components summing to 1.0)
DEFAULT_WEIGHTS = {
    "frequency": 0.24,
    "relevance": 0.30,
    "diversity": 0.15,
    "recency": 0.15,
    "consolidation": 0.10,
    "conceptual": 0.06,
}

# Dreaming mode presets
DREAMING_PRESETS: dict[str, dict[str, Any]] = {
    "off": {"enabled": False},
    "core": {
        "enabled": True,
        "min_score": 0.75,
        "min_recall_count": 3,
        "min_unique_queries": 2,
        "check_interval_turns": 10,
    },
    "rem": {
        "enabled": True,
        "min_score": 0.85,
        "min_recall_count": 4,
        "min_unique_queries": 3,
        "check_interval_turns": 6,
    },
    "deep": {
        "enabled": True,
        "min_score": 0.80,
        "min_recall_count": 3,
        "min_unique_queries": 3,
        "check_interval_turns": 8,
    },
}

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
_DREAMS_DIR = os.path.join(_DATA_DIR, "memory", ".dreams")
_RECALL_STORE_PATH = os.path.join(_DREAMS_DIR, "short-term-recall.json")


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _query_hash(query: str) -> str:
    return hashlib.sha1(query.strip().lower().encode()).hexdigest()[:12]


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── RecallEntry ──────────────────────────────────────────────────
class RecallEntry:
    """Single memory chunk that has been recalled at least once."""

    __slots__ = (
        "key", "content_preview", "source", "recall_count",
        "total_score", "max_score", "first_recalled_at",
        "last_recalled_at", "query_hashes", "recall_days",
        "concept_tags", "promoted_at",
    )

    def __init__(
        self,
        key: str,
        content_preview: str = "",
        source: str = "memory",
    ):
        self.key = key
        self.content_preview = content_preview[:200]
        self.source = source
        self.recall_count: int = 0
        self.total_score: float = 0.0
        self.max_score: float = 0.0
        self.first_recalled_at: str = ""
        self.last_recalled_at: str = ""
        self.query_hashes: list[str] = []
        self.recall_days: list[str] = []
        self.concept_tags: list[str] = []
        self.promoted_at: str | None = None

    def record_recall(self, query: str, score: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        today = _today_str()

        self.recall_count += 1
        self.total_score += score
        self.max_score = max(self.max_score, score)
        if not self.first_recalled_at:
            self.first_recalled_at = now
        self.last_recalled_at = now

        qh = _query_hash(query)
        if qh not in self.query_hashes:
            self.query_hashes.append(qh)
            if len(self.query_hashes) > MAX_QUERY_HASHES:
                self.query_hashes = self.query_hashes[-MAX_QUERY_HASHES:]

        if today not in self.recall_days:
            self.recall_days.append(today)
            if len(self.recall_days) > MAX_RECALL_DAYS:
                self.recall_days = self.recall_days[-MAX_RECALL_DAYS:]

    @property
    def unique_queries(self) -> int:
        return len(self.query_hashes)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(1, self.recall_count)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "content_preview": self.content_preview,
            "source": self.source,
            "recall_count": self.recall_count,
            "total_score": self.total_score,
            "max_score": self.max_score,
            "first_recalled_at": self.first_recalled_at,
            "last_recalled_at": self.last_recalled_at,
            "query_hashes": self.query_hashes,
            "recall_days": self.recall_days,
            "concept_tags": self.concept_tags,
            "promoted_at": self.promoted_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RecallEntry:
        entry = cls(
            key=data["key"],
            content_preview=data.get("content_preview", ""),
            source=data.get("source", "memory"),
        )
        entry.recall_count = data.get("recall_count", 0)
        entry.total_score = data.get("total_score", 0.0)
        entry.max_score = data.get("max_score", 0.0)
        entry.first_recalled_at = data.get("first_recalled_at", "")
        entry.last_recalled_at = data.get("last_recalled_at", "")
        entry.query_hashes = data.get("query_hashes", [])
        entry.recall_days = data.get("recall_days", [])
        entry.concept_tags = data.get("concept_tags", [])
        entry.promoted_at = data.get("promoted_at", None)
        return entry


# ── RecallTracker ────────────────────────────────────────────────
class RecallTracker:
    """Tracks memory recall events across retrieval operations."""

    def __init__(self) -> None:
        self._entries: dict[str, RecallEntry] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(_RECALL_STORE_PATH):
            return
        try:
            with open(_RECALL_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, entry_data in data.get("entries", {}).items():
                self._entries[key] = RecallEntry.from_dict(entry_data)
            logger.debug("RecallTracker: loaded %d entries", len(self._entries))
        except Exception as e:
            logger.warning("RecallTracker: failed to load store: %s", e)

    def record(self, memory_id: str, content: str, query: str, score: float, source: str = "memory") -> None:
        self._ensure_loaded()
        key = f"{source}:{memory_id}"
        if key not in self._entries:
            self._entries[key] = RecallEntry(key=key, content_preview=content, source=source)
        self._entries[key].record_recall(query, score)

    def save(self) -> None:
        self._ensure_loaded()
        os.makedirs(_DREAMS_DIR, exist_ok=True)
        store = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
        }
        try:
            with open(_RECALL_STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("RecallTracker: failed to save: %s", e)

    def get_entries(self) -> dict[str, RecallEntry]:
        self._ensure_loaded()
        return dict(self._entries)


# ── Scoring Components ───────────────────────────────────────────

def _frequency_component(recall_count: int) -> float:
    """Log-scaled frequency: log1p(count) / log1p(10)."""
    return _clamp(math.log1p(recall_count) / math.log1p(10))


def _relevance_component(entry: RecallEntry) -> float:
    """Average recall score (0-1)."""
    return _clamp(entry.avg_score)


def _diversity_component(unique_queries: int) -> float:
    """Normalized by dividing by 5."""
    return _clamp(unique_queries / 5)


def _recency_component(last_recalled_at: str, half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS) -> float:
    """Exponential decay: exp(-(ln2 / halfLife) * ageDays)."""
    if not last_recalled_at:
        return 0.0
    try:
        last_ts = datetime.fromisoformat(last_recalled_at.replace("Z", "+00:00"))
        age_days = max(0, (datetime.now(timezone.utc) - last_ts).total_seconds() / DAY_SECONDS)
        lam = math.log(2) / half_life_days
        return _clamp(math.exp(-lam * age_days))
    except (ValueError, TypeError):
        return 0.0


def _consolidation_component(recall_days: list[str]) -> float:
    """Reward entries recalled on multiple, spaced days."""
    if len(recall_days) == 0:
        return 0.0
    if len(recall_days) == 1:
        return 0.2

    try:
        parsed = sorted(
            datetime.strptime(d, "%Y-%m-%d").timestamp()
            for d in recall_days
            if d
        )
    except ValueError:
        return 0.2

    if len(parsed) <= 1:
        return 0.2

    span_days = max(0, (parsed[-1] - parsed[0]) / DAY_SECONDS)
    spacing = _clamp(math.log1p(len(parsed) - 1) / math.log1p(4))
    span = _clamp(span_days / 7)
    return _clamp(0.55 * spacing + 0.45 * span)


def _conceptual_component(concept_tags: list[str]) -> float:
    """Normalized by dividing by 6."""
    return _clamp(len(concept_tags) / 6)


# ── DreamingEngine ───────────────────────────────────────────────

class DreamingEngine:
    """Scores recall entries and promotes high-value candidates.

    Runs during Phase 5 consolidation (periodically, not every turn)
    to promote frequently-recalled memories to long-term storage.
    """

    def __init__(
        self,
        tracker: RecallTracker,
        mode: str = "core",
        weights: dict[str, float] | None = None,
    ):
        self._tracker = tracker
        self._mode = mode
        preset = DREAMING_PRESETS.get(mode, DREAMING_PRESETS["core"])
        self._enabled = preset.get("enabled", True)
        self._min_score = preset.get("min_score", 0.75)
        self._min_recall_count = preset.get("min_recall_count", 3)
        self._min_unique_queries = preset.get("min_unique_queries", 2)
        self._check_interval = preset.get("check_interval_turns", 10)
        self._weights = weights or dict(DEFAULT_WEIGHTS)
        self._turns_since_last = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def should_dream(self) -> bool:
        """Check if it's time to run a dreaming cycle."""
        if not self._enabled:
            return False
        self._turns_since_last += 1
        if self._turns_since_last >= self._check_interval:
            self._turns_since_last = 0
            return True
        return False

    def rank_candidates(self, limit: int = 10) -> list[dict]:
        """Score and rank all recall entries, return top candidates.

        Returns list of dicts with keys: key, score, components, entry.
        """
        entries = self._tracker.get_entries()
        candidates = []
        w = self._weights

        for key, entry in entries.items():
            # Gate: already promoted
            if entry.promoted_at:
                continue
            # Gate: minimum recall count
            if entry.recall_count < self._min_recall_count:
                continue
            # Gate: minimum unique queries
            if entry.unique_queries < self._min_unique_queries:
                continue

            # Compute components
            freq = _frequency_component(entry.recall_count)
            rel = _relevance_component(entry)
            div = _diversity_component(entry.unique_queries)
            rec = _recency_component(entry.last_recalled_at)
            con = _consolidation_component(entry.recall_days)
            cpt = _conceptual_component(entry.concept_tags)

            score = (
                w["frequency"] * freq
                + w["relevance"] * rel
                + w["diversity"] * div
                + w["recency"] * rec
                + w["consolidation"] * con
                + w["conceptual"] * cpt
            )

            # Gate: minimum score
            if score < self._min_score:
                continue

            candidates.append({
                "key": key,
                "score": score,
                "components": {
                    "frequency": freq,
                    "relevance": rel,
                    "diversity": div,
                    "recency": rec,
                    "consolidation": con,
                    "conceptual": cpt,
                },
                "entry": entry,
            })

        # Sort by score desc, recall_count desc, key asc
        candidates.sort(key=lambda c: (-c["score"], -c["entry"].recall_count, c["key"]))
        return candidates[:limit]

    def apply_promotions(self, candidates: list[dict]) -> str:
        """Mark candidates as promoted and return markdown snippet for MEMORY.md.

        Returns the text to append to MEMORY.md (empty string if nothing to promote).
        """
        if not candidates:
            return ""

        today = _today_str()
        lines = [f"\n## Promoted From Short-Term Recall ({today})\n"]

        for c in candidates:
            entry: RecallEntry = c["entry"]
            preview = entry.content_preview.replace("\n", " ").strip()
            if len(preview) > 120:
                preview = preview[:117] + "..."
            lines.append(
                f"- {preview} "
                f"[score={c['score']:.3f} recalls={entry.recall_count} "
                f"avg={entry.avg_score:.3f}]"
            )
            entry.promoted_at = datetime.now(timezone.utc).isoformat()

        self._tracker.save()
        result = "\n".join(lines)
        logger.info(
            "Dreaming: promoted %d candidates to MEMORY.md (mode=%s)",
            len(candidates), self._mode,
        )
        return result

    async def run_cycle(self) -> str:
        """Run one complete dreaming cycle.

        Returns the promotion text appended to MEMORY.md (empty if nothing promoted).
        """
        candidates = self.rank_candidates()
        if not candidates:
            logger.debug("Dreaming: no candidates met promotion thresholds")
            return ""
        return self.apply_promotions(candidates)
