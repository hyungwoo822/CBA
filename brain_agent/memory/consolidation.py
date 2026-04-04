from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable, TYPE_CHECKING

import numpy as np

from brain_agent.memory.episodic_store import EpisodicStore
from brain_agent.memory.forgetting import ForgettingEngine
from brain_agent.memory.hippocampal_staging import HippocampalStaging

if TYPE_CHECKING:
    from brain_agent.memory.semantic_store import SemanticStore

logger = logging.getLogger(__name__)

HOMEOSTATIC_FACTOR = 0.95
PRUNING_THRESHOLD = 0.05
EMOTIONAL_BOOST = 0.35


@dataclass
class ConsolidationResult:
    """Summary of a single consolidation cycle."""

    transferred: int = 0
    pruned: int = 0
    semantic_extracted: int = 0
    reflections_generated: int = 0


class ConsolidationEngine:
    """Transfers memories from hippocampal staging to episodic long-term store.

    Applies emotional prioritisation, homeostatic scaling, pruning,
    episodic-to-semantic extraction (Winocur & Moscovitch 2011), and
    reflection (Park et al. 2023) during each consolidation cycle.
    """

    def __init__(
        self,
        staging: HippocampalStaging,
        episodic_store: EpisodicStore,
        forgetting: ForgettingEngine,
        threshold: int = 5,
        semantic_store: "SemanticStore | None" = None,
        pfc_fn: Callable[[str], Awaitable[str | None]] | None = None,
        similarity_fn: Callable[[list[float], list[float]], float] | None = None,
        get_acetylcholine: Callable[[], float] | None = None,
    ):
        self._staging = staging
        self._episodic = episodic_store
        self._forgetting = forgetting
        self._threshold = threshold
        self._semantic = semantic_store
        self._pfc_fn = pfc_fn
        self._similarity_fn = similarity_fn or self._default_cosine
        self._get_ach = get_acetylcholine

    @staticmethod
    def _default_cosine(a: list[float], b: list[float]) -> float:
        """Fallback cosine similarity when no external function is provided."""
        if not a or not b or len(a) != len(b):
            return 0.0
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = float(np.dot(va, vb))
        norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return dot / norm if norm > 0 else 0.0

    async def should_consolidate(self) -> bool:
        """Return True when staging pressure exceeds the threshold."""
        return await self._staging.count_unconsolidated() >= self._threshold

    async def consolidate(self) -> ConsolidationResult:
        """Run one consolidation cycle.

        1. Transfer unconsolidated staging memories to episodic store,
           boosting strength for high-arousal memories.
        2. Apply homeostatic scaling to all existing episodes and prune
           those that fall below the strength threshold.
        3. Episodic-to-semantic transition via PFC (Winocur & Moscovitch 2011).
        4. Reflection — generate higher-level insights (Park et al. 2023).
        """
        result = ConsolidationResult()

        # --- Phase 1: transfer from staging to episodic ---
        # ACh gating (Hasselmo 2006): Low ACh during SWS supports
        # hippocampus→neocortex transfer. High ACh blocks transfer
        # (favors new encoding over consolidation).
        ach_level = self._get_ach() if self._get_ach else 0.3
        # Low ACh (≤0.3): transfer boost up to 1.4x (SWS optimal)
        # High ACh (≥0.7): transfer penalty down to 0.6x (encoding mode)
        # Baseline (0.5): 1.0x (no modulation)
        ach_transfer_factor = 1.0 + 0.8 * (0.5 - ach_level)  # [0.6, 1.4]
        ach_transfer_factor = max(0.6, min(1.4, ach_transfer_factor))

        memories = await self._staging.get_unconsolidated()
        memories.sort(
            key=lambda m: m["emotional_tag"].get("arousal", 0), reverse=True
        )

        for mem in memories:
            strength = mem["strength"] * ach_transfer_factor
            arousal = mem["emotional_tag"].get("arousal", 0)
            if arousal > 0.5:
                strength *= 1.0 + (arousal * EMOTIONAL_BOOST)

            # Enrich episodic content with structured context
            ents = mem["entities"]
            content = mem["content"]
            if isinstance(ents, dict) and ents.get("intent"):
                parts = [content]
                kw = ents.get("keywords", [])
                if kw:
                    parts.append(f"[keywords: {', '.join(kw[:5])}]")
                if arousal > 0.5:
                    valence = mem["emotional_tag"].get("valence", 0)
                    tone = "positive" if valence > 0.2 else "negative" if valence < -0.2 else "neutral"
                    parts.append(f"[emotional: {tone}, arousal={arousal:.1f}]")
                content = "\n".join(parts)

            await self._episodic.save(
                content=content,
                context_embedding=mem["context_embedding"],
                entities=mem["entities"],
                emotional_tag=mem["emotional_tag"],
                interaction_id=mem["last_interaction"],
                session_id=mem["last_session"],
                strength=strength,
                access_count=mem["access_count"],
            )
            await self._staging.mark_consolidated(mem["id"])
            result.transferred += 1

        # --- Phase 2: homeostatic scaling of existing episodes ---
        all_episodes = await self._episodic.get_all()
        for ep in all_episodes:
            new_strength = ep["strength"] * HOMEOSTATIC_FACTOR
            if new_strength < PRUNING_THRESHOLD:
                result.pruned += 1
            else:
                await self._episodic.update_strength(ep["id"], new_strength)

        if result.pruned > 0:
            await self._episodic.delete_below_strength(PRUNING_THRESHOLD)

        # --- Phase 3: Episodic→Semantic transition (Winocur & Moscovitch 2011) ---
        if self._pfc_fn and self._semantic:
            try:
                from brain_agent.memory.semantic_extractor import (
                    find_episode_clusters,
                    build_extraction_prompt,
                    parse_extraction_response,
                )

                recent_episodes = await self._episodic.get_recent(limit=50)
                clusters = find_episode_clusters(
                    recent_episodes, self._similarity_fn,
                )
                for cluster in clusters:
                    try:
                        prompt = build_extraction_prompt(cluster)
                        response = await self._pfc_fn(prompt)
                        if response:
                            fact, relations = parse_extraction_response(response)
                            if fact:
                                await self._semantic.add(
                                    fact, category="extracted_fact",
                                )
                                for rel in relations:
                                    if len(rel) >= 3:
                                        try:
                                            w = float(rel[3]) if len(rel) >= 4 else 0.7
                                        except (ValueError, TypeError):
                                            w = 0.7
                                        cat = rel[4] if len(rel) >= 5 and isinstance(rel[4], str) else "GENERAL"
                                        await self._semantic.add_relationship(
                                            rel[0], rel[1], rel[2], weight=w, category=cat,
                                        )
                                result.semantic_extracted += 1
                    except Exception:
                        logger.debug(
                            "Error during semantic extraction for cluster",
                            exc_info=True,
                        )
            except Exception:
                logger.debug(
                    "Error during episodic→semantic phase", exc_info=True,
                )

        # --- Phase 4: Reflection (Park et al. 2023) ---
        if self._pfc_fn and self._semantic:
            try:
                from brain_agent.memory.reflection import (
                    build_reflection_prompt,
                    parse_insights,
                    MIN_EPISODES_FOR_REFLECTION,
                )

                recent = await self._episodic.get_recent(limit=20)
                if len(recent) >= MIN_EPISODES_FOR_REFLECTION:
                    prompt = build_reflection_prompt(recent)
                    response = await self._pfc_fn(prompt)
                    if response:
                        insights = parse_insights(response)
                        for insight in insights:
                            await self._semantic.add(
                                insight,
                                category="reflection",
                                strength=1.5,
                            )
                            result.reflections_generated += 1
            except Exception:
                logger.debug(
                    "Error during reflection phase", exc_info=True,
                )

        return result
