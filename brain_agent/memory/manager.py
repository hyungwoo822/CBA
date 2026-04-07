from __future__ import annotations

import asyncio
import os
from typing import Callable

import numpy as np

from brain_agent.memory.sensory_buffer import SensoryBuffer, SensoryItem
from brain_agent.memory.working_memory import WorkingMemory, WorkingMemoryItem
from brain_agent.memory.hippocampal_staging import HippocampalStaging
from brain_agent.memory.episodic_store import EpisodicStore
from brain_agent.memory.semantic_store import SemanticStore
from brain_agent.memory.procedural_store import ProceduralStore
from brain_agent.memory.forgetting import ForgettingEngine
from brain_agent.memory.retrieval import RetrievalEngine
from brain_agent.memory.consolidation import ConsolidationEngine, ConsolidationResult
from brain_agent.memory.brain_state import BrainStateStore
from brain_agent.memory.dreaming import RecallTracker


class MemoryManager:
    """Facade that coordinates all memory subsystems.

    Provides a single entry point for encoding, retrieval, consolidation,
    and lifecycle management across sensory, working, staging, episodic,
    semantic, and procedural stores.
    """

    def __init__(
        self,
        db_dir: str,
        embed_fn: Callable[[str], list[float]],
        working_capacity: int = 4,
        consolidation_threshold: int = 5,
    ):
        self._db_dir = db_dir
        self._embed_fn = embed_fn
        self._interaction_counter = 0
        self._session_id = ""

        # Subsystems
        self.sensory = SensoryBuffer()
        self.working = WorkingMemory(capacity=working_capacity)
        self.staging = HippocampalStaging(
            db_path=os.path.join(db_dir, "staging.db"), embed_fn=embed_fn
        )
        self.episodic = EpisodicStore(
            db_path=os.path.join(db_dir, "episodic.db")
        )
        self.semantic = SemanticStore(
            chroma_path=os.path.join(db_dir, "chroma"),
            graph_db_path=os.path.join(db_dir, "graph.db"),
            embed_fn=embed_fn,
        )
        self.procedural = ProceduralStore(
            db_path=os.path.join(db_dir, "procedural.db"),
        )
        self.brain_state = BrainStateStore(
            db_path=os.path.join(db_dir, "brain_state.db"),
        )
        self.forgetting = ForgettingEngine()
        self.retrieval = RetrievalEngine()
        self.recall_tracker = RecallTracker()
        self._get_cortisol: Callable[[], float] | None = None
        self.consolidation = ConsolidationEngine(
            staging=self.staging,
            episodic_store=self.episodic,
            forgetting=self.forgetting,
            threshold=consolidation_threshold,
            semantic_store=self.semantic,
            pfc_fn=None,  # Set later via set_pfc_fn() when PFC is available
            similarity_fn=self._cosine_sim,
            get_acetylcholine=None,  # Set later via set_neuromodulators()
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        await self.staging.initialize()
        await self.episodic.initialize()
        await self.semantic.initialize()
        await self.procedural.initialize()
        await self.brain_state.initialize()

    async def close(self) -> None:
        await self.staging.close()
        await self.episodic.close()
        await self.semantic.close()
        await self.procedural.close()
        await self.brain_state.close()

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def set_context(self, interaction_id: int, session_id: str) -> None:
        self._interaction_counter = interaction_id
        self._session_id = session_id

    def set_pfc_fn(self, fn: Callable | None) -> None:
        """Inject PFC callback for consolidation phases (episodic→semantic, reflection).

        The callback should be ``async (str) -> str | None``.
        Called by the pipeline once PFC is available.
        """
        self.consolidation._pfc_fn = fn

    def set_neuromodulators(self, get_ach: Callable[[], float]) -> None:
        """Inject neuromodulator accessor for ACh-gated consolidation.

        Hasselmo (2006): Low ACh during SWS supports hippocampus→neocortex
        memory transfer. High ACh during waking favors new encoding.
        """
        self.consolidation._get_ach = get_ach

    def set_cortisol_accessor(self, fn: Callable[[], float]) -> None:
        """Inject cortisol accessor for retrieval inhibition (de Quervain 2000)."""
        self._get_cortisol = fn

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def encode(
        self,
        content: str,
        entities: dict,
        emotional_tag: dict | None = None,
        interaction_id: int | None = None,
        session_id: str | None = None,
        learning_rate: float | None = None,
        modality: str | None = None,
    ) -> str:
        # Hippocampus hemisphere awareness (Milner 1971):
        #   Left hippocampus: verbal/textual encoding
        #   Right hippocampus: visual/spatial encoding
        if modality is not None:
            entities = dict(entities)  # avoid mutating caller's dict
            entities["modality"] = modality

        embedding = self._embed_fn(content)
        mem_id = await self.staging.encode(
            content=content,
            entities=entities,
            interaction_id=interaction_id if interaction_id is not None else self._interaction_counter,
            session_id=session_id or self._session_id,
            emotional_tag=emotional_tag,
        )

        # Learning rate modulates initial encoding strength (Hasselmo 2006 ACh).
        # Higher ACh = more plastic = stronger initial encoding.
        # strength = 1.0 + 0.5 * (learning_rate - 0.5)  =>  range [0.75, 1.25]
        if learning_rate is not None:
            strength = 1.0 + 0.5 * (learning_rate - 0.5)
            await self.staging.update_strength(mem_id, strength)

        # Retroactive interference: weaken similar existing memories
        # (Interference Theory)
        existing = await self.staging.get_unconsolidated()
        for ex in existing:
            if ex["id"] == mem_id:
                continue
            ex_emb = ex.get("context_embedding", [])
            if ex_emb:
                sim = self._cosine_sim(embedding, ex_emb)
                if sim > 0.85:
                    new_strength = self.forgetting.apply_interference(
                        ex["strength"], sim,
                    )
                    await self.staging.update_strength(ex["id"], new_strength)

        return mem_id

    async def consolidate(self) -> ConsolidationResult:
        return await self.consolidation.consolidate()

    async def retrieve_identity(self) -> dict:
        """Retrieve identity facts regardless of query.

        Always returns self_model and user_model facts from the semantic
        knowledge graph. These are "schema" facts that are always relevant,
        unlike episodic memories which are query-dependent.

        References:
          - Ghosh & Gilboa (2014): Schemas are always active in mPFC
          - Frith & Frith (2006): User model always active in TPJ
        """
        self_facts = await self.semantic.get_identity_facts("self_model")
        user_facts = await self.semantic.get_identity_facts("user_model")
        return {
            "self_model": self_facts,
            "user_model": user_facts,
        }

    async def retrieve(
        self,
        query: str,
        context: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve relevant memories using multi-factor scoring.

        Searches episodic and semantic stores, scores candidates with
        RetrievalEngine, applies forgetting (Ebbinghaus retention), and
        returns top-K results sorted by composite score.
        """
        query_embedding = self._embed_fn(query)
        context_embedding = self._embed_fn(context) if context else query_embedding
        candidates: list[dict] = []

        # ── Parallel retrieval: semantic + episodic + spread activation ──
        query_nodes = [w for w in query.split() if len(w) >= 3]

        async def _fetch_semantic() -> list[dict]:
            try:
                return await self.semantic.search(query, top_k=top_k * 2)
            except Exception:
                return []

        async def _fetch_episodic() -> list[dict]:
            try:
                return await self.episodic.get_recent(limit=top_k * 4)
            except Exception:
                return []

        async def _fetch_activation() -> dict[str, float]:
            try:
                if query_nodes:
                    act = await self.semantic.spread_activation(
                        start_nodes=query_nodes, max_hops=2, decay=0.85,
                    )
                    for n in query_nodes:
                        act.pop(n, None)
                    return act
            except Exception:
                pass
            return {}

        semantic_results, recent, activated = await asyncio.gather(
            _fetch_semantic(), _fetch_episodic(), _fetch_activation(),
        )

        # 1. Semantic store — vector similarity
        for mem in semantic_results:
            relevance = 1.0 - min(1.0, mem.get("distance", 0.5) or 0.5)
            candidates.append({
                "id": mem["id"],
                "content": mem["content"],
                "source": "semantic",
                "relevance": relevance,
                "importance": mem.get("metadata", {}).get("strength", 0.5),
                "access_count": int(mem.get("metadata", {}).get("access_count", 0)),
                "recency_distance": 0.0,
                "context_similarity": relevance,
            })

        # 2. Episodic store — recent episodes scored with embedding similarity
        for ep in recent:
            ep_embedding = ep.get("context_embedding", [])
            if ep_embedding and query_embedding:
                relevance = self._cosine_sim(query_embedding, ep_embedding)
                ctx_sim = self._cosine_sim(context_embedding, ep_embedding)
            else:
                relevance = 0.0
                ctx_sim = 0.0

            recency_dist = float(
                max(0, self._interaction_counter - ep.get("last_interaction", 0))
            )
            strength = ep.get("strength", 1.0)

            # Apply Ebbinghaus forgetting — skip effectively forgotten memories
            retention = self.forgetting.retention(recency_dist, strength)
            if retention < 0.01:
                continue

            etag = ep.get("emotional_tag", {})
            arousal = etag.get("arousal", 0.0) if isinstance(etag, dict) else 0.0

            candidates.append({
                "id": ep["id"],
                "content": ep["content"],
                "source": "episodic",
                "relevance": relevance * retention,
                "importance": arousal,
                "access_count": ep.get("access_count", 0),
                "recency_distance": recency_dist,
                "context_similarity": ctx_sim,
            })

        # 3. Spreading activation — parallel search for activated nodes
        if activated:
            nodes_to_search = [(n, lv) for n, lv in activated.items() if lv >= 0.1]

            async def _search_node(node: str, act_level: float) -> list[dict]:
                results = []
                try:
                    for mem in await self.semantic.search(node, top_k=3):
                        relevance = 1.0 - min(1.0, mem.get("distance", 0.5) or 0.5)
                        results.append({
                            "id": mem["id"],
                            "content": mem["content"],
                            "source": "semantic",
                            "relevance": relevance,
                            "importance": mem.get("metadata", {}).get("strength", 0.5),
                            "access_count": int(mem.get("metadata", {}).get("access_count", 0)),
                            "recency_distance": 0.0,
                            "context_similarity": relevance,
                            "activation_boost": act_level,
                        })
                except Exception:
                    pass
                return results

            # 3a. Parallel search for all activated nodes
            node_results = await asyncio.gather(
                *[_search_node(n, lv) for n, lv in nodes_to_search]
            )
            seen_ids = {c["id"] for c in candidates}
            for batch in node_results:
                for mem in batch:
                    if mem["id"] not in seen_ids:
                        seen_ids.add(mem["id"])
                        candidates.append(mem)

            # 3b. Boost existing candidates whose content mentions activated nodes
            for c in candidates:
                if "activation_boost" in c:
                    continue
                content_lower = c["content"].lower()
                boost = 0.0
                for node, act_level in activated.items():
                    if node.lower() in content_lower:
                        boost = max(boost, act_level)
                c["activation_boost"] = boost

        # 4. Score all candidates via RetrievalEngine
        # Cortisol inhibits retrieval (de Quervain 2000)
        cortisol_penalty = 1.0
        if self._get_cortisol:
            cort = self._get_cortisol()
            if cort > 0.7:
                cortisol_penalty = 1.0 - (cort - 0.7) * 1.5  # 0.7→1.0, 0.9→0.7, 1.0→0.55
                cortisol_penalty = max(0.3, cortisol_penalty)  # Floor at 30%

        for c in candidates:
            c["score"] = self.retrieval.compute_score(
                recency_distance=c["recency_distance"],
                relevance=c["relevance"],
                importance=c["importance"],
                access_count=c["access_count"],
                context_similarity=c["context_similarity"],
                activation_boost=c.get("activation_boost", 0.0),
            ) * cortisol_penalty

        # 5. Sort by score, return top-K
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_results = candidates[:top_k]

        # 6. Retrieval-Induced Forgetting — suppress episodic competitors
        #    (Anderson 1994)
        async def _apply_rif(c: dict) -> None:
            ep = await self.episodic.get_by_id(c["id"])
            if ep:
                new_str = self.forgetting.retrieval_induced_forgetting(ep["strength"])
                await self.episodic.update_strength(c["id"], new_str)

        rif_tasks = [
            _apply_rif(c) for c in candidates[top_k:]
            if c["source"] == "episodic" and c["relevance"] > 0.3
        ]
        if rif_tasks:
            await asyncio.gather(*rif_tasks)

        # 7. SM-2 retrieval boost + reconsolidation for retrieved episodic
        #    memories (Wozniak 1990, Nader 2000)
        async def _boost_and_reconsolidate(c: dict) -> None:
            await self.episodic.on_retrieval(c["id"], boost=1.5)
            await self.episodic.reconsolidate(
                c["id"], self._interaction_counter, self._session_id,
            )

        boost_tasks = [
            _boost_and_reconsolidate(c) for c in top_results
            if c["source"] == "episodic"
        ]
        if boost_tasks:
            await asyncio.gather(*boost_tasks)

        # 8. Track recalls for dreaming (Diekelmann & Born 2010)
        for c in top_results:
            self.recall_tracker.record(
                memory_id=c["id"],
                content=c.get("content", ""),
                query=query,
                score=c.get("score", 0.0),
                source=c.get("source", "memory"),
            )
        if top_results:
            self.recall_tracker.save()

        return top_results

    async def store_semantic_facts(
        self,
        entities: list[str],
        relations: list[list[str]],
        facts: list[str] | None = None,
        origin: str = "unknown",
    ) -> None:
        """Immediately store extracted facts and relations in semantic memory.

        Called per-request after PFC entity extraction, rather than waiting
        for consolidation. This ensures semantic memory builds incrementally.

        Parameters
        ----------
        origin : str
            Source of the facts: ``"user_input"``, ``"agent_response"``, or
            ``"unknown"`` (default).
        """
        # Store individual entity concepts as vector documents
        # Skip very short tokens (single chars, particles) — no semantic value
        for entity in entities:
            if entity and len(entity.strip()) >= 2:
                await self.semantic.add(
                    entity.strip(), category="entity", strength=0.9,
                )

        # Store relations in knowledge graph
        # Relations may be [s, r, t], [s, r, t, confidence], or [s, r, t, confidence, category]
        for rel in relations:
            if len(rel) >= 3:
                source, relation, target = rel[0], rel[1], rel[2]
                try:
                    weight = float(rel[3]) if len(rel) >= 4 else 0.8
                except (ValueError, TypeError):
                    weight = 0.8
                weight = max(0.1, min(1.0, weight))
                category = rel[4] if len(rel) >= 5 and isinstance(rel[4], str) else "GENERAL"
                await self.semantic.add_relationship(
                    source, relation, target, weight=weight, category=category,
                    origin=origin,
                )

        # Store fact documents in vector store
        if facts:
            for fact in facts:
                if fact.strip():
                    await self.semantic.add(
                        fact.strip(), category="extracted_fact", strength=1.0,
                    )

    async def update_knowledge_graph(
        self,
        entities: list[str],
        relations: list[list[str]],
    ) -> None:
        """Populate knowledge graph from extracted entities and relations.

        Brain mapping: Hippocampal binding of cortically-extracted relations
        into the semantic knowledge graph (Eichenbaum 2000, Collins & Loftus 1975).
        """
        for rel in relations:
            if len(rel) >= 3:
                source, relation, target = rel[0], rel[1], rel[2]
                try:
                    weight = float(rel[3]) if len(rel) >= 4 else 0.8
                except (ValueError, TypeError):
                    weight = 0.8
                weight = max(0.1, min(1.0, weight))
                category = rel[4] if len(rel) >= 5 and isinstance(rel[4], str) else "GENERAL"
                await self.semantic.add_relationship(
                    source, relation, target, weight=weight, category=category
                )

    async def match_procedure(self, input_text: str) -> dict | None:
        """Check procedural store for a cached action sequence."""
        return await self.procedural.match(input_text)

    async def stats(self) -> dict:
        proc_count = 0
        try:
            proc_count = len(await self.procedural.get_all())
        except Exception:
            pass
        return {
            "sensory": len(self.sensory.get_all()),
            "working": len(self.working.get_slots()),
            "staging": await self.staging.count_unconsolidated(),
            "episodic": len(await self.episodic.get_all()),
            "semantic": await self.semantic.count(),
            "procedural": proc_count,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = float(np.dot(va, vb))
        norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return dot / norm if norm > 0 else 0.0

    @staticmethod
    def _to_wm_item(sensory_item: SensoryItem) -> WorkingMemoryItem:
        text = sensory_item.data.get("text", str(sensory_item.data))
        return WorkingMemoryItem(content=text, slot="phonological")
