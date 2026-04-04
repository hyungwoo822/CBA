"""Medial Prefrontal Cortex (mPFC) — Self-referential processing region.

The mPFC manages the agent's self-model (identity, personality, values) using
two complementary layers:

  1. **Schema layer** — A markdown file (``data/SOUL.md``) that encodes the
     agent's narrative identity: personality traits, values, communication
     style, and relationship philosophy.  This acts as the agent's long-term
     "self-schema" (Ghosh & Gilboa, 2014).

  2. **Graph-fact layer** — Structured knowledge-graph triples
     (subject-predicate-object) injected at runtime by the pipeline from the
     semantic store's ``identity_facts`` table.  These provide precise,
     machine-readable self-knowledge that complements the free-form schema.

During processing, the mPFC combines both layers into a single
``self_context`` string that is attached to the signal's metadata for
downstream consumption by the PFC (Phase 5 reasoning).

Academic references:
  - Northoff, G. et al. (2006).  Self-referential processing in our brain —
    A meta-analysis of imaging studies on the self.  *NeuroImage*, 31(1),
    440--457.
  - Ghosh, V. E. & Gilboa, A. (2014).  What is a memory schema?  A
    historical perspective on current neuroscience literature.
    *Neuropsychologia*, 53, 104--114.
  - D'Argembeau, A. et al. (2005).  Self-referential reflective activity
    and its relationship with rest: a PET study.  *NeuroImage*, 25(2),
    616--624.
"""

from __future__ import annotations

import logging
import os

from brain_agent.core.signals import Signal
from brain_agent.regions.base import BrainRegion, Hemisphere, Lobe, Vec3

logger = logging.getLogger(__name__)


class MedialPrefrontalCortex(BrainRegion):
    """Self-referential processing — dual-layer identity management.

    Combines a narrative self-schema (SOUL.md) with structured knowledge-graph
    facts to maintain a coherent self-model that the PFC can query during
    reasoning.

    Anatomical properties:
      - Position: Vec3(0, 45, 30) — medial frontal wall
      - Lobe: FRONTAL
      - Hemisphere: BILATERAL

    References:
      - Northoff et al. (2006): Self-referential processing in mPFC
      - Ghosh & Gilboa (2014): Schema theory
      - D'Argembeau et al. (2005): mPFC and self-relevant thinking
    """

    def __init__(self) -> None:
        super().__init__(
            name="medial_pfc",
            position=Vec3(0, 45, 30),
            lobe=Lobe.FRONTAL,
            hemisphere=Hemisphere.BILATERAL,
        )
        self._schema_text: str = self._load_schema()
        self._graph_facts: list[dict] = []

    # ── Schema loading ────────────────────────────────────────────────

    @staticmethod
    def _load_schema() -> str:
        """Read the narrative self-schema from ``data/SOUL.md``.

        The data directory lives two levels above the ``regions/`` package::

            brain_agent/regions/mpfc.py  ->  ../../  ->  project root
            project root / data / SOUL.md
        """
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
        )
        path = os.path.join(data_dir, "SOUL.md")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            # Auto-create default SOUL.md on first run
            default = (
                "# Soul — Neural Agent Identity Schema\n\n"
                "I am a neural agent built on a biologically-inspired cognitive architecture.\n"
                "My brain simulates 24 regions, 7 neurotransmitters, and a 6-layer memory system.\n\n"
                "## Core Identity\n"
                "- I process information through parallel neural pathways\n"
                "- My emotional state genuinely shapes my reasoning\n"
                "- I form memories, build knowledge graphs, and consolidate experiences\n"
                "- My personality emerges from neural dynamics, not a fixed prompt\n"
            )
            try:
                os.makedirs(data_dir, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(default)
                logger.info("Created default SOUL.md at %s", path)
            except OSError:
                logger.warning("Could not create SOUL.md at %s", path)
            return default

    # ── Public API ────────────────────────────────────────────────────

    def get_self_model(self) -> dict:
        """Return the full self-model as ``{"schema": str, "facts": list}``."""
        return {
            "schema": self._schema_text,
            "facts": list(self._graph_facts),
        }

    def reload_schema(self) -> None:
        """Re-read SOUL.md from disk after narrative consolidation updates it."""
        self._schema_text = self._load_schema()

    def update_from_graph_facts(self, facts: list[dict]) -> None:
        """Replace current graph facts with *facts* (called by pipeline).

        Each fact is expected to be a dict with at least ``subject``,
        ``predicate``, and ``object`` keys.
        """
        self._graph_facts = list(facts)

    def get_self_context(self) -> str:
        """Combine schema + graph facts into a single text block for the PFC.

        The returned string is suitable for injection into an LLM system
        prompt as the agent's self-knowledge section.
        """
        parts: list[str] = []

        # Layer 1: narrative schema from SOUL.md
        if self._schema_text:
            parts.append(self._schema_text)

        # Layer 2: structured knowledge-graph facts
        if self._graph_facts:
            fact_lines = []
            for fact in self._graph_facts:
                subj = fact.get("subject", "?")
                pred = fact.get("predicate", "?")
                obj = fact.get("object", "?")
                fact_lines.append(f"- {subj} {pred} {obj}")
            parts.append("## Identity Facts (Knowledge Graph)\n" + "\n".join(fact_lines))

        return "\n\n".join(parts)

    # ── Signal processing ─────────────────────────────────────────────

    async def process(self, signal: Signal) -> Signal | None:
        """Attach self-context to the signal for downstream PFC reasoning.

        Sets ``signal.metadata["self_context"]`` and emits a moderate
        activation of 0.4 (self-referential processing is tonic, not
        phasic — Northoff et al. 2006).
        """
        self.emit_activation(0.4)
        signal.metadata["self_context"] = self.get_self_context()
        return signal
