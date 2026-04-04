"""Temporoparietal Junction (TPJ) --- Theory of Mind region.

Brain mapping: Right TPJ, inferior parietal lobule near the posterior
superior temporal sulcus (MNI ~50, -35, 30).

Functions:
  - Theory of Mind: maintaining a model of the user's mental states,
    beliefs, intentions, and identity
  - Social cognition: integrating identity facts with conversational
    schema to produce a coherent user representation
  - Perspective-taking: enabling the agent to reason about what the
    user knows, wants, and feels

AI agent function: Manages a dual-layer user model composed of a
markdown schema file (USER.md) and structured knowledge-graph facts
injected by the pipeline.  Produces a unified ``user_context`` string
consumed by the prefrontal cortex during response generation.

References:
  - Frith & Frith (2006): The neural basis of mentalizing
  - Saxe & Kanwisher (2003): People thinking about thinking people ---
    the role of the temporoparietal junction in theory of mind
  - Van Overwalle (2009): Social cognition and the brain: a
    meta-analysis
"""
from __future__ import annotations

import os
from pathlib import Path

from brain_agent.core.signals import Signal
from brain_agent.regions.base import BrainRegion, Hemisphere, Lobe, Vec3

# Resolve the project-level ``data/`` directory.
_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DATA_DIR = _PROJECT_ROOT / "data"


class TemporoparietalJunction(BrainRegion):
    """Right-lateralised Theory of Mind region.

    Maintains a user model through two complementary layers:

    1. **Schema** -- a markdown profile loaded from ``data/USER.md``
       that captures high-level identity, preferences, and personality.
    2. **Graph facts** -- structured ``(subject, predicate, object)``
       triples injected at runtime from the semantic store's identity
       facts table.

    The :meth:`get_user_context` method merges both layers into a
    single textual representation suitable for prompt injection.
    """

    def __init__(self) -> None:
        super().__init__(
            name="tpj",
            position=Vec3(50, -35, 30),
            lobe=Lobe.PARIETAL,
            hemisphere=Hemisphere.RIGHT,
        )
        self._schema_text: str = self._load_schema()
        self._graph_facts: list[dict] = []

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _load_schema() -> str:
        """Read the user profile markdown from ``data/USER.md``.

        Returns an empty string if the file does not exist so the
        region can still function without the template.
        """
        path = _DATA_DIR / "USER.md"
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Auto-create default USER.md on first run
            default = (
                "# User Profile\n\n"
                "## About\n"
                "- Preferences and personality will be learned through conversation\n\n"
                "## Communication Style\n"
                "- To be discovered through interaction\n\n"
                "## Interests\n"
                "- To be discovered through interaction\n"
            )
            try:
                _DATA_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(default, encoding="utf-8")
            except OSError:
                pass
            return default

    # ── Public API ────────────────────────────────────────────────────

    def reload_schema(self) -> None:
        """Re-read USER.md from disk after consolidation updates it."""
        self._schema_text = self._load_schema()

    def get_user_model(self) -> dict:
        """Return the raw dual-layer user model.

        Returns:
            dict with ``schema`` (str) and ``facts`` (list[dict]).
        """
        return {
            "schema": self._schema_text,
            "facts": list(self._graph_facts),
        }

    def update_from_graph_facts(self, facts: list[dict]) -> None:
        """Replace the current graph-fact layer with *facts*.

        Called by the pipeline after loading identity facts from the
        semantic store.

        Args:
            facts: list of dicts, each expected to have at least
                   ``subject``, ``predicate``, and ``object`` keys.
        """
        self._graph_facts = list(facts)

    def get_user_context(self) -> str:
        """Merge schema and graph facts into a unified context string.

        The result is intended for injection into PFC prompts so the
        agent can reason with an up-to-date model of the user.
        """
        parts: list[str] = []

        if self._schema_text:
            parts.append(self._schema_text.strip())

        if self._graph_facts:
            lines = ["## Knowledge-Graph Identity Facts"]
            for fact in self._graph_facts:
                subj = fact.get("subject", "?")
                pred = fact.get("predicate", "?")
                obj = fact.get("object", "?")
                lines.append(f"- {subj} | {pred} | {obj}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    # ── Signal processing ─────────────────────────────────────────────

    async def process(self, signal: Signal) -> Signal | None:
        """Attach the current user context to the signal metadata.

        Sets ``signal.metadata["user_context"]`` and emits a steady
        activation of 0.4 (the TPJ is tonically active during social
        processing).
        """
        signal.metadata["user_context"] = self.get_user_context()
        self.emit_activation(0.4)
        return signal
