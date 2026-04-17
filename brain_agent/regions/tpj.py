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

    Maintains a user model through identity facts stored in the
    knowledge graph (single source of truth).

    The :meth:`get_user_context` method renders identity facts into a
    textual representation suitable for prompt injection.

    References:
      - Frith & Frith (2006): The neural basis of mentalizing
      - Saxe & Kanwisher (2003): TPJ in theory of mind
    """

    def __init__(self) -> None:
        super().__init__(
            name="tpj",
            position=Vec3(50, -35, 30),
            lobe=Lobe.PARIETAL,
            hemisphere=Hemisphere.RIGHT,
        )
        self._identity_facts: list[dict] = []
        self._graph_facts: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────

    def reload_schema(self) -> None:
        """No-op for backwards compatibility.

        Identity facts are updated via update_from_identity_facts() at
        runtime, not loaded from disk.
        """
        pass

    def get_user_model(self) -> dict:
        """Return the raw user model layers."""
        facts = list(self._identity_facts) + list(self._graph_facts)
        return {
            "schema": self.get_user_context(),
            "facts": facts,
            "identity_facts": list(self._identity_facts),
            "graph_facts": list(self._graph_facts),
        }

    def update_from_identity_facts(self, facts: list[dict]) -> None:
        """Replace identity facts (from semantic_store.get_identity_facts).

        Args:
            facts: list of dicts with ``key``, ``value``, ``confidence``.
        """
        self._identity_facts = list(facts)

    def update_from_graph_facts(self, facts: list[dict]) -> None:
        """Replace the current graph-fact layer with *facts*.

        Called by the pipeline after loading identity facts from the
        semantic store.
        """
        self._graph_facts = list(facts)

    def get_user_context(self) -> str:
        """Render identity facts into a unified context string.

        The result is intended for injection into PFC prompts so the
        agent can reason with an up-to-date model of the user.
        """
        parts: list[str] = []

        if self._identity_facts:
            lines = ["# User Profile"]
            categorized: dict[str, list[str]] = {}
            uncategorized: list[str] = []

            for f in self._identity_facts:
                key = f.get("key", "")
                value = f.get("value", "")
                if ":" in key:
                    prefix, detail = key.split(":", 1)
                    categorized.setdefault(prefix.strip(), []).append(
                        f"- {detail.strip()}: {value}"
                    )
                else:
                    uncategorized.append(f"- **{key}**: {value}")

            if uncategorized:
                lines.extend(uncategorized)
            for cat, items in sorted(categorized.items()):
                lines.append(f"\n## {cat.title()}")
                lines.extend(items)
            parts.append("\n".join(lines))

        if self._graph_facts:
            lines = ["## Relationship Graph"]
            for fact in self._graph_facts:
                subj = fact.get("subject", "?")
                pred = fact.get("predicate", "?")
                obj = fact.get("object", "?")
                lines.append(f"- {subj} | {pred} | {obj}")
            parts.append("\n".join(lines))

        if not parts:
            return "# User Profile\n- No stored user facts yet."

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
