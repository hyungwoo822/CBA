"""Personal workspace adapter over SemanticStore identity_facts.

The personal workspace is modeled over the existing ``identity_facts`` table
owned by ``SemanticStore``. This adapter owns no storage: all reads and writes
pass through to already-instantiated memory stores.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from brain_agent.memory.workspace_store import PERSONAL_WORKSPACE_ID

if TYPE_CHECKING:
    from brain_agent.memory.ontology_store import OntologyStore
    from brain_agent.memory.semantic_store import SemanticStore
    from brain_agent.memory.workspace_store import WorkspaceStore


_FACT_TYPE_TO_LABEL: dict[str, str] = {
    "user_model": "user",
    "self_model": "agent",
}
_LABEL_TO_FACT_TYPE: dict[str, str] = {
    label: fact_type for fact_type, label in _FACT_TYPE_TO_LABEL.items()
}


class PersonalAdapter:
    """Adapter from identity_facts to a personal workspace node interface."""

    def __init__(
        self,
        workspace_store: "WorkspaceStore",
        ontology_store: "OntologyStore",
        semantic_store: "SemanticStore",
    ):
        self._workspace = workspace_store
        self._ontology = ontology_store
        self._semantic = semantic_store

    # ------------------------------------------------------------------
    # Backward-compatible passthroughs
    # ------------------------------------------------------------------

    async def get_user_facts(self) -> list[dict]:
        """Return identity_facts(user_model) in SemanticStore's native shape."""
        return await self._semantic.get_identity_facts("user_model")

    async def get_self_facts(self) -> list[dict]:
        """Return identity_facts(self_model) in SemanticStore's native shape."""
        return await self._semantic.get_identity_facts("self_model")

    async def add_user_fact(
        self,
        key: str,
        value: str,
        confidence: float = 1.0,
    ) -> None:
        """Route to SemanticStore.add_identity_fact('user_model', ...)."""
        await self._semantic.add_identity_fact(
            "user_model",
            key,
            value,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Workspace-node forward API
    # ------------------------------------------------------------------

    async def render_as_nodes(
        self,
        workspace_id: str = PERSONAL_WORKSPACE_ID,
    ) -> list[dict]:
        """Render identity_facts as Person nodes for the personal workspace.

        Returns at most two nodes: one for the user and one for the agent.
        Each node collapses all facts of its fact_type into a properties dict.
        Non-personal workspace ids return an empty list because identity_facts
        is a personal-workspace artifact only.
        """
        if workspace_id != PERSONAL_WORKSPACE_ID:
            return []

        nodes: list[dict] = []
        for fact_type, label in _FACT_TYPE_TO_LABEL.items():
            facts = await self._semantic.get_identity_facts(fact_type)
            if not facts:
                continue

            properties: dict[str, str] = {}
            property_meta: dict[str, dict] = {}
            for fact in facts:
                key = fact["key"]
                properties[key] = fact["value"]
                property_meta[key] = {
                    "confidence": fact.get("confidence", 1.0),
                    "source": fact.get("source", "unknown"),
                    "updated_at": fact.get("updated_at", ""),
                }

            nodes.append(
                {
                    "type": "Person",
                    "label": label,
                    "workspace_id": PERSONAL_WORKSPACE_ID,
                    "properties": properties,
                    "property_meta": property_meta,
                }
            )

        return nodes

    async def write_from_nodes(self, nodes: list[dict]) -> None:
        """Write personal Person node properties back to identity_facts.

        ``label='user'`` maps to ``user_model`` and ``label='agent'`` maps to
        ``self_model``. Unknown labels raise instead of being silently dropped.
        Missing per-property metadata defaults to confidence 1.0 and source
        ``personal_adapter``.
        """
        for node in nodes:
            label = node.get("label")
            if label not in _LABEL_TO_FACT_TYPE:
                raise ValueError(
                    f"unknown label {label!r}: expected 'user' or 'agent'"
                )

            fact_type = _LABEL_TO_FACT_TYPE[label]
            properties: dict[str, str] = node.get("properties", {}) or {}
            property_meta: dict[str, dict] = node.get("property_meta", {}) or {}

            for key, value in properties.items():
                per_key_meta = property_meta.get(key, {})
                confidence = float(per_key_meta.get("confidence", 1.0))
                source = per_key_meta.get("source", "personal_adapter")
                await self._semantic.add_identity_fact(
                    fact_type,
                    key,
                    str(value),
                    source=source,
                    confidence=confidence,
                )
