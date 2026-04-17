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

    # Forward API (render_as_nodes, write_from_nodes) is added in Task 2/3.
