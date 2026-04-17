"""Tests for PersonalAdapter identity_facts to workspace-node bridge."""
import pytest

from brain_agent.memory.personal_adapter import PersonalAdapter
from brain_agent.memory.workspace_store import PERSONAL_WORKSPACE_ID


@pytest.fixture
async def adapter(memory_manager):
    """PersonalAdapter wired over the real MemoryManager stack."""
    return PersonalAdapter(
        workspace_store=memory_manager.workspace,
        ontology_store=memory_manager.ontology,
        semantic_store=memory_manager.semantic,
    )


async def test_get_user_facts_returns_what_semantic_stored(adapter, memory_manager):
    """Adding via semantic_store.add_identity_fact surfaces via adapter.get_user_facts."""
    await memory_manager.semantic.add_identity_fact(
        "user_model", "name", "Alice", source="test", confidence=0.9,
    )
    facts = await adapter.get_user_facts()
    assert len(facts) == 1
    assert facts[0]["key"] == "name"
    assert facts[0]["value"] == "Alice"
    assert facts[0]["source"] == "test"
    assert facts[0]["confidence"] == pytest.approx(0.9)


async def test_get_self_facts_returns_what_semantic_stored(adapter, memory_manager):
    await memory_manager.semantic.add_identity_fact(
        "self_model", "role", "coding-assistant", source="seed", confidence=1.0,
    )
    facts = await adapter.get_self_facts()
    assert len(facts) == 1
    assert facts[0]["key"] == "role"
    assert facts[0]["value"] == "coding-assistant"


async def test_add_user_fact_writes_to_identity_facts(adapter, memory_manager):
    """add_user_fact routes to semantic_store.add_identity_fact('user_model', ...)."""
    await adapter.add_user_fact("city", "Seoul", confidence=0.8)
    raw = await memory_manager.semantic.get_identity_facts("user_model")
    assert len(raw) == 1
    assert raw[0]["key"] == "city"
    assert raw[0]["value"] == "Seoul"
    assert raw[0]["confidence"] == pytest.approx(0.8)


async def test_add_user_fact_default_confidence_is_one(adapter, memory_manager):
    await adapter.add_user_fact("pet", "cat")
    raw = await memory_manager.semantic.get_identity_facts("user_model")
    assert raw[0]["confidence"] == pytest.approx(1.0)


async def test_add_user_fact_upsert_semantics(adapter, memory_manager):
    """Same key twice updates, does not duplicate; matches identity_facts UNIQUE."""
    await adapter.add_user_fact("name", "Alice", confidence=0.5)
    await adapter.add_user_fact("name", "Alicia", confidence=0.9)
    facts = await adapter.get_user_facts()
    assert len(facts) == 1
    assert facts[0]["value"] == "Alicia"
    assert facts[0]["confidence"] == pytest.approx(0.9)
