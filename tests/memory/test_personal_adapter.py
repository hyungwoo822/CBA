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


async def test_render_as_nodes_returns_user_node(adapter, memory_manager):
    """user_model facts collapse into a single Person/label='user' node."""
    await memory_manager.semantic.add_identity_fact("user_model", "name", "Alice")
    await memory_manager.semantic.add_identity_fact("user_model", "city", "Seoul")
    nodes = await adapter.render_as_nodes()

    user_nodes = [n for n in nodes if n["label"] == "user"]
    assert len(user_nodes) == 1
    user = user_nodes[0]
    assert user["type"] == "Person"
    assert user["label"] == "user"
    assert user["workspace_id"] == PERSONAL_WORKSPACE_ID
    assert user["properties"] == {"city": "Seoul", "name": "Alice"}
    assert set(user["property_meta"].keys()) == {"name", "city"}
    assert "confidence" in user["property_meta"]["name"]


async def test_render_as_nodes_returns_agent_node(adapter, memory_manager):
    await memory_manager.semantic.add_identity_fact("self_model", "role", "assistant")
    nodes = await adapter.render_as_nodes()

    agent_nodes = [n for n in nodes if n["label"] == "agent"]
    assert len(agent_nodes) == 1
    agent = agent_nodes[0]
    assert agent["type"] == "Person"
    assert agent["label"] == "agent"
    assert agent["properties"] == {"role": "assistant"}


async def test_render_as_nodes_returns_both_when_present(adapter, memory_manager):
    await memory_manager.semantic.add_identity_fact("user_model", "name", "Alice")
    await memory_manager.semantic.add_identity_fact("self_model", "role", "assistant")
    nodes = await adapter.render_as_nodes()
    labels = {n["label"] for n in nodes}
    assert labels == {"user", "agent"}


async def test_render_as_nodes_empty_when_no_facts(adapter):
    """No identity_facts means no nodes, not empty-properties nodes."""
    nodes = await adapter.render_as_nodes()
    assert nodes == []


async def test_render_as_nodes_empty_for_non_personal_workspace(
    adapter, memory_manager,
):
    """Adapter is bound to personal; business workspaces return []."""
    biz = await memory_manager.workspace.create_workspace(name="Billing Service")
    await memory_manager.semantic.add_identity_fact("user_model", "name", "Alice")
    nodes = await adapter.render_as_nodes(workspace_id=biz["id"])
    assert nodes == []


async def test_render_as_nodes_default_workspace_id_is_personal(
    adapter, memory_manager,
):
    """Calling render_as_nodes() with no argument defaults to personal."""
    await memory_manager.semantic.add_identity_fact("user_model", "name", "Alice")
    nodes_default = await adapter.render_as_nodes()
    nodes_explicit = await adapter.render_as_nodes(workspace_id=PERSONAL_WORKSPACE_ID)
    assert nodes_default == nodes_explicit
    assert len(nodes_default) == 1


async def test_write_from_nodes_user_label_writes_user_model(adapter, memory_manager):
    node = {
        "type": "Person",
        "label": "user",
        "workspace_id": PERSONAL_WORKSPACE_ID,
        "properties": {"name": "Alice", "city": "Seoul"},
    }
    await adapter.write_from_nodes([node])
    raw = await memory_manager.semantic.get_identity_facts("user_model")
    by_key = {f["key"]: f["value"] for f in raw}
    assert by_key == {"name": "Alice", "city": "Seoul"}


async def test_write_from_nodes_agent_label_writes_self_model(adapter, memory_manager):
    node = {
        "type": "Person",
        "label": "agent",
        "workspace_id": PERSONAL_WORKSPACE_ID,
        "properties": {"role": "assistant"},
    }
    await adapter.write_from_nodes([node])
    raw = await memory_manager.semantic.get_identity_facts("self_model")
    assert raw[0]["key"] == "role"
    assert raw[0]["value"] == "assistant"


async def test_write_from_nodes_unknown_label_raises(adapter):
    node = {
        "type": "Person",
        "label": "stranger",
        "workspace_id": PERSONAL_WORKSPACE_ID,
        "properties": {"name": "X"},
    }
    with pytest.raises(ValueError, match="unknown label"):
        await adapter.write_from_nodes([node])


async def test_write_from_nodes_applies_property_meta_confidence(
    adapter, memory_manager,
):
    """property_meta.confidence propagates to identity_facts.confidence."""
    node = {
        "type": "Person",
        "label": "user",
        "workspace_id": PERSONAL_WORKSPACE_ID,
        "properties": {"name": "Alice"},
        "property_meta": {"name": {"confidence": 0.42, "source": "from-node"}},
    }
    await adapter.write_from_nodes([node])
    raw = await memory_manager.semantic.get_identity_facts("user_model")
    assert raw[0]["confidence"] == pytest.approx(0.42)
    assert raw[0]["source"] == "from-node"


async def test_write_from_nodes_default_confidence(adapter, memory_manager):
    """Missing property_meta defaults confidence to 1.0 and source to personal_adapter."""
    node = {
        "type": "Person",
        "label": "user",
        "workspace_id": PERSONAL_WORKSPACE_ID,
        "properties": {"name": "Alice"},
    }
    await adapter.write_from_nodes([node])
    raw = await memory_manager.semantic.get_identity_facts("user_model")
    assert raw[0]["confidence"] == pytest.approx(1.0)
    assert raw[0]["source"] == "personal_adapter"


async def test_round_trip_integrity(adapter, memory_manager):
    """add -> render_as_nodes -> write_from_nodes -> get_user_facts preserves values."""
    await memory_manager.semantic.add_identity_fact(
        "user_model", "name", "Alice", source="seed", confidence=0.9,
    )
    await memory_manager.semantic.add_identity_fact(
        "user_model", "city", "Seoul", source="seed", confidence=0.7,
    )

    snapshot = await adapter.render_as_nodes()
    await adapter.write_from_nodes(snapshot)

    facts_after = await adapter.get_user_facts()
    by_key = {f["key"]: f for f in facts_after}
    assert by_key["name"]["value"] == "Alice"
    assert by_key["name"]["confidence"] == pytest.approx(0.9)
    assert by_key["city"]["value"] == "Seoul"
    assert by_key["city"]["confidence"] == pytest.approx(0.7)
