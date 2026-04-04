import pytest
from brain_agent.memory.semantic_store import SemanticStore


@pytest.fixture
async def store(tmp_path, mock_embedding):
    s = SemanticStore(
        chroma_path=str(tmp_path / "chroma"),
        graph_db_path=str(tmp_path / "graph.db"),
        embed_fn=mock_embedding,
    )
    await s.initialize()
    yield s
    await s.close()


async def test_add_identity_fact_self(store):
    """mPFC self-model facts can be stored and retrieved."""
    await store.add_identity_fact(
        fact_type="self_model",
        key="personality",
        value="curious and analytical",
        source="introspection",
        confidence=0.9,
    )
    facts = await store.get_identity_facts("self_model")
    assert len(facts) == 1
    assert facts[0]["key"] == "personality"
    assert facts[0]["value"] == "curious and analytical"
    assert facts[0]["source"] == "introspection"
    assert facts[0]["confidence"] == 0.9


async def test_add_identity_fact_user(store):
    """TPJ user-model facts can be stored and retrieved."""
    await store.add_identity_fact(
        fact_type="user_model",
        key="name",
        value="Alice",
        source="conversation",
        confidence=1.0,
    )
    facts = await store.get_identity_facts("user_model")
    assert len(facts) == 1
    assert facts[0]["key"] == "name"
    assert facts[0]["value"] == "Alice"
    assert facts[0]["source"] == "conversation"
    assert facts[0]["confidence"] == 1.0


async def test_update_existing_identity_fact(store):
    """UPSERT: updating same (fact_type, key) overwrites value.

    Ref: Ghosh & Gilboa (2014) schema updating in mPFC.
    """
    await store.add_identity_fact(
        fact_type="user_model",
        key="preferred_language",
        value="English",
        source="conversation",
        confidence=0.8,
    )
    await store.add_identity_fact(
        fact_type="user_model",
        key="preferred_language",
        value="Korean",
        source="explicit_statement",
        confidence=1.0,
    )
    facts = await store.get_identity_facts("user_model")
    assert len(facts) == 1
    assert facts[0]["key"] == "preferred_language"
    assert facts[0]["value"] == "Korean"
    assert facts[0]["source"] == "explicit_statement"
    assert facts[0]["confidence"] == 1.0


async def test_get_identity_facts_filters_by_type(store):
    """get_identity_facts only returns facts matching the requested type."""
    await store.add_identity_fact(
        fact_type="self_model",
        key="core_value",
        value="helpfulness",
        source="design",
    )
    await store.add_identity_fact(
        fact_type="user_model",
        key="name",
        value="Bob",
        source="conversation",
    )
    await store.add_identity_fact(
        fact_type="self_model",
        key="personality",
        value="calm",
        source="introspection",
    )

    self_facts = await store.get_identity_facts("self_model")
    user_facts = await store.get_identity_facts("user_model")

    assert len(self_facts) == 2
    assert len(user_facts) == 1

    # self_model facts ordered by key
    assert self_facts[0]["key"] == "core_value"
    assert self_facts[1]["key"] == "personality"

    # user_model facts
    assert user_facts[0]["key"] == "name"
    assert user_facts[0]["value"] == "Bob"
