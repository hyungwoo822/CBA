# tests/memory/test_identity_retrieval.py
import pytest
import hashlib
from brain_agent.memory.semantic_store import SemanticStore


def mock_embed(text):
    h = hashlib.md5(text.encode()).digest()
    return [float(b) / 255.0 for b in h] * 24


@pytest.mark.asyncio
async def test_retrieve_identity_returns_both_models(tmp_path):
    store = SemanticStore(
        chroma_path=str(tmp_path / "chroma"),
        graph_db_path=str(tmp_path / "graph.db"),
        embed_fn=mock_embed,
    )
    await store.initialize()
    await store.add_identity_fact("self_model", "personality", "curious and warm")
    await store.add_identity_fact("user_model", "name", "진혁")
    await store.add_identity_fact("user_model", "language", "Korean")

    self_facts = await store.get_identity_facts("self_model")
    user_facts = await store.get_identity_facts("user_model")

    assert len(self_facts) == 1
    assert len(user_facts) == 2
    assert any(f["value"] == "진혁" for f in user_facts)
    await store.close()
