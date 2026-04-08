import pytest
from brain_agent.mcp.knowledge_server import KnowledgeGraphTools
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


async def test_get_neighbors(store):
    await store.add_relationship("Python", "is_a", "Language", weight=0.9)
    result = await KnowledgeGraphTools.get_neighbors(store, "Python")
    assert result["count"] == 1
    assert result["neighbors"][0]["relation"] == "is_a"


async def test_list_communities(store):
    await store.add_relationship("a", "r", "b", weight=0.9)
    await store.add_relationship("b", "r", "c", weight=0.8)
    result = await KnowledgeGraphTools.list_communities(store)
    assert result["count"] >= 1


async def test_find_hubs(store):
    for i in range(5):
        await store.add_relationship("hub", f"r{i}", f"leaf_{i}")
    result = await KnowledgeGraphTools.find_hubs(store, top_n=3)
    assert result["count"] >= 1
    assert result["hubs"][0]["id"] == "hub"


async def test_find_bridges_empty(store):
    result = await KnowledgeGraphTools.find_bridges(store)
    assert result["count"] == 0


async def test_get_assemblies(store):
    await store.add_hyperedge(["A", "B", "C"], "test_assembly")
    result = await KnowledgeGraphTools.get_assemblies(store, "B")
    assert result["count"] == 1
    assert result["assemblies"][0]["label"] == "test_assembly"
