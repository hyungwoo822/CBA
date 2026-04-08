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


async def test_add_hyperedge(store):
    await store.add_hyperedge(
        members=["FastAPI", "Auth", "JWT"],
        label="auth_system",
        category="FUNCTIONAL",
    )
    edges = await store.get_hyperedges()
    assert len(edges) == 1
    assert set(edges[0]["members"]) == {"Auth", "FastAPI", "JWT"}
    assert edges[0]["label"] == "auth_system"


async def test_get_assemblies_for_node(store):
    await store.add_hyperedge(["A", "B", "C"], "cluster1")
    await store.add_hyperedge(["B", "D", "E"], "cluster2")
    assemblies = await store.get_assemblies_for_node("B")
    assert len(assemblies) == 2


async def test_assemblies_for_absent_node(store):
    await store.add_hyperedge(["A", "B", "C"], "cluster1")
    assemblies = await store.get_assemblies_for_node("Z")
    assert len(assemblies) == 0


async def test_hyperedge_no_duplicates(store):
    await store.add_hyperedge(["A", "B", "C"], "cluster1")
    await store.add_hyperedge(["A", "B", "C"], "cluster1")
    edges = await store.get_hyperedges()
    assert len(edges) == 1


async def test_hyperedge_strength(store):
    await store.add_hyperedge(["A", "B"], "pair", strength=0.8)
    edges = await store.get_hyperedges()
    assert edges[0]["strength"] == pytest.approx(0.8)
