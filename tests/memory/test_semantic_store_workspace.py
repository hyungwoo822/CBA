"""Tests for workspace-aware SemanticStore API."""
import pytest

from brain_agent.memory.semantic_store import SemanticStore
from brain_agent.migrations.steps import m001_workspace_columns as m001


@pytest.fixture
async def semantic(tmp_path, mock_embedding):
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=str(tmp_path),
    )
    s = SemanticStore(
        chroma_path=str(tmp_path / "chroma"),
        graph_db_path=str(tmp_path / "graph.db"),
        embed_fn=mock_embedding,
    )
    await s.initialize()
    await m001.apply(
        brain_state_db=str(tmp_path / "brain_state.db"),
        data_dir=str(tmp_path),
    )
    yield s
    await s.close()


async def test_add_relationship_default_workspace(semantic):
    await semantic.add_relationship("alice", "likes", "coffee")
    rels = await semantic.get_relationships("alice")
    assert rels and rels[0]["workspace_id"] == "personal"


async def test_add_relationship_explicit_workspace(semantic):
    await semantic.add_relationship(
        "alice",
        "works_on",
        "billing",
        workspace_id="ws_biz",
    )
    rels = await semantic.get_relationships("alice", workspace_id="ws_biz")
    assert len(rels) == 1
    assert rels[0]["workspace_id"] == "ws_biz"


async def test_get_relationships_filters_by_workspace(semantic):
    await semantic.add_relationship(
        "alice", "likes", "coffee", workspace_id="personal"
    )
    await semantic.add_relationship("alice", "owns", "repo", workspace_id="ws_biz")
    personal = await semantic.get_relationships("alice", workspace_id="personal")
    biz = await semantic.get_relationships("alice", workspace_id="ws_biz")
    assert len(personal) == 1
    assert personal[0]["relation"] == "likes"
    assert len(biz) == 1
    assert biz[0]["relation"] == "owns"


async def test_get_relationships_none_returns_all(semantic):
    await semantic.add_relationship(
        "alice", "likes", "coffee", workspace_id="personal"
    )
    await semantic.add_relationship("alice", "owns", "repo", workspace_id="ws_biz")
    both = await semantic.get_relationships("alice", workspace_id=None)
    assert len(both) == 2


async def test_add_relationship_cross_workspace(semantic):
    await semantic.add_relationship(
        "alice",
        "references",
        "billing_service",
        workspace_id="personal",
        target_workspace_id="ws_biz",
    )
    rels = await semantic.get_relationships("alice", workspace_id="personal")
    assert rels[0]["target_workspace_id"] == "ws_biz"


async def test_add_relationship_epistemic_and_importance_and_never_decay(semantic):
    await semantic.add_relationship(
        "service",
        "depends_on",
        "db",
        workspace_id="ws_biz",
        epistemic_source="observed",
        importance_score=0.9,
        never_decay=True,
    )
    rels = await semantic.get_relationships("service", workspace_id="ws_biz")
    assert rels[0]["epistemic_source"] == "observed"
    assert rels[0]["importance_score"] == 0.9
    assert rels[0]["never_decay"] == 1


async def test_add_chroma_document_tagged_with_workspace(semantic):
    doc_id = await semantic.add(
        "billing service handles invoices",
        workspace_id="ws_biz",
    )
    res = semantic._collection.get(ids=[doc_id], include=["metadatas"])
    meta = res["metadatas"][0]
    assert meta["workspace_id"] == "ws_biz"


async def test_search_with_workspace_filter(semantic):
    await semantic.add("personal note about coffee", workspace_id="personal")
    await semantic.add("billing invoices are due", workspace_id="ws_biz")
    biz_hits = await semantic.search("invoice", top_k=5, workspace_id="ws_biz")
    personal_hits = await semantic.search("coffee", top_k=5, workspace_id="personal")
    for hit in biz_hits:
        assert hit["metadata"].get("workspace_id", "personal") == "ws_biz"
    for hit in personal_hits:
        assert hit["metadata"].get("workspace_id", "personal") == "personal"


async def test_search_without_workspace_returns_all(semantic):
    await semantic.add("note A", workspace_id="personal")
    await semantic.add("note B", workspace_id="ws_biz")
    all_hits = await semantic.search("note", top_k=5, workspace_id=None)
    ws_set = {hit["metadata"].get("workspace_id", "personal") for hit in all_hits}
    assert {"personal", "ws_biz"}.issubset(ws_set)


async def test_export_as_networkx_workspace_filter(semantic):
    await semantic.add_relationship("a", "r1", "b", workspace_id="personal")
    await semantic.add_relationship("c", "r2", "d", workspace_id="ws_biz")
    g_personal = await semantic.export_as_networkx(workspace_id="personal")
    nodes_p = set(g_personal.nodes())
    assert "a" in nodes_p and "b" in nodes_p
    assert "c" not in nodes_p and "d" not in nodes_p


async def test_export_as_networkx_cross_ref_included_by_default(semantic):
    await semantic.add_relationship(
        "a",
        "r",
        "b",
        workspace_id="personal",
        target_workspace_id="ws_biz",
    )
    g = await semantic.export_as_networkx(
        workspace_id="personal", include_cross_refs=True
    )
    assert g.has_edge("a", "b")


async def test_mark_superseded_sets_valid_to(semantic):
    await semantic.add_relationship(
        "alice",
        "lives_in",
        "seoul",
        workspace_id="personal",
    )
    rels = await semantic.get_relationships("alice", workspace_id="personal")
    edge_id = rels[0]["id"]
    await semantic.mark_superseded(edge_id, valid_to="2026-04-17T00:00:00Z")
    after = await semantic.get_relationships("alice", workspace_id="personal")
    assert after[0]["valid_to"] == "2026-04-17T00:00:00Z"
