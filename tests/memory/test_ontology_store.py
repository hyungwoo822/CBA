"""Tests for OntologyStore."""
import pytest

from brain_agent.memory.ontology_seed import UNIVERSAL_WORKSPACE_ID
from brain_agent.memory.ontology_store import OntologyStore


@pytest.fixture
async def store(tmp_path):
    store = OntologyStore(db_path=str(tmp_path / "ontology.db"))
    await store.initialize()
    yield store
    await store.close()


async def test_seed_universal_inserts_all_types(store):
    await store.seed_universal()
    node_types = await store.get_node_types(UNIVERSAL_WORKSPACE_ID)
    relation_types = await store.get_relation_types(UNIVERSAL_WORKSPACE_ID)
    assert len(node_types) == 7
    assert len(relation_types) == 10
    node_names = {node_type["name"] for node_type in node_types}
    assert "Person" in node_names
    assert "Event" in node_names


async def test_seed_universal_idempotent(store):
    await store.seed_universal()
    await store.seed_universal()
    node_types = await store.get_node_types(UNIVERSAL_WORKSPACE_ID)
    relation_types = await store.get_relation_types(UNIVERSAL_WORKSPACE_ID)
    assert len(node_types) == 7
    assert len(relation_types) == 10


async def test_seeded_types_are_canonical(store):
    await store.seed_universal()
    types = await store.get_node_types(UNIVERSAL_WORKSPACE_ID)
    assert all(node_type["confidence"] == "CANONICAL" for node_type in types)


async def test_register_node_type_new(store):
    await store.seed_universal()
    ws = "ws_alpha"
    node_type = await store.register_node_type(
        ws, "ApiEndpoint", source_snippet="our /orders endpoint"
    )
    assert node_type["name"] == "ApiEndpoint"
    assert node_type["workspace_id"] == ws
    assert node_type["confidence"] == "PROVISIONAL"
    assert node_type["occurrence_count"] == 1
    assert node_type["source_snippet"] == "our /orders endpoint"


async def test_register_node_type_returns_existing_on_duplicate(store):
    await store.seed_universal()
    ws = "ws_alpha"
    first = await store.register_node_type(ws, "ApiEndpoint")
    second = await store.register_node_type(ws, "ApiEndpoint")
    assert first["id"] == second["id"]


async def test_register_node_type_with_parent_name(store):
    await store.seed_universal()
    ws = "ws_alpha"
    node_type = await store.register_node_type(
        ws, "ServiceEndpoint", parent_name="Artifact"
    )
    artifact = await store._get_node_type_by_name("__universal__", "Artifact")
    assert node_type["parent_type_id"] == artifact["id"]


async def test_register_node_type_unknown_parent_raises(store):
    await store.seed_universal()
    with pytest.raises(ValueError, match="Parent type not found"):
        await store.register_node_type("ws_alpha", "X", parent_name="NoSuchType")


async def test_get_node_types_unions_universal(store):
    await store.seed_universal()
    ws = "ws_beta"
    await store.register_node_type(ws, "CustomThing")
    types = await store.get_node_types(ws)
    names = {node_type["name"] for node_type in types}
    assert "Entity" in names
    assert "CustomThing" in names


async def test_resolve_node_type_workspace_wins_over_universal(store):
    await store.seed_universal()
    ws = "ws_gamma"
    await store.register_node_type(ws, "Person", source_snippet="custom Person")
    resolved = await store.resolve_node_type(ws, "Person")
    assert resolved["workspace_id"] == ws


async def test_resolve_node_type_falls_back_to_universal(store):
    await store.seed_universal()
    resolved = await store.resolve_node_type("ws_delta", "Person")
    assert resolved is not None
    assert resolved["workspace_id"] == "__universal__"


async def test_register_relation_type(store):
    await store.seed_universal()
    relation_type = await store.register_relation_type(
        "ws_e", "owns", domain_type="Person", range_type="Artifact"
    )
    assert relation_type["name"] == "owns"
    assert relation_type["confidence"] == "PROVISIONAL"
    person = await store._get_node_type_by_name("__universal__", "Person")
    assert relation_type["domain_type_id"] == person["id"]


async def test_increment_occurrence_below_threshold(store):
    await store.seed_universal()
    node_type = await store.register_node_type("ws_i", "Widget")
    result = await store.increment_occurrence(node_type["id"])
    assert result["confidence"] == "PROVISIONAL"
    assert result["occurrence_count"] == 2
    assert result["promoted"] is False


async def test_increment_occurrence_at_threshold_promotes_to_stable(store):
    await store.seed_universal()
    node_type = await store.register_node_type("ws_i", "Widget")
    await store.increment_occurrence(node_type["id"])
    result = await store.increment_occurrence(node_type["id"])
    assert result["confidence"] == "STABLE"
    assert result["occurrence_count"] == 3
    assert result["promoted"] is True


async def test_increment_occurrence_beyond_threshold_stays_stable(store):
    await store.seed_universal()
    node_type = await store.register_node_type("ws_i", "Widget")
    await store.increment_occurrence(node_type["id"])
    await store.increment_occurrence(node_type["id"])
    result = await store.increment_occurrence(node_type["id"])
    assert result["confidence"] == "STABLE"
    assert result["occurrence_count"] == 4
    assert result["promoted"] is False


async def test_promote_confidence_stable_to_canonical(store):
    await store.seed_universal()
    node_type = await store.register_node_type("ws_i", "Widget")
    await store.promote_confidence(node_type["id"], "STABLE")
    await store.promote_confidence(node_type["id"], "CANONICAL", promoted_by="user")
    latest = await store._get_node_type_by_name("ws_i", "Widget")
    assert latest["confidence"] == "CANONICAL"


async def test_promote_confidence_backwards_raises(store):
    await store.seed_universal()
    node_type = await store.register_node_type("ws_i", "Widget")
    await store.promote_confidence(node_type["id"], "CANONICAL")
    with pytest.raises(ValueError, match="backwards"):
        await store.promote_confidence(node_type["id"], "PROVISIONAL")


async def test_resolve_type_or_fallback_below_min_confidence(store):
    await store.seed_universal()
    await store.register_node_type("ws_i", "Widget")
    result = await store.resolve_type_or_fallback(
        "ws_i", "Widget", min_confidence="STABLE"
    )
    assert result["name"] == "Concept"


async def test_resolve_type_or_fallback_meets_min_confidence(store):
    await store.seed_universal()
    node_type = await store.register_node_type("ws_i", "Widget")
    await store.promote_confidence(node_type["id"], "STABLE")
    result = await store.resolve_type_or_fallback(
        "ws_i", "Widget", min_confidence="STABLE"
    )
    assert result["name"] == "Widget"


async def test_resolve_type_or_fallback_unknown_name(store):
    await store.seed_universal()
    result = await store.resolve_type_or_fallback(
        "ws_i", "TotallyUnknown", min_confidence="PROVISIONAL"
    )
    assert result["name"] == "Concept"


async def test_propose_node_type(store):
    await store.seed_universal()
    proposal = await store.propose_node_type(
        "ws_p",
        "DraftIdea",
        definition={"description": "not sure yet"},
        confidence="AMBIGUOUS",
        source_input="maybe DraftIdea is a kind of Concept",
    )
    assert proposal["kind"] == "node_type"
    assert proposal["proposed_name"] == "DraftIdea"
    assert proposal["status"] == "pending"


async def test_propose_duplicate_returns_existing(store):
    await store.seed_universal()
    first = await store.propose_node_type(
        "ws_p", "DraftIdea", definition={}, confidence="INFERRED"
    )
    second = await store.propose_node_type(
        "ws_p", "DraftIdea", definition={}, confidence="INFERRED"
    )
    assert first["id"] == second["id"]


async def test_list_pending(store):
    await store.seed_universal()
    await store.propose_node_type("ws_p", "A", definition={}, confidence="INFERRED")
    await store.propose_node_type("ws_p", "B", definition={}, confidence="INFERRED")
    pending = await store.list_pending("ws_p")
    names = {proposal["proposed_name"] for proposal in pending}
    assert names == {"A", "B"}


async def test_approve_proposal_creates_canonical_type(store):
    await store.seed_universal()
    proposal = await store.propose_node_type(
        "ws_p", "Widget", definition={"props": ["weight"]}, confidence="INFERRED"
    )
    approved = await store.approve_proposal(proposal["id"], approved_by="user")
    assert approved["status"] == "approved"
    node_type = await store._get_node_type_by_name("ws_p", "Widget")
    assert node_type is not None
    assert node_type["confidence"] == "CANONICAL"


async def test_approve_proposal_twice_raises(store):
    await store.seed_universal()
    proposal = await store.propose_node_type(
        "ws_p", "Widget", definition={}, confidence="INFERRED"
    )
    await store.approve_proposal(proposal["id"])
    with pytest.raises(ValueError, match="not pending"):
        await store.approve_proposal(proposal["id"])


async def test_reject_proposal(store):
    await store.seed_universal()
    proposal = await store.propose_node_type(
        "ws_p", "Widget", definition={}, confidence="AMBIGUOUS"
    )
    await store.reject_proposal(proposal["id"])
    pending = await store.list_pending("ws_p")
    assert len(pending) == 0


async def test_propose_relation_type_then_approve(store):
    await store.seed_universal()
    proposal = await store.propose_relation_type(
        "ws_p",
        "owns",
        definition={"domain": "Person", "range": "Artifact"},
        confidence="INFERRED",
    )
    await store.approve_proposal(proposal["id"])
    relation_type = await store._get_relation_type_by_name("ws_p", "owns")
    assert relation_type is not None
    assert relation_type["confidence"] == "CANONICAL"


async def test_parent_chain_universal_only(store):
    await store.seed_universal()
    person = await store._get_node_type_by_name("__universal__", "Person")
    chain = await store.resolve_parent_chain(person["id"])
    names = [node_type["name"] for node_type in chain]
    assert names == ["Person", "Entity"]


async def test_parent_chain_workspace_and_universal(store):
    await store.seed_universal()
    node_type = await store.register_node_type(
        "ws_pc", "Developer", parent_name="Person"
    )
    chain = await store.resolve_parent_chain(node_type["id"])
    names = [item["name"] for item in chain]
    assert names == ["Developer", "Person", "Entity"]


async def test_parent_chain_cycle_raises(store):
    await store.seed_universal()
    a = await store.register_node_type("ws_cy", "A")
    b = await store.register_node_type("ws_cy", "B", parent_name="A")
    await store._db.execute(
        "UPDATE node_types SET parent_type_id = ? WHERE id = ?",
        (b["id"], a["id"]),
    )
    await store._db.commit()
    with pytest.raises(ValueError, match="Cycle"):
        await store.resolve_parent_chain(a["id"])


async def test_validate_node_properties_required_missing(store):
    await store.seed_universal()
    event = await store._get_node_type_by_name("__universal__", "Event")
    valid, errors = await store.validate_node_properties(event["id"], {"actor": "Alice"})
    assert valid is False
    assert any("happened_at" in error for error in errors)


async def test_validate_node_properties_required_present(store):
    await store.seed_universal()
    event = await store._get_node_type_by_name("__universal__", "Event")
    valid, errors = await store.validate_node_properties(
        event["id"], {"happened_at": "2026-04-17T10:00:00Z", "actor": "Alice"}
    )
    assert valid is True
    assert errors == []


async def test_validate_node_properties_no_schema_passes(store):
    await store.seed_universal()
    entity = await store._get_node_type_by_name("__universal__", "Entity")
    valid, errors = await store.validate_node_properties(entity["id"], {})
    assert valid is True
