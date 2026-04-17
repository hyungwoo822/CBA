"""Tests for OntologyStore.apply_template."""
import pytest

from brain_agent.memory.ontology_seed import UNIVERSAL_WORKSPACE_ID
from brain_agent.memory.ontology_store import OntologyStore
from brain_agent.memory.workspace_store import WorkspaceStore


@pytest.fixture
async def stores(tmp_path):
    workspace_store = WorkspaceStore(db_path=str(tmp_path / "workspaces.db"))
    await workspace_store.initialize()
    ontology_store = OntologyStore(db_path=str(tmp_path / "ontology.db"))
    await ontology_store.initialize()
    yield workspace_store, ontology_store
    await ontology_store.close()
    await workspace_store.close()


async def test_apply_software_project_registers_10_node_types(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="Billing")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    types = [
        item
        for item in await ontology_store.get_node_types(workspace["id"])
        if item["workspace_id"] == workspace["id"]
    ]
    assert len(types) == 10
    names = {item["name"] for item in types}
    assert "Requirement" in names
    assert "DataModel" in names


async def test_apply_software_project_registers_10_relation_types(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="Billing2")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    relations = [
        item
        for item in await ontology_store.get_relation_types(workspace["id"])
        if item["workspace_id"] == workspace["id"]
    ]
    assert len(relations) == 10
    names = {item["name"] for item in relations}
    assert "implements" in names
    assert "trades_off_against" in names


async def test_apply_template_sets_confidence_canonical(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="Billing3")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    local_types = [
        item
        for item in await ontology_store.get_node_types(workspace["id"])
        if item["workspace_id"] == workspace["id"]
    ]
    local_relations = [
        item
        for item in await ontology_store.get_relation_types(workspace["id"])
        if item["workspace_id"] == workspace["id"]
    ]
    assert all(item["confidence"] == "CANONICAL" for item in local_types)
    assert all(item["confidence"] == "CANONICAL" for item in local_relations)


async def test_apply_template_sets_source_id_template(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="BillingSource")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    local_types = [
        item
        for item in await ontology_store.get_node_types(workspace["id"])
        if item["workspace_id"] == workspace["id"]
    ]
    local_relations = [
        item
        for item in await ontology_store.get_relation_types(workspace["id"])
        if item["workspace_id"] == workspace["id"]
    ]
    assert all(item["source_id"] == "template" for item in local_types)
    assert all(item["source_id"] == "template" for item in local_relations)


async def test_apply_template_parent_chain_requirement_to_statement(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="Billing4")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    requirement = await ontology_store.resolve_node_type(workspace["id"], "Requirement")
    statement = await ontology_store.resolve_node_type(workspace["id"], "Statement")
    assert requirement["parent_type_id"] == statement["id"]
    assert statement["workspace_id"] == UNIVERSAL_WORKSPACE_ID


async def test_apply_template_resolves_domain_range(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="Billing5")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    implements = await ontology_store.resolve_relation_type(
        workspace["id"], "implements"
    )
    module = await ontology_store.resolve_node_type(workspace["id"], "Module")
    requirement = await ontology_store.resolve_node_type(workspace["id"], "Requirement")
    assert implements["domain_type_id"] == module["id"]
    assert implements["range_type_id"] == requirement["id"]


async def test_apply_template_records_version_on_workspace(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="Billing6")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    updated = await workspace_store.get_workspace(workspace["id"])
    assert updated["template_id"] == "software-project"
    assert updated["template_version"] == "1.0"


async def test_apply_template_auto_seeds_universal_if_missing(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="Billing7")
    universal_before = await ontology_store.get_node_types(UNIVERSAL_WORKSPACE_ID)
    assert len(universal_before) == 0
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    universal_after = await ontology_store.get_node_types(UNIVERSAL_WORKSPACE_ID)
    assert len(universal_after) == 7


async def test_apply_template_unknown_name_raises(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="Billing8")
    with pytest.raises(ValueError, match="Unknown template"):
        await ontology_store.apply_template(
            workspace["id"], "no-such-template", workspace_store=workspace_store
        )


async def test_apply_template_two_workspaces_isolated(stores):
    workspace_store, ontology_store = stores
    workspace_a = await workspace_store.create_workspace(name="ProjA")
    workspace_b = await workspace_store.create_workspace(name="ProjB")
    await ontology_store.apply_template(
        workspace_a["id"], "software-project", workspace_store=workspace_store
    )
    await ontology_store.apply_template(
        workspace_b["id"], "research-notes", workspace_store=workspace_store
    )
    local_a = [
        item
        for item in await ontology_store.get_node_types(workspace_a["id"])
        if item["workspace_id"] == workspace_a["id"]
    ]
    local_b = [
        item
        for item in await ontology_store.get_node_types(workspace_b["id"])
        if item["workspace_id"] == workspace_b["id"]
    ]
    assert {item["name"] for item in local_a} == {
        "Requirement",
        "Decision",
        "Module",
        "Interface",
        "Constraint",
        "Risk",
        "NonGoal",
        "Workflow",
        "DomainTerm",
        "DataModel",
    }
    assert {item["name"] for item in local_b} == {
        "Hypothesis",
        "Experiment",
        "Result",
        "Citation",
        "Finding",
        "Methodology",
    }


async def test_apply_template_idempotent(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="BillingIdem")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    local_types = [
        item
        for item in await ontology_store.get_node_types(workspace["id"])
        if item["workspace_id"] == workspace["id"]
    ]
    assert len(local_types) == 10


async def test_compose_two_templates_sums_disjoint_types(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="Mixed")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    await ontology_store.apply_template(
        workspace["id"], "personal-knowledge", workspace_store=workspace_store
    )
    local_types = [
        item
        for item in await ontology_store.get_node_types(workspace["id"])
        if item["workspace_id"] == workspace["id"]
    ]
    assert len(local_types) == 15


async def test_compose_template_id_is_last_applied(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="Mixed2")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    await ontology_store.apply_template(
        workspace["id"], "research-notes", workspace_store=workspace_store
    )
    updated = await workspace_store.get_workspace(workspace["id"])
    assert updated["template_id"] == "research-notes"
    assert updated["template_version"] == "1.0"


async def test_compose_overlapping_node_unions_schema(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="Merge")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    before = await ontology_store.resolve_node_type(workspace["id"], "Decision")
    assert "rationale" in before["schema"]["required"]

    await ontology_store._apply_template_node_types(
        workspace["id"],
        [
            {
                "name": "Decision",
                "parent": "Statement",
                "schema": {
                    "props": ["reversibility"],
                    "required": ["decided_by"],
                },
            }
        ],
    )

    after = await ontology_store.resolve_node_type(workspace["id"], "Decision")
    props = set(after["schema"]["props"])
    required = set(after["schema"]["required"])
    assert "rationale" in required
    assert "decided_by" in required
    assert "reversibility" in props
    assert {"date", "rationale", "alternatives", "decided_by"} <= props


async def test_compose_overlapping_relation_domain_range_last_wins(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="MergeRel")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    await ontology_store._apply_template_relation_types(
        workspace["id"],
        [
            {
                "name": "belongs_to",
                "transitive": False,
                "symmetric": False,
                "inverse_of": None,
                "domain": "Decision",
                "range": "Module",
            }
        ],
    )
    relation = await ontology_store.resolve_relation_type(workspace["id"], "belongs_to")
    decision = await ontology_store.resolve_node_type(workspace["id"], "Decision")
    module = await ontology_store.resolve_node_type(workspace["id"], "Module")
    assert relation["domain_type_id"] == decision["id"]
    assert relation["range_type_id"] == module["id"]
