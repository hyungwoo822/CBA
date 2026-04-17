"""End-to-end smoke tests for bundled ontology templates."""
import pytest

from brain_agent.memory.ontology_seed import UNIVERSAL_WORKSPACE_ID
from brain_agent.memory.ontology_store import OntologyStore
from brain_agent.memory.workspace_store import WorkspaceStore


@pytest.fixture
async def fresh(tmp_path):
    workspace_store = WorkspaceStore(db_path=str(tmp_path / "workspaces.db"))
    await workspace_store.initialize()
    ontology_store = OntologyStore(db_path=str(tmp_path / "ontology.db"))
    await ontology_store.initialize()
    yield workspace_store, ontology_store
    await ontology_store.close()
    await workspace_store.close()


async def test_three_templates_three_workspaces(fresh):
    workspace_store, ontology_store = fresh
    engineering = await workspace_store.create_workspace(name="Eng")
    research = await workspace_store.create_workspace(name="Lab")
    personal = await workspace_store.create_workspace(name="Life")
    await ontology_store.apply_template(
        engineering["id"], "software-project", workspace_store=workspace_store
    )
    await ontology_store.apply_template(
        research["id"], "research-notes", workspace_store=workspace_store
    )
    await ontology_store.apply_template(
        personal["id"], "personal-knowledge", workspace_store=workspace_store
    )

    def local_rows(workspace_id: str, rows: list[dict]) -> list[dict]:
        return [item for item in rows if item["workspace_id"] == workspace_id]

    engineering_rows = await ontology_store.get_node_types(engineering["id"])
    research_rows = await ontology_store.get_node_types(research["id"])
    personal_rows = await ontology_store.get_node_types(personal["id"])
    assert len(local_rows(engineering["id"], engineering_rows)) == 10
    assert len(local_rows(research["id"], research_rows)) == 6
    assert len(local_rows(personal["id"], personal_rows)) == 5
    assert (
        len(
            [
                item
                for item in engineering_rows
                if item["workspace_id"] == UNIVERSAL_WORKSPACE_ID
            ]
        )
        == 7
    )
    assert (
        len(
            [
                item
                for item in research_rows
                if item["workspace_id"] == UNIVERSAL_WORKSPACE_ID
            ]
        )
        == 7
    )
    assert (
        len(
            [
                item
                for item in personal_rows
                if item["workspace_id"] == UNIVERSAL_WORKSPACE_ID
            ]
        )
        == 7
    )


async def test_template_version_round_trip(fresh):
    workspace_store, ontology_store = fresh
    workspace = await workspace_store.create_workspace(name="VerRT")
    await ontology_store.apply_template(
        workspace["id"], "research-notes", workspace_store=workspace_store
    )
    updated = await workspace_store.get_workspace(workspace["id"])
    assert updated["template_id"] == "research-notes"
    assert updated["template_version"] == "1.0"


async def test_inverse_of_resolution_in_templates(fresh):
    workspace_store, ontology_store = fresh
    workspace = await workspace_store.create_workspace(name="InvRT")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    supersedes = await ontology_store.resolve_relation_type(
        workspace["id"], "supersedes"
    )
    superseded_by = await ontology_store.resolve_relation_type(
        workspace["id"], "superseded_by"
    )
    assert supersedes["inverse_of"] == superseded_by["id"]
    assert superseded_by["inverse_of"] == supersedes["id"]


async def test_template_types_carry_deprecated_flag(fresh):
    workspace_store, ontology_store = fresh
    workspace = await workspace_store.create_workspace(name="DepFlag")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    requirement = await ontology_store.resolve_node_type(workspace["id"], "Requirement")
    assert requirement["deprecated"] == 0
