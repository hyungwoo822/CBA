"""Tests for OntologyStore.upgrade_template and downgrade_template."""
from unittest.mock import patch

import pytest

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


def _patch_template_resolver():
    from brain_agent.memory.templates import software_project as v1_0
    from tests.memory.templates.fixtures import (
        software_project_v1_1 as v1_1,
        software_project_v2_0 as v2_0,
    )

    def fake_get_template_at(name: str, version: str) -> dict:
        if name != "software-project":
            raise ValueError(f"fixture only covers software-project, got {name}")
        if version == "1.0":
            return v1_0.TEMPLATE
        if version == "1.1":
            return v1_1.TEMPLATE
        if version == "2.0":
            return v2_0.TEMPLATE
        raise ValueError(f"no fixture for {version}")

    return fake_get_template_at


async def test_upgrade_dry_run_returns_diff_without_db_change(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="DryRun")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    with patch.object(
        ontology_store,
        "_get_template_at_version",
        side_effect=_patch_template_resolver(),
    ):
        diff = await ontology_store.upgrade_template(
            workspace["id"],
            target_version="1.1",
            workspace_store=workspace_store,
            dry_run=True,
        )
    assert "Metric" in [item["name"] for item in diff["added"]["node_types"]]
    assert "measures" in [item["name"] for item in diff["added"]["relation_types"]]
    local_types = [
        item
        for item in await ontology_store.get_node_types(workspace["id"])
        if item["workspace_id"] == workspace["id"]
    ]
    assert "Metric" not in {item["name"] for item in local_types}
    updated = await workspace_store.get_workspace(workspace["id"])
    assert updated["template_version"] == "1.0"


async def test_upgrade_minor_applies_added_types(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="MinorBump")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    with patch.object(
        ontology_store,
        "_get_template_at_version",
        side_effect=_patch_template_resolver(),
    ):
        diff = await ontology_store.upgrade_template(
            workspace["id"],
            target_version="1.1",
            workspace_store=workspace_store,
        )
    assert "Metric" in [item["name"] for item in diff["added"]["node_types"]]
    metric = await ontology_store.resolve_node_type(workspace["id"], "Metric")
    assert metric is not None
    assert metric["workspace_id"] == workspace["id"]
    measures = await ontology_store.resolve_relation_type(workspace["id"], "measures")
    assert measures is not None
    updated = await workspace_store.get_workspace(workspace["id"])
    assert updated["template_version"] == "1.1"


async def test_upgrade_minor_modifies_schema_with_added_required_emits_warning(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="SchemaBump")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    from tests.memory.templates.fixtures import software_project_v1_1 as v1_1

    patched_template = dict(v1_1.TEMPLATE)
    patched_template["version"] = "1.2"
    patched_template["node_types"] = [
        dict(
            node_type,
            schema={
                "props": node_type["schema"].get("props", []) + ["ttl"],
                "required": node_type["schema"].get("required", []) + ["decided_by"],
            },
        )
        if node_type["name"] == "Decision"
        else node_type
        for node_type in patched_template["node_types"]
    ]

    def resolver(name: str, version: str) -> dict:
        if version == "1.2":
            return patched_template
        return _patch_template_resolver()(name, version)

    with patch.object(ontology_store, "_get_template_at_version", side_effect=resolver):
        diff = await ontology_store.upgrade_template(
            workspace["id"],
            target_version="1.2",
            workspace_store=workspace_store,
        )
    messages = " ".join(diff["warnings"])
    assert "Decision" in messages
    assert "decided_by" in messages


async def test_upgrade_major_without_confirm_raises(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="MajorBump")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    with patch.object(
        ontology_store,
        "_get_template_at_version",
        side_effect=_patch_template_resolver(),
    ):
        with pytest.raises(ValueError, match="confirm"):
            await ontology_store.upgrade_template(
                workspace["id"],
                target_version="2.0",
                workspace_store=workspace_store,
                confirm=False,
            )


async def test_upgrade_major_with_confirm_proceeds(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="MajorBumpOK")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    with patch.object(
        ontology_store,
        "_get_template_at_version",
        side_effect=_patch_template_resolver(),
    ):
        await ontology_store.upgrade_template(
            workspace["id"],
            target_version="2.0",
            workspace_store=workspace_store,
            confirm=True,
        )
    non_goal = await ontology_store.resolve_node_type(workspace["id"], "NonGoal")
    assert non_goal is not None
    assert non_goal["deprecated"] == 1
    updated = await workspace_store.get_workspace(workspace["id"])
    assert updated["template_version"] == "2.0"


async def test_upgrade_soft_deletes_type_in_use(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="SoftDel")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    before = await ontology_store.resolve_node_type(workspace["id"], "NonGoal")
    assert before["deprecated"] == 0
    with patch.object(
        ontology_store,
        "_get_template_at_version",
        side_effect=_patch_template_resolver(),
    ):
        await ontology_store.upgrade_template(
            workspace["id"],
            target_version="2.0",
            workspace_store=workspace_store,
            confirm=True,
        )
    after = await ontology_store.resolve_node_type(workspace["id"], "NonGoal")
    assert after is not None
    assert after["deprecated"] == 1


async def test_upgrade_diff_shape(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="DiffShape")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    with patch.object(
        ontology_store,
        "_get_template_at_version",
        side_effect=_patch_template_resolver(),
    ):
        diff = await ontology_store.upgrade_template(
            workspace["id"],
            target_version="1.1",
            workspace_store=workspace_store,
            dry_run=True,
        )
    for section in ("added", "removed", "modified"):
        assert section in diff
        assert "node_types" in diff[section]
        assert "relation_types" in diff[section]
    assert "warnings" in diff
    assert isinstance(diff["warnings"], list)


async def test_downgrade_always_raises(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="NoDowngrade")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    with pytest.raises(ValueError, match="Downgrades not supported"):
        await ontology_store.downgrade_template(
            workspace["id"], target_version="0.9", workspace_store=workspace_store
        )


async def test_downgrade_does_not_touch_workspace(stores):
    workspace_store, ontology_store = stores
    workspace = await workspace_store.create_workspace(name="NoDowngrade2")
    await ontology_store.apply_template(
        workspace["id"], "software-project", workspace_store=workspace_store
    )
    try:
        await ontology_store.downgrade_template(
            workspace["id"], target_version="0.9", workspace_store=workspace_store
        )
    except ValueError:
        pass
    still = await workspace_store.get_workspace(workspace["id"])
    assert still["template_version"] == "1.0"
