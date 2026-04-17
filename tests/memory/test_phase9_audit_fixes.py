"""Phase 9 regression tests for Phase 0-8 audit Tier-1 fixes."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from brain_agent.dashboard.routers.curation import ResolveBody
from brain_agent.memory.hippocampal_staging import HippocampalStaging


@pytest.mark.asyncio
async def test_get_unconsolidated_defaults_to_personal_workspace(tmp_path):
    """Legacy callers with no workspace arg must keep seeing personal only."""
    staging = HippocampalStaging(str(tmp_path / "staging.db"), lambda _text: [0.1])
    await staging.initialize()
    try:
        await staging.encode(
            content="personal item",
            entities={},
            interaction_id=1,
            session_id="s1",
            workspace_id="personal",
        )
        await staging.encode(
            content="project item",
            entities={},
            interaction_id=2,
            session_id="s1",
            workspace_id="project_alpha",
        )

        items = await staging.get_unconsolidated()

        assert len(items) == 1
        assert items[0]["content"] == "personal item"
    finally:
        await staging.close()


def test_resolve_body_accepts_confidence_label_string():
    """ResolveBody.resolution_confidence is a CONFIDENCE_RANK label string."""
    body = ResolveBody(
        resolution="value_a",
        resolved_by="user",
        resolution_confidence="USER_GROUND_TRUTH",
    )
    assert body.resolution_confidence == "USER_GROUND_TRUTH"


def test_resolve_body_rejects_unknown_confidence_label():
    """Unknown confidence labels must be rejected so typos don't slip through."""
    with pytest.raises(ValidationError):
        ResolveBody(
            resolution="value_a",
            resolved_by="user",
            resolution_confidence="NOT_A_REAL_LABEL",
        )


@pytest.mark.asyncio
async def test_template_compose_preserves_prior_domain_range_inverse_on_none(
    memory_manager,
):
    """Template overlays that omit relation metadata must not clobber it."""
    workspace = await memory_manager.workspace.create_workspace(name="Phase 9")
    ontology = memory_manager.ontology

    await ontology.register_relation_type(
        workspace["id"],
        "collaborates_with",
        domain_type="Person",
        range_type="Person",
        source_id="first_template",
    )
    await ontology.register_relation_type(
        workspace["id"],
        "collaborated_by",
        domain_type="Person",
        range_type="Person",
        source_id="first_template",
    )
    await ontology._apply_template_relation_types(
        workspace["id"],
        [
            {
                "name": "collaborates_with",
                "domain": "Person",
                "range": "Person",
                "inverse_of": "collaborated_by",
            },
            {
                "name": "collaborated_by",
                "domain": "Person",
                "range": "Person",
                "inverse_of": "collaborates_with",
            },
        ],
    )

    row_before = await ontology._get_relation_type_by_name(
        workspace["id"], "collaborates_with"
    )
    assert row_before is not None
    assert row_before["domain_type_id"] is not None
    assert row_before["range_type_id"] is not None
    assert row_before["inverse_of"] is not None

    await ontology._apply_template_relation_types(
        workspace["id"],
        [
            {
                "name": "collaborates_with",
            }
        ],
    )

    row_after = await ontology._get_relation_type_by_name(
        workspace["id"], "collaborates_with"
    )
    assert row_after is not None
    assert row_after["domain_type_id"] == row_before["domain_type_id"]
    assert row_after["range_type_id"] == row_before["range_type_id"]
    assert row_after["inverse_of"] == row_before["inverse_of"]
