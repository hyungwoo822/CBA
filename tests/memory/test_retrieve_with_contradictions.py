"""S1 and S2 retrieval post-processing tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_contradictions_appended_to_retrieval_result():
    from brain_agent.memory.manager import MemoryManager

    mm = MagicMock(spec=MemoryManager)
    mm.retrieve = AsyncMock(return_value={
        "candidates": [{"id": 1, "content": "user lives in Seoul"}],
        "edges": [{"subject": "user", "relation": "lives_in", "target": "Seoul"}],
        "nodes": [],
    })
    mm.contradictions = MagicMock()
    mm.contradictions.get_for_subject_batch = AsyncMock(return_value={
        "user": [{"subject": "user", "key": "lives_in", "value_a": "Seoul", "value_b": "Busan"}],
    })
    mm.ontology = MagicMock()
    mm.ontology.resolve_node_type_by_id = AsyncMock(return_value=None)

    result = await MemoryManager.retrieve_with_contradictions(
        mm, query="user", workspace_id="personal", top_k=5,
    )

    assert len(result["contradictions"]) == 1
    assert result["contradictions"][0]["value_a"] == "Seoul"


@pytest.mark.asyncio
async def test_gaps_detected_for_missing_required_properties():
    from brain_agent.memory.manager import MemoryManager

    mm = MagicMock(spec=MemoryManager)
    mm.retrieve = AsyncMock(return_value={
        "candidates": [],
        "edges": [],
        "nodes": [{
            "id": "ev1",
            "label": "hospital_visit",
            "type_id": "t_event",
            "properties": {"actor": "user"},
        }],
    })
    mm.contradictions = MagicMock()
    mm.contradictions.get_for_subject_batch = AsyncMock(return_value={})
    mm.ontology = MagicMock()
    mm.ontology.resolve_node_type_by_id = AsyncMock(return_value={
        "name": "Event",
        "schema": {"props": ["happened_at", "actor"], "required": ["happened_at"]},
    })

    result = await MemoryManager.retrieve_with_contradictions(
        mm, query="hospital", workspace_id="personal", top_k=5,
    )

    assert result["gaps"] == [{
        "node": "hospital_visit",
        "missing": "happened_at",
        "type": "Event",
    }]


@pytest.mark.asyncio
async def test_no_contradictions_empty_list():
    from brain_agent.memory.manager import MemoryManager

    mm = MagicMock(spec=MemoryManager)
    mm.retrieve = AsyncMock(return_value={"candidates": [], "edges": [], "nodes": []})
    mm.contradictions = MagicMock()
    mm.contradictions.get_for_subject_batch = AsyncMock(return_value={})
    mm.ontology = MagicMock()
    mm.ontology.resolve_node_type_by_id = AsyncMock(return_value=None)

    result = await MemoryManager.retrieve_with_contradictions(
        mm, query="hello", workspace_id="personal", top_k=5,
    )

    assert result["contradictions"] == []
    assert result["gaps"] == []


@pytest.mark.asyncio
async def test_workspace_id_passed_through_to_retrieve():
    from brain_agent.memory.manager import MemoryManager

    mm = MagicMock(spec=MemoryManager)
    mm.retrieve = AsyncMock(return_value={"candidates": [], "edges": [], "nodes": []})
    mm.contradictions = MagicMock()
    mm.contradictions.get_for_subject_batch = AsyncMock(return_value={})
    mm.ontology = MagicMock()

    await MemoryManager.retrieve_with_contradictions(
        mm, query="hi", workspace_id="billing-service", top_k=3,
    )

    call_kwargs = mm.retrieve.await_args.kwargs
    assert call_kwargs["workspace_id"] == "billing-service"
    assert call_kwargs["top_k"] == 3
