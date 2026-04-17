"""Tests for Stage 3 Validate."""
from unittest.mock import AsyncMock

import pytest

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.validator import Validator


@pytest.fixture
def semantic_store():
    store = AsyncMock()
    store.get_relationships = AsyncMock(return_value=[])
    store.find_events_near = AsyncMock(return_value=[])
    store.search = AsyncMock(return_value=[])
    store.is_hub_node = AsyncMock(return_value=False)
    return store


@pytest.fixture
def ontology_store():
    store = AsyncMock()
    store.get_node_schema = AsyncMock(return_value={"required": []})
    return store


@pytest.fixture
def validator(semantic_store, ontology_store):
    return Validator(semantic_store, ontology_store, ExtractionConfig())


async def test_no_existing_edges_no_contradictions(validator):
    result = await validator.validate(
        nodes=[],
        edges=[{"source": "a", "relation": "r", "target": "b", "confidence": "EXTRACTED"}],
        workspace_id="personal",
        narrative_chunk="a r b",
        input_kinds=["spec_drop"],
    )
    assert result.contradictions == []


async def test_same_subject_relation_different_target_is_contradiction(validator, semantic_store):
    semantic_store.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "e-1",
                "source": "a",
                "relation": "r",
                "target": "b",
                "valid_to": None,
                "confidence": "EXTRACTED",
            }
        ]
    )
    result = await validator.validate(
        nodes=[],
        edges=[{"source": "a", "relation": "r", "target": "c", "confidence": "EXTRACTED"}],
        workspace_id="personal",
        narrative_chunk="a r c",
        input_kinds=["spec_drop"],
    )
    assert len(result.contradictions) == 1
    contradiction = result.contradictions[0]
    assert contradiction["subject"] == "a"
    assert contradiction["key"] == "r"
    assert contradiction["value_a"] == "b"
    assert contradiction["value_b"] == "c"
    assert contradiction["severity"] in {"minor", "moderate", "severe"}


async def test_superseded_edge_is_skipped(validator, semantic_store):
    semantic_store.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "e-2",
                "source": "a",
                "relation": "r",
                "target": "b",
                "valid_to": "2025-01-01T00:00:00Z",
                "confidence": "EXTRACTED",
            }
        ]
    )
    result = await validator.validate(
        nodes=[],
        edges=[{"source": "a", "relation": "r", "target": "c", "confidence": "EXTRACTED"}],
        workspace_id="personal",
        narrative_chunk="",
        input_kinds=["spec_drop"],
    )
    assert result.contradictions == []


async def test_missing_required_property_raises_question(validator, ontology_store):
    ontology_store.get_node_schema = AsyncMock(return_value={"required": ["launch_date"]})
    result = await validator.validate(
        nodes=[{"type": "Event", "label": "product_launch", "properties": {}, "confidence": "EXTRACTED"}],
        edges=[],
        workspace_id="personal",
        narrative_chunk="",
        input_kinds=["spec_drop"],
    )
    assert len(result.open_questions) == 1
    assert "launch_date" in result.open_questions[0]["question"]
    assert result.open_questions[0]["raised_by"] == "ambiguity_detector"


async def test_question_cap_enforced(validator, ontology_store):
    ontology_store.get_node_schema = AsyncMock(return_value={"required": ["p1", "p2", "p3", "p4", "p5"]})
    result = await validator.validate(
        nodes=[{"type": "Event", "label": "e1", "properties": {}, "confidence": "EXTRACTED"}],
        edges=[],
        workspace_id="personal",
        narrative_chunk="",
        input_kinds=["spec_drop"],
    )
    assert len(result.open_questions) <= 3


async def test_pattern_separation_similar_events_raises_question(validator, semantic_store):
    semantic_store.find_events_near = AsyncMock(
        return_value=[
            {"id": "ev-old", "label": "product_launch", "happened_at": "2025-04-01T10:00:00Z"}
        ]
    )
    result = await validator.validate(
        nodes=[
            {
                "type": "Event",
                "label": "product_launch_v2",
                "properties": {"happened_at": "2025-04-01T14:00:00Z"},
                "confidence": "EXTRACTED",
                "id": "ev-new",
            }
        ],
        edges=[],
        workspace_id="personal",
        narrative_chunk="",
        input_kinds=["spec_drop"],
    )
    assert any(question["raised_by"] == "pattern_separation" for question in result.open_questions)


async def test_fok_pre_retrieval_on_unanswerable_question(validator, semantic_store):
    semantic_store.search = AsyncMock(return_value=[])
    result = await validator.validate(
        nodes=[],
        edges=[],
        workspace_id="personal",
        narrative_chunk="How many retries does billing allow?",
        input_kinds=["question"],
    )
    assert any(question["raised_by"] == "fok_pre_retrieval" for question in result.open_questions)


async def test_fok_suppressed_when_hits_strong(validator, semantic_store):
    semantic_store.search = AsyncMock(return_value=[{"similarity": 0.85, "content": "billing retries = 3"}])
    result = await validator.validate(
        nodes=[],
        edges=[],
        workspace_id="personal",
        narrative_chunk="How many retries?",
        input_kinds=["question"],
    )
    assert not any(question["raised_by"] == "fok_pre_retrieval" for question in result.open_questions)


async def test_fok_only_runs_when_question_kind(validator, semantic_store):
    semantic_store.search = AsyncMock(return_value=[])
    result = await validator.validate(
        nodes=[],
        edges=[],
        workspace_id="personal",
        narrative_chunk="random spec text",
        input_kinds=["spec_drop"],
    )
    assert not any(question["raised_by"] == "fok_pre_retrieval" for question in result.open_questions)


async def test_alias_skips_contradiction(validator, semantic_store):
    semantic_store.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "e-3",
                "source": "a",
                "relation": "r",
                "target": "python3",
                "valid_to": None,
                "confidence": "EXTRACTED",
            }
        ]
    )
    result = await validator.validate(
        nodes=[],
        edges=[{"source": "a", "relation": "r", "target": "python", "confidence": "EXTRACTED"}],
        workspace_id="personal",
        narrative_chunk="",
        input_kinds=["spec_drop"],
    )
    assert result.contradictions == []
