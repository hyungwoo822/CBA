"""Tests for Stage 2.5 Temporal Resolve."""
from unittest.mock import AsyncMock

import pytest

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.temporal_resolver import TemporalResolver


@pytest.fixture
def semantic_store():
    store = AsyncMock()
    store.get_relationships = AsyncMock(return_value=[])
    return store


@pytest.fixture
def resolver(semantic_store, mock_llm):
    return TemporalResolver(semantic_store, mock_llm, ExtractionConfig())


async def test_no_existing_edge_adds_as_new(resolver, semantic_store):
    semantic_store.get_relationships = AsyncMock(return_value=[])
    edge = {"source": "hyungpu", "relation": "use", "target": "python"}
    result = await resolver.resolve([edge], "I use Python", "personal")
    assert result.new_edges == [edge]
    assert result.update_ops == []
    assert result.reinforced_edges == []


async def test_same_target_reinforces(resolver, semantic_store):
    existing = {
        "id": "e-1",
        "source": "hyungpu",
        "relation": "use",
        "target": "python",
        "valid_to": None,
        "type_id": "rel-1",
    }
    semantic_store.get_relationships = AsyncMock(return_value=[existing])
    edge = {"source": "hyungpu", "relation": "use", "target": "python"}
    result = await resolver.resolve([edge], "I still use Python", "personal")
    assert result.reinforced_edges == [existing]
    assert result.new_edges == []
    assert result.update_ops == []


async def test_explicit_past_marker_supersedes(resolver, semantic_store):
    existing = {
        "id": "e-2",
        "source": "hyungpu",
        "relation": "use",
        "target": "python",
        "valid_to": None,
    }
    semantic_store.get_relationships = AsyncMock(return_value=[existing])
    edge = {"source": "hyungpu", "relation": "use", "target": "go"}
    result = await resolver.resolve([edge], "previously Python, now Go", "personal")
    assert len(result.update_ops) == 1
    assert result.update_ops[0]["type"] == "supersede"
    assert result.update_ops[0]["edge_id"] == "e-2"
    assert result.update_ops[0]["valid_to"] is not None
    assert result.new_edges[0]["target"] == "go"
    assert result.new_edges[0].get("valid_from") is not None


async def test_no_markers_llm_update_supersedes(resolver, semantic_store, mock_llm):
    semantic_store.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "e-3",
                "source": "hyungpu",
                "relation": "use",
                "target": "python",
                "valid_to": None,
            }
        ]
    )
    mock_llm.enqueue_content("update")
    edge = {"source": "hyungpu", "relation": "use", "target": "rust"}
    result = await resolver.resolve([edge], "rust is better", "personal")
    assert len(result.update_ops) == 1
    assert len(result.new_edges) == 1
    assert len(mock_llm.calls) == 1


async def test_no_markers_llm_contradiction_passes_through(resolver, semantic_store, mock_llm):
    semantic_store.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "e-4",
                "source": "hyungpu",
                "relation": "use",
                "target": "python",
                "valid_to": None,
            }
        ]
    )
    mock_llm.enqueue_content("contradiction")
    edge = {"source": "hyungpu", "relation": "use", "target": "rust"}
    result = await resolver.resolve([edge], "rust", "personal")
    assert result.update_ops == []
    assert result.new_edges[0].get("temporal_ambiguous") is not True
    assert result.new_edges[0]["target"] == "rust"


async def test_no_markers_llm_ambiguous_tags_edge(resolver, semantic_store, mock_llm):
    semantic_store.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "e-5",
                "source": "hyungpu",
                "relation": "use",
                "target": "python",
                "valid_to": None,
            }
        ]
    )
    mock_llm.enqueue_content("ambiguous")
    edge = {"source": "hyungpu", "relation": "use", "target": "rust"}
    result = await resolver.resolve([edge], "rust", "personal")
    assert result.new_edges[0]["temporal_ambiguous"] is True


async def test_already_superseded_edge_ignored(resolver, semantic_store):
    semantic_store.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "e-6",
                "source": "hyungpu",
                "relation": "use",
                "target": "python",
                "valid_to": "2025-01-01T00:00:00Z",
            }
        ]
    )
    edge = {"source": "hyungpu", "relation": "use", "target": "go"}
    result = await resolver.resolve([edge], "use go", "personal")
    assert result.new_edges == [edge]
    assert result.update_ops == []


async def test_llm_uses_configured_model(resolver, semantic_store, mock_llm):
    semantic_store.get_relationships = AsyncMock(
        return_value=[
            {"id": "e-7", "source": "a", "relation": "r", "target": "b", "valid_to": None}
        ]
    )
    mock_llm.enqueue_content("update")
    resolver._cfg.temporal_classify_model = "gpt-4o-mini"
    await resolver.resolve([{"source": "a", "relation": "r", "target": "c"}], "r c", "personal")
    assert mock_llm.calls[0]["model"] == "gpt-4o-mini"
