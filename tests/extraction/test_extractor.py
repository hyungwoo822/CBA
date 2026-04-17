"""Tests for Stage 2 Workspace-Aware Extract."""
import json
from unittest.mock import AsyncMock

import pytest

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.extractor import Extractor


@pytest.fixture
def fake_ontology_store():
    store = AsyncMock()
    store.get_node_types = AsyncMock(
        return_value=[
            {"name": "Person", "parent": None},
            {"name": "Food", "parent": None},
            {"name": "Concept", "parent": None},
            {"name": "Event", "parent": None},
        ]
    )
    store.get_relation_types = AsyncMock(
        return_value=[
            {"name": "prefer", "description": "user prefers x"},
            {"name": "refers_to", "description": "generic reference"},
            {"name": "happens_at", "description": "event timing"},
        ]
    )
    store.register_node_type = AsyncMock()
    store.propose_node_type = AsyncMock()
    return store


@pytest.fixture
def extractor(fake_ontology_store, mock_llm):
    return Extractor(fake_ontology_store, mock_llm, ExtractionConfig())


async def test_happy_path_valid_json(extractor, mock_llm):
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [
                    {"type": "Person", "label": "hyungpu", "properties": {}, "confidence": "EXTRACTED"},
                    {"type": "Food", "label": "jjambbong", "properties": {}, "confidence": "EXTRACTED"},
                ],
                "edges": [
                    {
                        "source": "hyungpu",
                        "relation": "prefer",
                        "target": "jjambbong",
                        "confidence": "EXTRACTED",
                        "epistemic_source": "asserted",
                        "importance_score": 0.6,
                        "never_decay": 0,
                    }
                ],
                "new_type_proposals": [],
                "narrative_chunk": "hyungpu likes jjambbong",
            }
        )
    )
    result = await extractor.extract("hyungpu likes jjambbong", "personal")
    assert len(result.nodes) == 2
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge["epistemic_source"] == "asserted"
    assert edge["importance_score"] == 0.6
    assert edge["never_decay"] == 0


async def test_invalid_json_retries_then_falls_back(extractor, mock_llm):
    mock_llm.enqueue_content("not valid json at all")
    mock_llm.enqueue_content("still not json")
    result = await extractor.extract("hyungpu likes jjambbong", "personal")
    assert result.narrative_chunk == "hyungpu likes jjambbong"
    assert result.edges == []
    assert len(mock_llm.calls) == 2


async def test_retry_succeeds_on_second_call(extractor, mock_llm):
    mock_llm.enqueue_content("garbage")
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [{"type": "Person", "label": "hyungpu", "properties": {}, "confidence": "EXTRACTED"}],
                "edges": [],
                "new_type_proposals": [],
                "narrative_chunk": "hyungpu",
            }
        )
    )
    result = await extractor.extract("hyungpu", "personal")
    assert len(mock_llm.calls) == 2
    assert len(result.nodes) == 1
    retry_user_content = "\n".join(
        message["content"]
        for message in mock_llm.calls[1]["messages"]
        if message["role"] == "user"
    )
    assert "previous output" in retry_user_content.lower() or "error" in retry_user_content.lower()


async def test_ontology_violation_retries(extractor, mock_llm):
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [{"type": "UnknownType", "label": "x", "properties": {}, "confidence": "EXTRACTED"}],
                "edges": [],
                "new_type_proposals": [],
                "narrative_chunk": "x",
            }
        )
    )
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [{"type": "Concept", "label": "x", "properties": {}, "confidence": "EXTRACTED"}],
                "edges": [],
                "new_type_proposals": [],
                "narrative_chunk": "x",
            }
        )
    )
    result = await extractor.extract("x", "personal")
    assert result.nodes[0]["type"] == "Concept"
    assert len(mock_llm.calls) == 2


async def test_edge_defaults_for_missing_s3_s7_s8(extractor, mock_llm):
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [
                    {"type": "Person", "label": "a", "properties": {}, "confidence": "EXTRACTED"},
                    {"type": "Concept", "label": "b", "properties": {}, "confidence": "EXTRACTED"},
                ],
                "edges": [{"source": "a", "relation": "refers_to", "target": "b", "confidence": "EXTRACTED"}],
                "new_type_proposals": [],
                "narrative_chunk": "a refers to b",
            }
        )
    )
    result = await extractor.extract("a refers to b", "personal")
    edge = result.edges[0]
    assert edge["epistemic_source"] == "asserted"
    assert edge["importance_score"] == 0.5
    assert edge["never_decay"] == 0


async def test_prompt_embeds_ontology_types(extractor, mock_llm):
    mock_llm.enqueue_content(
        json.dumps({"nodes": [], "edges": [], "new_type_proposals": [], "narrative_chunk": "x"})
    )
    await extractor.extract("x", "personal")
    system_msg = mock_llm.calls[0]["messages"][0]["content"]
    assert "Person" in system_msg
    assert "Food" in system_msg
    assert "prefer" in system_msg


async def test_new_type_proposal_extracted_routes_to_result_not_store(
    extractor,
    fake_ontology_store,
    mock_llm,
):
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [],
                "edges": [],
                "new_type_proposals": [
                    {
                        "kind": "node",
                        "name": "Endpoint",
                        "definition": "API route",
                        "confidence": "EXTRACTED",
                        "source_snippet": "API endpoint",
                    }
                ],
                "narrative_chunk": "x",
            }
        )
    )
    result = await extractor.extract("x", "personal")
    assert len(result.new_type_proposals) == 1
    fake_ontology_store.register_node_type.assert_not_called()
    fake_ontology_store.propose_node_type.assert_not_called()


async def test_uses_configured_model(extractor, mock_llm):
    mock_llm.enqueue_content(
        json.dumps({"nodes": [], "edges": [], "new_type_proposals": [], "narrative_chunk": "x"})
    )
    extractor._cfg.extract_model = "claude-haiku-4-5"
    await extractor.extract("x", "personal")
    assert mock_llm.calls[0]["model"] == "claude-haiku-4-5"


async def test_auto_resolves_default_model(extractor, mock_llm):
    mock_llm.enqueue_content(
        json.dumps({"nodes": [], "edges": [], "new_type_proposals": [], "narrative_chunk": "x"})
    )
    extractor._cfg.extract_model = "auto"
    await extractor.extract("x", "personal")
    assert mock_llm.calls[0]["model"] == "mock-default"
