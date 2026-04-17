"""Tests for ExtractionOrchestrator with mocked stores."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.orchestrator import ExtractionOrchestrator


def _stub_memory():
    memory = MagicMock()
    memory._interaction_counter = 0
    memory.raw_vault.ingest = AsyncMock(return_value={"id": "src-1", "kind": "user_utterance"})

    memory.workspace.get_session_workspace = AsyncMock(return_value="personal")

    async def _get_workspace(identifier):
        if identifier in {"personal", "Personal Knowledge"}:
            return {"id": "personal", "name": "Personal Knowledge"}
        return None

    memory.workspace.get_workspace = _get_workspace

    memory.ontology.get_node_types = AsyncMock(
        return_value=[
            {"name": "Person"},
            {"name": "Food"},
            {"name": "Concept"},
            {"name": "Event"},
        ]
    )
    memory.ontology.get_relation_types = AsyncMock(
        return_value=[
            {"name": "prefer"},
            {"name": "refers_to"},
            {"name": "use"},
            {"name": "r"},
        ]
    )
    memory.ontology.get_node_schema = AsyncMock(return_value={"required": []})
    memory.ontology.register_node_type = AsyncMock()
    memory.ontology.propose_node_type = AsyncMock()
    memory.ontology.register_relation_type = AsyncMock()
    memory.ontology.propose_relation_type = AsyncMock()
    memory.ontology.increment_occurrence = AsyncMock()

    memory.semantic.get_relationships = AsyncMock(return_value=[])
    memory.semantic.find_events_near = AsyncMock(return_value=[])
    memory.semantic.search = AsyncMock(return_value=[])
    memory.semantic.is_hub_node = AsyncMock(return_value=False)
    memory.semantic.mark_superseded = AsyncMock()
    memory.semantic.encode_edge = AsyncMock()
    memory.episodic.encode = AsyncMock()

    memory.staging.encode = AsyncMock()
    memory.staging.encode_edge = AsyncMock()
    memory.staging.reinforce = AsyncMock()

    memory.contradictions.detect = AsyncMock()
    memory.open_questions.add_question = AsyncMock()

    return memory


@pytest.fixture
def memory():
    return _stub_memory()


@pytest.fixture
def orchestrator(memory, mock_llm):
    return ExtractionOrchestrator(memory, mock_llm, ExtractionConfig())


async def test_greeting_skips_stages_and_persists_only_raw(orchestrator, memory, mock_llm):
    result = await orchestrator.extract("hello hyungpu", session_id="s1")
    assert result.response_mode == "normal"
    assert result.nodes == []
    assert result.edges == []
    memory.raw_vault.ingest.assert_called_once()
    assert len(mock_llm.calls) == 0
    memory.staging.encode_edge.assert_not_called()


async def test_full_extract_path_writes_to_staging_only(orchestrator, memory, mock_llm):
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
                        "importance_score": 0.7,
                        "never_decay": 0,
                    }
                ],
                "new_type_proposals": [],
                "narrative_chunk": "hyungpu prefers jjambbong",
            }
        )
    )
    result = await orchestrator.extract("hyungpu prefers jjambbong important", "s1")
    assert result.response_mode == "normal"
    assert len(result.edges) == 1
    memory.staging.encode_edge.assert_called_once()
    memory.staging.encode.assert_called_once()
    memory.semantic.encode_edge.assert_not_called()
    memory.episodic.encode.assert_not_called()


async def test_temporal_supersede_path(orchestrator, memory, mock_llm):
    memory.semantic.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "old-1",
                "source": "hyungpu",
                "relation": "use",
                "target": "python",
                "valid_to": None,
                "confidence": "EXTRACTED",
            }
        ]
    )
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [
                    {"type": "Person", "label": "hyungpu", "properties": {}, "confidence": "EXTRACTED"},
                    {"type": "Concept", "label": "go", "properties": {}, "confidence": "EXTRACTED"},
                ],
                "edges": [
                    {
                        "source": "hyungpu",
                        "relation": "use",
                        "target": "go",
                        "confidence": "EXTRACTED",
                        "epistemic_source": "asserted",
                        "importance_score": 0.5,
                        "never_decay": 0,
                    }
                ],
                "new_type_proposals": [],
                "narrative_chunk": "previously python, now go",
            }
        )
    )
    result = await orchestrator.extract("previously python, now go", "s1")
    assert result.response_mode != "block"
    memory.semantic.mark_superseded.assert_called_once()
    memory.staging.encode_edge.assert_called_once()
    memory.semantic.encode_edge.assert_not_called()


async def test_block_mode_returns_empty_response(orchestrator, memory, mock_llm):
    memory.semantic.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "old-1",
                "source": "hyungpu",
                "relation": "prefer",
                "target": "python",
                "valid_to": None,
                "confidence": "EXTRACTED",
            }
        ]
    )
    memory.semantic.is_hub_node = AsyncMock(return_value=True)
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [
                    {"type": "Person", "label": "hyungpu", "properties": {}, "confidence": "EXTRACTED"},
                    {"type": "Concept", "label": "rust", "properties": {}, "confidence": "EXTRACTED"},
                ],
                "edges": [
                    {
                        "source": "hyungpu",
                        "relation": "prefer",
                        "target": "rust",
                        "confidence": "EXTRACTED",
                        "epistemic_source": "asserted",
                        "importance_score": 0.5,
                        "never_decay": 0,
                    }
                ],
                "new_type_proposals": [],
                "narrative_chunk": "hyungpu prefers rust",
            }
        )
    )
    mock_llm.enqueue_content("contradiction")
    result = await orchestrator.extract(
        text="hyungpu prefers rust",
        session_id="s1",
        agent_response="should not be refined",
    )
    assert result.response_mode == "block"
    assert result.response_text == ""
    memory.contradictions.detect.assert_called_once()


async def test_refine_invoked_for_personal_non_block(orchestrator, mock_llm):
    mock_llm.enqueue_content("polished")
    result = await orchestrator.extract(
        text="hello",
        session_id="s1",
        agent_response="hello draft",
        language="ko",
    )
    assert result.response_text == "polished"
    assert result.response_mode == "normal"


async def test_new_type_proposal_extracted_routes_to_register(orchestrator, memory, mock_llm):
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
                "narrative_chunk": "endpoint",
            }
        )
    )
    await orchestrator.extract("endpoint", "s1")
    memory.ontology.register_node_type.assert_called_once()
    memory.ontology.propose_node_type.assert_not_called()


async def test_new_type_proposal_ambiguous_routes_to_propose(orchestrator, memory, mock_llm):
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [],
                "edges": [],
                "new_type_proposals": [
                    {
                        "kind": "node",
                        "name": "FuzzyThing",
                        "definition": "?",
                        "confidence": "AMBIGUOUS",
                        "source_snippet": "fuzzy",
                    }
                ],
                "narrative_chunk": "fuzzy",
            }
        )
    )
    await orchestrator.extract("fuzzy", "s1")
    memory.ontology.propose_node_type.assert_called_once()
    memory.ontology.register_node_type.assert_not_called()


async def test_contradictions_and_questions_routed_to_stores(orchestrator, memory, mock_llm):
    memory.semantic.get_relationships = AsyncMock(
        return_value=[
            {"id": "old-1", "source": "a", "relation": "r", "target": "b", "valid_to": None, "confidence": "EXTRACTED"}
        ]
    )
    memory.ontology.get_node_schema = AsyncMock(return_value={"required": ["when"]})
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [{"type": "Event", "label": "a", "properties": {}, "confidence": "EXTRACTED"}],
                "edges": [
                    {
                        "source": "a",
                        "relation": "r",
                        "target": "c",
                        "confidence": "EXTRACTED",
                        "epistemic_source": "asserted",
                        "importance_score": 0.5,
                        "never_decay": 0,
                    }
                ],
                "new_type_proposals": [],
                "narrative_chunk": "a r c",
            }
        )
    )
    mock_llm.enqueue_content("contradiction")
    await orchestrator.extract("a r c", "s1")
    memory.contradictions.detect.assert_called_once()
    memory.open_questions.add_question.assert_called()


async def test_reinforced_edge_increments_occurrence(orchestrator, memory, mock_llm):
    memory.semantic.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "old-1",
                "source": "hyungpu",
                "relation": "prefer",
                "target": "python",
                "valid_to": None,
                "confidence": "EXTRACTED",
                "type_id": "rel-type-1",
            }
        ]
    )
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [
                    {"type": "Person", "label": "hyungpu", "properties": {}, "confidence": "EXTRACTED"},
                    {"type": "Concept", "label": "python", "properties": {}, "confidence": "EXTRACTED"},
                ],
                "edges": [
                    {
                        "source": "hyungpu",
                        "relation": "prefer",
                        "target": "python",
                        "confidence": "EXTRACTED",
                        "epistemic_source": "asserted",
                        "importance_score": 0.5,
                        "never_decay": 0,
                    }
                ],
                "new_type_proposals": [],
                "narrative_chunk": "I still prefer python",
            }
        )
    )
    await orchestrator.extract("I still prefer python", "s1")
    memory.ontology.increment_occurrence.assert_called_once_with("rel-type-1")
    memory.staging.encode_edge.assert_not_called()


async def test_workspace_ask_emitted_on_high_conf_override(orchestrator, memory, mock_llm):
    async def _get_workspace(identifier):
        if identifier in {"personal", "Personal Knowledge"}:
            return {"id": "personal", "name": "Personal Knowledge"}
        if identifier in {"billing-service", "Billing Service"}:
            return {"id": "billing-service", "name": "Billing Service"}
        return None

    memory.workspace.get_workspace = _get_workspace
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [],
                "edges": [],
                "new_type_proposals": [],
                "narrative_chunk": "retry 3",
            }
        )
    )
    result = await orchestrator.extract(
        text="retry policy changes to 3",
        session_id="s1",
        comprehension={"workspace_hint": "billing-service", "confidence": 0.95},
    )
    assert result.workspace_id == "personal"
    assert any("billing service" in question.lower() for question in result.clarification_questions)


async def test_persist_never_writes_to_semantic_or_episodic_directly(
    orchestrator,
    memory,
    mock_llm,
):
    memory.semantic.get_relationships = AsyncMock(
        return_value=[
            {
                "id": "e-1",
                "source": "hyungpu",
                "relation": "use",
                "target": "python",
                "valid_to": None,
                "confidence": "EXTRACTED",
                "type_id": "rt-1",
            }
        ]
    )
    mock_llm.enqueue_content(
        json.dumps(
            {
                "nodes": [
                    {"type": "Person", "label": "hyungpu", "properties": {}, "confidence": "EXTRACTED"},
                    {"type": "Concept", "label": "go", "properties": {}, "confidence": "EXTRACTED"},
                ],
                "edges": [
                    {
                        "source": "hyungpu",
                        "relation": "use",
                        "target": "python",
                        "confidence": "EXTRACTED",
                        "epistemic_source": "asserted",
                        "importance_score": 0.5,
                        "never_decay": 0,
                    },
                    {
                        "source": "hyungpu",
                        "relation": "use",
                        "target": "go",
                        "confidence": "EXTRACTED",
                        "epistemic_source": "asserted",
                        "importance_score": 0.5,
                        "never_decay": 0,
                    },
                ],
                "new_type_proposals": [
                    {
                        "kind": "node",
                        "name": "Language",
                        "definition": "programming language",
                        "confidence": "EXTRACTED",
                        "source_snippet": "lang",
                    }
                ],
                "narrative_chunk": "now python but go going forward",
            }
        )
    )
    await orchestrator.extract("now python but go going forward", "s1")
    assert memory.staging.encode_edge.call_count >= 1
    assert memory.staging.encode.call_count >= 1
    memory.semantic.encode_edge.assert_not_called()
    memory.episodic.encode.assert_not_called()
