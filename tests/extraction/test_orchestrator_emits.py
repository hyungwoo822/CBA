"""Extraction orchestrator curation event emission tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.extractor import _ExtractOutput
from brain_agent.extraction.orchestrator import ExtractionOrchestrator
from brain_agent.extraction.types import (
    TemporalResolveResult,
    TriageResult,
    ValidationResult,
)


class FakeEmitter:
    def __init__(self):
        self.calls = {
            "clarification_requested": 0,
            "contradiction_detected": 0,
            "ontology_proposal": 0,
        }

    async def clarification_requested(self, **_kwargs):
        self.calls["clarification_requested"] += 1

    async def contradiction_detected(self, **_kwargs):
        self.calls["contradiction_detected"] += 1

    async def ontology_proposal(self, **_kwargs):
        self.calls["ontology_proposal"] += 1


async def test_orchestrator_emits_on_persist(mock_llm):
    memory = MagicMock()
    memory._interaction_counter = 1
    memory.ontology.increment_occurrence = AsyncMock()
    memory.ontology.propose_node_type = AsyncMock(
        return_value={
            "id": "p1",
            "kind": "node_type",
            "proposed_name": "Foo",
            "confidence": "PROVISIONAL",
            "workspace_id": "personal",
            "source_input": "foo",
        }
    )
    memory.semantic.mark_superseded = AsyncMock()
    memory.staging.encode_edge = AsyncMock(return_value="edge-1")
    memory.staging.encode = AsyncMock(return_value="mem-1")
    memory.staging.reinforce = AsyncMock()
    memory.contradictions.detect = AsyncMock(
        return_value={
            "id": "c1",
            "workspace_id": "personal",
            "subject_node": "s",
            "key_or_relation": "is",
            "value_a": "a",
            "value_b": "b",
            "severity": "severe",
        }
    )
    memory.open_questions.add_question = AsyncMock(
        return_value={
            "id": "q1",
            "workspace_id": "personal",
            "question": "why?",
            "severity": "severe",
            "context_input": "ctx",
            "raised_by": "unknown_fact",
        }
    )
    emitter = FakeEmitter()
    orchestrator = ExtractionOrchestrator(
        memory,
        mock_llm,
        ExtractionConfig(),
        emitter=emitter,
    )

    await orchestrator._persist(
        triage=TriageResult(
            target_workspace_id="personal",
            input_kinds=["fact"],
        ),
        extracted=_ExtractOutput(
            nodes=[],
            edges=[],
            new_type_proposals=[
                {
                    "kind": "node",
                    "name": "Foo",
                    "definition": "foo",
                    "confidence": "PROVISIONAL",
                    "source_snippet": "foo",
                }
            ],
            narrative_chunk="text",
        ),
        validated=ValidationResult(
            contradictions=[
                {
                    "subject": "s",
                    "key": "is",
                    "value_a": "a",
                    "value_b": "b",
                    "severity": "severe",
                }
            ],
            open_questions=[
                {
                    "question": "why?",
                    "raised_by": "unknown_fact",
                    "severity": "severe",
                    "context_input": "ctx",
                }
            ],
        ),
        temporal=TemporalResolveResult(new_edges=[]),
        source={"id": "src1"},
        session_id="s1",
    )

    assert emitter.calls["clarification_requested"] == 1
    assert emitter.calls["contradiction_detected"] == 1
    assert emitter.calls["ontology_proposal"] == 1
