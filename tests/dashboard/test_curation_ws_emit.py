"""DashboardEmitter curation event tests."""
from __future__ import annotations

from brain_agent.dashboard.emitter import DashboardEmitter
from brain_agent.dashboard.server import event_bus


async def test_clarification_requested_emit():
    event_bus._buffer.clear()
    emitter = DashboardEmitter()
    await emitter.clarification_requested(
        question_id="q1",
        question="why?",
        severity="moderate",
        workspace_id="ws1",
        context_input="...",
        raised_by="pipeline",
    )
    recent = event_bus.get_recent(1)
    assert recent[0]["type"] == "clarification_requested"
    assert recent[0]["payload"]["question_id"] == "q1"


async def test_contradiction_detected_emit():
    event_bus._buffer.clear()
    emitter = DashboardEmitter()
    await emitter.contradiction_detected(
        contradiction_id="c1",
        subject="s",
        value_a="a",
        value_b="b",
        severity="severe",
        workspace_id="ws1",
    )
    recent = event_bus.get_recent(1)
    assert recent[0]["type"] == "contradiction_detected"
    assert recent[0]["payload"]["contradiction_id"] == "c1"


async def test_ontology_proposal_emit():
    event_bus._buffer.clear()
    emitter = DashboardEmitter()
    await emitter.ontology_proposal(
        proposal_id="p1",
        kind="node_type",
        proposed_name="Foo",
        confidence="PROVISIONAL",
        workspace_id="ws1",
        source_snippet="...",
    )
    recent = event_bus.get_recent(1)
    assert recent[0]["type"] == "ontology_proposal"
    assert recent[0]["payload"]["proposal_id"] == "p1"


async def test_resolution_emits():
    event_bus._buffer.clear()
    emitter = DashboardEmitter()
    await emitter.question_answered("q1", "ws1")
    await emitter.contradiction_resolved("c1", "A", "ws1")
    await emitter.proposal_decided("p1", "approved", "ws1")
    event_types = [event["type"] for event in event_bus.get_recent(10)]
    assert "question_answered" in event_types
    assert "contradiction_resolved" in event_types
    assert "proposal_decided" in event_types
