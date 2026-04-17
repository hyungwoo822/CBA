"""Phase 6 end-to-end smoke tests for mixed-workspace decay behavior."""

import pytest


async def test_end_to_end_mixed_workspaces(memory_manager):
    ws_biz = await memory_manager.workspace.create_workspace(
        name="Billing Service",
        decay_policy="none",
    )
    ws_slow = await memory_manager.workspace.create_workspace(
        name="Research Notes",
        decay_policy="slow",
    )
    personal = await memory_manager.workspace.get_workspace("personal")

    await memory_manager.ontology.register_node_type(
        personal["id"],
        "Requirement",
        parent_name="Statement",
        decay_override="none",
    )

    await memory_manager.staging.encode(
        content="biz-fact",
        entities={},
        interaction_id=1,
        session_id="s1",
        emotional_tag={"valence": 0, "arousal": 0.1},
        strength=1.0,
        workspace_id=ws_biz["id"],
        importance_score=0.5,
    )
    await memory_manager.staging.encode(
        content="research-note",
        entities={},
        interaction_id=2,
        session_id="s1",
        emotional_tag={"valence": 0, "arousal": 0.1},
        strength=1.0,
        workspace_id=ws_slow["id"],
        importance_score=0.5,
    )
    await memory_manager.staging.encode(
        content="personal-req",
        entities={"topic": {"type": "Requirement"}},
        interaction_id=3,
        session_id="s1",
        emotional_tag={"valence": 0, "arousal": 0.1},
        strength=1.0,
        workspace_id=personal["id"],
        importance_score=0.5,
    )
    await memory_manager.staging.encode(
        content="protected",
        entities={},
        interaction_id=4,
        session_id="s1",
        emotional_tag={"valence": 0, "arousal": 0.1},
        strength=1.0,
        workspace_id=personal["id"],
        never_decay=1,
        importance_score=0.5,
    )

    await memory_manager.consolidation.consolidate()

    eps = {e["content"]: e for e in await memory_manager.episodic.get_all()}
    assert eps["biz-fact"]["strength"] == pytest.approx(1.0, abs=1e-4)
    assert eps["research-note"]["strength"] == pytest.approx(
        0.9925 * 0.9925,
        abs=1e-3,
    )
    assert eps["personal-req"]["strength"] == pytest.approx(1.0, abs=1e-4)
    assert eps["protected"]["strength"] == pytest.approx(1.0, abs=1e-4)


async def test_edge_decay_respects_workspace_and_never_decay(memory_manager):
    ws_biz = await memory_manager.workspace.create_workspace(
        name="Biz Graph",
        decay_policy="none",
    )
    personal = await memory_manager.workspace.get_workspace("personal")

    await memory_manager.semantic.add_relationship(
        "a",
        "r",
        "b",
        weight=1.0,
        workspace_id=personal["id"],
        never_decay=0,
        importance_score=0.5,
    )
    await memory_manager.semantic.add_relationship(
        "c",
        "r",
        "d",
        weight=1.0,
        workspace_id=ws_biz["id"],
        never_decay=0,
        importance_score=0.5,
    )
    await memory_manager.semantic.add_relationship(
        "e",
        "r",
        "f",
        weight=1.0,
        workspace_id=personal["id"],
        never_decay=1,
        importance_score=0.5,
    )

    await memory_manager.semantic.decay_edge_weights(factor=0.5)
    edges = await memory_manager.semantic.get_edges()

    personal_edge = next(e for e in edges if e["source_node"] == "a")
    biz_edge = next(e for e in edges if e["source_node"] == "c")
    protected_edge = next(e for e in edges if e["source_node"] == "e")

    assert personal_edge["weight"] == pytest.approx(0.625, abs=1e-3)
    assert biz_edge["weight"] == pytest.approx(1.0, abs=1e-6)
    assert protected_edge["weight"] == pytest.approx(1.0, abs=1e-6)


async def test_dream_cycle_still_all_workspaces_after_phase6(memory_manager):
    ws_biz = await memory_manager.workspace.create_workspace(
        name="Biz Dream",
        decay_policy="none",
    )
    tracker = memory_manager.recall_tracker
    tracker._entries = {}
    tracker._loaded = True
    tracker.record(
        memory_id="m1",
        content="biz thing",
        query="q1",
        score=0.95,
        source="memory",
        workspace_id=ws_biz["id"],
    )
    tracker.record(
        memory_id="m1",
        content="biz thing",
        query="q2",
        score=0.95,
        source="memory",
        workspace_id=ws_biz["id"],
    )
    tracker.record(
        memory_id="m1",
        content="biz thing",
        query="q3",
        score=0.95,
        source="memory",
        workspace_id=ws_biz["id"],
    )
    tracker.record(
        memory_id="m1",
        content="biz thing",
        query="q7",
        score=0.95,
        source="memory",
        workspace_id=ws_biz["id"],
    )
    tracker.record(
        memory_id="m1",
        content="biz thing",
        query="q8",
        score=0.95,
        source="memory",
        workspace_id=ws_biz["id"],
    )
    tracker.record(
        memory_id="m2",
        content="personal thing",
        query="q4",
        score=0.95,
        source="memory",
        workspace_id="personal",
    )
    tracker.record(
        memory_id="m2",
        content="personal thing",
        query="q5",
        score=0.95,
        source="memory",
        workspace_id="personal",
    )
    tracker.record(
        memory_id="m2",
        content="personal thing",
        query="q6",
        score=0.95,
        source="memory",
        workspace_id="personal",
    )
    tracker.record(
        memory_id="m2",
        content="personal thing",
        query="q9",
        score=0.95,
        source="memory",
        workspace_id="personal",
    )
    tracker.record(
        memory_id="m2",
        content="personal thing",
        query="q10",
        score=0.95,
        source="memory",
        workspace_id="personal",
    )

    from brain_agent.memory.dreaming import DreamingEngine

    engine = DreamingEngine(tracker, mode="core")
    promoted = await engine.run_cycle()
    keys = {c["key"] for c in promoted}
    assert "memory:m1" in keys
    assert "memory:m2" in keys
