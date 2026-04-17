"""Tests for Phase 6 ConsolidationEngine decay-policy resolution."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from brain_agent.memory.consolidation import ConsolidationEngine
from brain_agent.memory.episodic_store import EpisodicStore
from brain_agent.memory.forgetting import ForgettingEngine
from brain_agent.memory.hippocampal_staging import HippocampalStaging


@pytest.fixture
async def staging(tmp_path, mock_embedding):
    store = HippocampalStaging(
        db_path=str(tmp_path / "staging.db"),
        embed_fn=mock_embedding,
    )
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def episodic(tmp_path):
    store = EpisodicStore(db_path=str(tmp_path / "episodic.db"))
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def engine(staging, episodic):
    return ConsolidationEngine(
        staging=staging,
        episodic_store=episodic,
        forgetting=ForgettingEngine(),
    )


async def test_get_workspace_fallback_to_normal_when_unwired(engine):
    ws = await engine._get_workspace("anything")
    assert ws["decay_policy"] == "normal"


async def test_get_workspace_returns_store_row(engine):
    mock_store = MagicMock()
    mock_store.get_workspace = AsyncMock(
        return_value={"id": "biz", "decay_policy": "none"}
    )
    engine._workspace_store = mock_store
    ws = await engine._get_workspace("biz")
    assert ws["decay_policy"] == "none"


async def test_get_workspace_missing_row_falls_back_to_normal(engine):
    mock_store = MagicMock()
    mock_store.get_workspace = AsyncMock(return_value=None)
    engine._workspace_store = mock_store
    ws = await engine._get_workspace("nonexistent")
    assert ws["decay_policy"] == "normal"


async def test_resolve_entity_types_without_ontology_returns_empty(engine):
    types = await engine._resolve_entity_types(
        {"subject": {"type": "Requirement"}},
        workspace_id="ws",
    )
    assert types == []


async def test_resolve_entity_types_returns_ontology_rows(engine):
    mock_onto = MagicMock()
    mock_onto.resolve_node_type = AsyncMock(
        return_value={"name": "Requirement", "decay_override": "none"}
    )
    engine._ontology_store = mock_onto
    types = await engine._resolve_entity_types(
        {"subject": {"type": "Requirement"}},
        workspace_id="ws",
    )
    assert len(types) == 1
    assert types[0]["decay_override"] == "none"


async def test_get_decay_policy_most_restrictive_type_wins(engine):
    mock_ws = MagicMock()
    mock_ws.get_workspace = AsyncMock(
        return_value={"id": "personal", "decay_policy": "normal"}
    )
    mock_onto = MagicMock()
    mock_onto.resolve_node_type = AsyncMock(
        side_effect=[
            {"name": "Fact", "decay_override": "slow"},
            {"name": "Requirement", "decay_override": "none"},
        ]
    )
    engine._workspace_store = mock_ws
    engine._ontology_store = mock_onto
    policy = await engine._get_decay_policy(
        "personal",
        {"a": {"type": "Fact"}, "b": {"type": "Requirement"}},
    )
    assert policy == "none"


async def test_get_decay_policy_falls_back_to_workspace(engine):
    mock_ws = MagicMock()
    mock_ws.get_workspace = AsyncMock(
        return_value={"id": "personal", "decay_policy": "slow"}
    )
    mock_onto = MagicMock()
    mock_onto.resolve_node_type = AsyncMock(
        return_value={"name": "Generic", "decay_override": None}
    )
    engine._workspace_store = mock_ws
    engine._ontology_store = mock_onto
    policy = await engine._get_decay_policy(
        "personal",
        {"a": {"type": "Generic"}},
    )
    assert policy == "slow"


async def _stage_one(
    staging,
    *,
    content="m",
    entities=None,
    strength=1.0,
    never_decay=0,
    importance_score=0.5,
    workspace_id="personal",
    arousal=0.1,
):
    mem_id = await staging.encode(
        content=content,
        entities=entities or {},
        interaction_id=1,
        session_id="s1",
        emotional_tag={"valence": 0, "arousal": arousal},
        strength=strength,
        workspace_id=workspace_id,
        never_decay=never_decay,
        importance_score=importance_score,
    )
    return mem_id


async def test_transfer_never_decay_preserves_strength(engine, staging, episodic):
    await _stage_one(staging, strength=0.8, never_decay=1)
    await engine.consolidate()
    eps = await episodic.get_all()
    assert len(eps) == 1
    assert eps[0]["strength"] == pytest.approx(0.8, abs=1e-6)


async def test_transfer_workspace_none_preserves_strength(engine, staging, episodic):
    mock_ws = MagicMock()
    mock_ws.get_workspace = AsyncMock(
        return_value={"id": "biz", "decay_policy": "none"}
    )
    engine._workspace_store = mock_ws
    await _stage_one(staging, strength=1.0, workspace_id="biz")
    await engine.consolidate()
    eps = await episodic.get_all()
    assert eps[0]["strength"] == pytest.approx(1.0, abs=1e-6)


async def test_transfer_workspace_slow_applies_0_99(engine, staging, episodic):
    mock_ws = MagicMock()
    mock_ws.get_workspace = AsyncMock(
        return_value={"id": "ws_slow", "decay_policy": "slow"}
    )
    engine._workspace_store = mock_ws
    await _stage_one(
        staging,
        strength=1.0,
        workspace_id="ws_slow",
        importance_score=0.5,
    )
    await engine.consolidate()
    eps = await episodic.get_all()
    assert eps[0]["strength"] == pytest.approx(0.9925 * 0.9925, abs=1e-4)


async def test_transfer_workspace_normal_uses_ach_factor(engine, staging, episodic):
    mock_ws = MagicMock()
    mock_ws.get_workspace = AsyncMock(
        return_value={"id": "personal", "decay_policy": "normal"}
    )
    engine._workspace_store = mock_ws
    engine._get_ach = lambda: 0.5
    await _stage_one(staging, strength=1.0, importance_score=0.0)
    await engine.consolidate()
    eps = await episodic.get_all()
    assert eps[0]["strength"] == pytest.approx(0.95, abs=1e-4)


async def test_transfer_type_override_none_beats_workspace_normal(
    engine,
    staging,
    episodic,
):
    mock_ws = MagicMock()
    mock_ws.get_workspace = AsyncMock(
        return_value={"id": "personal", "decay_policy": "normal"}
    )
    mock_onto = MagicMock()
    mock_onto.resolve_node_type = AsyncMock(
        return_value={"name": "Requirement", "decay_override": "none"}
    )
    engine._workspace_store = mock_ws
    engine._ontology_store = mock_onto
    engine._get_ach = lambda: 0.3
    await _stage_one(
        staging,
        strength=1.0,
        entities={"topic": {"type": "Requirement"}},
    )
    await engine.consolidate()
    eps = await episodic.get_all()
    assert eps[0]["strength"] == pytest.approx(1.0, abs=1e-6)


async def test_transfer_importance_doubles_retention(engine, staging, episodic):
    mock_ws = MagicMock()
    mock_ws.get_workspace = AsyncMock(
        return_value={"id": "ws", "decay_policy": "normal"}
    )
    engine._workspace_store = mock_ws
    engine._get_ach = lambda: 1.0
    await _stage_one(
        staging,
        content="lo",
        strength=1.0,
        importance_score=0.0,
    )
    await _stage_one(
        staging,
        content="hi",
        strength=1.0,
        importance_score=1.0,
    )
    await engine.consolidate()
    eps = await episodic.get_all()
    hi = next(e for e in eps if e["content"] == "hi")
    lo = next(e for e in eps if e["content"] == "lo")
    assert lo["strength"] == pytest.approx(0.6 * 0.95, abs=0.05)
    assert hi["strength"] == pytest.approx(0.8 * 0.975, abs=0.05)


async def test_homeostatic_skips_never_decay(engine, episodic):
    await episodic.save(
        content="protected",
        context_embedding=[0.0] * 8,
        entities={},
        emotional_tag={"valence": 0, "arousal": 0.1},
        interaction_id=0,
        session_id="s1",
        strength=0.5,
        workspace_id="personal",
        never_decay=1,
        importance_score=0.5,
    )
    await engine.consolidate()
    eps = await episodic.get_all()
    assert eps[0]["strength"] == pytest.approx(0.5, abs=1e-6)


async def test_homeostatic_workspace_none_no_decay(engine, episodic):
    mock_ws = MagicMock()
    mock_ws.get_workspace = AsyncMock(
        return_value={"id": "biz", "decay_policy": "none"}
    )
    mock_ws.list_workspaces = AsyncMock(
        return_value=[{"id": "biz", "decay_policy": "none"}]
    )
    engine._workspace_store = mock_ws
    await episodic.save(
        content="biz-fact",
        context_embedding=[0.0] * 8,
        entities={},
        emotional_tag={"valence": 0, "arousal": 0.1},
        interaction_id=0,
        session_id="s1",
        strength=0.5,
        workspace_id="biz",
        never_decay=0,
        importance_score=0.5,
    )
    await engine.consolidate()
    eps = await episodic.get_all()
    assert eps[0]["strength"] == pytest.approx(0.5, abs=1e-6)


async def test_homeostatic_workspace_slow_uses_0_99(engine, episodic):
    mock_ws = MagicMock()
    mock_ws.get_workspace = AsyncMock(
        return_value={"id": "ws_slow", "decay_policy": "slow"}
    )
    mock_ws.list_workspaces = AsyncMock(
        return_value=[{"id": "ws_slow", "decay_policy": "slow"}]
    )
    engine._workspace_store = mock_ws
    await episodic.save(
        content="x",
        context_embedding=[0.0] * 8,
        entities={},
        emotional_tag={"valence": 0, "arousal": 0.1},
        interaction_id=0,
        session_id="s1",
        strength=1.0,
        workspace_id="ws_slow",
        never_decay=0,
        importance_score=0.5,
    )
    await engine.consolidate()
    eps = await episodic.get_all()
    assert eps[0]["strength"] == pytest.approx(0.9925, abs=1e-4)


async def test_homeostatic_mixed_workspaces(engine, episodic):
    mock_ws = MagicMock()

    def _get_ws(name):
        return {
            "biz": {"id": "biz", "decay_policy": "none"},
            "ws_slow": {"id": "ws_slow", "decay_policy": "slow"},
            "personal": {"id": "personal", "decay_policy": "normal"},
        }[name]

    mock_ws.get_workspace = AsyncMock(side_effect=_get_ws)
    mock_ws.list_workspaces = AsyncMock(
        return_value=[
            {"id": "biz", "decay_policy": "none"},
            {"id": "ws_slow", "decay_policy": "slow"},
            {"id": "personal", "decay_policy": "normal"},
        ]
    )
    engine._workspace_store = mock_ws

    for ws_id in ("biz", "ws_slow", "personal"):
        await episodic.save(
            content=ws_id,
            context_embedding=[0.0] * 8,
            entities={},
            emotional_tag={"valence": 0, "arousal": 0.1},
            interaction_id=0,
            session_id="s1",
            strength=1.0,
            workspace_id=ws_id,
            never_decay=0,
            importance_score=0.5,
        )
    await engine.consolidate()
    eps = {e["content"]: e for e in await episodic.get_all()}
    assert eps["biz"]["strength"] == pytest.approx(1.0, abs=1e-6)
    assert eps["ws_slow"]["strength"] == pytest.approx(0.9925, abs=1e-4)
    assert eps["personal"]["strength"] == pytest.approx(0.9625, abs=1e-4)
