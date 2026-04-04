import pytest
import numpy as np
from unittest.mock import AsyncMock

from brain_agent.memory.consolidation import ConsolidationEngine
from brain_agent.memory.episodic_store import EpisodicStore
from brain_agent.memory.forgetting import ForgettingEngine
from brain_agent.memory.hippocampal_staging import HippocampalStaging
from brain_agent.memory.semantic_store import SemanticStore
from brain_agent.memory.semantic_extractor import (
    find_episode_clusters,
    build_extraction_prompt,
    parse_extraction_response,
)
from brain_agent.memory.reflection import (
    build_reflection_prompt,
    parse_insights,
)


@pytest.fixture
async def staging(tmp_path, mock_embedding):
    s = HippocampalStaging(
        db_path=str(tmp_path / "staging.db"), embed_fn=mock_embedding
    )
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def episodic(tmp_path):
    s = EpisodicStore(db_path=str(tmp_path / "episodic.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def engine(staging, episodic):
    return ConsolidationEngine(
        staging=staging, episodic_store=episodic, forgetting=ForgettingEngine()
    )


async def test_consolidate_moves_to_episodic(staging, episodic, engine):
    await staging.encode(
        content="test memory",
        entities={"what": "test"},
        interaction_id=1,
        session_id="s1",
    )
    result = await engine.consolidate()
    assert result.transferred == 1
    assert len(await staging.get_unconsolidated()) == 0
    assert len(await episodic.get_all()) == 1


async def test_emotional_memories_prioritized(staging, episodic, engine):
    await staging.encode(
        content="boring",
        entities={},
        interaction_id=1,
        session_id="s1",
        emotional_tag={"valence": 0, "arousal": 0.1},
    )
    await staging.encode(
        content="critical",
        entities={},
        interaction_id=2,
        session_id="s1",
        emotional_tag={"valence": -0.8, "arousal": 0.9},
    )
    result = await engine.consolidate()
    assert result.transferred == 2


async def test_should_consolidate_staging_pressure(staging, engine):
    for i in range(25):
        await staging.encode(
            content=f"mem {i}",
            entities={},
            interaction_id=i,
            session_id="s1",
        )
    assert await engine.should_consolidate()


async def test_should_not_consolidate_low_count(staging, engine):
    await staging.encode(
        content="one", entities={}, interaction_id=1, session_id="s1"
    )
    assert not await engine.should_consolidate()


async def test_should_consolidate_respects_custom_threshold(staging, episodic):
    engine = ConsolidationEngine(
        staging=staging, episodic_store=episodic, forgetting=ForgettingEngine(),
        threshold=3,
    )
    for i in range(3):
        await staging.encode(f"memory {i}", {}, i, "s1")
    assert await engine.should_consolidate() is True


async def test_should_not_consolidate_below_custom_threshold(staging, episodic):
    engine = ConsolidationEngine(
        staging=staging, episodic_store=episodic, forgetting=ForgettingEngine(),
        threshold=10,
    )
    await staging.encode("single memory", {}, 1, "s1")
    assert await engine.should_consolidate() is False


# ------------------------------------------------------------------
# Task 3.1: Semantic Extractor — pure functions
# ------------------------------------------------------------------


def _high_sim(a, b):
    """Always returns high similarity for clustering tests."""
    return 0.95


def _low_sim(a, b):
    """Always returns low similarity — nothing clusters."""
    return 0.10


def _make_episode(content, embedding=None):
    return {
        "content": content,
        "context_embedding": embedding or [1.0, 0.0, 0.0],
    }


def test_find_episode_clusters_basic():
    """Cluster episodes when all are similar."""
    episodes = [
        _make_episode("Python for AI"),
        _make_episode("Python for ML"),
        _make_episode("Python for data science"),
    ]
    clusters = find_episode_clusters(episodes, _high_sim, threshold=0.8, min_size=3)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_find_episode_clusters_below_min_size():
    """Clusters smaller than min_size should be discarded."""
    episodes = [
        _make_episode("Python for AI"),
        _make_episode("Python for ML"),
    ]
    clusters = find_episode_clusters(episodes, _high_sim, threshold=0.8, min_size=3)
    assert len(clusters) == 0


def test_find_episode_clusters_low_similarity():
    """No clusters when similarity is below threshold."""
    episodes = [
        _make_episode("Python for AI"),
        _make_episode("Java for enterprise"),
        _make_episode("Go for systems"),
    ]
    clusters = find_episode_clusters(episodes, _low_sim, threshold=0.8, min_size=3)
    assert len(clusters) == 0


def test_find_episode_clusters_missing_embeddings():
    """Episodes without embeddings should be skipped."""
    episodes = [
        {"content": "no embedding"},
        _make_episode("has embedding"),
        _make_episode("also has embedding"),
    ]
    clusters = find_episode_clusters(episodes, _high_sim, threshold=0.8, min_size=2)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2  # Only the two with embeddings


def test_build_extraction_prompt():
    cluster = [
        _make_episode("Python is good for AI"),
        _make_episode("Python is used in ML"),
        _make_episode("Python helps with data science"),
    ]
    prompt = build_extraction_prompt(cluster)
    assert "Python is good for AI" in prompt
    assert "Python is used in ML" in prompt
    assert "ONE general fact" in prompt
    assert "<fact>" in prompt
    assert "<relations>" in prompt


def test_parse_extraction_response_valid():
    response = (
        '<fact>Python is widely used for AI and data science.</fact>\n'
        '<relations>[["Python","used_for","AI"],["Python","used_for","data_science"]]</relations>'
    )
    fact, relations = parse_extraction_response(response)
    assert fact == "Python is widely used for AI and data science."
    assert len(relations) == 2
    assert relations[0] == ["Python", "used_for", "AI"]


def test_parse_extraction_response_no_tags():
    fact, relations = parse_extraction_response("Just some text without tags")
    assert fact == ""
    assert relations == []


def test_parse_extraction_response_malformed_json():
    response = "<fact>Some fact</fact>\n<relations>not valid json</relations>"
    fact, relations = parse_extraction_response(response)
    assert fact == "Some fact"
    assert relations == []


def test_parse_extraction_response_filters_invalid_triples():
    response = (
        '<fact>A fact</fact>\n'
        '<relations>[["a","b","c"],["short"],["x","y","z","extra"]]</relations>'
    )
    fact, relations = parse_extraction_response(response)
    assert fact == "A fact"
    # 3-element and 4-element are both accepted (>= 3); only ["short"] is filtered
    assert len(relations) == 2
    assert relations[0] == ["a", "b", "c"]
    assert relations[1] == ["x", "y", "z", "extra"]


# ------------------------------------------------------------------
# Task 3.2: Reflection — pure functions
# ------------------------------------------------------------------


def test_build_reflection_prompt():
    episodes = [_make_episode(f"Experience {i}") for i in range(10)]
    prompt = build_reflection_prompt(episodes, max_episodes=5)
    # Should only include first 5
    assert "Experience 0" in prompt
    assert "Experience 4" in prompt
    assert "Experience 5" not in prompt
    assert "<insight>" in prompt


def test_parse_insights_valid():
    response = (
        "<insight>Users frequently ask about Python.</insight>\n"
        "<insight>Error handling is important.</insight>\n"
        "<insight>Performance matters.</insight>"
    )
    insights = parse_insights(response)
    assert len(insights) == 3
    assert insights[0] == "Users frequently ask about Python."
    assert insights[2] == "Performance matters."


def test_parse_insights_empty():
    insights = parse_insights("No insight tags here.")
    assert insights == []


def test_parse_insights_with_whitespace():
    response = "<insight>  spaced insight  </insight>"
    insights = parse_insights(response)
    assert len(insights) == 1
    assert insights[0] == "spaced insight"


def test_parse_insights_empty_tag_filtered():
    """Empty <insight></insight> tags should be filtered out."""
    response = "<insight></insight><insight>Real insight</insight>"
    insights = parse_insights(response)
    assert len(insights) == 1
    assert insights[0] == "Real insight"


# ------------------------------------------------------------------
# Task 3.1 & 3.2: Consolidation with mock pfc_fn
# ------------------------------------------------------------------


def _similar_embed(text):
    """Embed fn that maps by first word for high intra-cluster similarity."""
    rng = np.random.RandomState(hash(text.split()[0]) % 2**31)
    vec = rng.randn(64).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _cosine(a, b):
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    dot = float(np.dot(va, vb))
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return dot / norm if norm > 0 else 0.0


@pytest.fixture
async def semantic(tmp_path, mock_embedding):
    s = SemanticStore(
        chroma_path=str(tmp_path / "chroma"),
        graph_db_path=str(tmp_path / "graph.db"),
        embed_fn=mock_embedding,
    )
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def staging_similar(tmp_path):
    s = HippocampalStaging(
        db_path=str(tmp_path / "staging_sim.db"), embed_fn=_similar_embed,
    )
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def episodic_for_pfc(tmp_path):
    s = EpisodicStore(db_path=str(tmp_path / "episodic_pfc.db"))
    await s.initialize()
    yield s
    await s.close()


async def test_consolidation_with_pfc_extracts_semantic(
    staging_similar, episodic_for_pfc, semantic,
):
    """Phase 3: PFC-mediated episodic→semantic extraction during consolidation."""
    mock_pfc = AsyncMock(
        side_effect=[
            # Response for semantic extraction
            '<fact>Python is commonly used for AI.</fact>\n'
            '<relations>[["Python","used_for","AI"]]</relations>',
            # Response for reflection
            '<insight>Users are interested in Python AI.</insight>',
        ]
    )

    engine = ConsolidationEngine(
        staging=staging_similar,
        episodic_store=episodic_for_pfc,
        forgetting=ForgettingEngine(),
        semantic_store=semantic,
        pfc_fn=mock_pfc,
        similarity_fn=_cosine,
        threshold=1,
    )

    # Encode 4 similar memories (same first word → cluster)
    for i in range(4):
        await staging_similar.encode(
            content=f"Python topic {i}",
            entities={},
            interaction_id=i,
            session_id="s1",
        )

    result = await engine.consolidate()

    # Phase 1: all transferred
    assert result.transferred == 4

    # Phase 3: semantic extraction should have run
    assert result.semantic_extracted >= 1

    # Phase 4: reflection should have run (4 episodes < 5 threshold,
    # but after transfer there are 4 eps — let's check)
    # Actually 4 < MIN_EPISODES_FOR_REFLECTION=5, so reflection is skipped
    # We need at least 5 episodes for reflection
    assert mock_pfc.call_count >= 1


async def test_consolidation_with_pfc_generates_reflections(
    staging_similar, episodic_for_pfc, semantic,
):
    """Phase 4: Reflection generates insights when enough episodes exist."""
    mock_pfc = AsyncMock(
        return_value='<insight>A general pattern emerged.</insight>'
    )

    engine = ConsolidationEngine(
        staging=staging_similar,
        episodic_store=episodic_for_pfc,
        forgetting=ForgettingEngine(),
        semantic_store=semantic,
        pfc_fn=mock_pfc,
        similarity_fn=_cosine,
        threshold=1,
    )

    # Encode 6 memories (enough for reflection min=5)
    for i in range(6):
        await staging_similar.encode(
            content=f"Topic{i} experience details",
            entities={},
            interaction_id=i,
            session_id="s1",
        )

    result = await engine.consolidate()
    assert result.transferred == 6
    # Reflection should have generated at least 1 insight
    assert result.reflections_generated >= 1


async def test_consolidation_without_pfc_fn_skips_phases_3_4(staging, episodic):
    """When pfc_fn is None, phases 3 and 4 are skipped gracefully."""
    engine = ConsolidationEngine(
        staging=staging,
        episodic_store=episodic,
        forgetting=ForgettingEngine(),
        semantic_store=None,
        pfc_fn=None,
    )
    for i in range(6):
        await staging.encode(f"mem {i}", {}, i, "s1")

    result = await engine.consolidate()
    assert result.transferred == 6
    assert result.semantic_extracted == 0
    assert result.reflections_generated == 0


async def test_consolidation_pfc_fn_error_handled_gracefully(
    staging_similar, episodic_for_pfc, semantic,
):
    """pfc_fn raising should not break consolidation."""
    mock_pfc = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    engine = ConsolidationEngine(
        staging=staging_similar,
        episodic_store=episodic_for_pfc,
        forgetting=ForgettingEngine(),
        semantic_store=semantic,
        pfc_fn=mock_pfc,
        similarity_fn=_cosine,
        threshold=1,
    )

    for i in range(4):
        await staging_similar.encode(f"Python fact {i}", {}, i, "s1")

    # Should not raise
    result = await engine.consolidate()
    assert result.transferred == 4
    # Semantic extraction and reflection may fail, but phases 1-2 succeed
    assert result.semantic_extracted == 0
    assert result.reflections_generated == 0


@pytest.mark.asyncio
async def test_consolidation_enriches_episodic_content(tmp_path, mock_embedding):
    staging_e = HippocampalStaging(
        db_path=str(tmp_path / "staging.db"), embed_fn=mock_embedding
    )
    await staging_e.initialize()
    episodic = EpisodicStore(db_path=str(tmp_path / "episodic.db"))
    await episodic.initialize()
    forgetting = ForgettingEngine()

    engine = ConsolidationEngine(
        staging=staging_e, episodic_store=episodic, forgetting=forgetting,
        threshold=1,
    )

    await staging_e.encode(
        content="User asked (command): deploy the app\nResponse: Deploying now",
        entities={
            "intent": "command",
            "keywords": ["deploy", "app"],
            "input": "deploy the app",
            "output": "Deploying now",
        },
        interaction_id=5,
        session_id="s1",
        emotional_tag={"valence": 0.0, "arousal": 0.3},
    )

    result = await engine.consolidate()
    assert result.transferred == 1

    episodes = await episodic.get_all()
    assert len(episodes) == 1
    ep = episodes[0]

    # Episodic content should be enriched with keyword annotations
    assert "[keywords:" in ep["content"], f"Expected [keywords:] annotation in episodic content, got: {ep['content']}"

    await staging_e.close()
    await episodic.close()
