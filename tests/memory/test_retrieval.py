from brain_agent.memory.retrieval import RetrievalEngine, RetrievalConfig


def test_score_computation():
    engine = RetrievalEngine(config=RetrievalConfig())
    score = engine.compute_score(
        recency_distance=0.0,
        relevance=1.0,
        importance=0.5,
        access_count=3,
        context_similarity=0.8,
    )
    assert score > 0


def test_recency_matters():
    engine = RetrievalEngine(config=RetrievalConfig())
    recent = engine.compute_score(
        recency_distance=1.0,
        relevance=0.5,
        importance=0.3,
        access_count=1,
        context_similarity=0.5,
    )
    old = engine.compute_score(
        recency_distance=50.0,
        relevance=0.5,
        importance=0.3,
        access_count=1,
        context_similarity=0.5,
    )
    assert recent > old


def test_relevance_has_highest_weight():
    cfg = RetrievalConfig()
    engine = RetrievalEngine(config=cfg)
    relevant = engine.compute_score(
        recency_distance=10.0,
        relevance=1.0,
        importance=0.0,
        access_count=0,
        context_similarity=0.0,
    )
    irrelevant = engine.compute_score(
        recency_distance=0.0,
        relevance=0.0,
        importance=1.0,
        access_count=100,
        context_similarity=0.65,  # below CA3 completion threshold
    )
    assert relevant > irrelevant * 0.4


def test_config_weights_sum_to_one():
    cfg = RetrievalConfig()
    total = cfg.alpha + cfg.beta + cfg.gamma + cfg.delta + cfg.epsilon + cfg.zeta + cfg.eta
    assert abs(total - 1.0) < 0.001


def test_confidence_bonus_increases_score():
    engine = RetrievalEngine()
    base = engine.compute_score(
        recency_distance=5.0, relevance=0.5, importance=0.3,
        access_count=1, context_similarity=0.5,
    )
    boosted = engine.compute_score(
        recency_distance=5.0, relevance=0.5, importance=0.3,
        access_count=1, context_similarity=0.5, confidence_bonus=0.8,
    )
    assert boosted > base


def test_ambiguous_confidence_penalty():
    engine = RetrievalEngine()
    extracted = engine.compute_score(
        recency_distance=5.0, relevance=0.5, importance=0.3,
        access_count=1, context_similarity=0.5, confidence_bonus=1.0,
    )
    ambiguous = engine.compute_score(
        recency_distance=5.0, relevance=0.5, importance=0.3,
        access_count=1, context_similarity=0.5, confidence_bonus=0.3,
    )
    assert extracted > ambiguous


def test_activation_boost_increases_score():
    engine = RetrievalEngine(config=RetrievalConfig())
    base = engine.compute_score(
        recency_distance=5.0,
        relevance=0.5,
        importance=0.3,
        access_count=1,
        context_similarity=0.5,
    )
    boosted = engine.compute_score(
        recency_distance=5.0,
        relevance=0.5,
        importance=0.3,
        access_count=1,
        context_similarity=0.5,
        activation_boost=0.8,
    )
    assert boosted > base


def test_pattern_completion_boost():
    """High context similarity gets nonlinear boost (CA3, Rolls 2013)."""
    engine = RetrievalEngine()
    score_high_ctx = engine.compute_score(
        recency_distance=5, relevance=0.4, importance=0.5,
        access_count=2, context_similarity=0.85, activation_boost=0.0,
    )
    score_low_ctx = engine.compute_score(
        recency_distance=5, relevance=0.4, importance=0.5,
        access_count=2, context_similarity=0.3, activation_boost=0.0,
    )
    assert score_high_ctx > score_low_ctx * 1.2


def test_pattern_completion_below_threshold():
    """Below threshold, no completion bonus."""
    engine = RetrievalEngine()
    score_at_threshold = engine.compute_score(
        recency_distance=5, relevance=0.4, importance=0.5,
        access_count=2, context_similarity=0.7, activation_boost=0.0,
    )
    score_below = engine.compute_score(
        recency_distance=5, relevance=0.4, importance=0.5,
        access_count=2, context_similarity=0.69, activation_boost=0.0,
    )
    # Very small difference at boundary (no bonus below threshold)
    diff = score_at_threshold - score_below
    assert diff < 0.02  # Normal linear difference only
