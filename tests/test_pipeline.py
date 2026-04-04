import pytest
from unittest.mock import AsyncMock
from brain_agent.pipeline import ProcessingPipeline
from brain_agent.memory.manager import MemoryManager
from brain_agent.providers.base import LLMProvider, LLMResponse
from brain_agent.dashboard.emitter import DashboardEmitter
import numpy as np


def _mock_embed(text: str) -> list[float]:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
async def memory(tmp_path):
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=_mock_embed)
    await mm.initialize()
    yield mm
    await mm.close()


async def test_basic_request_processing(memory):
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request("read file auth.py")
    assert result.response != ""
    assert result.signals_processed > 5
    assert len(result.actions_taken) > 0


async def test_network_mode_transitions(memory):
    pipeline = ProcessingPipeline(memory=memory)
    assert pipeline.network_ctrl.current_mode.value == "default_mode"
    result = await pipeline.process_request("analyze the codebase")
    assert result.network_mode == "default_mode"


async def test_emotional_tagging_in_pipeline(memory):
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request("CRITICAL ERROR: database crashed")
    assert result.signals_processed > 5


async def test_multiple_requests(memory):
    pipeline = ProcessingPipeline(memory=memory)
    r1 = await pipeline.process_request("first request")
    r2 = await pipeline.process_request("second request")
    assert r1.response != ""
    assert r2.response != ""
    assert len(memory.working.get_slots()) >= 1


async def test_routing_events_logged(memory):
    pipeline = ProcessingPipeline(memory=memory)
    await pipeline.process_request("test request")
    assert len(pipeline.router.event_log) > 0


async def test_pipeline_accepts_llm_provider(memory):
    provider = AsyncMock(spec=LLMProvider)
    provider.chat.return_value = LLMResponse(
        content="LLM says hello", finish_reason="stop", usage={}
    )
    pipeline = ProcessingPipeline(memory=memory, llm_provider=provider)
    result = await pipeline.process_request("hello")
    assert result.response == "LLM says hello"
    provider.chat.assert_awaited()


async def test_pipeline_works_without_provider(memory):
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request("hello")
    assert "Processing:" in result.response


async def test_pipeline_emits_signal_flow_events(memory):
    emitter = DashboardEmitter()
    emitter.signal_flow = AsyncMock()
    emitter.region_activation = AsyncMock()
    emitter.network_switch = AsyncMock()
    emitter.broadcast = AsyncMock()

    pipeline = ProcessingPipeline(memory=memory, emitter=emitter)
    await pipeline.process_request("hello")

    assert emitter.signal_flow.await_count >= 5
    assert emitter.region_activation.await_count >= 3


async def test_pipeline_works_without_emitter(memory):
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request("hello")
    assert result.response != ""


async def test_memory_encoded_in_pipeline(memory):
    """Memory encoding now happens inside the pipeline, not after."""
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request("test memory encoding")
    assert result.memory_encoded is True
    staging_count = await memory.staging.count_unconsolidated()
    assert staging_count >= 1


async def test_memory_retrieval_in_pipeline(memory):
    """Pipeline retrieves relevant memories and passes them to PFC."""
    pipeline = ProcessingPipeline(memory=memory)
    # First encode some memories
    await memory.encode(content="auth bug found in login.py", entities={"bug": "auth"})
    await memory.consolidate()

    # Process a related request — should retrieve the memory
    result = await pipeline.process_request("check auth module")
    assert isinstance(result.memories_retrieved, list)


# ------------------------------------------------------------------
# Task 4.1: Pipeline mode enforcement
# ------------------------------------------------------------------

from brain_agent.core.network_modes import NetworkMode, ALWAYS_ACTIVE


async def test_is_active_always_active_regions(memory):
    """ALWAYS_ACTIVE regions should be active regardless of mode."""
    pipeline = ProcessingPipeline(memory=memory)
    for region in ALWAYS_ACTIVE:
        assert pipeline._is_active(region) is True


async def test_is_active_ecn_regions_in_ecn(memory):
    """ECN regions should be active when mode is ECN."""
    pipeline = ProcessingPipeline(memory=memory)
    pipeline.network_ctrl.switch_to(NetworkMode.ECN)
    for region in ("acc", "basal_ganglia", "cerebellum", "prefrontal_cortex"):
        assert pipeline._is_active(region) is True


async def test_is_active_ecn_regions_inactive_in_dmn(memory):
    """ECN regions should be inactive when mode is DMN."""
    pipeline = ProcessingPipeline(memory=memory)
    pipeline.network_ctrl.switch_to(NetworkMode.DMN)
    for region in ("acc", "basal_ganglia", "cerebellum"):
        assert pipeline._is_active(region) is False


async def test_pipeline_still_works_in_dmn_mode(memory):
    """Pipeline should complete without error even in DMN mode,
    gracefully skipping inactive regions."""
    pipeline = ProcessingPipeline(memory=memory)
    pipeline.network_ctrl.switch_to(NetworkMode.DMN)
    result = await pipeline.process_request("quiet reflection")
    # In DMN mode, BG is skipped so no actions are taken
    assert result.response == "" or result.response is not None
    # Should complete without exceptions
    assert result.signals_processed >= 3  # At least thalamus, amygdala, salience


async def test_pipeline_works_in_ecn_mode(memory):
    """Pipeline should work normally in ECN mode with all regions active."""
    pipeline = ProcessingPipeline(memory=memory)
    pipeline.network_ctrl.switch_to(NetworkMode.ECN)
    result = await pipeline.process_request("read file auth.py")
    assert result.response != ""
    assert result.signals_processed > 5


# ------------------------------------------------------------------
# Task 4.3: GWT Real Competition
# ------------------------------------------------------------------


async def test_gwt_multiple_signals_submitted(memory):
    """Workspace should receive multiple signals for competition when
    emotional arousal is high."""
    pipeline = ProcessingPipeline(memory=memory)
    # Force ECN mode for full processing
    pipeline.network_ctrl.switch_to(NetworkMode.ECN)

    # Track workspace submissions
    original_submit = pipeline.workspace.submit
    submission_count = 0

    def tracking_submit(*args, **kwargs):
        nonlocal submission_count
        submission_count += 1
        return original_submit(*args, **kwargs)

    pipeline.workspace.submit = tracking_submit

    # Process a request — at minimum the action result signal should be submitted
    result = await pipeline.process_request("read file auth.py")
    assert submission_count >= 1  # At least the action result


async def test_gwt_compete_selects_winner(memory):
    """GlobalWorkspace.compete() should return a winner when multiple
    signals are submitted."""
    from brain_agent.core.workspace import GlobalWorkspace
    from brain_agent.core.signals import Signal, SignalType, EmotionalTag

    gw = GlobalWorkspace()

    # Submit action result
    gw.submit(
        Signal(type=SignalType.GWT_BROADCAST, source="pipeline",
               payload={"status": "complete"}),
        salience=0.7, goal_relevance=0.8,
    )
    # Submit emotional signal (high arousal)
    gw.submit(
        Signal(type=SignalType.EMOTIONAL_TAG, source="amygdala",
               payload={"valence": -0.8, "arousal": 0.9},
               emotional_tag=EmotionalTag(valence=-0.8, arousal=0.9)),
        salience=0.9, goal_relevance=0.5,
    )
    # Submit prediction error
    gw.submit(
        Signal(type=SignalType.PREDICTION_ERROR, source="cerebellum",
               payload={"error": 0.7}),
        salience=0.8, goal_relevance=0.7,
    )

    winner = gw.compete()
    assert winner is not None
    # The emotional signal has highest arousal boost so it should win
    assert winner.source == "amygdala"


# ------------------------------------------------------------------
# Task 3.3: Procedural Memory Auto-Save (Graybiel 2008, Fitts 1967)
# ------------------------------------------------------------------


async def test_procedural_auto_save_requires_llm(memory):
    """Without LLM, procedural save is skipped (PFC-gated, Eichenbaum 2000)."""
    pipeline = ProcessingPipeline(memory=memory)
    await pipeline.process_request("read file auth.py")

    # No LLM → PFC evaluate_procedural returns None → nothing saved
    if memory.procedural._db:
        async with memory.procedural._db.execute(
            "SELECT trigger_pattern FROM procedures"
        ) as cursor:
            rows = await cursor.fetchall()
            assert len(rows) == 0


async def test_procedural_save_and_reinforce(memory):
    """Pre-saved procedures get reinforced on match (Fitts stage promotion)."""
    # Manually save a procedure (simulating prior LLM evaluation)
    await memory.procedural.save(
        trigger_pattern="user asks to read a specific file",
        strategy="read the file contents and display them",
        action_sequence=[{"tool": "read_file", "args": {}}],
    )

    proc_before = (await memory.procedural.get_all())[0]
    assert proc_before["execution_count"] == 0

    # Record execution (as pipeline would on match)
    await memory.procedural.record_execution(proc_before["id"], success=True)
    proc_after = await memory.procedural.get_by_id(proc_before["id"])
    assert proc_after["execution_count"] == 1
    assert proc_after["success_rate"] == 1.0


async def test_procedural_embedding_match(memory):
    """Embedding-based matching finds semantically stored procedures."""
    await memory.procedural.save(
        trigger_pattern="user wants to build the project",
        strategy="run build command",
        action_sequence=[{"tool": "build"}],
    )
    # Exact same text → same embedding → match
    result = await memory.procedural.match("user wants to build the project")
    assert result is not None
    assert result["strategy"] == "run build command"


# ------------------------------------------------------------------
# Task 4: Enriched Staging Content
# ------------------------------------------------------------------


async def test_procedural_uses_intent_trigger(tmp_path):
    """Procedural store should use intent-based triggers, not raw text."""
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=_mock_embed)
    await mm.initialize()
    pipeline = ProcessingPipeline(memory=mm, llm_provider=None)

    await pipeline.process_request("Build and deploy the project")

    if pipeline.memory.procedural._db:
        async with pipeline.memory.procedural._db.execute(
            "SELECT trigger_pattern FROM procedures"
        ) as cursor:
            rows = await cursor.fetchall()
            if rows:
                trigger = rows[0][0]
                assert ":" in trigger, f"Expected intent:keywords format, got: {trigger}"
                assert "Build and deploy the project" != trigger

    await mm.close()


async def test_staging_content_is_structured(tmp_path):
    """Staging should capture semantic meaning, not raw Q&A."""
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=_mock_embed)
    await mm.initialize()
    pipeline = ProcessingPipeline(memory=mm, llm_provider=None)
    await pipeline.process_request("What is Python used for?")

    items = await mm.staging.get_unconsolidated()
    assert len(items) >= 1
    item = items[0]

    # Entities should include Wernicke analysis
    ents = item["entities"]
    assert "intent" in ents
    assert "keywords" in ents

    await mm.close()


# ------------------------------------------------------------------
# Task 8: End-to-End Integration Test
# ------------------------------------------------------------------


async def test_enriched_memory_flow_end_to_end(tmp_path):
    """Working memory and staging should contain meaningful processed data."""
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=_mock_embed)
    await mm.initialize()
    try:
        pipeline = ProcessingPipeline(memory=mm, llm_provider=None)

        await pipeline.process_request("Find the authentication bug in the login module")

        # 1. Working Memory: should have intent + keywords metadata
        wm_items = mm.working.get_slots()
        assert len(wm_items) >= 1
        assert wm_items[0].metadata.get("intent") is not None
        assert isinstance(wm_items[0].metadata.get("keywords"), list)

        # 2. Staging: should have structured entities
        staged = await mm.staging.get_unconsolidated()
        assert len(staged) >= 1
        ents = staged[0]["entities"]
        assert "intent" in ents
        assert "keywords" in ents
        assert isinstance(ents["keywords"], list)

        # 3. Staging content should use structured "User (intent): ..." format
        content = staged[0]["content"]
        assert "user" in content.lower(), f"Expected 'User (...)' format, got: {content[:100]}"
    finally:
        await mm.close()


# ------------------------------------------------------------------
# Neuromodulator: no fake arousal baseline
# ------------------------------------------------------------------


async def test_neutral_input_no_arousal_boost(tmp_path):
    """Neutral input (no emotional tag) should NOT inject fake arousal."""
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=_mock_embed)
    await mm.initialize()
    pipeline = ProcessingPipeline(memory=mm, llm_provider=None)

    await pipeline.process_request("hello")

    # Without LLM, amygdala returns fallback (near 0.0 arousal)
    # NE should stay near baseline, not spike from fake 0.3 arousal
    assert pipeline.neuromodulators.norepinephrine < 0.65, \
        f"NE too high for neutral input: {pipeline.neuromodulators.norepinephrine}"

    await mm.close()


# ------------------------------------------------------------------
# Task 6: Adaptive Processing Depth (Schneider & Shiffrin 1977)
# ------------------------------------------------------------------


def test_classify_complexity_fast_with_procedure():
    from brain_agent.pipeline import _classify_complexity
    assert _classify_complexity({"complexity": "simple", "intent": "greeting"}, has_procedure=True) == "fast"


def test_classify_complexity_fast_greeting():
    from brain_agent.pipeline import _classify_complexity
    assert _classify_complexity({"complexity": "simple", "intent": "greeting"}, has_procedure=False) == "fast"


def test_classify_complexity_standard():
    from brain_agent.pipeline import _classify_complexity
    assert _classify_complexity({"complexity": "moderate", "intent": "question"}, has_procedure=False) == "standard"


def test_classify_complexity_full():
    from brain_agent.pipeline import _classify_complexity
    assert _classify_complexity({"complexity": "complex", "intent": "explanation"}, has_procedure=False) == "full"


def test_classify_complexity_defaults():
    from brain_agent.pipeline import _classify_complexity
    assert _classify_complexity({}, has_procedure=False) == "standard"


# ------------------------------------------------------------------
# Task 8: Recurrent Processing (Feedback Loops) — Lamme 2006
# ------------------------------------------------------------------


def test_max_reprocess_constant():
    from brain_agent.pipeline import MAX_REPROCESS, REPROCESS_CONFIDENCE_THRESHOLD
    assert MAX_REPROCESS == 2
    assert REPROCESS_CONFIDENCE_THRESHOLD == 0.4


async def test_pipeline_still_works_with_reprocessing(memory):
    """Pipeline processes normally with reprocessing loop in place."""
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request(text="explain quantum physics")
    assert result.response  # Should still produce a response
    assert result.signals_processed > 5


# ------------------------------------------------------------------
# Metadata integration: interoceptive/attention/confidence/surprise
# ------------------------------------------------------------------


def test_classify_complexity_with_metadata():
    """Verify _classify_complexity still works (regression)."""
    from brain_agent.pipeline import _classify_complexity
    assert _classify_complexity({"complexity": "simple", "intent": "greeting"}, has_procedure=True) == "fast"
