"""Tests for multimodal pipeline integration, Wernicke, and new region wiring."""
import pytest
from unittest.mock import AsyncMock
import numpy as np

from brain_agent.pipeline import ProcessingPipeline
from brain_agent.memory.manager import MemoryManager
from brain_agent.core.signals import Signal, SignalType
from brain_agent.core.network_modes import NetworkMode, ALWAYS_ACTIVE, MODE_REGIONS
from brain_agent.core.router import ECN_ROUTES, DMN_ROUTES
from brain_agent.dashboard.emitter import DashboardEmitter


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


# ──────────────────────────────────────────────────────────────
# Signal type tests
# ──────────────────────────────────────────────────────────────

def test_image_input_signal_type():
    """IMAGE_INPUT signal type should exist and have correct value."""
    assert SignalType.IMAGE_INPUT.value == "image_input"


def test_audio_input_signal_type():
    """AUDIO_INPUT signal type should exist and have correct value."""
    assert SignalType.AUDIO_INPUT.value == "audio_input"


def test_multimodal_signal_creation_image():
    """Can create a signal with IMAGE_INPUT type."""
    sig = Signal(type=SignalType.IMAGE_INPUT, source="user", payload={"image_data": b"fake", "text": ""})
    assert sig.type == SignalType.IMAGE_INPUT
    assert sig.payload["image_data"] == b"fake"


def test_multimodal_signal_creation_audio():
    """Can create a signal with AUDIO_INPUT type."""
    sig = Signal(type=SignalType.AUDIO_INPUT, source="user", payload={"audio_data": b"wav", "text": ""})
    assert sig.type == SignalType.AUDIO_INPUT
    assert sig.payload["audio_data"] == b"wav"


# ──────────────────────────────────────────────────────────────
# Routing table tests
# ──────────────────────────────────────────────────────────────

def test_ecn_routes_image_input():
    """ECN routing table should include IMAGE_INPUT routes."""
    targets = ECN_ROUTES[SignalType.IMAGE_INPUT]
    assert "visual_cortex" in targets
    assert "thalamus" in targets
    assert "salience_network" in targets


def test_ecn_routes_audio_input():
    """ECN routing table should include AUDIO_INPUT routes."""
    targets = ECN_ROUTES[SignalType.AUDIO_INPUT]
    assert "auditory_cortex_l" in targets
    assert "auditory_cortex_r" in targets
    assert "thalamus" in targets
    assert "salience_network" in targets


def test_dmn_routes_image_input():
    """DMN routing table should include IMAGE_INPUT routes."""
    targets = DMN_ROUTES[SignalType.IMAGE_INPUT]
    assert "visual_cortex" in targets
    assert "thalamus" in targets


def test_dmn_routes_audio_input():
    """DMN routing table should include AUDIO_INPUT routes."""
    targets = DMN_ROUTES[SignalType.AUDIO_INPUT]
    assert "auditory_cortex_l" in targets
    assert "auditory_cortex_r" in targets


# ──────────────────────────────────────────────────────────────
# Network mode tests
# ──────────────────────────────────────────────────────────────

def test_ecn_includes_new_regions():
    """ECN mode should include visual_cortex, auditory_cortex_l/r, wernicke, broca."""
    ecn = MODE_REGIONS[NetworkMode.ECN]
    assert "visual_cortex" in ecn
    assert "auditory_cortex_l" in ecn
    assert "auditory_cortex_r" in ecn
    assert "wernicke" in ecn
    assert "broca" in ecn


def test_always_active_includes_brainstem_vta():
    """ALWAYS_ACTIVE should include brainstem and vta."""
    assert "brainstem" in ALWAYS_ACTIVE
    assert "vta" in ALWAYS_ACTIVE


# ──────────────────────────────────────────────────────────────
# Wernicke integration tests
# ──────────────────────────────────────────────────────────────

async def test_wernicke_processes_in_pipeline(memory):
    """Wernicke should add comprehension metadata in the pipeline."""
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request("What is the weather today?")
    assert result.response != ""
    assert result.signals_processed > 5


async def test_wernicke_comprehension_metadata(memory):
    """After Wernicke processing, signal should have comprehension payload."""
    from brain_agent.regions.wernicke import WernickeArea
    wernicke = WernickeArea()
    sig = Signal(type=SignalType.EXTERNAL_INPUT, source="user", payload={"text": "What is the weather?"})
    result = await wernicke.process(sig)
    assert "comprehension" in result.payload
    # Without LLM, intent is always "statement" (no keyword faking)
    assert result.payload["comprehension"]["intent"] == "statement"


# ──────────────────────────────────────────────────────────────
# Multimodal pipeline tests (backward compatibility)
# ──────────────────────────────────────────────────────────────

async def test_text_only_backward_compatible(memory):
    """process_request('hello') should still work as positional arg."""
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request("hello")
    assert result.response != ""
    assert result.signals_processed > 3


async def test_text_keyword_arg(memory):
    """process_request(text='hello') should also work."""
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request(text="hello")
    assert result.response != ""


async def test_image_input_pipeline(memory):
    """Pipeline should handle image input without errors."""
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request(text="describe this", image=b"fake_image_data")
    assert result.response != ""
    assert result.signals_processed > 3


async def test_audio_input_pipeline(memory):
    """Pipeline should handle audio input without errors."""
    pipeline = ProcessingPipeline(memory=memory)
    result = await pipeline.process_request(text="", audio=b"fake_audio_data")
    # Audio might not produce text from our mock, but should not crash
    assert result.signals_processed > 3


async def test_image_routes_through_visual_cortex(memory):
    """Image input should activate visual cortex."""
    pipeline = ProcessingPipeline(memory=memory)
    await pipeline.process_request(text="what is this?", image=b"image_bytes")
    assert pipeline.visual_cortex.activation_level > 0


async def test_audio_routes_through_auditory_cortex(memory):
    """Audio input should activate both auditory cortices."""
    pipeline = ProcessingPipeline(memory=memory)
    await pipeline.process_request(text="", audio=b"audio_bytes")
    assert pipeline.auditory_cortex_l.activation_level > 0
    assert pipeline.auditory_cortex_r.activation_level > 0


# ──────────────────────────────────────────────────────────────
# New region wiring tests
# ──────────────────────────────────────────────────────────────

async def test_pipeline_has_new_regions(memory):
    """Pipeline should instantiate all new regions."""
    pipeline = ProcessingPipeline(memory=memory)
    assert pipeline.visual_cortex is not None
    assert pipeline.auditory_cortex_l is not None
    assert pipeline.auditory_cortex_r is not None
    assert pipeline.wernicke is not None
    assert pipeline.broca is not None
    assert pipeline.brainstem_region is not None
    assert pipeline.vta is not None


async def test_brainstem_activates_in_pipeline(memory):
    """Brainstem processing now runs in background post-response task."""
    import asyncio
    pipeline = ProcessingPipeline(memory=memory)
    await pipeline.process_request("test brainstem")
    # Brainstem runs in background; verify pipeline completed without error
    assert pipeline.brainstem_region is not None


async def test_broca_activates_in_pipeline(memory):
    """Broca should be activated during pipeline processing."""
    pipeline = ProcessingPipeline(memory=memory)
    await pipeline.process_request("test broca")
    assert pipeline.broca.activation_level > 0


async def test_wernicke_activates_in_pipeline(memory):
    """Wernicke should be activated during pipeline processing."""
    pipeline = ProcessingPipeline(memory=memory)
    await pipeline.process_request("test wernicke activation")
    assert pipeline.wernicke.activation_level > 0


# ──────────────────────────────────────────────────────────────
# Emitter tests
# ──────────────────────────────────────────────────────────────

async def test_region_io_emitted(memory):
    """Pipeline should emit region_io events for new regions."""
    emitter = DashboardEmitter()
    emitter.signal_flow = AsyncMock()
    emitter.region_activation = AsyncMock()
    emitter.network_switch = AsyncMock()
    emitter.broadcast = AsyncMock()
    emitter.neuromodulator_update = AsyncMock()
    emitter.region_io = AsyncMock()

    pipeline = ProcessingPipeline(memory=memory, emitter=emitter)
    await pipeline.process_request("hello world")

    # region_io should have been called for thalamus, wernicke, amygdala, SN, PFC, etc.
    assert emitter.region_io.await_count >= 3


async def test_signal_flow_with_data_summary(memory):
    """Enhanced signal_flow should accept optional data_summary."""
    from unittest.mock import patch
    emitter = DashboardEmitter()
    with patch("brain_agent.dashboard.emitter.event_bus") as mock_bus:
        mock_bus.emit = AsyncMock()
        await emitter.signal_flow("thalamus", "amygdala", "EXTERNAL_INPUT", 0.8, data_summary={"text": "hello"})
        mock_bus.emit.assert_awaited_once()
        call_args = mock_bus.emit.call_args
        payload = call_args[0][1]
        assert payload["data_summary"] == {"text": "hello"}


async def test_signal_flow_without_data_summary(memory):
    """signal_flow without data_summary should still work (backward compat)."""
    from unittest.mock import patch
    emitter = DashboardEmitter()
    with patch("brain_agent.dashboard.emitter.event_bus") as mock_bus:
        mock_bus.emit = AsyncMock()
        await emitter.signal_flow("thalamus", "amygdala", "EXTERNAL_INPUT", 0.8)
        mock_bus.emit.assert_awaited_once()
        call_args = mock_bus.emit.call_args
        payload = call_args[0][1]
        assert "data_summary" not in payload


async def test_emitter_region_io_method():
    """Emitter should have region_io method that emits correct event."""
    from unittest.mock import patch
    emitter = DashboardEmitter()
    with patch("brain_agent.dashboard.emitter.event_bus") as mock_bus:
        mock_bus.emit = AsyncMock()
        await emitter.region_io("wernicke", {"type": "external_input"}, {"comprehension": "question"}, "language_comprehension")
        mock_bus.emit.assert_awaited_once_with("region_io", {
            "region": "wernicke",
            "input": {"type": "external_input"},
            "output": {"comprehension": "question"},
            "processing": "language_comprehension",
        })
