"""Tests for Posterior Superior Temporal Sulcus (pSTS) — multisensory binding."""
import pytest
from brain_agent.regions.psts import PosteriorSuperiorTemporalSulcus
from brain_agent.regions.base import Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


@pytest.fixture
def psts():
    return PosteriorSuperiorTemporalSulcus()


def test_anatomy(psts):
    assert psts.name == "psts"
    assert psts.lobe == Lobe.TEMPORAL
    assert psts.hemisphere == Hemisphere.LEFT


async def test_text_only_binding(psts):
    """Text-only input should produce a basic binding."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "Hello world", "comprehension": {"intent": "greeting", "keywords": ["hello", "world"]}},
    )
    result = psts.integrate(sig)
    binding = result.metadata["multisensory_binding"]
    assert "auditory_ventral" in binding["modalities"]
    assert binding["dominant_modality"] == "auditory_ventral"
    assert binding["num_streams"] >= 1


async def test_multimodal_binding(psts):
    """Multimodal input should merge streams and report higher congruence."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={
            "text": "A cat sitting on a mat",
            "comprehension": {"intent": "statement", "keywords": ["cat", "sitting", "mat"]},
            "visual_features": {"description": "a cat on a mat", "size_bytes": 1024},
            "prosody": {"emotional_tone": "neutral"},
        },
    )
    result = psts.integrate(sig)
    binding = result.metadata["multisensory_binding"]
    assert len(binding["modalities"]) >= 3
    assert binding["congruence_score"] > 0.5
    assert "semantic" in binding["binding"]
    assert "visual" in binding["binding"]


async def test_empty_input(psts):
    """Empty signal should produce minimal binding."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={},
    )
    result = psts.integrate(sig)
    binding = result.metadata["multisensory_binding"]
    assert binding["num_streams"] == 0 or binding["dominant_modality"] == "none"


async def test_activation_scales_with_modalities(psts):
    """Activation should increase with more modalities."""
    sig_simple = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "hello", "comprehension": {"intent": "greeting"}},
    )
    psts.integrate(sig_simple)
    activation_simple = psts.activation_level

    sig_multi = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={
            "text": "hello",
            "comprehension": {"intent": "greeting"},
            "visual_features": {"size": 100},
            "prosody": {"tone": "warm"},
        },
    )
    psts.integrate(sig_multi)
    activation_multi = psts.activation_level

    assert activation_multi > activation_simple
