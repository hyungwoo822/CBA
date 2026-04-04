import pytest
from brain_agent.regions.angular_gyrus import AngularGyrus
from brain_agent.core.signals import Signal, SignalType


@pytest.fixture
def ag():
    return AngularGyrus()


# ── Anatomy ──────────────────────────────────────────────────────────

def test_anatomy(ag):
    assert ag.name == "angular_gyrus"
    assert ag.lobe.value == "parietal"
    assert ag.hemisphere.value == "left"
    assert ag.position.x == -35
    assert ag.position.y == -25
    assert ag.position.z == 30


# ── Single modality passthrough ─────────────────────────────────────

async def test_text_only_signal(ag):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "The quick brown fox jumps over the lazy dog"},
    )
    result = await ag.process(sig)
    assert result is not None

    si = result.metadata["semantic_integration"]
    assert si["modalities_present"] == ["text"]
    assert "text_comprehension" in si["cross_modal_binding"]
    assert si["cross_modal_binding"]["text_comprehension"]["word_count"] == 9
    assert si["integration_confidence"] > 0


async def test_no_modality_passthrough(ag):
    sig = Signal(
        type=SignalType.PLAN,
        source="pfc",
        payload={"goal": "do stuff"},
    )
    result = await ag.process(sig)
    assert result is sig
    assert "semantic_integration" not in result.metadata
    assert ag.activation_level == pytest.approx(0.05)


# ── Multi-modal integration ─────────────────────────────────────────

async def test_text_plus_visual(ag):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "Look at this chart"},
        metadata={"visual_features": {"objects": ["bar_chart"], "colors": ["blue"]}},
    )
    result = await ag.process(sig)

    si = result.metadata["semantic_integration"]
    assert sorted(si["modalities_present"]) == ["text", "visual"]
    binding = si["cross_modal_binding"]
    assert "text_comprehension" in binding
    assert "visual_semantics" in binding
    assert binding["visual_semantics"]["objects"] == ["bar_chart"]
    assert si["integration_confidence"] > 0.5


async def test_text_plus_auditory(ag):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "I am happy"},
        metadata={"prosody": {"pitch": "high", "tempo": "fast"}},
    )
    result = await ag.process(sig)

    si = result.metadata["semantic_integration"]
    assert sorted(si["modalities_present"]) == ["auditory", "text"]
    binding = si["cross_modal_binding"]
    assert "auditory_semantics" in binding
    assert binding["auditory_semantics"]["pitch"] == "high"


async def test_three_modalities(ag):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "Watch and listen"},
        metadata={
            "visual_features": {"scene": "outdoor"},
            "prosody": {"pitch": "neutral"},
        },
    )
    result = await ag.process(sig)

    si = result.metadata["semantic_integration"]
    assert sorted(si["modalities_present"]) == ["auditory", "text", "visual"]
    assert si["integration_confidence"] > 0.7


# ── Confidence scoring ──────────────────────────────────────────────

async def test_confidence_increases_with_modalities(ag):
    sig_text = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "hello"},
    )
    r1 = await ag.process(sig_text)
    conf_text = r1.metadata["semantic_integration"]["integration_confidence"]

    sig_multi = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "hello"},
        metadata={"visual_features": {"data": True}},
    )
    r2 = await ag.process(sig_multi)
    conf_multi = r2.metadata["semantic_integration"]["integration_confidence"]

    assert conf_multi > conf_text


# ── Activation level ────────────────────────────────────────────────

async def test_activation_higher_for_multimodal(ag):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "check image"},
        metadata={"visual_features": {"scene": "indoor"}},
    )
    await ag.process(sig)
    assert ag.activation_level > 0.5
