from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from brain_agent.regions.auditory_cortex import AuditoryCortexLeft, AuditoryCortexRight
from brain_agent.regions.base import Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


@pytest.fixture
def left():
    return AuditoryCortexLeft()


@pytest.fixture
def right():
    return AuditoryCortexRight()


# ── Left hemisphere (speech/language) ────────────────────────────────

def test_left_anatomy(left):
    assert left.lobe == Lobe.TEMPORAL
    assert left.hemisphere == Hemisphere.LEFT
    assert left.position.x == -35
    assert left.position.y == -10
    assert left.position.z == 10


async def test_left_extracts_transcript(left):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"audio_data": b"audio", "transcript": "hello world"},
    )
    result = await left.process(sig)
    assert result.payload["text"] == "hello world"
    assert result.payload["modality"] == "audio"


async def test_left_passthrough_no_audio(left):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "hello"},
    )
    result = await left.process(sig)
    assert result is sig
    assert result.payload.get("modality") != "audio"


async def test_left_no_transcript(left):
    """Without llm_provider, raw audio with no transcript produces no text."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"audio_data": b"audio"},
    )
    result = await left.process(sig)
    assert result.payload["modality"] == "audio"
    assert "text" not in result.payload


async def test_left_no_transcript_activation(left):
    """Activation is 0.4 when no transcript is available."""
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"audio_data": b"audio"},
    )
    await left.process(sig)
    assert left.activation_level == pytest.approx(0.4)


async def test_left_stt_transcription():
    """With llm_provider, raw audio bytes trigger Whisper STT."""
    mock_provider = MagicMock()
    region = AuditoryCortexLeft(llm_provider=mock_provider)

    mock_response = MagicMock()
    mock_response.text = "transcribed text"

    mock_litellm = MagicMock()
    mock_litellm.atranscription = AsyncMock(return_value=mock_response)

    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        sig = Signal(
            type=SignalType.EXTERNAL_INPUT,
            source="thalamus",
            payload={"audio_data": b"raw audio bytes"},
        )
        result = await region.process(sig)

    assert result.payload["text"] == "transcribed text"
    assert result.payload["transcript"] == "transcribed text"
    assert result.payload["modality"] == "audio"
    assert region.activation_level == pytest.approx(0.7)


async def test_left_stt_failure_graceful():
    """When STT fails, processing continues without transcript."""
    mock_provider = MagicMock()
    region = AuditoryCortexLeft(llm_provider=mock_provider)

    mock_litellm = MagicMock()
    mock_litellm.atranscription = AsyncMock(side_effect=RuntimeError("API error"))

    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        sig = Signal(
            type=SignalType.EXTERNAL_INPUT,
            source="thalamus",
            payload={"audio_data": b"raw audio bytes"},
        )
        result = await region.process(sig)

    assert result.payload["modality"] == "audio"
    assert "text" not in result.payload
    assert region.activation_level == pytest.approx(0.4)


async def test_left_activation(left):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"audio_data": b"audio", "transcript": "test"},
    )
    await left.process(sig)
    assert left.activation_level == pytest.approx(0.7)


# ── Right hemisphere (prosody/emotional tone) ────────────────────────

def test_right_anatomy(right):
    assert right.lobe == Lobe.TEMPORAL
    assert right.hemisphere == Hemisphere.RIGHT
    assert right.position.x == 35
    assert right.position.y == -10
    assert right.position.z == 10


async def test_right_extracts_emotional_tone(right):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"audio_data": b"audio", "emotional_tone": "angry", "stress_level": 0.8},
    )
    result = await right.process(sig)
    assert result.payload["emotional_tone"] == "angry"
    assert result.payload["stress_level"] == pytest.approx(0.8)
    assert result.payload["modality"] == "audio"


async def test_right_defaults_neutral(right):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"audio_data": b"audio"},
    )
    result = await right.process(sig)
    assert result.payload["emotional_tone"] == "neutral"
    assert result.payload["stress_level"] == pytest.approx(0.0)


async def test_right_clamps_stress_level(right):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"audio_data": b"audio", "stress_level": 1.5},
    )
    result = await right.process(sig)
    assert result.payload["stress_level"] == pytest.approx(1.0)


async def test_right_passthrough_no_audio(right):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"text": "hello"},
    )
    result = await right.process(sig)
    assert result is sig


async def test_right_audio_features(right):
    """Right hemisphere computes basic audio features from raw bytes."""
    data = b"\x00" * 32000
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"audio_data": data},
    )
    result = await right.process(sig)
    assert result.payload["audio_features"]["size_bytes"] == 32000
    assert result.payload["audio_features"]["duration_estimate"] == 2.0


async def test_right_activation_scales_with_stress(right):
    sig = Signal(
        type=SignalType.EXTERNAL_INPUT,
        source="thalamus",
        payload={"audio_data": b"audio", "stress_level": 1.0},
    )
    await right.process(sig)
    assert right.activation_level == pytest.approx(0.9)
